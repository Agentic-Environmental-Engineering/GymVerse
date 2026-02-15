from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class CodeFunctionInferenceEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            "function_tier": (1, 4),        # Higher tier enables more complex function families/compositions → harder
            "input_size": (3, 25),          # Target input size (string length or int magnitude proxy) → larger/harder
            "alphabet_size": (5, 26),       # String alphabet size → larger alphabet increases hypothesis space → harder
            "test_budget": (8, 2),          # REVERSED: fewer tests → harder
            "hint_budget": (2, 0),          # REVERSED: fewer hints → harder
        }

        # Variance per parameter
        self.param_variance = {
            "function_tier": 0,      # small discrete range
            "input_size": 3,         # ~±3 over a 22-range (~14%)
            "alphabet_size": 3,      # ~±3 over a 21-range (~14%)
            "test_budget": 1,        # integer count, ±1
            "hint_budget": 0,        # tiny range, keep fixed for stability
        }

        # Placeholder evolvable attributes
        self.function_tier: int = 1
        self.input_size: int = 3
        self.alphabet_size: int = 5
        self.test_budget: int = 8
        self.hint_budget: int = 2

        # Other state
        self.turn_count: int = 0
        self.input_type: str = "str"  # 'str' or 'int'
        self.function_desc: Dict[str, Any] = {}
        self.target_input: Any = None
        self.correct_output: Any = None
        self.tests_left: int = 0
        self.hints_left: int = 0
        self.test_history: List[Tuple[str, str]] = []
        self.hints_used: int = 0
        self.last_submission: Optional[str] = None
        self.last_action_kind: Optional[str] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
            # Clamp respecting reversed or normal
            lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are given a hidden pure function f that maps an input to an output. Your task is to find f(target_input).\n"
            "You may run black-box tests on inputs of the same type as the target to observe f(input). You have limited test and hint budgets.\n"
            "Actions (use exactly one per turn, in \\boxed{...}):\n"
            "- test_str:<text>    → run a test with the given string (for string-type tasks)\n"
            "- test_int:<number>  → run a test with the given integer (for integer-type tasks)\n"
            "- test:<value>       → shorthand: if the task is string, value is text; if integer, value is a number\n"
            "- hint               → receive a clue (consumes 1 hint)\n"
            "- submit:<answer>    → submit your final answer for the target input. If correct → success; else → failure\n"
            "Rules:\n"
            "- You must use \\boxed{...} format. Malformed inputs terminate with a penalty.\n"
            "- test_* actions consume 1 test. Exceeding budgets or using wrong type is a protocol violation and may terminate.\n"
            "- Intermediate actions have 0 reward. Correct submission yields 1.0, incorrect submission -1.0.\n"
            f"Example actions: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        # Show target and summary to keep the agent oriented
        hist_lines = []
        recent = self.test_history[-3:] if len(self.test_history) > 3 else self.test_history
        for q, r in recent:
            hist_lines.append(f"- test({q}) -> {r}")
        hist_text = "\n".join(hist_lines) if hist_lines else "(no tests yet)"
        input_type = self.input_type
        return (
            f"TURN={self.turn_count} | tests_left={self.tests_left} | hints_left={self.hints_left}\n"
            f"Target input type: {input_type}\n"
            f"Target input: {self._render_value(self.target_input)}\n"
            f"Recent tests:\n{hist_text}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.tests_left = self.test_budget
        self.hints_left = self.hint_budget
        self.test_history = []
        self.hints_used = 0
        self.last_submission = None
        self.last_action_kind = None

        self._sample_function()
        self.target_input = self._generate_target_input()
        self.correct_output = self._apply_function(self.target_input)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        self.last_action_kind = None

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, INVALID ACTION FORMAT. Use \\boxed{{...}} with a valid command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        self.last_action_kind = cmd

        # Unsupported command
        if cmd not in {"test", "hint", "submit"}:
            obs = f"At turn {self.turn_count}, UNSUPPORTED ACTION '{cmd}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # test action
        if cmd == "test":
            # type check and budget
            if self.tests_left <= 0:
                obs = f"At turn {self.turn_count}, PROTOCOL VIOLATION: no tests left."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            test_type = parsed.get("type")
            val = parsed.get("value")
            # Validate type
            if self.input_type == "str" and test_type != "str":
                obs = f"At turn {self.turn_count}, PROTOCOL VIOLATION: expected string test."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if self.input_type == "int" and test_type != "int":
                obs = f"At turn {self.turn_count}, PROTOCOL VIOLATION: expected integer test."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            self.tests_left -= 1
            try:
                out_val = self._apply_function(val)
            except Exception as e:
                obs = f"At turn {self.turn_count}, INTERNAL ERROR during test."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            q_render = self._render_value(val)
            r_render = self._render_value(out_val)
            self.test_history.append((q_render, r_render))
            obs = f"TEST RESULT: f({q_render}) = {r_render}. tests_left={self.tests_left}, hints_left={self.hints_left}."
            # Continue
            if self.turn_count >= self.max_turns:
                obs_timeout = f"{obs} TIMEOUT: Reached max turns ({self.max_turns})."
                return obs_timeout, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        # hint action
        if cmd == "hint":
            if self.hints_left <= 0:
                obs = f"At turn {self.turn_count}, PROTOCOL VIOLATION: no hints left."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.hints_left -= 1
            self.hints_used += 1
            hint_text = self._generate_hint(self.hints_used)
            obs = f"HINT: {hint_text} (hints_left={self.hints_left})."
            if self.turn_count >= self.max_turns:
                obs_timeout = f"{obs} TIMEOUT: Reached max turns ({self.max_turns})."
                return obs_timeout, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        # submit action
        if cmd == "submit":
            answer = parsed.get("value")
            self.last_submission = self._render_value(answer)
            correct = self._values_equal(answer, self.correct_output)
            if correct:
                obs = "SUCCESS: Correct submission."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"FAILED SUBMISSION: Your answer {self._render_value(answer)} is incorrect."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        # Fallback (should not reach)
        obs = "UNEXPECTED STATE."
        return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{([\s\S]+?)\}', re.IGNORECASE)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()

        # Normalize
        content_lower = content.lower()

        # hint
        if content_lower == "hint":
            return {"cmd": "hint"}

        # submit:<value>
        if content_lower.startswith("submit:"):
            payload = content[len("submit:"):].strip()
            if self.input_type == "int":
                # parse int
                if re.fullmatch(r'[-+]?\d+', payload):
                    return {"cmd": "submit", "value": int(payload)}
                else:
                    # Try to extract integer inside
                    m = re.search(r'[-+]?\d+', payload)
                    if m:
                        return {"cmd": "submit", "value": int(m.group(0))}
                    else:
                        return {"cmd": "submit", "value": payload}  # will be judged incorrect
            else:
                # string task
                return {"cmd": "submit", "value": payload}

        # test_str:<text>
        if content_lower.startswith("test_str:"):
            payload = content[len("test_str:"):].strip()
            return {"cmd": "test", "type": "str", "value": payload}

        # test_int:<num>
        if content_lower.startswith("test_int:"):
            payload = content[len("test_int:"):].strip()
            if re.fullmatch(r'[-+]?\d+', payload):
                return {"cmd": "test", "type": "int", "value": int(payload)}
            else:
                return None

        # test:<value> -> infer by current type
        if content_lower.startswith("test:"):
            payload = content[len("test:"):].strip()
            if self.input_type == "int":
                if re.fullmatch(r'[-+]?\d+', payload):
                    return {"cmd": "test", "type": "int", "value": int(payload)}
                else:
                    return None
            else:
                return {"cmd": "test", "type": "str", "value": payload}

        # Unknown
        return {"cmd": "unsupported"}

    def sample_random_action(self) -> str:
        # Provide an example based on current input type and budgets
        if self.input_type == "int":
            ex = random.randint(0, 9)
            return f"\\boxed{{test_int:{ex}}}"
        else:
            letters = "abcdefghijklmnopqrstuvwxyz"[:max(3, min(self.alphabet_size, 26))]
            s = "".join(random.choice(letters) for _ in range(3))
            return f"\\boxed{{test_str:{s}}}"

    # ============ Domain helpers ============

    def _render_value(self, v: Any) -> str:
        if isinstance(v, str):
            return f'"{v}"'
        return str(v)

    def _values_equal(self, a: Any, b: Any) -> bool:
        # strict equality; for strings exact match; for ints exact
        return a == b

    def _sample_function(self):
        # Define families per tier
        # Each family returns a dict with: kind ('str'/'int'), name, params, and 'ops' (list of ops to compose) or single.
        tier = int(self.function_tier)
        # Candidate families
        families = []

        # Tier 1 - simple transforms
        families.extend([
            ("int_add_k", "int"),
            ("int_mul_k", "int"),
            ("str_reverse", "str"),
            ("str_upper", "str"),
        ])
        if tier >= 2:
            families.extend([
                ("int_affine", "int"),
                ("str_remove_vowels", "str"),
                ("str_rotate_k", "str"),
            ])
        if tier >= 3:
            families.extend([
                ("str_caesar_k", "str"),
                ("int_reverse_digits", "int"),
                ("str_sort_chars", "str"),
                ("str_duplicate_chars", "str"),
            ])
        if tier >= 4:
            families.extend([
                ("compose_str_caesar_reverse", "str"),
                ("compose_int_reverse_add", "int"),
                ("compose_str_rotate_remove_vowels", "str"),
                ("compose_int_affine_reverse", "int"),
            ])

        fam, kind = random.choice(families)
        desc: Dict[str, Any] = {"kind": kind, "name": fam, "params": {}}

        if fam == "int_add_k":
            desc["params"]["k"] = random.randint(1, 9)
        elif fam == "int_mul_k":
            desc["params"]["k"] = random.randint(2, 5)
        elif fam == "int_affine":
            desc["params"]["a"] = random.choice([2, 3, 4])
            desc["params"]["b"] = random.randint(1, 9)
        elif fam == "str_rotate_k":
            desc["params"]["k"] = random.randint(1, 5)
        elif fam == "str_caesar_k":
            desc["params"]["k"] = random.randint(1, 5)
        elif fam == "str_duplicate_chars":
            pass
        elif fam == "str_remove_vowels":
            pass
        elif fam == "str_reverse":
            pass
        elif fam == "str_upper":
            pass
        elif fam == "int_reverse_digits":
            pass
        elif fam == "str_sort_chars":
            pass
        elif fam == "compose_str_caesar_reverse":
            desc["params"]["k"] = random.randint(1, 5)
        elif fam == "compose_int_reverse_add":
            desc["params"]["k"] = random.randint(1, 9)
        elif fam == "compose_str_rotate_remove_vowels":
            desc["params"]["k"] = random.randint(1, 5)
        elif fam == "compose_int_affine_reverse":
            desc["params"]["a"] = random.choice([2, 3, 4])
            desc["params"]["b"] = random.randint(1, 9)

        self.function_desc = desc
        self.input_type = kind

    def _generate_target_input(self) -> Any:
        if self.input_type == "str":
            L = max(1, self.input_size)
            # clamp to a reasonable length
            L = min(40, L)
            alphabet = "abcdefghijklmnopqrstuvwxyz"[:max(3, min(self.alphabet_size, 26))]
            # To ensure some variability, allow repeated chars
            s = "".join(random.choice(alphabet) for _ in range(L))
            return s
        else:
            # integer input: map input_size to digits 1..6
            digits = max(1, min(6, 1 + self.input_size // 5))
            lo = 0
            hi = 10 ** digits - 1
            return random.randint(lo, hi)

    def _apply_function(self, x: Any) -> Any:
        d = self.function_desc
        name = d["name"]
        params = d["params"]

        def str_reverse(s: str) -> str:
            return s[::-1]

        def str_upper(s: str) -> str:
            return s.upper()

        def str_remove_vowels(s: str) -> str:
            return "".join(ch for ch in s if ch.lower() not in "aeiou")

        def str_rotate_k(s: str, k: int) -> str:
            if len(s) == 0:
                return s
            k = k % len(s)
            return s[k:] + s[:k]

        def str_caesar_k(s: str, k: int) -> str:
            out = []
            for ch in s:
                if 'a' <= ch <= 'z':
                    idx = ord(ch) - ord('a')
                    out.append(chr(ord('a') + (idx + k) % 26))
                elif 'A' <= ch <= 'Z':
                    idx = ord(ch) - ord('A')
                    out.append(chr(ord('A') + (idx + k) % 26))
                else:
                    out.append(ch)
            return "".join(out)

        def str_duplicate_chars(s: str) -> str:
            return "".join(ch * 2 for ch in s)

        def str_sort_chars(s: str) -> str:
            return "".join(sorted(s))

        def int_add_k(n: int, k: int) -> int:
            return n + k

        def int_mul_k(n: int, k: int) -> int:
            return n * k

        def int_affine(n: int, a: int, b: int) -> int:
            return a * n + b

        def int_reverse_digits(n: int) -> int:
            s = str(abs(n))
            rev = s[::-1].lstrip("0") or "0"
            val = int(rev)
            return val if n >= 0 else -val

        # Dispatch
        if name == "str_reverse":
            return str_reverse(x)
        if name == "str_upper":
            return str_upper(x)
        if name == "str_remove_vowels":
            return str_remove_vowels(x)
        if name == "str_rotate_k":
            return str_rotate_k(x, params["k"])
        if name == "str_caesar_k":
            return str_caesar_k(x, params["k"])
        if name == "str_duplicate_chars":
            return str_duplicate_chars(x)
        if name == "str_sort_chars":
            return str_sort_chars(x)

        if name == "int_add_k":
            return int_add_k(x, params["k"])
        if name == "int_mul_k":
            return int_mul_k(x, params["k"])
        if name == "int_affine":
            return int_affine(x, params["a"], params["b"])
        if name == "int_reverse_digits":
            return int_reverse_digits(x)

        # Compositions
        if name == "compose_str_caesar_reverse":
            return str_reverse(str_caesar_k(x, params["k"]))
        if name == "compose_int_reverse_add":
            return int_add_k(int_reverse_digits(x), params["k"])
        if name == "compose_str_rotate_remove_vowels":
            return str_remove_vowels(str_rotate_k(x, params["k"]))
        if name == "compose_int_affine_reverse":
            return int_reverse_digits(int_affine(x, params["a"], params["b"]))

        # Unknown
        raise ValueError("Unknown function")

    def _generate_hint(self, hint_idx: int) -> str:
        d = self.function_desc
        name = d["name"]
        kind = d["kind"]
        params = d["params"]

        # Base category hints
        category = "numeric" if kind == "int" else "text"
        composition = name.startswith("compose_")

        # Tiered hinting
        if hint_idx == 1:
            if composition:
                return f"The function is a composition of two {category} operations."
            else:
                return f"The function is a single-step {category} transformation."
        elif hint_idx == 2:
            # family hint
            if "caesar" in name:
                return "It shifts letters by a fixed offset."
            if "rotate" in name:
                return "It rotates the string by a fixed number of positions."
            if "reverse" in name and kind == "str":
                return "It reverses the order of characters."
            if "reverse" in name and kind == "int":
                return "It reverses the decimal digits."
            if "remove_vowels" in name:
                return "It removes vowels."
            if "upper" in name:
                return "It changes character casing."
            if "duplicate" in name:
                return "It duplicates each character."
            if "sort_chars" in name:
                return "It sorts characters."
            if "affine" in name:
                return "It applies a linear arithmetic transformation."
            if "mul" in name:
                return "It multiplies by a constant."
            if "add" in name:
                return "It adds a constant."
            return "It is a common basic transformation."
        else:
            # slightly more specific without revealing params
            if "caesar" in name:
                return "The offset is small (between 1 and 5)."
            if "rotate" in name:
                return "The rotation count is small (between 1 and 5)."
            if "affine" in name:
                return "The multiplier is a small integer (2–4)."
            return "Try contrasting simple inputs to isolate the pattern."


class CodeFunctionInferenceEnvWithFeedback(CodeFunctionInferenceEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Use \\boxed{...} with a valid command, e.g., \\boxed{test_str:abc} or \\boxed{submit:result}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Allowed commands: test_str:<text>, test_int:<number>, test:<value>, hint, submit:<answer>."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no tests left" in text:
                error_detail["violation"] = "no_tests_left"
                hint = "You ran out of tests. Use hint (if available) or submit your best answer."
            elif "expected string test" in text:
                error_detail["violation"] = "wrong_type_str_expected"
                hint = "Use test_str:<text> or test:<text> for string tasks."
            elif "expected integer test" in text:
                error_detail["violation"] = "wrong_type_int_expected"
                hint = "Use test_int:<number> or test:<number> for integer tasks."
            elif "no hints left" in text:
                error_detail["violation"] = "no_hints_left"
                hint = "No hints remain. Use remaining tests to isolate the pattern or submit."
            else:
                error_detail["violation"] = "generic_protocol"
                hint = "Check command type and budgets."
        elif "failed submission" in text:
            error_type = "WrongDecision"
            error_detail["outcome"] = "incorrect_final_answer"
            if self.input_type == "str":
                hint = "Compare simple string tests (e.g., single letters) to infer the transform, then recompute."
            else:
                hint = "Try tests on small integers to identify arithmetic pattern, then recompute."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Plan fewer, more informative tests. Identify family (rotation/shift/arithmetic) quickly, then submit."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        else:
            error_type = "OK"
            error_detail["status"] = "normal_step"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["tests_left"] = getattr(self, "tests_left", None)
            diagnostic["hints_left"] = getattr(self, "hints_left", None)
            diagnostic["input_type"] = getattr(self, "input_type", None)
            diagnostic["last_action"] = getattr(self, "last_action_kind", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by running a simple test, e.g., test a short string or a small integer to infer the family.",
            "turn": 0,
            "tests_left": getattr(self, "tests_left", None),
            "hints_left": getattr(self, "hints_left", None),
            "input_type": getattr(self, "input_type", None),
        }
        return obs, info