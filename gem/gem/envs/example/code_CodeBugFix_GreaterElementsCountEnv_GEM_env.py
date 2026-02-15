from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeBugFixEnv(Env):
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
            'input_max': (10, 100),               # Valid integer inputs range [0..input_max]; larger domain -> harder probing and reasoning
            'num_distractor_lines': (0, 6),       # More non-essential code/comments -> harder to locate the bug
            'test_budget': (10, 4),               # REVERSED: fewer test runs allowed -> harder to gather evidence
            'allow_show_all': (1, 0),             # REVERSED: bulk code view disabled at higher complexity -> harder information access
            'doc_examples': (2, 0),               # REVERSED: fewer doc examples -> less guidance, harder
            'bug_difficulty': (1, 4),             # Harder bug families appear at higher levels -> subtler failure modes
        }

        # Parameter variance
        self.param_variance = {
            'input_max': 8,               # ~9% variance over range
            'num_distractor_lines': 1,    # ±1 line variation
            'test_budget': 1,             # ±1 test variation
            'allow_show_all': 0,          # binary constraint, no randomization
            'doc_examples': 0,            # small discrete range, no randomization
            'bug_difficulty': 0,          # small discrete range, no randomization
        }

        # Placeholder attributes set in _apply_complexity_params
        self.input_max: int = 0
        self.num_distractor_lines: int = 0
        self.test_budget: int = 0
        self.allow_show_all: int = 1
        self.doc_examples: int = 0
        self.bug_difficulty: int = 1

        # Other state
        self.turn_count: int = 0
        self.tests_used: int = 0
        self.submitted: bool = False
        self.func_name: str = "transform"
        self.bug_type: str = ""
        self.code_lines: list = []
        self.docstring: str = ""
        self._last_action: Optional[Dict[str, Any]] = None

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
            # Clamp with reversed support
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _generate_instance(self):
        # Choose bug type by difficulty
        if self.bug_difficulty <= 1:
            family = ["parity_inverted", "exponent_swap", "wrong_operator_plus"]
        elif self.bug_difficulty == 2:
            family = ["off_by_one_increment"]
        elif self.bug_difficulty == 3:
            family = ["missing_zero_case"]
        else:
            family = ["threshold_special_case"]
        self.bug_type = random.choice(family)

        # Build docstring/spec with examples
        examples = []
        if self.doc_examples >= 1:
            examples.append(f"Example: {self.func_name}(2) -> 4 (even -> square)")
        if self.doc_examples >= 2:
            examples.append(f"Example: {self.func_name}(3) -> 27 (odd -> cube)")
        examples_text = ("\n" + "\n".join(examples)) if examples else ""
        self.docstring = (
            f"{self.func_name}(n): returns n^2 if n is even, otherwise returns n^3."
            f" Valid inputs are integers in [0..{self.input_max}]."
            f"{examples_text}"
        )

        # Generate buggy code snippet lines
        # Intended correct:
        # def transform(n):
        #     if n % 2 == 0:
        #         return n * n
        #     else:
        #         return n * n * n
        even_return = "n * n"
        odd_return = "n * n * n"
        if self.bug_type == "parity_inverted":
            cond = "n % 2 != 0"  # wrong check
            even_body = even_return
            odd_body = odd_return
        elif self.bug_type == "exponent_swap":
            cond = "n % 2 == 0"
            even_body = odd_return  # swapped
            odd_body = even_return
        elif self.bug_type == "wrong_operator_plus":
            cond = "n % 2 == 0"
            even_body = "n * n + n"  # erroneous plus
            odd_body = odd_return
        elif self.bug_type == "off_by_one_increment":
            cond = "n % 2 == 0"
            even_body = "(n + 1) * (n + 1)"
            odd_body = "((n + 1) * (n + 1) * (n + 1))"
        elif self.bug_type == "missing_zero_case":
            cond = "n % 2 == 0"
            even_body = "1 if n == 0 else n * n"  # mishandles 0 as 1
            odd_body = odd_return
        else:  # threshold_special_case
            threshold = max(2, self.input_max // 2)
            cond = "n % 2 == 0"
            even_body = f"(n * n if n <= {threshold} else n * 3)"  # bad special-case
            odd_body = f"(n * n * n if n <= {threshold} else n + 3)"  # bad special-case

        base_lines = [
            f"def {self.func_name}(n):",
            f"    if {cond}:",
            f"        return {even_body}",
            f"    else:",
            f"        return {odd_body}",
        ]

        distractors = []
        for i in range(self.num_distractor_lines):
            kind = random.choice(["comment", "unused_fn", "dead_code"])
            if kind == "comment":
                distractors.append(f"# TODO: consider optimizing power operation (line {i})")
            elif kind == "unused_fn":
                distractors.append("def _helper(x):\n    return x  # unused")
            else:
                distractors.append("flag = False  # unused flag")
        # Interleave distractors
        self.code_lines = []
        insertion_points = set(random.sample(range(len(base_lines) + len(distractors)), len(distractors))) if distractors else set()
        bi = 0
        di = 0
        total_len = len(base_lines) + len(distractors)
        for pos in range(total_len):
            if pos in insertion_points and di < len(distractors):
                self.code_lines.append(distractors[di])
                di += 1
            else:
                if bi < len(base_lines):
                    self.code_lines.append(base_lines[bi])
                    bi += 1

    def _correct_output(self, n: int) -> int:
        return (n * n) if (n % 2 == 0) else (n * n * n)

    def _buggy_output(self, n: int) -> int:
        if self.bug_type == "parity_inverted":
            if n % 2 != 0:
                return n * n
            else:
                return n * n * n
        elif self.bug_type == "exponent_swap":
            if n % 2 == 0:
                return n * n * n
            else:
                return n * n
        elif self.bug_type == "wrong_operator_plus":
            if n % 2 == 0:
                return n * n + n
            else:
                return n * n * n
        elif self.bug_type == "off_by_one_increment":
            m = n + 1
            if n % 2 == 0:
                return m * m
            else:
                return m * m * m
        elif self.bug_type == "missing_zero_case":
            if n % 2 == 0:
                return 1 if n == 0 else n * n
            else:
                return n * n * n
        else:  # threshold_special_case
            threshold = max(2, self.input_max // 2)
            if n % 2 == 0:
                return n * n if n <= threshold else n * 3
            else:
                return n * n * n if n <= threshold else n + 3

    def _get_instructions(self) -> str:
        return (
            "You are diagnosing a bug in a small function.\n"
            "Goal: identify the correct bug label for the provided function.\n"
            "Valid actions (use \\boxed{...}):\n"
            "- TEST k            : run the buggy function on integer k (0..input_max)\n"
            "- SHOW i            : show code line i (1-indexed)\n"
            "- SHOW ALL          : show the entire code (may be disabled at higher complexity)\n"
            "- DESCRIBE          : view the function's specification/docstring\n"
            "- LIST              : view valid input range\n"
            "- SUBMIT BUG=<label>: submit your final bug label and end the episode\n"
            "Allowed labels: parity_inverted, exponent_swap, wrong_operator_plus,\n"
            "                off_by_one_increment, missing_zero_case, threshold_special_case\n"
            f"For example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        rem_tests = max(0, self.test_budget - self.tests_used)
        lines_info = f"{len(self.code_lines)} code lines available"
        show_all_status = "enabled" if self.allow_show_all == 1 else "disabled"
        return (
            f"State: turn={self.turn_count}, tests_used={self.tests_used}/{self.test_budget}, "
            f"{lines_info}, SHOW ALL is {show_all_status}.\n"
            "Enter your action using \\boxed{...}. Example: \\boxed{TEST 3} or \\boxed{SUBMIT BUG=parity_inverted}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.tests_used = 0
        self.submitted = False
        self._generate_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        parsed = self._parse_action(action)
        self._last_action = parsed

        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        typ = parsed.get("type")
        obs = ""
        reward = 0.0

        if typ == "submit":
            label = parsed.get("label", "").strip()
            if label not in [
                "parity_inverted", "exponent_swap", "wrong_operator_plus",
                "off_by_one_increment", "missing_zero_case", "threshold_special_case"
            ]:
                obs = f"Unsupported bug label '{label}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.submitted = True
            if label == self.bug_type:
                obs = f"Correct! bug identified: {label}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect bug label: {label}. True bug was '{self.bug_type}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif typ == "test":
            k = parsed.get("value")
            if not isinstance(k, int) or k < 0 or k > self.input_max:
                obs = f"Unsupported action: TEST expects integer in [0..{self.input_max}]."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if self.tests_used >= self.test_budget:
                obs = "Protocol violation: test budget exceeded. No more TEST actions allowed."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.tests_used += 1
            buggy = self._buggy_output(k)
            obs = f"Ran {self.func_name}({k}) -> {buggy} [buggy output]."
            reward = 0.0

        elif typ == "show_all":
            if self.allow_show_all == 0:
                obs = "Protocol violation: SHOW ALL is disabled at current complexity."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            content = "\n".join(f"{i+1}: {line}" for i, line in enumerate(self.code_lines))
            obs = f"Code (full):\n{content}"

        elif typ == "show":
            i = parsed.get("line")
            if not isinstance(i, int) or i < 1 or i > len(self.code_lines):
                obs = f"Unsupported action: line index must be in [1..{len(self.code_lines)}]."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            obs = f"Line {i}: {self.code_lines[i-1]}"

        elif typ == "describe":
            obs = f"Spec:\n{self.docstring}"

        elif typ == "list":
            obs = f"Valid inputs: integers in [0..{self.input_max}]"

        else:
            obs = "Unsupported action."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            return f"Reached max turns ({self.max_turns}).", 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        s = extracted.strip()
        s_lower = s.lower()

        if s_lower.startswith("test"):
            m = re.match(r'(?i)test\s+(-?\d+)', s)
            if m:
                val = int(m.group(1))
                return {"type": "test", "value": val}
            return None

        if s_lower.startswith("show all"):
            return {"type": "show_all"}

        if s_lower.startswith("show"):
            m = re.match(r'(?i)show\s+(\d+)', s)
            if m:
                line = int(m.group(1))
                return {"type": "show", "line": line}
            return None

        if s_lower.startswith("describe"):
            return {"type": "describe"}

        if s_lower.startswith("list"):
            return {"type": "list"}

        if s_lower.startswith("submit"):
            # Accept "SUBMIT BUG=<label>" or "SUBMIT BUG <label>"
            m = re.match(r'(?i)submit\s+bug(?:\s*=\s*|\s+)([a-z_]+)', s)
            if m:
                label = m.group(1).strip().lower()
                return {"type": "submit", "label": label}
            return None

        return None

    def sample_random_action(self) -> str:
        choice = random.choice(["TEST", "SHOW", "DESCRIBE", "LIST", "SUBMIT"])
        if choice == "TEST":
            k = random.randint(0, max(0, self.input_max))
            return f"\\boxed{{TEST {k}}}"
        elif choice == "SHOW":
            if self.code_lines:
                i = random.randint(1, len(self.code_lines))
                return f"\\boxed{{SHOW {i}}}"
            else:
                return "\\boxed{DESCRIBE}"
        elif choice == "DESCRIBE":
            return "\\boxed{DESCRIBE}"
        elif choice == "LIST":
            return "\\boxed{LIST}"
        else:
            label = random.choice([
                "parity_inverted", "exponent_swap", "wrong_operator_plus",
                "off_by_one_increment", "missing_zero_case", "threshold_special_case"
            ])
            return f"\\boxed{{SUBMIT BUG={label}}}"


class CodeBugFixEnvWithFeedback(CodeBugFixEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if ("invalid action format" in text) or ("use \\boxed" in text):
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Use \\boxed{...} and a valid command like TEST k or SUBMIT BUG=<label>."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "test budget exceeded" in text:
                error_detail["violation"] = "test_budget_exceeded"
                remaining = max(0, self.test_budget - self.tests_used)
                hint = f"No more TEST actions allowed. Remaining tests: {remaining}. Submit your bug label or use SHOW/DESCRIBE."
            elif "show all is disabled" in text:
                error_detail["violation"] = "show_all_disabled"
                hint = "Use SHOW <line_number> to inspect specific lines instead of SHOW ALL."

        elif "unsupported action" in text and "bug label" not in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = self._last_action.get("type") if self._last_action else None
            hint = "Valid actions: TEST k, SHOW i, SHOW ALL, DESCRIBE, LIST, SUBMIT BUG=<label>."

        elif "unsupported bug label" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "submit"
            hint = "Use one of: parity_inverted, exponent_swap, wrong_operator_plus, off_by_one_increment, missing_zero_case, threshold_special_case."

        elif "incorrect bug label" in text:
            error_type = "WrongDecision"
            # Extract expected if present
            m = re.search(r"true bug was '([a-z_]+)'", text)
            if m:
                error_detail["expected"] = m.group(1)
            got = None
            if self._last_action and self._last_action.get("type") == "submit":
                got = self._last_action.get("label")
            error_detail["got"] = got
            # Provide adaptive hint
            if self.bug_difficulty >= 4:
                hint = "Test values around the threshold (near half of input_max) and compare even vs odd outputs."
            elif self.bug_difficulty == 3:
                hint = "Check the behavior at n=0 and verify even branch returns 0 properly."
            elif self.bug_difficulty == 2:
                hint = "Verify whether n is incremented before computing powers by testing consecutive numbers (e.g., 2 vs 3)."
            else:
                hint = "Inspect parity check and returned expressions on even/odd lines; use SHOW 2-5 and TEST both even and odd."

        elif "reached max turns" in text:
            error_type = "Timeout"
            hint = "Plan faster: prioritize SHOW lines and a few targeted TESTs, then SUBMIT."

        elif "correct! bug identified" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            error_detail["turn"] = getattr(self, "turn_count", None)
            error_detail["tests_used"] = getattr(self, "tests_used", None)
            error_detail["test_budget"] = getattr(self, "test_budget", None)
            error_detail["allow_show_all"] = getattr(self, "allow_show_all", None)
            diagnostic["error_detail"] = error_detail
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {
                "outcome": "episode_start",
                "turn": 0,
                "tests_used": self.tests_used,
                "test_budget": self.test_budget,
                "allow_show_all": self.allow_show_all,
            },
            "hint": "Start by DESCRIBE to read the spec, then SHOW 2-5 to inspect logic. Use a couple of TESTs on even/odd values.",
        }
        return obs, info