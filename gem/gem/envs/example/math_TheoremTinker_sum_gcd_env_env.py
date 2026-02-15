from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class TheoremTinkerEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters (math-native)
        self.complexity_params = {
            # number of terms in top-level expression: more terms harder
            "num_terms": (2, 7),
            # maximum integer magnitude in coefficients/constants: larger magnitude harder
            "max_coeff_abs": (5, 50),
            # operator variety index: controls allowed operators; more variety harder
            "operator_variety": (1, 4),
            # polynomial degree cap for generated polynomials: higher degree harder
            "poly_degree_cap": (1, 5),
            # presence of modular arithmetic: higher modulus and presence increases difficulty
            "modulus": (0, 1000),  # 0 means no modulo; otherwise compute final answer mod this
            # nested depth of parentheses/sub-expressions: deeper nesting harder
            "nest_depth": (0, 3),
        }

        # Variance settings
        self.param_variance = {
            "num_terms": 1,            # medium discrete range
            "max_coeff_abs": 5,        # ~10% of range
            "operator_variety": 0,     # small (1-4) keep stable
            "poly_degree_cap": 0,      # small discrete range
            "modulus": 50,             # ~5% variation on large range; clamped
            "nest_depth": 1,           # small discrete range
        }

        # Placeholder attributes
        self.num_terms: int = 0
        self.max_coeff_abs: int = 0
        self.operator_variety: int = 0
        self.poly_degree_cap: int = 0
        self.modulus: int = 0
        self.nest_depth: int = 0

        # Other state
        self.turn_count: int = 0
        self.active: bool = True
        self.history: List[str] = []
        self.expression_str: str = ""
        self.current_value: Optional[int] = None
        self.hidden_answer: Optional[int] = None
        self.allowed_ops: List[str] = []
        self.has_submitted: bool = False
        self.last_action_result: str = ""
        self.random_seed: Optional[int] = None

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
                    lo = min(min_val, max_val)
                    hi = max(min_val, max_val)
                    if actual_value < lo:
                        actual_value = lo
                    if actual_value > hi:
                        actual_value = hi
            setattr(self, param_name, int(round(actual_value)))

        # Ensure feasibility: if modulus in very small range treat as none or >= 7
        if self.modulus != 0 and self.modulus < 7:
            self.modulus = 7

    def _allowed_operator_set(self) -> List[str]:
        # operator_variety determines which ops can appear in the generation
        # 1: +,-
        # 2: +,-,*
        # 3: +,-,*,^ (polynomial degrees), parentheses
        # 4: +,-,*,^, factorial limited, and mixed nested combos
        base = []
        if self.operator_variety >= 1:
            base.extend(["add", "sub"])
        if self.operator_variety >= 2:
            base.append("mul")
        if self.operator_variety >= 3:
            base.append("pow")  # small integer powers
        if self.operator_variety >= 4:
            base.append("fact")  # limited factorial on small integers
        return base

    def _safe_pow(self, a: int, b: int) -> int:
        b = max(0, min(b, self.poly_degree_cap + 2))
        # clamp to avoid explosion
        a = max(-self.max_coeff_abs, min(self.max_coeff_abs, a))
        val = 1
        for _ in range(b):
            val *= a
            # clamp intermediate
            val = max(-10**12, min(10**12, val))
        return val

    def _safe_fact(self, n: int) -> int:
        # limit n to 0..10 for safety
        n = max(0, min(10, n))
        v = 1
        for i in range(2, n + 1):
            v *= i
        return v

    def _gen_term(self) -> Tuple[str, int]:
        ops = self.allowed_ops
        # choose a base integer
        a = random.randint(-self.max_coeff_abs, self.max_coeff_abs)
        if a == 0:
            a = random.choice([1, -1])

        # possibly apply unary op like factorial (only if allowed)
        if "fact" in ops and random.random() < 0.2:
            n = abs(a)
            expr = f"({n})!"
            val = self._safe_fact(n)
            return expr, val

        # possibly power
        if "pow" in ops and random.random() < 0.3:
            b = random.randint(1, max(1, self.poly_degree_cap + 1))
            expr = f"({a})^{b}"
            val = self._safe_pow(a, b)
            return expr, val

        # otherwise simple integer term
        return str(a), a

    def _combine(self, t1: Tuple[str, int], t2: Tuple[str, int]) -> Tuple[str, int]:
        ops = self.allowed_ops
        candidates = []
        if "add" in ops:
            candidates.append("+")
        if "sub" in ops:
            candidates.append("-")
        if "mul" in ops:
            candidates.append("*")
        if "pow" in ops and random.random() < 0.15:
            # sometimes do small power with t2 as exponent if positive small
            if 0 <= t2[1] <= self.poly_degree_cap + 2:
                # clamp base
                base = max(-self.max_coeff_abs, min(self.max_coeff_abs, t1[1]))
                val = self._safe_pow(base, t2[1])
                expr = f"({t1[0]})^{({t2[0]})}"
                return expr, val
        if not candidates:
            candidates = ["+"]

        op = random.choice(candidates)
        if op == "+":
            expr = f"({t1[0]} + {t2[0]})"
            val = t1[1] + t2[1]
        elif op == "-":
            expr = f"({t1[0]} - {t2[0]})"
            val = t1[1] - t2[1]
        else:
            expr = f"({t1[0]} * {t2[0]})"
            val = t1[1] * t2[1]
        return expr, val

    def _maybe_nest(self, expr_val: Tuple[str, int]) -> Tuple[str, int]:
        # Introduce nested structure: wrap with an op and another term
        depth = random.randint(0, self.nest_depth)
        cur = expr_val
        for _ in range(depth):
            t = self._gen_term()
            cur = self._combine(cur, t)
            # clamp values to avoid explosion
            cur = (cur[0], max(-10**12, min(10**12, cur[1])))
        return cur

    def _generate_expression(self):
        self.allowed_ops = self._allowed_operator_set()
        term = self._gen_term()
        term = self._maybe_nest(term)
        expr, val = term
        for _ in range(self.num_terms - 1):
            t = self._gen_term()
            t = self._maybe_nest(t)
            expr, val = self._combine((expr, val), t)
            val = max(-10**12, min(10**12, val))

        # hidden answer may be modulo or raw
        if self.modulus != 0:
            hidden = val % self.modulus
        else:
            hidden = val
        self.expression_str = expr
        self.current_value = None  # not yet evaluated by agent
        self.hidden_answer = hidden

    def _get_instructions(self) -> str:
        tools = [
            "- eval: compute the integer value of the current expression.",
            "- simplify: algebraic simplification steps (textual).",
            "- expand: expand products/powers when applicable (textual).",
            "- factor: factor integer or simple polynomial-like integers (textual).",
            "- substitute value=<int>: replace a literal occurrence with another integer.",
            "- modulo m=<int>: set modulus for final answer if allowed (only if modulus is 0 initially).",
            "- submit value=<int>: submit your final answer in boxed form."
        ]
        fmt = (
            "You are solving an arithmetic expression evaluation task.\n"
            "Goal: compute the value of the given expression exactly"
        )
        if self.modulus != 0:
            fmt += f" modulo {self.modulus}"
        fmt += ".\n"
        fmt += "Use tools to transform, analyze, and finally submit the answer.\n"
        fmt += "Rules:\n"
        fmt += "- Use exactly one action per turn in \\boxed{...} format.\n"
        fmt += "- Only one submit is allowed; submitting ends the episode.\n"
        fmt += "- 'modulo' is only allowed if the current modulus is 0.\n"
        fmt += "Available actions:\n" + "\n".join(tools) + "\n"
        fmt += "Action syntax examples:\n"
        fmt += r"- \boxed{eval}" + "\n"
        fmt += r"- \boxed{simplify}" + "\n"
        fmt += r"- \boxed{expand}" + "\n"
        fmt += r"- \boxed{factor}" + "\n"
        fmt += r"- \boxed{substitute value=3}" + "\n"
        fmt += r"- \boxed{modulo m=17}" + "\n"
        fmt += r"- \boxed{submit value=42}" + "\n"
        return fmt

    def get_task_suffix(self) -> str:
        status = []
        status.append(f"Turns used: {self.turn_count}/{self.max_turns}")
        status.append(f"Expression: {self.expression_str}")
        if self.current_value is not None:
            if self.modulus != 0:
                status.append(f"Last eval: {self.current_value} (not reduced); target is modulo {self.modulus}")
            else:
                status.append(f"Last eval: {self.current_value}")
        else:
            status.append("Last eval: none")
        status.append(f"Modulus: {self.modulus if self.modulus != 0 else 'none'}")
        status.append("Enter your action in \\boxed{...} format.")
        return "\n".join(status)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
            self.random_seed = seed
        self._apply_complexity_params()
        self.turn_count = 0
        self.active = True
        self.history = []
        self.last_action_result = ""
        self.has_submitted = False
        self._generate_expression()
        instruction = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return instruction, info

    def _format_invalid(self, msg: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs = f"INVALID ACTION FORMAT: {msg}"
        return (
            obs,
            LanguageGameReward.format_error_reward,
            True,
            False,
            {"suffix": self.get_task_suffix()},
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if not self.active:
            obs = "Episode already ended."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            return self._format_invalid("Use \\boxed{...} with a supported action.")

        name = parsed.get("action", "")
        supported = {"eval", "simplify", "expand", "factor", "substitute", "modulo", "submit"}
        if name not in supported:
            obs = f"UNSUPPORTED ACTION: '{name}' is not recognized."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if name == "eval":
            # Deterministic evaluation: we have hidden ground truth; keep a non-reduced last eval
            self.current_value = self._evaluate_expression_internal()
            obs = f"Evaluated current expression to {self.current_value}."
            self.last_action_result = obs

        elif name == "simplify":
            # For this numeric world, simplification is cosmetic; describe a plausible step
            obs = "Applied simplification: combined constant terms where applicable."
            self.last_action_result = obs

        elif name == "expand":
            obs = "Applied expansion: distributed products/powers in conceptual form."
            self.last_action_result = obs

        elif name == "factor":
            obs = "Attempted factoring: identified common integer factors when present."
            self.last_action_result = obs

        elif name == "substitute":
            if "value" not in parsed:
                return self._format_invalid("substitute requires 'value=<int>'.")
            try:
                newv = int(parsed["value"])
            except Exception:
                return self._format_invalid("substitute value must be an integer.")
            # We only allow substituting one literal occurrence: change one visible integer coefficient randomly
            # This does not change hidden answer (as it changes the task); so we constrain it to a no-op description
            obs = "No-op: substitution is restricted on fixed tasks; expression unchanged."
            self.last_action_result = obs

        elif name == "modulo":
            if self.modulus != 0:
                obs = "Protocol violation: modulus already set; cannot change."
                terminated = True
                reward = 0.0
                self.active = False
                self.last_action_result = obs
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            if "m" not in parsed:
                return self._format_invalid("modulo requires 'm=<int>'.")
            try:
                m = int(parsed["m"])
            except Exception:
                return self._format_invalid("m must be an integer.")
            if m <= 1:
                obs = "Protocol violation: modulus must be >= 2."
                terminated = True
                reward = 0.0
                self.active = False
                self.last_action_result = obs
                return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}
            self.modulus = m
            # Recompute hidden answer accordingly
            val = self._evaluate_expression_internal()
            self.hidden_answer = val % self.modulus
            obs = f"Modulus set to {self.modulus}. Hidden target updated accordingly."
            self.last_action_result = obs

        elif name == "submit":
            if "value" not in parsed:
                return self._format_invalid("submit requires 'value=<int>'.")
            try:
                ans = int(parsed["value"])
            except Exception:
                return self._format_invalid("submitted value must be an integer.")
            self.has_submitted = True
            self.active = False
            terminated = True
            # Correctness check relative to current modulus setting
            target = self.hidden_answer
            if ans == target:
                obs = f"Success! Submitted {ans} equals the target."
                reward = 1.0
            else:
                obs = f"Failed! Submitted {ans} does not match the target."
                reward = 0.0
            self.last_action_result = obs

        # Timeout check only if not terminated already
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            terminated = True
            truncated = True
            self.active = False

        return obs if obs else "OK", reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _evaluate_expression_internal(self) -> int:
        # Directly evaluate via stored construction by regenerating numeric value using same seed trace:
        # We don't store full parse tree, but we computed hidden_answer from a sequence.
        # Since hidden_answer was computed from raw integer operations, to stay deterministic we re-derive:
        # Strategy: We cannot reconstruct intermediate exactly; instead, we use hidden answer source:
        # - If modulus is 0: the hidden_answer equals the raw value computed earlier.
        # - If modulus != 0: hidden_answer = raw % modulus; we need raw. To preserve consistency,
        #   we store an approximate last raw by reverse: if current_value exists, reuse it; else recompute by sampling again is invalid.
        # Solution: During generation, we can save the raw_value before modulus.
        # Implement: store self._raw_value at generation time.
        # If None, fallback: if modulus==0, hidden_answer is raw; else we return hidden_answer (not reduced) as best proxy.
        # Add storage now.
        # This method will assume self._raw_value exists; otherwise fallback.
        if hasattr(self, "_raw_value") and self._raw_value is not None:
            return int(self._raw_value)
        # Fallback: best-effort
        if self.modulus == 0:
            return int(self.hidden_answer if self.hidden_answer is not None else 0)
        # Unknown raw; return something consistent for eval calls (non-reduced eval)
        return int(self.hidden_answer if self.hidden_answer is not None else 0)

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        parts = inner.split()
        tokens: Dict[str, Any] = {}
        tokens["action"] = parts[0].lower()
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip().lower()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        choices = [
            r"\boxed{eval}",
            r"\boxed{simplify}",
            r"\boxed{expand}",
            r"\boxed{factor}",
            r"\boxed{substitute value=1}",
        ]
        if self.modulus == 0:
            choices.append(r"\boxed{modulo m=17}")
        random.shuffle(choices)
        return choices[0]

    # Override _generate_expression to also store raw value for evaluation consistency
    def _generate_expression(self):
        self.allowed_ops = self._allowed_operator_set()
        term = self._gen_term()
        term = self._maybe_nest(term)
        expr, val = term
        for _ in range(self.num_terms - 1):
            t = self._gen_term()
            t = self._maybe_nest(t)
            expr, val = self._combine((expr, val), t)
            val = max(-10**12, min(10**12, val))
        raw = val
        if self.modulus != 0:
            hidden = raw % self.modulus
        else:
            hidden = raw
        self.expression_str = expr
        self.current_value = None
        self.hidden_answer = hidden
        self._raw_value = raw


class TheoremTinkerEnvWithFeedback(TheoremTinkerEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "bad_boxed_or_params"
            hint = "Use \\boxed{action_name key=value}. Example: \\boxed{submit value=12}"

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown"
            hint = "Allowed actions: eval, simplify, expand, factor, substitute, modulo, submit."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "modulus already set" in text:
                error_detail["violation"] = "reset_modulus"
                hint = "Only set modulus once and only if current Modulus is none."
            elif "modulus must be >= 2" in text:
                error_detail["violation"] = "bad_modulus"
                hint = "Choose an integer modulus m >= 2."
            else:
                error_detail["violation"] = "unknown_rule"
                hint = "Follow the action rules described in the instructions."

        elif "failed!" in text:
            error_type = "WrongDecision"
            error_detail["expected_modulus"] = self.modulus if self.modulus != 0 else "none"
            if hasattr(self, "hidden_answer"):
                error_detail["target_known"] = True
            else:
                error_detail["target_known"] = False
            hint = "Use \\boxed{eval} to compute the raw value, apply modulo if needed, then \\boxed{submit value=...}."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act earlier: eval the expression and submit within the turn limit."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "modulus": self.modulus if self.modulus != 0 else None,
                "has_submitted": getattr(self, "has_submitted", False),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{eval}. If a modulus is present, reduce before submitting.",
            "turn": 0,
            "state": {
                "modulus": self.modulus if self.modulus != 0 else None,
                "has_submitted": False,
            },
        }
        return obs, info