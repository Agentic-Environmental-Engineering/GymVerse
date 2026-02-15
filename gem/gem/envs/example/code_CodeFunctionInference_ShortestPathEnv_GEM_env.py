from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class CodeFunctionInferenceEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 40,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 40

        # Evolvable parameters
        self.complexity_params = {
            "input_range_max": (5, 30),     # Domain size [-M, M]; larger domain = more options and reasoning complexity
            "query_budget": (8, 3),         # REVERSED: fewer queries = harder (less information)
            "coef_abs_max": (4, 12),        # Larger coefficient magnitudes = wider numeric spread = harder to infer casually
            "lock_radius": (0, 2),          # More locked inputs around target = fewer nearby probes = harder
        }
        self.param_variance = {
            "input_range_max": 3,   # ~10% of range
            "query_budget": 1,      # small discrete range
            "coef_abs_max": 1,      # small discrete range
            "lock_radius": 0,       # small range → no randomization
        }

        # Placeholder attributes set in _apply_complexity_params
        self.input_range_max: int = 0
        self.query_budget: int = 0
        self.coef_abs_max: int = 0
        self.lock_radius: int = 0

        # Other state
        self.turn_count: int = 0
        self.family_type: str = ""
        self.breakpoint: int = 0
        self.coeffs: Dict[str, int] = {}
        self.target_input: int = 0
        self.locked_inputs: set = set()
        self.tests: Dict[int, int] = {}
        self.tests_used: int = 0
        self.terminated_flag: bool = False

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
            # Clamp within [min_v, max_v] regardless of order
            lo, hi = (min_v, max_v) if min_v <= max_v else (max_v, min_v)
            val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

    def _decide_family(self) -> str:
        if self.complexity <= 3:
            return "affine"
        elif self.complexity <= 7:
            return "quadratic"
        else:
            return "piecewise"

    def _gen_coeffs(self):
        cam = self.coef_abs_max
        if self.family_type == "affine":
            # a, b; avoid trivial all-zero
            a = random.randint(-cam, cam)
            b = random.randint(-cam, cam)
            if a == 0 and b == 0:
                a = 1
            self.coeffs = {"a": a, "b": b}
        elif self.family_type == "quadratic":
            a = random.randint(-cam, cam)
            b = random.randint(-cam, cam)
            c = random.randint(-cam, cam)
            if a == 0 and b == 0 and c == 0:
                a = 1
            self.coeffs = {"a": a, "b": b, "c": c}
        else:  # piecewise
            # breakpoint fixed at 0 for clarity
            self.breakpoint = 0
            a1 = random.randint(-cam, cam)
            b1 = random.randint(-cam, cam)
            a2 = random.randint(-cam, cam)
            b2 = random.randint(-cam, cam)
            if a1 == 0 and b1 == 0:
                a1 = 1
            if a2 == 0 and b2 == 0:
                a2 = 1
            self.coeffs = {"a1": a1, "b1": b1, "a2": a2, "b2": b2}

    def _eval_f(self, x: int) -> int:
        if self.family_type == "affine":
            return self.coeffs["a"] * x + self.coeffs["b"]
        elif self.family_type == "quadratic":
            return self.coeffs["a"] * x * x + self.coeffs["b"] * x + self.coeffs["c"]
        else:
            if x < self.breakpoint:
                return self.coeffs["a1"] * x + self.coeffs["b1"]
            else:
                return self.coeffs["a2"] * x + self.coeffs["b2"]

    def _family_description(self) -> str:
        if self.family_type == "affine":
            return "Family: affine f(x) = a*x + b (integer coefficients)."
        elif self.family_type == "quadratic":
            return "Family: quadratic f(x) = a*x^2 + b*x + c (integer coefficients)."
        else:
            return f"Family: piecewise-affine with breakpoint at x={self.breakpoint}: x<bp: a1*x+b1; else: a2*x+b2 (integer coefficients)."

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Code Function Inference Game\n"
            "- You face a hidden deterministic integer function f over a finite integer domain.\n"
            "- A target input T is disclosed, but evaluating f(T) directly is locked.\n"
            "- Use RUN x to query f(x) for allowed x (each RUN consumes budget).\n"
            "- Use STATUS to see budget, domain, and history.\n"
            "- When ready, SUBMIT y as your predicted f(T). The episode ends on submission.\n"
            "Formatting:\n"
            "- Every action must be in \\boxed{...}. Examples:\n"
            f"  {example}\n"
            "- Supported actions: RUN <int>, STATUS, HELP, SUBMIT <int>.\n"
        )

    def get_task_suffix(self) -> str:
        locked_preview = sorted(list(self.locked_inputs))[:7]
        locked_str = "[" + ", ".join(str(v) for v in locked_preview) + ("]" if len(locked_preview) <= 7 else ", ...]")
        hist = ", ".join(f"x={x}->y={y}" for x, y in list(self.tests.items())[-5:])
        if not hist:
            hist = "(no tests yet)"
        return (
            f"{self._family_description()}\n"
            f"Domain: integers in [-{self.input_range_max}, {self.input_range_max}]\n"
            f"Target input T = {self.target_input}; locked inputs around T (radius {self.lock_radius}): {locked_str}\n"
            f"Test budget used {self.tests_used}/{self.query_budget}. Recent tests: {hist}\n"
            "Enter your next action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.family_type = self._decide_family()
        self._gen_coeffs()

        # Sample target input away from borders to keep sufficient allowed inputs
        M = self.input_range_max
        margin = max(2, self.lock_radius + 1)
        if margin > M:
            margin = M
        self.target_input = random.randint(-M + margin, M - margin)
        # Build locked set
        self.locked_inputs = set()
        for d in range(-self.lock_radius, self.lock_radius + 1):
            self.locked_inputs.add(self.target_input + d)
        # Clamp locked to domain
        self.locked_inputs = {x for x in self.locked_inputs if -M <= x <= M}

        self.tests = {}
        self.tests_used = 0
        self.terminated_flag = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "unknown":
            obs = f"Unsupported action: '{parsed.get('raw','')}'. Episode terminated."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "help":
            obs = "Help: " + self._get_instructions().strip()
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "status":
            obs = "Status report.\n" + self.get_task_suffix()
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "run":
            x = parsed["x"]
            M = self.input_range_max
            if self.tests_used >= self.query_budget:
                obs = f"Protocol violation: no test budget remaining ({self.tests_used}/{self.query_budget})."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if not isinstance(x, int):
                obs = "Protocol violation: RUN expects an integer."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if x < -M or x > M:
                obs = f"Protocol violation: x={x} outside domain [-{M}, {M}]."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if x in self.locked_inputs:
                obs = f"Protocol violation: x={x} is locked (near T). Choose a different input."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

            y = self._eval_f(x)
            self.tests[x] = y
            self.tests_used += 1
            obs = f"Ran test at x={x}, observed y={y}. Budget left: {self.query_budget - self.tests_used}."
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "submit":
            y = parsed["y"]
            true_y = self._eval_f(self.target_input)
            if y == true_y:
                obs = f"Success! Correct f(T)={true_y}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Your answer {y} does not match f(T)."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"Unchanged state at turn {self.turn_count}."
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        m = re.findall(r'\\boxed\{(.+?)\}', action, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        content = m[-1].strip()

        # STATUS
        if re.fullmatch(r'(status)', content, flags=re.IGNORECASE):
            return {"type": "status"}

        # HELP
        if re.fullmatch(r'(help)', content, flags=re.IGNORECASE):
            return {"type": "help"}

        # RUN x
        m_run = re.fullmatch(r'(?:run)\s+([+-]?\d+)', content, flags=re.IGNORECASE)
        if m_run:
            try:
                x = int(m_run.group(1))
                return {"type": "run", "x": x}
            except Exception:
                return {"type": "run", "x": None}

        # SUBMIT y
        m_sub = re.fullmatch(r'(?:submit)\s+([+-]?\d+)', content, flags=re.IGNORECASE)
        if m_sub:
            try:
                y = int(m_sub.group(1))
                return {"type": "submit", "y": y}
            except Exception:
                return {"type": "submit", "y": None}

        # Unknown inside boxed
        return {"type": "unknown", "raw": content}

    def sample_random_action(self) -> str:
        choices = []
        # Try to suggest a valid RUN far from locked set
        M = max(5, self.input_range_max or 5)
        T = getattr(self, "target_input", 0)
        r = getattr(self, "lock_radius", 0)
        candidates = [i for i in range(-M, M + 1) if i not in {T + d for d in range(-r, r + 1)}]
        if candidates:
            x = random.choice(candidates)
            choices.append(f"\\boxed{{RUN {x}}}")
        choices.append("\\boxed{STATUS}")
        # Provide an example submit using a placeholder
        guess = 0
        choices.append(f"\\boxed{{SUBMIT {guess}}}")
        return random.choice(choices)


class CodeFunctionInferenceEnvWithFeedback(CodeFunctionInferenceEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{RUN 3} or \\boxed{SUBMIT 42}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use only RUN <int>, STATUS, HELP, or SUBMIT <int>."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no test budget" in text:
                error_detail["violation"] = "no_budget"
                hint = "You have no tests left. Consider SUBMIT when confident."
            elif "outside domain" in text:
                error_detail["violation"] = "out_of_domain"
                hint = "Choose x within the domain shown in STATUS."
            elif "expects an integer" in text:
                error_detail["violation"] = "non_integer"
                hint = "Provide an integer, e.g., \\boxed{RUN -2}."
            elif "is locked" in text:
                error_detail["violation"] = "locked_input"
                hint = "Do not RUN the target or its locked neighbors. Probe other inputs."
            else:
                error_detail["violation"] = "unspecified"
                hint = "Check STATUS and select valid inputs."
        elif "failed!" in text:
            error_type = "WrongDecision"
            error_detail["outcome"] = "incorrect_submission"
            # Provide strategy hints based on family
            fam = getattr(self, "family_type", "")
            if fam == "affine":
                hint = "Two RUNs on distinct x determine a and b. Use them to compute f(T)."
            elif fam == "quadratic":
                hint = "Three RUNs on distinct x determine a, b, c. Use them to compute f(T)."
            else:
                bp = getattr(self, "breakpoint", 0)
                T = getattr(self, "target_input", 0)
                side = "left" if T < bp else "right"
                hint = f"T is on the {side} of the breakpoint {bp}. Two RUNs on that side determine its coefficients."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
        elif "reached max turns" in text and (terminated and truncated):
            error_type = "Timeout"
            error_detail["outcome"] = "timeout"
            hint = "Act within fewer steps. Use STATUS, then RUN strategically, then SUBMIT."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "family_type": getattr(self, "family_type", ""),
                "target_input": getattr(self, "target_input", None),
                "budget_used": getattr(self, "tests_used", None),
                "budget_total": getattr(self, "query_budget", None),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        fam = getattr(self, "family_type", "")
        first_hint = None
        if fam == "affine":
            first_hint = "Start with two RUNs on distinct x to solve for a and b."
        elif fam == "quadratic":
            first_hint = "Plan three RUNs on distinct x to solve for a, b, c."
        else:
            bp = getattr(self, "breakpoint", 0)
            T = getattr(self, "target_input", 0)
            side = "left" if T < bp else "right"
            first_hint = f"Focus RUNs on the {side} side of breakpoint {bp}; two points determine that segment."

        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": first_hint,
            "turn": 0,
            "state": {
                "family_type": getattr(self, "family_type", ""),
                "target_input": getattr(self, "target_input", None),
                "budget_used": 0,
                "budget_total": getattr(self, "query_budget", None),
            },
        }
        return obs, info