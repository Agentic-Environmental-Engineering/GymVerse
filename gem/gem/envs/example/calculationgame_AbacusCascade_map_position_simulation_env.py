from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AbacusCascadeEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = bool(enable_param_randomization)
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # number of terms in the expression (operands count). More terms = deeper calculation chain = harder
            "num_terms": (3, 11),
            # maximum absolute value of generated integers. Larger magnitudes increase mental load and overflow of intuition
            "max_abs_value": (9, 97),
            # number of distinct operators allowed from the toolbox; more operators increases cognitive branching
            "operator_variety": (2, 5),
            # probability (percentage) of inserting parentheses around subexpressions. More parentheses = non-linear precedence
            "paren_percent": (0, 70),
            # probability (percentage) to include a single modulus operation in the chain; modulus adds nonlinearity
            "mod_percent": (0, 60),
            # probability (percentage) to allow integer division (floored). Division introduces truncation complexity
            "div_percent": (0, 60),
        }

        # Variance settings (± range around the interpolated center)
        self.param_variance = {
            "num_terms": 1,
            "max_abs_value": 5,
            "operator_variety": 1,
            "paren_percent": 5,
            "mod_percent": 5,
            "div_percent": 5,
        }

        # Placeholders populated by _apply_complexity_params at reset()
        self.num_terms: int = 0
        self.max_abs_value: int = 0
        self.operator_variety: int = 0
        self.paren_percent: int = 0
        self.mod_percent: int = 0
        self.div_percent: int = 0

        # Domain state
        self.turn_count: int = 0
        self.active: bool = False
        self.expression: str = ""
        self.target_value: Optional[int] = None
        self.allowed_ops: list = []
        self.had_mod: bool = False
        self.had_div: bool = False
        self.history: list = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
            # clamp both normal and reversed (none reversed here)
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            actual = max(lo, min(hi, actual))
            setattr(self, name, int(round(actual)))

    def _generate_expression(self, rng: random.Random) -> Tuple[str, int, list, bool, bool]:
        # Determine operator set by variety
        all_ops = ['+', '-', '*', '//', '%']
        # Ensure at least + and * appear as base ops
        base_ops = ['+', '*']
        ops_pool = list(all_ops)
        rng.shuffle(ops_pool)
        chosen = []
        # Guarantee basics first
        for op in base_ops:
            if op in ops_pool and op not in chosen:
                chosen.append(op)
            if len(chosen) >= self.operator_variety:
                break
        # Fill remaining up to operator_variety
        if len(chosen) < self.operator_variety:
            for op in ops_pool:
                if op not in chosen:
                    chosen.append(op)
                if len(chosen) >= self.operator_variety:
                    break

        allow_mod = ('%' in chosen) and (rng.randint(1, 100) <= self.mod_percent)
        allow_div = ('//' in chosen) and (rng.randint(1, 100) <= self.div_percent)

        effective_ops = [op for op in chosen if op not in ['//', '%']]
        if allow_div:
            effective_ops.append('//')
        if allow_mod:
            effective_ops.append('%')

        # Generate integers within range, avoid zeros where harmful
        def sample_int(nonzero=False, positive_only=False):
            if positive_only:
                val = rng.randint(1, self.max_abs_value)
                return val
            if nonzero:
                val = 0
                while val == 0:
                    val = rng.randint(-self.max_abs_value, self.max_abs_value)
                return val
            else:
                return rng.randint(-self.max_abs_value, self.max_abs_value)

        # Build a flat token list: number (op number) repeated
        terms = []
        # First operand
        first = sample_int(nonzero=True)
        terms.append(str(first))
        used_mod = False
        used_div = False

        for _ in range(self.num_terms - 1):
            op = rng.choice(effective_ops)
            # handle constraints for // and %
            if op == '//':
                used_div = True
                denom = sample_int(nonzero=True)
                # Adjust sign and magnitude to avoid trivial division by large numbers causing zero too early if small chain
                # but still allow realistic results
                terms.extend([op, str(denom)])
            elif op == '%':
                used_mod = True
                # modulus must be positive nonzero
                modv = sample_int(positive_only=True)
                # guarantee modulus > 1 to be meaningful
                if modv == 1:
                    modv = 2
                terms.extend([op, str(modv)])
            else:
                terms.extend([op, str(sample_int())])

        # Optionally insert parentheses
        # We will randomly wrap k disjoint binary spans with parentheses
        expr_tokens = terms[:]
        paren_probability = self.paren_percent
        # We create parentheses around subexpressions forming "(a op b)" units and then can nest if lucky
        def try_insert_parens(tokens):
            # find operator indices
            op_indices = [i for i, t in enumerate(tokens) if t in ['+', '-', '*', '//', '%']]
            rng.shuffle(op_indices)
            for oi in op_indices:
                if rng.randint(1, 100) > paren_probability:
                    continue
                left_i = oi - 1
                right_i = oi + 1
                if left_i >= 0 and right_i < len(tokens):
                    # Avoid duplicating parentheses
                    if tokens[left_i].startswith('(') or tokens[right_i].endswith(')'):
                        continue
                    tokens[left_i] = '(' + tokens[left_i]
                    tokens[right_i] = tokens[right_i] + ')'
                    # small chance to chain another operator on the right with the group if precedence suggests
                    # but we keep it simple and avoid invalid nesting
            return tokens

        expr_tokens = try_insert_parens(expr_tokens)
        expr_str = " ".join(expr_tokens)

        # Safely evaluate using Python integer semantics; ensure // is floor division
        # Convert to a safe eval by restricting builtins
        try:
            # Python eval with only operators and integers present is fine if we trust our generator
            value = eval(expr_str, {"__builtins__": {}}, {})
        except Exception:
            # Fallback: if invalid due to accidental bad parentheses, regenerate deterministically without parens
            expr_tokens = terms[:]
            expr_str = " ".join(expr_tokens)
            value = eval(expr_str, {"__builtins__": {}}, {})

        return expr_str, int(value), effective_ops, used_mod, used_div

    def _get_instructions(self) -> str:
        ops_list = ", ".join(self.allowed_ops) if self.allowed_ops else "(to be revealed)"
        return (
            "AbacusCascade: Compute the exact integer value of the generated arithmetic expression.\n"
            f"Expression: {self.expression}\n"
            f"Allowed operators present in this puzzle: {ops_list}\n"
            "- Operators follow standard precedence: parentheses > * // % > + -.\n"
            "- // means floor division on integers. % is modulus with positive modulus when present.\n"
            "- Negative numbers may appear. Parentheses may alter the natural precedence.\n"
            "Goal: Submit the final integer result of the entire expression.\n"
            "Action format: place your integer answer inside \\boxed{...}.\n"
            "Example: \\boxed{42}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Turns used: {self.turn_count}/{self.max_turns}\n"
            f"Expression: {self.expression}\n"
            "Submit your final integer result in \\boxed{...} format. Example: \\boxed{-13}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.active = True
        self.history = []

        rng = random.Random(random.randint(0, 10**9))
        expr, value, ops, used_mod, used_div = self._generate_expression(rng)
        self.expression = expr
        self.target_value = value
        self.allowed_ops = ops
        self.had_mod = used_mod
        self.had_div = used_div

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if not self.active:
            obs = "Episode already ended. Please reset."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a single integer."
            self.active = False
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        # Validate integer
        guess_str = parsed.get("raw", "").strip()
        # Allow optional leading +/-, digits
        if not re.fullmatch(r"[+-]?\d+", guess_str):
            obs = "PARSE ERROR: Answer must be an integer without spaces or commas."
            self.active = False
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        guess_val = int(guess_str)
        self.history.append(guess_val)

        if guess_val == self.target_value:
            obs = f"Success! Correct result: {self.target_value}."
            self.active = False
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        # Incorrect but continue if turns left
        if self.turn_count >= self.max_turns:
            obs = (
                f"TIMEOUT: Reached max turns ({self.max_turns}). "
                f"The correct result was {self.target_value}."
            )
            self.active = False
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        direction = ""
        if self.target_value is not None:
            if guess_val < self.target_value:
                direction = "Your guess is too small."
            elif guess_val > self.target_value:
                direction = "Your guess is too large."
        obs = f"Incorrect. {direction} Try again."
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        text = action.strip()
        m = re.search(r"\\boxed\{(.*)\}\s*$", text, flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        # inner should be the integer guess
        return {"action": "answer", "raw": inner}

    def sample_random_action(self) -> str:
        # Provide a plausible guess around zero or around last hint if available
        if self.history:
            last = self.history[-1]
            jitter = random.randint(-5, 5)
            return rf"\boxed{{{last + jitter}}}"
        # otherwise sample a small integer
        return rf"\boxed{{{random.randint(-10, 10)}}}"


class AbacusCascadeEnvWithFeedback(AbacusCascadeEnv):
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
            error_detail["issue"] = "missing_or_malformed_boxed"
            hint = "Place a single integer inside \\boxed{...}, e.g., \\boxed{17}."
        elif "parse error" in text:
            error_type = "FormatError"
            error_detail["issue"] = "non_integer_submission"
            hint = "Submit an integer with optional leading sign, no spaces or commas."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Only submit the final integer result inside \\boxed{...}."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Compute carefully but commit earlier guesses to refine with size hints."
        elif "incorrect" in text:
            error_type = "WrongDecision"
            # Extract last guess if available
            last_guess = self.history[-1] if getattr(self, "history", []) else None
            error_detail["got"] = last_guess
            error_detail["direction"] = "too small" if "too small" in text else ("too large" if "too large" in text else "unknown")
            # Domain hinting
            hi_ops = set(self.allowed_ops)
            if "//" in hi_ops or "%" in hi_ops:
                hint = "Mind floor division // and modulus % precedence: parentheses > * // % > + -."
            else:
                hint = "Re-check operator precedence and any parentheses before summing."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "episode already ended" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "acted_after_terminal"
            hint = "Reset the environment to start a new puzzle."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "expression": getattr(self, "expression", ""),
                "had_mod": getattr(self, "had_mod", False),
                "had_div": getattr(self, "had_div", False),
                "allowed_ops": getattr(self, "allowed_ops", []),
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
            "hint": "Start by parsing parentheses, then apply * // % before + -.",
            "turn": 0,
            "state": {
                "expression": getattr(self, "expression", ""),
                "had_mod": getattr(self, "had_mod", False),
                "had_div": getattr(self, "had_div", False),
                "allowed_ops": getattr(self, "allowed_ops", []),
            },
        }
        return obs, info