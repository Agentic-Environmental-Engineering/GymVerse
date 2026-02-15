from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class FormulaFoundryEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # number of base numbers provided; more numbers -> larger search space -> harder
            'num_numbers': (3, 7),
            # operator set richness index: 1=+, 2=+-, 3=+-*, 4=+-*/; richer ops -> harder search
            'operator_richness': (2, 4),
            # allowed parentheses pairs cap; higher allows deeper structures -> harder checking/construction
            'max_paren_pairs': (0, 3),
            # reversed: higher complexity reduces tolerance window; tighter target tolerance -> harder
            'target_tolerance': (2, 0),
            # reversed: fewer turns available -> harder
            'turn_budget': (8, 4),
            # magnitude of base numbers; larger numbers yield broader evaluation range -> harder
            'number_scale': (9, 25),
            # reversed: smaller max result magnitude cap increases need for exactness; but keep solvable
            'max_result_cap': (1000, 200),
        }

        # Variance settings
        self.param_variance = {
            'num_numbers': 1,
            'operator_richness': 0,   # small discrete set
            'max_paren_pairs': 1,
            'target_tolerance': 0,    # tiny range, keep fixed per level
            'turn_budget': 1,
            'number_scale': 3,
            'max_result_cap': 50,
        }

        # Placeholder attributes
        self.num_numbers: int = 0
        self.operator_richness: int = 0
        self.max_paren_pairs: int = 0
        self.target_tolerance: int = 0
        self.turn_budget: int = 0
        self.number_scale: int = 0
        self.max_result_cap: int = 0

        # State
        self.turn_count: int = 0
        self.numbers: List[int] = []
        self.target: int = 0
        self.allowed_ops: List[str] = []
        self.used_solution: Optional[str] = None
        self.history: List[str] = []
        self.solution_value: Optional[float] = None

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
            # Clamp for both normal and reversed ranges
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _generate_problem(self):
        # Determine operator set by richness
        richness_to_ops = {
            1: ['+'],
            2: ['+', '-'],
            3: ['+', '-', '*'],
            4: ['+', '-', '*', '/'],
        }
        self.allowed_ops = richness_to_ops.get(self.operator_richness, ['+', '-'])
        # Generate numbers
        # Ensure positive integers >=1
        self.numbers = [max(1, random.randint(1, self.number_scale)) for _ in range(self.num_numbers)]
        # Construct a hidden guaranteed solvable expression using all numbers once
        # Build a random expression tree using allowed operators and limited parentheses
        nums = [str(n) for n in self.numbers]
        ops = self.allowed_ops[:]
        # Build a left-associative base, then maybe wrap parentheses
        expr_parts = [nums[0]]
        for i in range(1, len(nums)):
            op = random.choice(ops)
            expr_parts.append(op)
            expr_parts.append(nums[i])
        base_expr = " ".join(expr_parts)

        # Optionally insert parentheses up to max_paren_pairs to vary target
        def maybe_parenthesize(expr, pairs):
            tokens = expr.split()
            if pairs <= 0 or len(tokens) < 5:
                return expr
            # choose spans covering op between two numbers
            spans = []
            for i in range(0, len(tokens)-2, 2):
                spans.append((i, i+2))
            chosen = random.sample(spans, k=min(pairs, len(spans)))
            # apply from rightmost to leftmost to keep indices valid
            chosen.sort(key=lambda x: x[0], reverse=True)
            for start, end in chosen:
                segment = " ".join(tokens[start:end+1])
                tokens = tokens[:start] + [f"( {segment} )"] + tokens[end+1:]
            return " ".join(tokens)

        paren_pairs = random.randint(0, self.max_paren_pairs) if self.max_paren_pairs > 0 else 0
        expr = maybe_parenthesize(base_expr, paren_pairs)

        # Evaluate safely
        value = self._safe_eval(expr)
        # Avoid None or infinite or too large; regenerate a few times if needed
        attempts = 0
        while (value is None or abs(value) > self.max_result_cap or (isinstance(value, float) and (value != value))) and attempts < 10:
            # regenerate numbers or operators slightly
            self.numbers = [max(1, random.randint(1, self.number_scale)) for _ in range(self.num_numbers)]
            nums = [str(n) for n in self.numbers]
            expr_parts = [nums[0]]
            for i in range(1, len(nums)):
                op = random.choice(ops)
                expr_parts.append(op)
                expr_parts.append(nums[i])
            base_expr = " ".join(expr_parts)
            paren_pairs = random.randint(0, self.max_paren_pairs) if self.max_paren_pairs > 0 else 0
            expr = maybe_parenthesize(base_expr, paren_pairs)
            value = self._safe_eval(expr)
            attempts += 1

        if value is None:
            # fallback simple sum target to ensure solvable: enforce '+' only
            expr = " + ".join([str(n) for n in self.numbers])
            value = self._safe_eval(expr)

        # Set target near value within tolerance window; sometimes exact
        jitter = random.randint(-self.target_tolerance, self.target_tolerance) if self.target_tolerance > 0 else 0
        self.target = int(round(value + jitter))
        # Store internal reference (not shown to agent)
        self.used_solution = expr
        self.solution_value = value

    def _safe_eval(self, expr: str) -> Optional[float]:
        # Whitelist tokens: numbers, ops, parentheses, spaces
        if not re.fullmatch(r"[0-9\.\s\+\-\*\/\(\)]+", expr):
            return None
        try:
            # Avoid division by zero by catching exceptions
            val = eval(expr, {"__builtins__": {}}, {})
            if isinstance(val, (int, float)):
                if val == float('inf') or val == float('-inf'):
                    return None
                return float(val)
            return None
        except Exception:
            return None

    def _get_instructions(self) -> str:
        ops_text = ", ".join(self.allowed_ops) if self.allowed_ops else "(unset)"
        return (
            "FormulaFoundry: Build a single arithmetic expression using each given number exactly once to hit the target.\n"
            f"- Target: {self.target}\n"
            f"- Numbers (use each exactly once): {self.numbers}\n"
            f"- Allowed operators: {ops_text}\n"
            f"- Max parentheses pairs allowed: {self.max_paren_pairs}\n"
            f"- Turns remaining: {max(0, self.turn_budget - self.turn_count)} (max per episode)\n"
            "Rules:\n"
            "1) Use every listed number exactly once; do not invent or repeat numbers.\n"
            "2) Only use allowed operators. Unary minus is allowed only as a sign on a number already listed (not new numbers).\n"
            "3) Parentheses pairs cannot exceed the stated limit.\n"
            "4) Division is real-number division; avoid division by zero.\n"
            "5) Submit one full expression per turn using boxed format.\n"
            "Formatting:\n"
            r"- Action must be \boxed{propose expr=YOUR_EXPRESSION}" "\n"
            "Example:\n"
            r"\boxed{propose expr=(3 + 7) * 2 - 4}" "\n"
        )

    def get_task_suffix(self) -> str:
        remain = max(0, self.turn_budget - self.turn_count)
        return (
            f"Problem:\n"
            f"- Target: {self.target}\n"
            f"- Numbers: {self.numbers}\n"
            f"- Operators: {self.allowed_ops}\n"
            f"- Max parentheses pairs: {self.max_paren_pairs}\n"
            f"- Turns remaining: {remain}\n"
            r"Submit in \boxed{propose expr=...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.history = []
        self.used_solution = None
        self.solution_value = None
        self._generate_problem()
        # Align environment turn budget with max_turns but keep internal budget primary
        self.max_turns = max(self.max_turns, self.turn_budget)
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = "INVALID ACTION FORMAT: Use \\boxed{propose expr=...}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get('action') not in ('propose',):
            obs = "UNSUPPORTED ACTION: Only 'propose' is allowed."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        expr = parsed.get('expr', '').strip()
        if not expr:
            obs = "PROTOCOL VIOLATION: Missing expr parameter."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Validate expression tokens and coverage
        token_numbers = re.findall(r"(?<![0-9\.])\-?\d+(?:\.\d+)?", expr)
        # Collect numbers used ignoring sign for coverage, but allow unary minus only on existing numbers
        used_raw = []
        invalid_neg = False
        for t in token_numbers:
            if '.' in t:
                # Non-integer number not allowed if not exactly equal to one of given numbers (with optional unary minus)
                try:
                    f = float(t)
                    # Only allow integer tokens; reject floats
                    invalid_neg = True
                except Exception:
                    invalid_neg = True
            try:
                v = int(t)
            except Exception:
                invalid_neg = True
                continue
            abs_v = abs(v)
            if abs_v not in self.numbers:
                invalid_neg = True
            used_raw.append(v)
        if invalid_neg:
            obs = "FORMAT ERROR: Numbers must be integers from the provided list; unary minus only allowed on those."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Check multiset equality: each provided number must be used exactly once
        from collections import Counter
        provided = Counter(self.numbers)
        used_counts = Counter([abs(v) for v in used_raw])
        if used_counts != provided:
            missing = list((provided - used_counts).elements())
            extra = list((used_counts - provided).elements())
            msg = "COVERAGE FAILURE:"
            if missing:
                msg += f" Missing numbers {missing}."
            if extra:
                msg += f" Extra/duplicated numbers {extra}."
            return msg, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Check allowed operators and parentheses limit
        ops_in_expr = re.findall(r"[\+\-\*\/]", expr)
        if any(o not in self.allowed_ops for o in ops_in_expr):
            obs = "OPERATOR ERROR: Used operator not in allowed set."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        paren_open = expr.count('(')
        paren_close = expr.count(')')
        if paren_open != paren_close:
            obs = "SYNTAX ERROR: Unbalanced parentheses."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        if paren_open > self.max_paren_pairs:
            obs = "PAREN LIMIT EXCEEDED: Too many parentheses pairs."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Evaluate
        value = self._safe_eval(expr)
        if value is None:
            obs = "EVALUATION ERROR: Invalid arithmetic (e.g., division by zero or bad syntax)."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if abs(value) > self.max_result_cap:
            obs = "RESULT MAGNITUDE EXCEEDED: Computed result outside allowed cap."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.history.append(expr)
        diff = abs(value - self.target)
        if diff <= self.target_tolerance:
            obs = f"Success! Expression {expr} evaluates to {value:.4g}, within tolerance {self.target_tolerance} of target {self.target}."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        # Not correct; continue if turns remain
        if self.turn_count >= self.turn_budget or self.turn_count >= self.max_turns:
            obs = f"Failed! Expression {expr} evaluates to {value:.4g}; target {self.target}, tolerance {self.target_tolerance}. Out of turns."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        hint_trend = "too low" if value < self.target else "too high"
        obs = f"Attempt {self.turn_count}: {expr} = {value:.4g} ({hint_trend}). Keep trying."
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        matches = re.findall(r"\\boxed\{(.+?)\}", action, flags=re.DOTALL)
        if not matches:
            return None
        inner = matches[-1].strip()
        parts = inner.split()
        if not parts:
            return None
        action_name = parts[0].lower()
        tokens: Dict[str, Any] = {'action': action_name}
        # expect key=value where value may include spaces; specifically expr=...
        # join back and parse expr=...
        rest = inner[len(parts[0]):].strip()
        # Find expr=...
        em = re.search(r"expr\s*=(.+)$", rest, flags=re.DOTALL)
        if em:
            expr = em.group(1).strip()
            tokens['expr'] = expr
        return tokens

    def sample_random_action(self) -> str:
        # Build a random naive expression using all numbers left-associatively with allowed ops
        if not self.numbers:
            return r"\boxed{propose expr=1+1}"
        ops = self.allowed_ops if self.allowed_ops else ['+']
        parts = [str(self.numbers[0])]
        for n in self.numbers[1:]:
            parts.append(random.choice(ops))
            parts.append(str(n))
        expr = " ".join(parts)
        return r"\boxed{propose expr=" + expr + "}"


class FormulaFoundryEnvWithFeedback(FormulaFoundryEnv):
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
            error_detail["issue"] = "missing_boxed_or_wrong_command"
            hint = r'Use \boxed{propose expr=...} with a full expression.'
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_action"
            hint = r'Only the propose action is supported: \boxed{propose expr=...}.'
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "missing_expr_param"
            hint = r'Include expr=YOUR_EXPRESSION inside the boxed action.'
        elif "format error" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "invalid_number_token"
            hint = "Use only the provided integers; apply unary minus only to those exact numbers."
        elif "coverage failure" in text:
            error_type = "WrongDecision"
            miss = re.findall(r"Missing numbers \[([0-9,\s]+)\]", obs)
            extra = re.findall(r"Extra/duplicated numbers \[([0-9,\s]+)\]", obs)
            error_detail["missing"] = miss[0] if miss else ""
            error_detail["extra"] = extra[0] if extra else ""
            hint = "Ensure each provided number appears exactly once in the expression."
        elif "operator error" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "disallowed_operator"
            hint = f"Use only these operators: {self.allowed_ops}."
        elif "syntax error" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "unbalanced_parentheses"
            hint = "Match every '(' with a ')' and stay within the parentheses limit."
        elif "paren limit exceeded" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "paren_limit"
            hint = f"Use at most {self.max_paren_pairs} parentheses pairs."
        elif "evaluation error" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "invalid_arithmetic"
            hint = "Avoid division by zero and keep expression syntax valid."
        elif "result magnitude exceeded" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "result_too_large"
            hint = f"Re-balance operations to keep result magnitude within {self.max_result_cap}."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
        elif "failed!" in text and "out of turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Aim closer to the target earlier; try different operator combinations or parentheses."
        elif "attempt" in text and "keep trying" in text:
            error_type = "OK"
            # Provide directional hint
            if "too high" in text:
                hint = "Result too high—consider replacing a multiplication with subtraction or use division."
            elif "too low" in text:
                hint = "Result too low—introduce multiplication or reduce subtractions."
            error_detail["progress"] = "attempt_feedback"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "target": self.target,
                "numbers": self.numbers,
                "operators": self.allowed_ops,
                "paren_limit": self.max_paren_pairs,
                "tolerance": self.target_tolerance,
                "turns_left": max(0, self.turn_budget - self.turn_count),
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
            "hint": "Start by chaining numbers with allowed operators; try to reach the target approximately, then refine.",
            "turn": 0,
            "state": {
                "target": self.target,
                "numbers": self.numbers,
                "operators": self.allowed_ops,
                "paren_limit": self.max_paren_pairs,
                "tolerance": self.target_tolerance,
                "turns_left": self.turn_budget,
            },
        }
        return obs, info
