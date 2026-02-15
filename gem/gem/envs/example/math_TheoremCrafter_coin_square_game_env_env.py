from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class TheoremCrafterEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 6,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 6

        # Evolvable parameters
        self.complexity_params = {
            # number of distinct task families to sample from: more families increases cognitive branching
            'task_family_span': (2, 6),
            # integer operand magnitude bound: larger absolute values increase arithmetic difficulty
            'operand_bound': (9, 60),
            # expression depth/steps for arithmetic expressions, harder as it requires more operations
            'expr_steps': (2, 6),
            # linear equation coefficient magnitude: larger -> harder isolation and potential fractions
            'lin_coeff_bound': (5, 20),
            # maximum modulus for modular arithmetic: larger mod -> harder reductions and CRT-like reasoning
            'modulus_bound': (11, 97),
        }

        self.param_variance = {
            'task_family_span': 0,      # small discrete range; keep stable per level
            'operand_bound': 5,         # ~10% variance across range
            'expr_steps': 1,            # +/-1 step variance
            'lin_coeff_bound': 2,       # small jitter
            'modulus_bound': 5,         # small jitter
        }

        # Placeholders for evolvable params
        self.task_family_span: int = 0
        self.operand_bound: int = 0
        self.expr_steps: int = 0
        self.lin_coeff_bound: int = 0
        self.modulus_bound: int = 0

        # Other state
        self.turn_count: int = 0
        self.active: bool = False
        self.instance: Dict[str, Any] = {}
        self.hidden_answer: Optional[str] = None
        self.format_rule: Dict[str, Any] = {}

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (mn, mx) in self.complexity_params.items():
            center = mn + (mx - mn) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
                    lo, hi = (mx, mn) if mn > mx else (mn, mx)
                    val = max(lo, min(hi, val))
            setattr(self, name, int(round(val)))

    def _get_instructions(self) -> str:
        return (
            "You are TheoremCrafter. Solve the presented math task and submit a single numeric answer.\n"
            "Rules:\n"
            "- Follow the exact formatting requirement (e.g., integer, reduced fraction a/b, value modulo m).\n"
            "- Submit only the final scalar using \\boxed{...}. No text, no units.\n"
            "- One submission per episode; invalid format ends the episode with a penalty.\n"
            "Action format:\n"
            "- Submit the value as \\boxed{ANSWER} where ANSWER is a number or reduced fraction a/b as instructed.\n"
            "Examples:\n"
            f"- {r'\\boxed{7}'}\n"
            f"- {r'\\boxed{-13}'}\n"
            f"- {r'\\boxed{5/12}'} (for reduced fractions only)\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        if not self.instance:
            return "No active task.\nEnter your answer in \\boxed{...}."
        lines.append("Current task:")
        lines.append(self.instance.get("prompt", ""))
        lines.append("")
        lines.append("Formatting requirement:")
        lines.append(self.instance.get("format_text", "Provide a single number."))
        lines.append("")
        lines.append("Enter your answer in \\boxed{...}.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.active = True
        self.instance = {}
        self.hidden_answer = None
        self.format_rule = {}
        self._sample_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    # --------- Task sampling and utilities ---------
    def _sample_instance(self):
        families = [
            "arith_expr",
            "linear_equation",
            "mod_arith",
            "gcd_lcm",
            "fraction_ops",
            "poly_eval",
        ]
        k = max(1, min(len(families), self.task_family_span))
        chosen = random.sample(families, k=k)
        family = random.choice(chosen)

        if family == "arith_expr":
            prompt, ans, fmt = self._make_arith_expr()
        elif family == "linear_equation":
            prompt, ans, fmt = self._make_linear_equation()
        elif family == "mod_arith":
            prompt, ans, fmt = self._make_mod_arith()
        elif family == "gcd_lcm":
            prompt, ans, fmt = self._make_gcd_lcm()
        elif family == "fraction_ops":
            prompt, ans, fmt = self._make_fraction_ops()
        else:
            prompt, ans, fmt = self._make_poly_eval()

        self.instance = {
            "family": family,
            "prompt": prompt,
            "format_text": fmt["text"],
        }
        self.hidden_answer = ans
        self.format_rule = fmt

    def _rand_int(self, bound):
        a = random.randint(-bound, bound)
        while a == 0:
            a = random.randint(-bound, bound)
        return a

    def _gcd(self, a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a

    def _reduce_frac(self, num, den):
        if den < 0:
            num, den = -num, -den
        g = self._gcd(num, den)
        return num // g, den // g

    def _int_or_frac_str(self, num, den):
        num, den = self._reduce_frac(num, den)
        if den == 1:
            return str(num)
        else:
            return f"{num}/{den}"

    # ----- Family generators -----
    def _make_arith_expr(self):
        steps = max(2, self.expr_steps)
        ops = ['+', '-', '*']
        bound = max(5, self.operand_bound)
        values = [random.randint(-bound, bound) for _ in range(steps + 1)]
        # ensure no all-zero triviality
        if all(v == 0 for v in values):
            values[0] = 1
        choices = [random.choice(ops) for _ in range(steps)]
        # build evaluable expression with parentheses to avoid ambiguity
        expr = str(values[0])
        current_value = values[0]
        for i, op in enumerate(choices):
            v = values[i + 1]
            if op == '*':
                expr = f"({expr})*({v})"
                current_value = current_value * v
            elif op == '+':
                expr = f"({expr})+({v})"
                current_value = current_value + v
            else:
                expr = f"({expr})-({v})"
                current_value = current_value - v

        prompt = f"Compute the value of the expression:\nE = {expr}\nProvide the integer result."
        ans = str(current_value)
        fmt = {"type": "integer", "text": "Answer must be a single integer (no spaces)."}
        return prompt, ans, fmt

    def _make_linear_equation(self):
        # ax + b = c or a(x + d) = e or two-step linear with potential fraction solution
        bound = max(3, self.lin_coeff_bound)
        a = self._rand_int(bound)
        b = random.randint(-bound, bound)
        c = random.randint(-bound, bound)
        # ensure not degenerate (a != 0)
        # solution x = (c - b)/a
        num = c - b
        den = a
        num, den = self._reduce_frac(num, den)
        ans_str = self._int_or_frac_str(num, den)
        # present as ax + b = c
        prompt = f"Solve for x: {a}*x + {b} = {c}. Give the answer as an integer or reduced fraction a/b."
        fmt = {"type": "fraction_or_int", "text": "Answer must be an integer or reduced fraction a/b."}
        return prompt, ans_str, fmt

    def _make_mod_arith(self):
        # compute arithmetic modulo m: e.g., (a*b + c) mod m or power small exponent
        m = random.randint(max(7, self.modulus_bound // 2), self.modulus_bound)
        a = random.randint(1, self.operand_bound)
        b = random.randint(1, self.operand_bound)
        c = random.randint(-self.operand_bound, self.operand_bound)
        # randomly choose a form
        form = random.choice(["ab_plus_c", "a_pow_b_plus_c"])
        if form == "ab_plus_c":
            expr_text = f"({a} * {b} + {c}) mod {m}"
            val = (a * b + c) % m
        else:
            # keep exponent modest to avoid huge ints
            b_small = random.randint(2, max(3, min(8, self.expr_steps + 3)))
            expr_text = f"({a}^{b_small} + {c}) mod {m}"
            val = (pow(a, b_small, m) + c) % m
            val %= m
        prompt = f"Compute the value of {expr_text} as an integer in [0, {m-1}]."
        ans = str(val)
        fmt = {"type": "mod", "mod": m, "text": f"Answer must be an integer between 0 and {m-1} inclusive."}
        return prompt, ans, fmt

    def _make_gcd_lcm(self):
        a = random.randint(2, max(8, self.operand_bound))
        b = random.randint(2, max(8, self.operand_bound))
        op = random.choice(["gcd", "lcm"])
        if op == "gcd":
            val = self._gcd(a, b)
            prompt = f"Compute gcd({a}, {b}). Provide the integer result."
            ans = str(val)
        else:
            g = self._gcd(a, b)
            lcm = abs(a * b) // g
            prompt = f"Compute lcm({a}, {b}). Provide the integer result."
            ans = str(lcm)
        fmt = {"type": "integer", "text": "Answer must be a single integer."}
        return prompt, ans, fmt

    def _make_fraction_ops(self):
        # compute sum/diff/product of two fractions and reduce
        def rand_frac():
            num = random.randint(-self.operand_bound, self.operand_bound)
            den = random.randint(2, max(3, self.operand_bound))
            if den == 0:
                den = 2
            if num == 0:
                num = 1
            n, d = self._reduce_frac(num, den)
            return n, d

        n1, d1 = rand_frac()
        n2, d2 = rand_frac()
        op = random.choice(['+', '-', '*'])
        if op == '+':
            num = n1 * d2 + n2 * d1
            den = d1 * d2
        elif op == '-':
            num = n1 * d2 - n2 * d1
            den = d1 * d2
        else:
            num = n1 * n2
            den = d1 * d2
        num, den = self._reduce_frac(num, den)
        ans = self._int_or_frac_str(num, den)
        prompt = f"Compute and reduce to lowest terms: ({n1}/{d1}) {op} ({n2}/{d2})."
        fmt = {"type": "fraction_or_int", "text": "Answer must be an integer or reduced fraction a/b."}
        return prompt, ans, fmt

    def _make_poly_eval(self):
        # evaluate polynomial with integer coefficients at integer point
        degree = max(1, min(4, self.expr_steps))  # cap degree for solvability
        coeffs = []
        for _ in range(degree + 1):
            c = random.randint(-self.operand_bound, self.operand_bound)
            coeffs.append(c)
        if all(c == 0 for c in coeffs):
            coeffs[0] = 1
        x0 = random.randint(-max(3, self.expr_steps + 2), max(3, self.expr_steps + 2))
        # Build text and compute
        terms = []
        for i, c in enumerate(coeffs):
            p = degree - i
            if p == 0:
                terms.append(f"{c}")
            elif p == 1:
                terms.append(f"{c}*x")
            else:
                terms.append(f"{c}*x^{p}")
        poly = " + ".join(terms)
        # Evaluate via Horner
        val = 0
        for c in coeffs:
            val = val * x0 + c
        prompt = f"Evaluate P(x) = {poly} at x = {x0}. Provide the integer result."
        fmt = {"type": "integer", "text": "Answer must be a single integer."}
        return prompt, str(val), fmt

    # --------- Core interaction ---------
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        if not self.active:
            obs = "Episode already finished."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        parsed = self._parse_action(action)
        if parsed is None or 'answer' not in parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a single numeric answer."
            self.active = False
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        user_ans = parsed['answer']

        # Validate format according to rule
        ok_format, norm_user, format_msg = self._validate_and_normalize(user_ans, self.format_rule)

        if not ok_format:
            obs = f"FORMAT VIOLATION: {format_msg}"
            self.active = False
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        # Compare to hidden answer (already normalized based on task creation)
        correct = (norm_user == self.hidden_answer)

        if correct:
            obs = "Success! Your answer is correct."
            self.active = False
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        # Allow multiple attempts until max_turns
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            self.active = False
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = "Incorrect. Try again while respecting the format."
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

    def _validate_and_normalize(self, text: str, rule: Dict[str, Any]) -> Tuple[bool, Optional[str], str]:
        t = text.strip()
        # numeric or fraction pattern
        int_pat = r'^[+-]?\d+$'
        frac_pat = r'^[+-]?\d+\/[+-]?\d+$'

        rtype = rule.get("type", "integer")
        if rtype == "integer":
            if not re.match(int_pat, t):
                return False, None, "Expected an integer (e.g., -12, 0, 45)."
            return True, str(int(t)), ""
        elif rtype == "fraction_or_int":
            if re.match(int_pat, t):
                return True, str(int(t)), ""
            if re.match(frac_pat, t):
                try:
                    num_s, den_s = t.split('/')
                    num = int(num_s)
                    den = int(den_s)
                    if den == 0:
                        return False, None, "Denominator cannot be zero."
                    num, den = self._reduce_frac(num, den)
                    if den == 1:
                        return True, str(num), ""
                    return True, f"{num}/{den}", ""
                except Exception:
                    return False, None, "Malformed fraction."
            return False, None, "Expected an integer or reduced fraction a/b."
        elif rtype == "mod":
            if not re.match(int_pat, t):
                return False, None, "Expected a nonnegative integer within the modulo range."
            m = int(rule.get("mod", 1))
            v = int(t)
            if not (0 <= v < m):
                return False, None, f"Value must be in [0, {m-1}]."
            return True, str(v), ""
        else:
            # Fallback: integer
            if not re.match(int_pat, t):
                return False, None, "Expected an integer."
            return True, str(int(t)), ""

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        s = action.strip()
        m = re.search(r"\\boxed\{(.+?)\}\s*$", s, flags=re.DOTALL)
        if not m:
            return None
        inside = m.group(1).strip()
        if inside == "":
            return None
        return {"action": "submit", "answer": inside}

    def sample_random_action(self) -> str:
        # sample consistent with current format
        t = self.format_rule.get("type", "integer")
        if t == "integer":
            guess = random.randint(-self.operand_bound, self.operand_bound)
            return rf"\boxed{{{guess}}}"
        elif t == "mod":
            m = int(self.format_rule.get("mod", 10))
            guess = random.randint(0, max(0, m - 1))
            return rf"\boxed{{{guess}}}"
        else:
            if random.random() < 0.5:
                guess = random.randint(-self.operand_bound, self.operand_bound)
                return rf"\boxed{{{guess}}}"
            # fraction
            a = random.randint(-self.operand_bound, self.operand_bound) or 1
            b = random.randint(2, max(3, self.operand_bound))
            g = self._gcd(a, b)
            a //= g
            b //= g
            if b == 1:
                return rf"\boxed{{{a}}}"
            return rf"\boxed{{{a}/{b}}}"


class TheoremCrafterEnvWithFeedback(TheoremCrafterEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed"
            hint = "Submit only your numeric answer inside \\boxed{...} without extra text."
        elif "format violation" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "format_rule_broken"
            fmt = self.instance.get("format_text", "")
            error_detail["requirement"] = fmt
            hint = f"Match the format exactly. Example following the rule: {self._format_example()}"
        elif "unsupported" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_action"
            hint = "Only submit a numeric answer; do not call functions."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan quickly. Compute step-by-step and submit within the turn limit."
        elif "incorrect" in text:
            error_type = "WrongDecision"
            fam = self.instance.get("family", "")
            error_detail["family"] = fam
            if fam == "fraction_ops" or fam == "linear_equation":
                hint = "Ensure the result is reduced; convert to integer if denominator becomes 1."
            elif fam == "mod_arith":
                m = self.format_rule.get("mod")
                hint = f"Reduce your result modulo {m} to be within [0, {m-1}]."
            elif fam == "arith_expr":
                hint = "Carefully respect parentheses and operation order; recompute step-by-step."
            elif fam == "gcd_lcm":
                hint = "Recheck prime factors or apply Euclidean algorithm carefully."
            else:
                hint = "Double-check substitutions and arithmetic, then resubmit."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["family"] = self.instance.get("family", None)
            diagnostic["format_rule"] = self.instance.get("format_text", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start", "family": self.instance.get("family")},
            "hint": "Read the format rule first, then compute carefully before submitting once.",
            "turn": 0,
            "family": self.instance.get("family", None),
            "format_rule": self.instance.get("format_text", None),
        }
        return obs, info

    def _format_example(self) -> str:
        t = self.format_rule.get("type", "integer")
        if t == "integer":
            return r"\boxed{12}"
        elif t == "fraction_or_int":
            return r"\boxed{5/7}"
        elif t == "mod":
            m = self.format_rule.get("mod", 10)
            ex = min(3, max(0, m - 1))
            return rf"\boxed{{{ex}}}"
        return r"\boxed{0}"