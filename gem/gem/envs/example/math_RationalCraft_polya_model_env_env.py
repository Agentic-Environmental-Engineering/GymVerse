from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class RationalCraftEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 24,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 24

        # Evolvable parameters
        self.complexity_params = {
            # number_count: how many numbers are provided. More numbers grows search space (harder).
            'number_count': (3, 7),
            # max_number: range of integers to sample from [2..max_number]; larger range increases branching/harder.
            'max_number': (6, 20),
            # operation_set_level: unlocks more ops; 1=+,- ; 2=+,-,* ; 3=+,-,*,/ ; 4=+,-,*,/,^ ; richer ops → harder.
            'operation_set_level': (2, 4),
            # must_use_all: 0 or 1 (REVERSED). 0=easier (not required), 1=harder (must use all numbers).
            'must_use_all': (0, 1),
            # hint_budget: number of hint requests allowed. REVERSED: fewer hints = harder.
            'hint_budget': (2, 0),
            # target_den_bound: bound on denominator in reduced form of target; larger bound → more complex target.
            'target_den_bound': (6, 24),
        }
        # Variance per parameter
        self.param_variance = {
            'number_count': 1,
            'max_number': 2,
            'operation_set_level': 0,
            'must_use_all': 0,
            'hint_budget': 1,
            'target_den_bound': 2,
        }

        # Placeholders
        self.number_count: int = 0
        self.max_number: int = 0
        self.operation_set_level: int = 0
        self.must_use_all: int = 0
        self.hint_budget: int = 0
        self.target_den_bound: int = 0

        # State
        self.turn_count: int = 0
        self.available_numbers = []  # list of ints
        self.used_indices = set()    # indices of numbers used
        self.ops = []                # allowed ops as symbols
        self.target_num = 0          # rational target numerator
        self.target_den = 1          # rational target denominator (positive)
        self.current_expr = ""       # last submitted expression text
        self.last_value = None       # (num, den) or None
        self.done = False

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
            # clamp considering reversed possible
            lo = min(min_v, max_v)
            hi = max(min_v, max_v)
            actual = max(lo, min(hi, actual))
            setattr(self, name, int(round(actual)))

    def _gcd(self, a: int, b: int) -> int:
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a

    def _reduce(self, num: int, den: int) -> Tuple[int, int]:
        if den == 0:
            return num, den
        if den < 0:
            num, den = -num, -den
        g = self._gcd(num, den)
        return num // g, den // g

    def _add(self, a, b):  # rationals (n,d)
        (n1, d1), (n2, d2) = a, b
        return self._reduce(n1 * d2 + n2 * d1, d1 * d2)

    def _sub(self, a, b):
        (n1, d1), (n2, d2) = a, b
        return self._reduce(n1 * d2 - n2 * d1, d1 * d2)

    def _mul(self, a, b):
        (n1, d1), (n2, d2) = a, b
        return self._reduce(n1 * n2, d1 * d2)

    def _div(self, a, b):
        (n1, d1), (n2, d2) = a, b
        if n2 == 0:
            return (1, 0)  # represent infinity/invalid
        return self._reduce(n1 * d2, d1 * n2)

    def _pow(self, a, k: int):
        n, d = a
        if k < 0:
            if n == 0:
                return (1, 0)
            k = -k
            n, d = d, n
        # small exponent restriction for safety
        if abs(k) > 5:
            return (1, 0)
        nn = n ** k
        dd = d ** k
        return self._reduce(nn, dd)

    def _ops_for_level(self, lvl: int):
        if lvl <= 1:
            return ['+', '-']
        elif lvl == 2:
            return ['+', '-', '*']
        elif lvl == 3:
            return ['+', '-', '*', '/']
        else:
            return ['+', '-', '*', '/', '^']

    def _random_target(self):
        # Create target as a reduced rational with denominator within bound.
        # Prefer constructible targets given provided numbers by combining a few.
        # Fallback: random a/b.
        # Attempt to sample target from small combinations of available numbers to keep feasibility.
        nums = self.available_numbers[:]
        random.shuffle(nums)
        allowed_ops = self.ops
        candidates = []

        def to_rat(x): return (x, 1)

        # build small pool
        base = [to_rat(x) for x in nums[:min(4, len(nums))]]
        candidates.extend(base)
        if '*' in allowed_ops or '/' in allowed_ops or '+' in allowed_ops or '-' in allowed_ops:
            for i in range(len(base)):
                for j in range(i + 1, len(base)):
                    a, b = base[i], base[j]
                    if '+' in allowed_ops:
                        candidates.append(self._add(a, b))
                        candidates.append(self._sub(a, b))
                        candidates.append(self._sub(b, a))
                    if '*' in allowed_ops:
                        candidates.append(self._mul(a, b))
                    if '/' in allowed_ops:
                        candidates.append(self._div(a, b))
                        candidates.append(self._div(b, a))
            # add a few triple combos
            triples = []
            if len(base) >= 3:
                for i in range(3):
                    a, b, c = random.sample(base, 3)
                    if random.random() < 0.5 and '*' in allowed_ops:
                        x = self._mul(a, b)
                    else:
                        x = self._add(a, b) if '+' in allowed_ops else self._sub(a, b)
                    if random.random() < 0.5 and '/' in allowed_ops:
                        y = self._div(x, c)
                    else:
                        y = self._add(x, c) if '+' in allowed_ops else x
                    triples.append(y)
            candidates.extend(triples)

        # filter by denominator bound
        bounded = []
        for n, d in candidates:
            if d != 0:
                n, d = self._reduce(n, d)
                if abs(d) <= self.target_den_bound and abs(n) <= 50:
                    bounded.append((n, d))
        if bounded:
            return random.choice(bounded)

        # fallback
        den = random.randint(2, max(2, self.target_den_bound))
        num = random.randint(-10, 10)
        if den < 0:
            den = -den
            num = -num
        num, den = self._reduce(num, den)
        return num, den

    def _get_instructions(self) -> str:
        ops_list = ', '.join(self.ops)
        must = "Yes" if self.must_use_all == 1 else "No"
        hint_info = f"You may request up to {self.hint_budget} hint(s) using \\boxed{{hint}}." if self.hint_budget > 0 else "No hints are available at this level."
        numbers_str = ' '.join(str(x) for x in self.available_numbers)
        target_str = f"{self.target_num}/{self.target_den}" if self.target_den != 1 else f"{self.target_num}"
        return (
            "RationalCraft: Construct an exact expression equal to the target rational.\n"
            f"- Available numbers (each can be used at most once): {numbers_str}\n"
            f"- Allowed operations: {ops_list} (use parentheses as needed)\n"
            f"- Must use all numbers? {must}\n"
            f"- Target value: {target_str}\n"
            f"- Expression rules: use integers from the provided set, operations from the allowed list, and parentheses. Division by zero is invalid. Exponent '^' (if allowed) must have an integer exponent in [-5,5].\n"
            f"- {hint_info}\n"
            "Submit either:\n"
            "- An expression: \\boxed{expr: (2+3)/5}\n"
            "- Or a final exact value: \\boxed{answer: a/b} (in reduced form). If you submit answer, it will be checked immediately.\n"
            "You can also submit: \\boxed{use idx=...} to mark a number index as used, or \\boxed{clear} to reset marked usage.\n"
            "Indices are 0-based: refer to available numbers by index if you want to track usage explicitly.\n"
            "Goal: produce an expression/value exactly equal to the target.\n"
        )

    def get_task_suffix(self) -> str:
        numbers = [f"[{i}]:{v}{'*' if i in self.used_indices else ''}" for i, v in enumerate(self.available_numbers)]
        used_note = "All numbers must be used." if self.must_use_all == 1 else "Not required to use all numbers."
        tval = f"{self.target_num}/{self.target_den}" if self.target_den != 1 else f"{self.target_num}"
        last_val = "none" if self.last_value is None else (f"{self.last_value[0]}/{self.last_value[1]}" if self.last_value[1] != 1 else f"{self.last_value[0]}")
        return (
            "State:\n"
            f"- Numbers: {' | '.join(numbers)}\n"
            f"- Requirement: {used_note}\n"
            f"- Target: {tval}\n"
            f"- Last expression: {self.current_expr or 'none'}\n"
            f"- Last value: {last_val}\n"
            "Enter an action in \\boxed{...}:\n"
            "  expr: <expression>\n"
            "  answer: <a/b or integer>\n"
            "  use idx=<index>\n"
            "  clear\n"
            "  hint\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        # sample numbers distinct within small range, allow duplicates rarely
        pool = list(range(2, max(3, self.max_number + 1)))
        self.available_numbers = random.sample(pool, min(self.number_count, len(pool)))
        # occasionally allow one duplicate to avoid triviality
        if len(self.available_numbers) < self.number_count and len(pool) > 0:
            while len(self.available_numbers) < self.number_count:
                self.available_numbers.append(random.choice(pool))
        random.shuffle(self.available_numbers)
        self.used_indices = set()
        self.ops = self._ops_for_level(self.operation_set_level)
        self.current_expr = ""
        self.last_value = None
        self.done = False
        # set target
        self.target_num, self.target_den = 0, 1
        # set target_den_bound already applied; choose a target
        self.target_num, self.target_den = self._random_target()
        # ensure solvable signal: do not generate impossible constraints "must use all" with too many numbers and weak ops for random target
        # we relax must_use_all if extremely tight: if must_use_all==1 and number_count>=7 and operation_set_level<=2, probabilistically set not required
        if self.must_use_all == 1 and self.number_count >= 7 and self.operation_set_level <= 2:
            if random.random() < 0.6:
                self.must_use_all = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _tokenize_expr(self, s: str):
        # simple tokenizer: integers, parentheses, operators + - * / ^
        tokens = []
        i = 0
        while i < len(s):
            c = s[i]
            if c.isspace():
                i += 1
            elif c.isdigit():
                j = i
                while j < len(s) and s[j].isdigit():
                    j += 1
                tokens.append(('INT', int(s[i:j])))
                i = j
            elif c in '()+-*/^':
                tokens.append((c, c))
                i += 1
            else:
                return None
        return tokens

    def _parse_factor(self, tokens, pos):
        # factor: INT | '(' expr ')' | factor '^' INT
        if pos >= len(tokens):
            return None, pos
        tok = tokens[pos]
        if tok[0] == 'INT':
            val = (tok[1], 1)
            pos += 1
        elif tok[0] == '(':
            val, pos = self._parse_expr(tokens, pos + 1)
            if val is None or pos >= len(tokens) or tokens[pos][0] != ')':
                return None, pos
            pos += 1
        else:
            return None, pos
        # exponent optional
        if pos < len(tokens) and tokens[pos][0] == '^':
            pos += 1
            if pos >= len(tokens) or tokens[pos][0] != 'INT':
                return None, pos
            exp = tokens[pos][1]
            pos += 1
            val = self._pow(val, exp)
        return val, pos

    def _parse_term(self, tokens, pos):
        val, pos = self._parse_factor(tokens, pos)
        if val is None:
            return None, pos
        while pos < len(tokens) and tokens[pos][0] in ('*', '/'):
            op = tokens[pos][0]
            if op not in self.ops:
                return None, pos
            pos += 1
            rhs, pos = self._parse_factor(tokens, pos)
            if rhs is None:
                return None, pos
            if op == '*':
                val = self._mul(val, rhs)
            else:
                val = self._div(val, rhs)
        return val, pos

    def _parse_expr(self, tokens, pos):
        val, pos = self._parse_term(tokens, pos)
        if val is None:
            return None, pos
        while pos < len(tokens) and tokens[pos][0] in ('+', '-'):
            op = tokens[pos][0]
            if op not in self.ops:
                return None, pos
            pos += 1
            rhs, pos = self._parse_term(tokens, pos)
            if rhs is None:
                return None, pos
            if op == '+':
                val = self._add(val, rhs)
            else:
                val = self._sub(val, rhs)
        return val, pos

    def _check_numbers_used(self, expr_text: str) -> Tuple[bool, set]:
        # verify that any integers in expression correspond to available numbers and usage <= counts
        toks = self._tokenize_expr(expr_text)
        if toks is None:
            return False, set()
        ints = [v for t, v in toks if t == 'INT']
        # build multiset mapping available numbers
        counts = {}
        for i, v in enumerate(self.available_numbers):
            counts[v] = counts.get(v, 0) + 1
        used = {}
        used_indices = set()
        # greedy assign integers to indices
        for val in ints:
            if counts.get(val, 0) <= 0:
                return False, set()
            counts[val] -= 1
            # mark one index with that value that's not already maxed
            for idx, v in enumerate(self.available_numbers):
                if v == val and idx not in used_indices:
                    used_indices.add(idx)
                    break
            used[val] = used.get(val, 0) + 1
        return True, used_indices

    def _eval_expression(self, expr_text: str) -> Optional[Tuple[int, int]]:
        toks = self._tokenize_expr(expr_text)
        if toks is None:
            return None
        # ensure ops used are allowed
        for t, v in toks:
            if t in ('+', '-', '*', '/', '^') and t not in self.ops:
                return None
        val, pos = self._parse_expr(toks, 0)
        if val is None or pos != len(toks):
            return None
        if val[1] == 0:
            return None
        return self._reduce(val[0], val[1])

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        # Recognize commands: expr: ..., answer: ..., use idx=K, clear, hint
        if inner.lower().startswith("expr:"):
            return {'action': 'expr', 'payload': inner[5:].strip()}
        if inner.lower().startswith("answer:"):
            return {'action': 'answer', 'payload': inner[7:].strip()}
        if inner.lower().startswith("use"):
            mm = re.search(r"idx\s*=\s*(\d+)", inner)
            if mm:
                return {'action': 'use', 'idx': int(mm.group(1))}
            else:
                return None
        if inner.lower() == "clear":
            return {'action': 'clear'}
        if inner.lower() == "hint":
            return {'action': 'hint'}
        # Well-formed boxed but unsupported command
        return {'action': 'unknown', 'raw': inner}

    def sample_random_action(self) -> str:
        # Randomly choose an action format
        choices = []
        if self.hint_budget > 0:
            choices.append(r"\boxed{hint}")
        if self.available_numbers:
            i = random.randrange(len(self.available_numbers))
            choices.append(rf"\boxed{{use idx={i}}}")
        # crude expression from first two numbers
        if len(self.available_numbers) >= 2:
            a, b = self.available_numbers[0], self.available_numbers[1]
            op = random.choice(self.ops)
            expr = f"({a}{op}{b})"
            choices.append(rf"\boxed{{expr: {expr}}}")
        # answer attempt: target itself
        t = f"{self.target_num}/{self.target_den}" if self.target_den != 1 else f"{self.target_num}"
        choices.append(rf"\boxed{{answer: {t}}}")
        return random.choice(choices)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.done:
            return "Episode already finished.", 0.0, True, False, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with one of: expr:, answer:, use idx=K, clear, hint."
            self.done = True
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed['action'] == 'hint':
            if self.hint_budget <= 0:
                obs = "No hints remaining."
                self.done = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix(), "violation": "hint_overuse"}
            self.hint_budget -= 1
            hint = self._generate_hint()
            obs = f"Hint: {hint}"
            if self.turn_count >= self.max_turns:
                self.done = True
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.2, False, False, {"suffix": self.get_task_suffix()}

        if parsed['action'] == 'clear':
            self.used_indices = set()
            obs = "Usage marks cleared."
            if self.turn_count >= self.max_turns:
                self.done = True
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if parsed['action'] == 'use':
            idx = parsed.get('idx', -1)
            if idx < 0 or idx >= len(self.available_numbers):
                obs = "Unsupported action: index out of range."
                self.done = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if idx in self.used_indices:
                obs = f"Index {idx} already marked used."
            else:
                self.used_indices.add(idx)
                obs = f"Marked index {idx} (value {self.available_numbers[idx]}) as used."
            if self.turn_count >= self.max_turns:
                self.done = True
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if parsed['action'] == 'expr':
            expr = parsed['payload']
            self.current_expr = expr
            # basic number usage validation
            ok, used_by_expr = self._check_numbers_used(expr)
            if not ok:
                obs = "Protocol violation: expression uses numbers not in the provided set or exceeds counts."
                self.done = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # evaluate
            val = self._eval_expression(expr)
            if val is None:
                obs = "Expression parse/evaluation failed or used disallowed operator/division by zero."
                self.done = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.last_value = val
            # success check if exactly equals target and usage requirement met
            meets_use = True
            if self.must_use_all == 1:
                # require that all provided numbers appear at least once in expr (tracked by used_by_expr)
                meets_use = (len(used_by_expr) == len(self.available_numbers))
            if val == (self.target_num, self.target_den) and meets_use:
                self.done = True
                obs = f"Success! Expression evaluates exactly to target {self._fmt_rat(val)}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                # continue, shaped reward if close
                closeness = self._closeness(val, (self.target_num, self.target_den))
                r = 0.5 if closeness <= 0.05 else 0.1 if closeness <= 0.2 else 0.0
                obs = f"Evaluated: {self._fmt_rat(val)}. Not equal to target."
                if self.turn_count >= self.max_turns:
                    self.done = True
                    return "Reached max turns", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, r, False, False, {"suffix": self.get_task_suffix()}

        if parsed['action'] == 'answer':
            payload = parsed['payload']
            ans = self._parse_rational_answer(payload)
            if ans is None:
                obs = "Invalid answer format. Provide integer or a/b in reduced form."
                self.done = True
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # must check reduced form equivalence; require exact equality
            if ans == (self.target_num, self.target_den):
                # usage requirement applies only to expr, not to final numeric answer (we allow direct exact report)
                self.done = True
                obs = f"Success! Exact target reported: {self._fmt_rat(ans)}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                self.done = True
                obs = f"Failed! Your answer {self._fmt_rat(ans)} is not the target."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Unknown or malformed but boxed: treat as unsupported action with neutral reward
        obs = "Unsupported action."
        self.done = True
        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _fmt_rat(self, r):
        n, d = r
        return f"{n}/{d}" if d != 1 else f"{n}"

    def _parse_rational_answer(self, s: str) -> Optional[Tuple[int, int]]:
        s = s.strip()
        if re.fullmatch(r"[+-]?\d+", s):
            return (int(s), 1)
        m = re.fullmatch(r"\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*", s)
        if not m:
            return None
        num = int(m.group(1))
        den = int(m.group(2))
        if den == 0:
            return None
        return self._reduce(num, den)

    def _closeness(self, a, b) -> float:
        # approximate closeness on real line
        n1, d1 = a
        n2, d2 = b
        x = n1 / d1
        y = n2 / d2
        diff = abs(x - y)
        scale = max(1.0, abs(y))
        return diff / scale

    def _generate_hint(self) -> str:
        # Provide actionable hint without giving full solution.
        hints = []
        t = self._fmt_rat((self.target_num, self.target_den))
        hints.append(f"Target is {t}. Consider building it as a sum/difference of fractions with small denominators.")
        if '/' in self.ops:
            hints.append("Division can create denominators; try forming a/b with small integers from the set.")
        if '*' in self.ops:
            hints.append("Products can adjust numerators; try scaling a base fraction to reach the target.")
        if '^' in self.ops:
            hints.append("Exponentiation is available; small exponents only. Use it sparingly to avoid invalid results.")
        if self.must_use_all == 1:
            hints.append("All numbers must appear in the expression at least once.")
        if self.last_value is not None:
            n, d = self.last_value
            if d != 0:
                closeness = self._closeness(self.last_value, (self.target_num, self.target_den))
                if closeness > 0.2:
                    hints.append("Your last value is far; try altering the denominator to match the target's structure.")
                else:
                    hints.append("You're close; adjust the numerator slightly using + or - with a small fraction.")
        return random.choice(hints)


class RationalCraftEnvWithFeedback(RationalCraftEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_malformed_boxed"
            hint = "Wrap your command in \\boxed{...} and use one of: expr:, answer:, use idx=K, clear, hint."
        elif "unsupported action" in text and "index out of range" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "index_out_of_range"
            hint = "Use a valid 0-based index shown in the Numbers list."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Allowed commands: expr:, answer:, use idx=K, clear, hint."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "used_unavailable_numbers"
            hint = "Only use the provided integers, each at most once. Check the Numbers list."
        elif "parse/evaluation failed" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "parse_or_disallowed_op_or_div0"
            hint = "Ensure only allowed operators are used, parentheses are balanced, and no division by zero."
        elif text.startswith("failed!"):
            error_type = "WrongDecision"
            error_detail["expected"] = f"{self.target_num}/{self.target_den}" if self.target_den != 1 else f"{self.target_num}"
            m = re.search(r"your answer ([^ ]+)", obs, flags=re.IGNORECASE)
            if m:
                error_detail["got"] = m.group(1)
            hint = "Compute exactly and reduce the fraction to simplest form before answering."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act earlier with expr: to test ideas, then finalize with answer: a/b."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "no hints remaining" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "hint_overuse"
            hint = "Conserve hints; build an expression using allowed operators."
        elif "evaluated:" in text and "not equal to target" in text:
            error_type = "OK"
            error_detail["progress"] = "near_miss" if reward >= 0.5 else "attempt_made"
            hint = "Adjust numerator/denominator via +,-,*,/ with remaining numbers."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "allowed_ops": self.ops,
                "must_use_all": bool(self.must_use_all),
                "hint_budget": self.hint_budget,
                "target": f"{self.target_num}/{self.target_den}" if self.target_den != 1 else f"{self.target_num}",
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
            "hint": "Start by forming a simple fraction close to the target using expr: (a/b).",
            "turn": 0,
            "state": {
                "allowed_ops": self.ops,
                "must_use_all": bool(self.must_use_all),
                "hint_budget": self.hint_budget,
                "target": f"{self.target_num}/{self.target_den}" if self.target_den != 1 else f"{self.target_num}",
            },
        }
        return obs, info
