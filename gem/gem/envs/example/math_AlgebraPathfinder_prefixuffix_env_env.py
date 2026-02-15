from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AlgebraPathfinderEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity_level = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters
        self.complexity_params = {
            # number of polynomial terms in initial expression (more terms = harder)
            "num_terms": (2, 7),
            # maximum exponent for x in any term (higher degrees = harder)
            "max_degree": (1, 5),
            # coefficient magnitude bound (larger coefficients = slightly harder mental algebra)
            "coef_bound": (3, 12),
            # number of distinct linear factors for factorization tasks (more factors = harder)
            "num_factors": (2, 5),
            # presence of nested parentheses depth for simplification tasks (deeper nesting = harder)
            "paren_depth": (0, 3),
            # number of equation steps for solve tasks (more structural pieces = harder)
            "solve_pieces": (1, 4),
        }
        self.param_variance = {
            "num_terms": 1,
            "max_degree": 1,
            "coef_bound": 2,
            "num_factors": 1,
            "paren_depth": 1,
            "solve_pieces": 1,
        }

        # Placeholder attributes
        self.num_terms: int = 0
        self.max_degree: int = 0
        self.coef_bound: int = 0
        self.num_factors: int = 0
        self.paren_depth: int = 0
        self.solve_pieces: int = 0

        # State
        self.turn_count: int = 0
        self.task_type: str = ""  # "simplify", "expand", "factor", "evaluate_at", "solve_for_x"
        self.current_expr: str = ""
        self.target_answer: str = ""
        self.hidden_x_value: Optional[int] = None  # used by evaluate_at
        self.history: List[str] = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity_level - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    if actual_value < lo:
                        actual_value = lo
                    if actual_value > hi:
                        actual_value = hi
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are navigating algebraic transformations to reach a canonical goal.\n"
            "Task types include: simplify, expand, factor, evaluate_at, solve_for_x.\n"
            "Available actions (use \\boxed{...}):\n"
            "- simplify: combine like terms, reduce fractions, remove unnecessary parentheses\n"
            "- expand: distribute products into a sum of terms\n"
            "- factor: factor polynomials into products of factors\n"
            "- substitute x=<int>: replace x with an integer and simplify\n"
            "- solve: attempt to solve the equation for x\n"
            "- submit answer=<value>: submit final numeric x or polynomial expression\n"
            "Rules:\n"
            "- Each action updates the expression deterministically.\n"
            "- Submit only when you believe the target is reached.\n"
            "- Format: \\boxed{action key=value} or \\boxed{action}\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        status = f"Turns: {self.turn_count}/{self.max_turns}"
        return (
            f"Task: {self.task_type}\n"
            f"Current expression: {self.current_expr}\n"
            f"{status}\n"
            "Enter your action in \\boxed{...} format."
        )

    # --- Domain utilities (lightweight, deterministic string-based algebra subset) ---
    def _canonical_poly(self, expr: str) -> str:
        # limited canonicalization: collect terms like ax^k, sort by degree desc, then by coeff
        # supports integers, x, x^k, +, -, *, parentheses minimally (only generated forms)
        # Parsing strategy: break into terms assuming it's already a sum of monomials "c*x^k"
        # This is a simplified canonicalizer for expressions we generate internally.
        tokens = expr.replace(" ", "")
        # Replace '-' with '+-' to split by '+'
        if tokens.startswith("-"):
            tokens = "-" + tokens[1:]
        tokens = tokens.replace("+-", "+-")
        parts = []
        buf = ""
        i = 0
        while i < len(tokens):
            ch = tokens[i]
            if ch == '+' and buf != "":
                parts.append(buf)
                buf = ""
            else:
                buf += ch
            i += 1
        if buf:
            parts.append(buf)

        def parse_monomial(s: str):
            # accept forms: "c", "c*x", "c*x^k", "x", "x^k", "-x", "-c", etc.
            if s == "":
                return (0, 0)  # zero term
            coef = 0
            deg = 0
            if "x" not in s:
                try:
                    coef = int(s)
                except:
                    coef = 0
                deg = 0
                return (coef, deg)
            # split by '*'
            # Cases: "-x", "x", "x^k", "c*x", "c*x^k", "-c*x", "-c*x^k"
            base_coef = 1
            sign = 1
            ss = s
            if ss.startswith("-"):
                sign = -1
                ss = ss[1:]
            if ss.startswith("+"):
                ss = ss[1:]
            if "*" in ss:
                pieces = ss.split("*")
            else:
                pieces = [ss]
            # Identify numeric coef and x-part
            for p in pieces:
                if p == "":
                    continue
                if p == "x":
                    deg = max(deg, 1) if deg != 0 else 1
                elif p.startswith("x^"):
                    try:
                        k = int(p[2:])
                    except:
                        k = 1
                    deg = k
                elif p.startswith("x"):
                    deg = max(deg, 1) if deg != 0 else 1
                else:
                    # numeric?
                    try:
                        base_coef *= int(p)
                    except:
                        # unexpected token -> treat as 0 to avoid breaking
                        return (0, 0)
            coef = sign * base_coef
            return (coef, deg)

        # collect like degrees
        acc: Dict[int, int] = {}
        for p in parts:
            if p == "":
                continue
            c, d = parse_monomial(p)
            if d not in acc:
                acc[d] = 0
            acc[d] += c

        # remove zero coefficients
        acc = {d: c for d, c in acc.items() if c != 0}
        if not acc:
            return "0"
        # sort by degree descending
        terms = []
        for d in sorted(acc.keys(), reverse=True):
            c = acc[d]
            if d == 0:
                terms.append(str(c))
            elif d == 1:
                if c == 1:
                    terms.append("x")
                elif c == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{c}*x")
            else:
                if c == 1:
                    terms.append(f"x^{d}")
                elif c == -1:
                    terms.append(f"-x^{d}")
                else:
                    terms.append(f"{c}*x^{d}")
        # join with plus, fix signs
        out = ""
        for t in terms:
            if out == "":
                out = t
            else:
                if t.startswith("-"):
                    out += t  # already has minus
                else:
                    out += "+" + t
        # normalize "+-" sequences
        out = out.replace("+-", "-")
        return out

    def _expand_pair(self, a: str, b: str) -> str:
        # expand (a)*(b) assuming a,b are linear sums "p+q+..." in canonical-ish form
        # We'll split by '+' after making '+-' explicit
        def split_sum(s: str) -> List[str]:
            s = s.replace(" ", "")
            s = s.replace("+-", "+-")
            parts = []
            buf = ""
            for ch in s:
                if ch == '+' and buf != "":
                    parts.append(buf)
                    buf = ""
                else:
                    buf += ch
            if buf:
                parts.append(buf)
            return parts

        A = split_sum(a)
        B = split_sum(b)

        # multiply all pairs as monomials using the canonical poly adder
        # Use _canonical_poly to add results: we will build sum string and canonicalize.
        prod_terms = []
        for t1 in A:
            for t2 in B:
                # multiply two monomials represented canonically
                c1 = self._canonical_poly(t1)  # ensures monomial form
                c2 = self._canonical_poly(t2)
                # Now parse c1 and c2 back to (coef,deg)
                def parse_m(s: str):
                    # c*x^d or x^d or c or x
                    # We can reuse parse_monomial by making a tiny wrapper:
                    return self._canonical_to_monomial(s)
                ccoef1, cdeg1 = parse_m(c1)
                ccoef2, cdeg2 = parse_m(c2)
                coef = ccoef1 * ccoef2
                deg = cdeg1 + cdeg2
                if deg == 0:
                    prod_terms.append(str(coef))
                elif deg == 1:
                    if coef == 1:
                        prod_terms.append("x")
                    elif coef == -1:
                        prod_terms.append("-x")
                    else:
                        prod_terms.append(f"{coef}*x")
                else:
                    if coef == 1:
                        prod_terms.append(f"x^{deg}")
                    elif coef == -1:
                        prod_terms.append(f"-x^{deg}")
                    else:
                        prod_terms.append(f"{coef}*x^{deg}")
        # sum up and canonicalize
        s = "+".join(prod_terms).replace("+-", "-")
        return self._canonical_poly(s)

    def _canonical_to_monomial(self, s: str):
        # s is already a canonical single monomial or sum; take sum of one term or total degree if sum
        # If sum, return aggregated (coef,deg) only if it is a single monomial; else return approx
        # Here we assume s is a monomial from _canonical_poly when used above.
        s = s.replace(" ", "")
        if "+" in s or (s.count("x") > 1):
            # not a single monomial -> return zero to be safe
            return (0, 0)
        # handle numbers
        if "x" not in s:
            try:
                return (int(s), 0)
            except:
                return (0, 0)
        # sign
        sign = 1
        if s.startswith("-"):
            sign = -1
            s = s[1:]
        # x-only cases
        if s == "x":
            return (sign * 1, 1)
        if s.startswith("x^"):
            try:
                d = int(s[2:])
            except:
                d = 1
            return (sign * 1, d)
        if s.startswith("x"):
            return (sign * 1, 1)
        # coefficient forms
        if "*x^" in s:
            coef_str, d_str = s.split("*x^")
            try:
                coef = int(coef_str)
                d = int(d_str)
            except:
                coef, d = 0, 0
            return (sign * coef, d)
        if "*x" in s:
            coef_str = s.split("*x")[0]
            try:
                coef = int(coef_str)
            except:
                coef = 0
            return (sign * coef, 1)
        return (0, 0)

    def _combine_like_terms(self, expr: str) -> str:
        return self._canonical_poly(expr)

    def _factor_quadratic(self, expr: str) -> Optional[str]:
        # try factoring ax^2+bx+c into (px+q)(rx+s) with integer p,q,r,s
        canon = self._canonical_poly(expr)
        # extract a,b,c
        # parse degrees 2,1,0
        def coef_of_degree(s: str, d: int) -> int:
            # rough parse by matching against canonical pattern
            s = s.replace(" ", "")
            # split into terms as before
            terms = []
            buf = ""
            for ch in s:
                if ch == '+' and buf != "":
                    terms.append(buf)
                    buf = ""
                else:
                    buf += ch
            if buf:
                terms.append(buf)
            total = 0
            for t in terms:
                cc, dd = self._canonical_to_monomial(t)
                if dd == d:
                    total += cc
            return total
        a = coef_of_degree(canon, 2)
        b = coef_of_degree(canon, 1)
        c = coef_of_degree(canon, 0)
        if a == 0:
            return None
        # attempt integer factorization
        # (px+q)(rx+s) = pr x^2 + (ps+qr)x + qs
        # search small factors within coef_bound
        bound = max(12, abs(a) * abs(c) * 2)
        candidates = list(range(-bound, bound + 1))
        for p in candidates:
            for r in candidates:
                if p * r != a:
                    continue
                # solve ps+qr = b and q*s = c
                for q in candidates:
                    if q == 0 and c != 0:
                        pass
                    for s_ in candidates:
                        if q * s_ != c:
                            continue
                        if p * s_ + q * r == b:
                            # build factors
                            def lin_str(alpha, beta):
                                if alpha == 1:
                                    lead = "x"
                                elif alpha == -1:
                                    lead = "-x"
                                else:
                                    lead = f"{alpha}*x"
                                if beta == 0:
                                    return lead
                                if beta > 0:
                                    return f"{lead}+{beta}"
                                else:
                                    return f"{lead}{beta}"
                            left = lin_str(p, q)
                            right = lin_str(r, s_)
                            return f"({left})*({right})"
        return None

    def _expand_all_parentheses(self, expr: str) -> str:
        # expand nested products of linear sums pairwise
        # match "(...)*(...)" iteratively until no parentheses remain
        s = expr.replace(" ", "")
        # naive detection: find innermost "(...)" groups and expand when "*(" follows/before
        # We'll repeatedly search for pattern "(A)*(B)" where A,B have no unmatched parentheses
        def find_pair(s: str):
            # locate first balanced pair sequence "(A)*(B)"
            for i in range(len(s)):
                if s[i] == '(':
                    depth = 1
                    j = i + 1
                    while j < len(s) and depth > 0:
                        if s[j] == '(':
                            depth += 1
                        elif s[j] == ')':
                            depth -= 1
                        j += 1
                    if depth != 0:
                        return None
                    A = s[i + 1 : j - 1]
                    # expect "*(" next
                    if j < len(s) - 1 and s[j] == '*' and s[j + 1] == '(':
                        # find matching for second
                        k = j + 2
                        depth2 = 1
                        l = k
                        while l < len(s) and depth2 > 0:
                            if s[l] == '(':
                                depth2 += 1
                            elif s[l] == ')':
                                depth2 -= 1
                            l += 1
                        if depth2 != 0:
                            return None
                        B = s[k : l - 1]
                        return (i, j + 1, l, A, B)
            return None

        while True:
            pair = find_pair(s)
            if not pair:
                break
            i, op_idx, l, A, B = pair
            expanded = self._expand_pair(self._canonical_poly(A), self._canonical_poly(B))
            s = s[:i] + expanded + s[l:]
            s = self._canonical_poly(s)
        return self._canonical_poly(s)

    def _evaluate_at_int(self, expr: str, xval: int) -> int:
        # evaluate canonical polynomial at integer x
        canon = self._canonical_poly(expr)
        # split into terms
        terms = []
        buf = ""
        for ch in canon:
            if ch == '+' and buf != "":
                terms.append(buf)
                buf = ""
            else:
                buf += ch
        if buf:
            terms.append(buf)
        total = 0
        for t in terms:
            c, d = self._canonical_to_monomial(t)
            total += c * (xval ** d)
        return total

    # --- Problem generation ---
    def _gen_simplify_instance(self):
        # Build nested sums and simple products then require simplify to canonical sum
        # Start with random monomials then add parentheses depth
        terms = []
        for _ in range(self.num_terms):
            coef = random.randint(-self.coef_bound, self.coef_bound)
            while coef == 0:
                coef = random.randint(-self.coef_bound, self.coef_bound)
            deg = random.randint(0, self.max_degree)
            if deg == 0:
                t = f"{coef}"
            elif deg == 1:
                if coef == 1:
                    t = "x"
                elif coef == -1:
                    t = "-x"
                else:
                    t = f"{coef}*x"
            else:
                if coef == 1:
                    t = f"x^{deg}"
                elif coef == -1:
                    t = f"-x^{deg}"
                else:
                    t = f"{coef}*x^{deg}"
            terms.append(t)
        base = "+".join(terms).replace("+-", "-")
        expr = base
        for _ in range(self.paren_depth):
            # wrap a random slice in parentheses and add with another slice
            expr = f"({expr})+({base})"
        expr = self._combine_like_terms(expr)
        # now inject a trivial zero-sum to require combination
        if random.random() < 0.5:
            expr = f"({expr})+({self._canonical_poly(expr.replace('-', '+-'))})".replace("+-", "-")
            expr = self._combine_like_terms(expr)
        target = self._canonical_poly(expr)
        return expr, target

    def _gen_expand_instance(self):
        # generate product of linear sums up to paren_depth+1 factors
        k = max(2, min(3 + self.paren_depth, 4))
        sums = []
        for _ in range(k):
            a = random.randint(-self.coef_bound, self.coef_bound)
            b = random.randint(-self.coef_bound, self.coef_bound)
            while a == 0 and b == 0:
                a = random.randint(-self.coef_bound, self.coef_bound)
                b = random.randint(-self.coef_bound, self.coef_bound)
            sa = ""  # string for ax
            if a == 0:
                sa = ""
            elif a == 1:
                sa = "x"
            elif a == -1:
                sa = "-x"
            else:
                sa = f"{a}*x"
            if b == 0:
                ssum = sa if sa != "" else "0"
            else:
                if sa == "" or sa == "0":
                    ssum = f"{b}"
                else:
                    if b > 0:
                        ssum = f"{sa}+{b}"
                    else:
                        ssum = f"{sa}{b}"
            sums.append(f"({ssum})")
        expr = "*".join(sums)
        expanded = self._expand_all_parentheses(expr)
        target = self._canonical_poly(expanded)
        return expr, target

    def _gen_factor_instance(self):
        # generate a quadratic or biquadratic that is factorable with integers
        # We'll create from known factors
        factors = []
        nf = self.num_factors
        # Keep it manageable: at most 2 quadratic-equivalent by multiplying linear factors pairwise
        for _ in range(nf):
            p = random.randint(-self.coef_bound, self.coef_bound)
            q = random.randint(-self.coef_bound, self.coef_bound)
            while p == 0 and q == 0:
                p = random.randint(-self.coef_bound, self.coef_bound)
                q = random.randint(-self.coef_bound, self.coef_bound)
            # linear factor p*x+q
            if p == 1:
                left = "x"
            elif p == -1:
                left = "-x"
            else:
                left = f"{p}*x"
            if q == 0:
                factor = f"({left})"
            elif q > 0:
                factor = f"({left}+{q})"
            else:
                factor = f"({left}{q})"
            factors.append(factor)
        expr = "*".join(factors)
        expanded = self._expand_all_parentheses(expr)
        # the target is the factored form in canonical ordering of factors (we can accept any order by canonical compare on expanded)
        target = self._canonical_poly(expanded)
        # The answer we expect is any factorization whose expansion matches target; for submission we accept either
        return expr, target

    def _gen_evaluate_instance(self):
        # produce polynomial and a hidden x value to evaluate
        expr, target = self._gen_expand_instance() if random.random() < 0.5 else self._gen_simplify_instance()
        xval = random.randint(-5, 5)
        self.hidden_x_value = xval
        value = self._evaluate_at_int(target, xval)
        return expr, str(value)

    def _gen_solve_instance(self):
        # build linear or quadratic solvable equations
        # structure: combine solve_pieces linear constraints into one equation
        # build left and right as polynomials then set LHS = RHS
        def random_lin():
            a = random.randint(-self.coef_bound, self.coef_bound)
            b = random.randint(-self.coef_bound, self.coef_bound)
            while a == 0:
                a = random.randint(-self.coef_bound, self.coef_bound)
            if a == 1:
                sa = "x"
            elif a == -1:
                sa = "-x"
            else:
                sa = f"{a}*x"
            if b == 0:
                return sa
            if b > 0:
                return f"{sa}+{b}"
            return f"{sa}{b}"

        left = random_lin()
        right = random_lin()
        # Optionally add another linear piece
        for _ in range(max(0, self.solve_pieces - 1)):
            if random.random() < 0.5:
                left = self._canonical_poly(f"({left})+({random_lin()})")
            else:
                right = self._canonical_poly(f"({right})+({random_lin()})")
        # Solve ax + b = cx + d => (a-c)x = d-b
        # Extract coefficients
        def lin_coef(expr: str):
            canon = self._canonical_poly(expr)
            a = 0
            b = 0
            # parse like earlier
            terms = []
            buf = ""
            for ch in canon:
                if ch == '+' and buf != "":
                    terms.append(buf)
                    buf = ""
                else:
                    buf += ch
            if buf:
                terms.append(buf)
            for t in terms:
                c, d = self._canonical_to_monomial(t)
                if d == 1:
                    a += c
                elif d == 0:
                    b += c
            return a, b
        a1, b1 = lin_coef(left)
        a2, b2 = lin_coef(right)
        A = a1 - a2
        B = b2 - b1
        # Ensure solvable and not degenerate
        if A == 0:
            if B == 0:
                # infinite solutions; adjust to avoid ambiguity
                bfix = random.randint(1, 5)
                right = self._canonical_poly(f"({right})+{bfix}")
                a1, b1 = lin_coef(left)
                a2, b2 = lin_coef(right)
                A = a1 - a2
                B = b2 - b1
            else:
                # no solution; adjust
                afix = random.choice([-1, 1]) * random.randint(1, 3)
                left = self._canonical_poly(f"({left})+{afix}*x")
                a1, b1 = lin_coef(left)
                A = a1 - a2
                B = b2 - b1
        # integer x preferred; if not integer, still accept exact fraction string "p/q"
        # Build equation string
        eq = f"{left}={right}"
        # target answer: exact rational
        # x = B/A
        from math import gcd
        g = gcd(B, A) if A != 0 else 1
        num = B // g if g != 0 else B
        den = A // g if g != 0 else A
        if den < 0:
            den = -den
            num = -num
        target = str(num) if den == 1 else f"{num}/{den}"
        return eq, target

    def _generate_instance(self):
        # Choose task type by complexity
        # lower complexity favors simplify/expand; higher introduces factor/solve/evaluate
        r = random.random()
        if self.complexity_level <= 3:
            self.task_type = "expand" if r < 0.5 else "simplify"
        elif self.complexity_level <= 6:
            self.task_type = "expand" if r < 0.34 else ("simplify" if r < 0.67 else "factor")
        elif self.complexity_level <= 8:
            self.task_type = random.choice(["expand", "simplify", "factor", "evaluate_at"])
        else:
            self.task_type = random.choice(["expand", "simplify", "factor", "evaluate_at", "solve_for_x"])

        if self.task_type == "simplify":
            expr, target = self._gen_simplify_instance()
            self.current_expr = expr
            self.target_answer = target
        elif self.task_type == "expand":
            expr, target = self._gen_expand_instance()
            self.current_expr = expr
            self.target_answer = target
        elif self.task_type == "factor":
            expr, target = self._gen_factor_instance()
            self.current_expr = expr
            self.target_answer = target
        elif self.task_type == "evaluate_at":
            expr, target = self._gen_evaluate_instance()
            self.current_expr = expr
            self.target_answer = target
        elif self.task_type == "solve_for_x":
            eq, target = self._gen_solve_instance()
            self.current_expr = eq
            self.target_answer = target
        else:
            self.task_type = "simplify"
            expr, target = self._gen_simplify_instance()
            self.current_expr = expr
            self.target_answer = target

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.history = []
        self.hidden_x_value = None
        self._generate_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "")
        terminated = False
        truncated = False
        reward = 0.0
        detail = ""

        # Apply actions
        if act not in ["simplify", "expand", "factor", "substitute", "solve", "submit"]:
            obs = f"UNSUPPORTED ACTION: {act}"
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        prev_expr = self.current_expr

        if act == "simplify":
            if self.task_type == "solve_for_x":
                # simplify both sides if equation
                if "=" in self.current_expr:
                    L, R = self.current_expr.split("=", 1)
                    L2 = self._combine_like_terms(L)
                    R2 = self._combine_like_terms(R)
                    self.current_expr = f"{L2}={R2}"
                else:
                    self.current_expr = self._combine_like_terms(self.current_expr)
            else:
                self.current_expr = self._combine_like_terms(self.current_expr)
            detail = "Applied simplify."

        elif act == "expand":
            if self.task_type == "solve_for_x":
                if "=" in self.current_expr:
                    L, R = self.current_expr.split("=", 1)
                    L2 = self._expand_all_parentheses(L)
                    R2 = self._expand_all_parentheses(R)
                    self.current_expr = f"{self._combine_like_terms(L2)}={self._combine_like_terms(R2)}"
                else:
                    self.current_expr = self._expand_all_parentheses(self.current_expr)
                    self.current_expr = self._combine_like_terms(self.current_expr)
            else:
                self.current_expr = self._expand_all_parentheses(self.current_expr)
                self.current_expr = self._combine_like_terms(self.current_expr)
            detail = "Applied expand."

        elif act == "factor":
            if self.task_type == "solve_for_x":
                if "=" in self.current_expr:
                    L, R = self.current_expr.split("=", 1)
                    # try factor each side's quadratics
                    Lf = self._factor_quadratic(L) or self._combine_like_terms(L)
                    Rf = self._factor_quadratic(R) or self._combine_like_terms(R)
                    self.current_expr = f"{Lf}={Rf}"
                else:
                    f = self._factor_quadratic(self.current_expr)
                    self.current_expr = f if f else self._combine_like_terms(self.current_expr)
            else:
                f = self._factor_quadratic(self.current_expr)
                self.current_expr = f if f else self._combine_like_terms(self.current_expr)
            detail = "Applied factor (attempt)."

        elif act == "substitute":
            if self.task_type not in ["evaluate_at", "simplify", "expand", "factor"]:
                obs = "PROTOCOL VIOLATION: substitute only allowed for expression tasks."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            sval = parsed.get("x")
            if sval is None:
                obs = "PROTOCOL VIOLATION: substitute requires x=<int>."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            try:
                xint = int(sval)
            except:
                obs = "PROTOCOL VIOLATION: x must be an integer."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # replace x with xint and simplify numerically
            if "=" in self.current_expr:
                obs = "PROTOCOL VIOLATION: substitute not allowed on equations."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            val = self._evaluate_at_int(self.current_expr, xint)
            self.current_expr = str(val)
            detail = f"Substituted x={xint}."

        elif act == "solve":
            if self.task_type != "solve_for_x":
                obs = "PROTOCOL VIOLATION: solve is only allowed for solve_for_x task."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # attempt solve like during generation
            if "=" not in self.current_expr:
                obs = "PROTOCOL VIOLATION: no equation to solve."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            L, R = self.current_expr.split("=", 1)
            # simplify and expand both sides first
            L = self._combine_like_terms(self._expand_all_parentheses(L))
            R = self._combine_like_terms(self._expand_all_parentheses(R))
            self.current_expr = f"{L}={R}"
            # same linear extraction
            def lin_coef(expr: str):
                canon = self._canonical_poly(expr)
                a = 0
                b = 0
                terms = []
                buf = ""
                for ch in canon:
                    if ch == '+' and buf != "":
                        terms.append(buf)
                        buf = ""
                    else:
                        buf += ch
                if buf:
                    terms.append(buf)
                for t in terms:
                    c, d = self._canonical_to_monomial(t)
                    if d == 1:
                        a += c
                    elif d == 0:
                        b += c
                return a, b
            a1, b1 = lin_coef(L)
            a2, b2 = lin_coef(R)
            A = a1 - a2
            B = b2 - b1
            if A == 0:
                obs = "Failed: equation appears degenerate or unsolvable in linear form."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            from math import gcd
            g = gcd(B, A) if A != 0 else 1
            num = B // g if g != 0 else B
            den = A // g if g != 0 else A
            if den < 0:
                den = -den
                num = -num
            sol = str(num) if den == 1 else f"{num}/{den}"
            # keep current_expr but the agent must submit; we provide small shaping for computing
            detail = f"Computed candidate solution x={sol}. Use submit answer={sol} to finalize."
            reward = 0.3

        elif act == "submit":
            ans = parsed.get("answer")
            if ans is None:
                obs = "PROTOCOL VIOLATION: submit requires answer=<value>."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # Evaluate correctness depending on task
            if self.task_type in ["simplify", "expand", "factor"]:
                # For expression targets, compare canonical expansions
                # Accept either expanded canonical equals target canonical
                cand = ans.strip()
                cand_canon = self._canonical_poly(cand)
                target_canon = self._canonical_poly(self.target_answer)
                if cand_canon == target_canon:
                    obs = f"Success! Correct expression: {cand_canon}"
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"Failed: Incorrect expression. Expected canonical equals {target_canon}"
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            elif self.task_type == "evaluate_at":
                # must be integer result
                try:
                    got_val = int(ans.strip())
                except:
                    obs = "Failed: answer must be an integer for evaluate_at."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                if str(got_val) == self.target_answer:
                    obs = f"Success! Correct value: {got_val}"
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"Failed: Incorrect value. Expected {self.target_answer}"
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            elif self.task_type == "solve_for_x":
                # expect rational or int string matching target
                cand = ans.strip()
                # normalize fraction
                if "/" in cand:
                    try:
                        n, d = cand.split("/", 1)
                        n = int(n.strip())
                        d = int(d.strip())
                        if d == 0:
                            raise ValueError
                        from math import gcd
                        g = gcd(n, d)
                        n //= g
                        d //= g
                        if d < 0:
                            d = -d
                            n = -n
                        cand_norm = str(n) if d == 1 else f"{n}/{d}"
                    except:
                        cand_norm = cand
                else:
                    try:
                        cand_norm = str(int(cand))
                    except:
                        cand_norm = cand
                if cand_norm == self.target_answer:
                    obs = f"Success! Correct solution: x={cand_norm}"
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"Failed: Incorrect solution. Expected x={self.target_answer}"
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "PROTOCOL VIOLATION: submit not valid for this task."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Non-submission step continues
        if self.turn_count >= self.max_turns:
            obs = f"TIMEOUT: Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        # Small shaped reward if expression moved closer to target (for expression tasks)
        if reward == 0.0 and self.task_type in ["simplify", "expand", "factor"]:
            before_close = int(self._canonical_poly(prev_expr) == self._canonical_poly(self.target_answer))
            after_close = int(self._canonical_poly(self.current_expr) == self._canonical_poly(self.target_answer))
            if after_close and not before_close:
                reward = 0.2

        obs = f"{detail} New expression: {self.current_expr}"
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

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
        action_name = parts[0].strip().lower()
        tokens: Dict[str, Any] = {"action": action_name}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        # normalize aliases
        if action_name == "substitute" and "x" not in tokens:
            # also accept x= in any case
            for k in list(tokens.keys()):
                if k.lower() == "x":
                    tokens["x"] = tokens.pop(k)
        if action_name == "submit" and "answer" not in tokens:
            for k in list(tokens.keys()):
                if k.lower() == "answer":
                    tokens["answer"] = tokens.pop(k)
        return tokens

    def sample_random_action(self) -> str:
        choices = [
            r"\boxed{simplify}",
            r"\boxed{expand}",
            r"\boxed{factor}",
            r"\boxed{substitute x=2}",
            r"\boxed{solve}",
            r"\boxed{submit answer=0}",
        ]
        return random.choice(choices)


class AlgebraPathfinderEnvWithFeedback(AlgebraPathfinderEnv):
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
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your action like \\boxed{simplify} or \\boxed{submit answer=...}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_action"
            hint = "Use one of: simplify, expand, factor, substitute x=<int>, solve, submit answer=..."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "substitute requires x=" in text:
                error_detail["violation"] = "missing_x_param"
                hint = "Provide an integer, e.g., \\boxed{substitute x=3}."
            elif "x must be an integer" in text:
                error_detail["violation"] = "non_integer_substitution"
                hint = "Use an integer for x, like -2, 0, or 5."
            elif "solve is only allowed" in text:
                error_detail["violation"] = "wrong_task_solve"
                hint = "Only use solve in solve_for_x tasks."
            elif "submit requires answer=" in text:
                error_detail["violation"] = "missing_answer_param"
                hint = "Submit like \\boxed{submit answer=...}."
            elif "substitute not allowed on equations" in text:
                error_detail["violation"] = "substitute_on_equation"
                hint = "Use simplify/expand/factor/solve on equations."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Follow the task rules shown in the instructions."
        elif "failed:" in text and "incorrect" in text:
            error_type = "WrongDecision"
            if "expected canonical equals" in text:
                error_detail["expected_form"] = "match_target_canonical"
                hint = "Canonicalize your expression: expand/factor appropriately and combine like terms before submitting."
            elif "expected x=" in text:
                # extract expected
                m = re.search(r"expected x=([-\d/]+)", obs, flags=re.IGNORECASE)
                if m:
                    error_detail["expected"] = m.group(1)
                hint = "Solve the equation step-by-step, then submit the simplified exact value (integer or reduced fraction)."
            elif "expected" in text:
                m = re.search(r"expected ([-\d/]+)", obs, flags=re.IGNORECASE)
                if m:
                    error_detail["expected"] = m.group(1)
                hint = "Double-check arithmetic and submit the exact integer value."
        elif "timeout" in text:
            error_type = "Timeout"
            hint = "Act sooner: simplify/expand toward a canonical form, then submit before the turn limit."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["task_type"] = getattr(self, "task_type", None)
            diagnostic["current_expr"] = getattr(self, "current_expr", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by applying simplify or expand to normalize the expression. Submit only when you have the final answer.",
            "turn": 0,
            "task_type": getattr(self, "task_type", None),
            "current_expr": getattr(self, "current_expr", None),
        }
        return obs, info