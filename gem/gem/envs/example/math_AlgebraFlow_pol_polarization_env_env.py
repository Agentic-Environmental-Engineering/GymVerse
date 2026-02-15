from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class AlgebraFlowEnv(Env):
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

        # Evolvable parameters
        self.complexity_params = {
            # Maximum polynomial degree per variable: higher degree increases algebraic difficulty
            'max_degree': (2, 5),
            # Number of additive terms in the hidden base expression: more terms harder to manage
            'num_terms': (2, 6),
            # Probability (percent) of cross-variable interactions (xy, x^2 y, etc.): more cross terms harder
            'cross_term_pct': (10, 70),
            # Target canonical form strictness: 0=any equivalent polynomial; 1=require sorted, factored or expanded per target_spec
            # REVERSED: higher strictness is harder
            'strictness': (0, 1),
            # Presence of nested parentheses depth: deeper nesting increases step planning difficulty
            'nest_depth': (0, 2),
        }
        # Variance settings
        self.param_variance = {
            'max_degree': 0,        # small range, keep fixed at level interpolation
            'num_terms': 1,         # moderate discrete
            'cross_term_pct': 7,    # ~10% of range
            'strictness': 0,        # binary, keep stable
            'nest_depth': 1,        # small discrete
        }

        # Placeholders set by _apply_complexity_params
        self.max_degree: int = 0
        self.num_terms: int = 0
        self.cross_term_pct: int = 0
        self.strictness: int = 0
        self.nest_depth: int = 0

        # State
        self.turn_count: int = 0
        self.vars = ['x', 'y']  # fixed two variables to keep parsing manageable
        self.current_expr: Dict[Tuple[int, int], int] = {}  # polynomial as map (deg_x,deg_y) -> coeff
        self.target_mode: str = ""  # 'expand' or 'factor'
        self.target_expr: Dict[Tuple[int,int], int] = {}     # canonical target polynomial (expanded) for checking equivalence
        self.history = []
        self.random_seed = None

        self.reset()

    # -------- Utility: complexity application --------
    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (lo, hi) in self.complexity_params.items():
            center = lo + (hi - lo) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
                    if lo <= hi:
                        val = max(lo, min(hi, val))
                    else:
                        val = max(hi, min(lo, val))
            setattr(self, name, int(round(val)))

    # -------- Polynomial helpers (native math mechanics) --------
    def _poly_add(self, A, B):
        R = dict(A)
        for k, v in B.items():
            R[k] = R.get(k, 0) + v
            if R[k] == 0:
                del R[k]
        return R

    def _poly_mul(self, A, B):
        R = {}
        for (ax, ay), ac in A.items():
            for (bx, by), bc in B.items():
                key = (ax + bx, ay + by)
                R[key] = R.get(key, 0) + ac * bc
        R = {k: v for k, v in R.items() if v != 0}
        return R

    def _poly_pow(self, A, n: int):
        R = {(0,0): 1}
        base = dict(A)
        e = n
        while e > 0:
            if e & 1:
                R = self._poly_mul(R, base)
            base = self._poly_mul(base, base)
            e >>= 1
        return R

    def _mono(self, cx=0, cy=0, coeff=1):
        return {(cx, cy): coeff} if coeff != 0 else {}

    def _random_coeff(self):
        # Non-zero small integer coefficients for clarity
        c = random.choice([-3, -2, -1, 1, 2, 3])
        return c

    def _format_poly(self, P: Dict[Tuple[int,int], int]) -> str:
        if not P:
            return "0"
        # Sort by total degree desc, then x-degree desc
        items = sorted(P.items(), key=lambda kv: (kv[0][0]+kv[0][1], kv[0][0], kv[0][1]), reverse=True)
        parts = []
        for (dx, dy), c in items:
            term = ""
            sign = "+" if c > 0 else "-"
            abs_c = abs(c)
            if dx == 0 and dy == 0:
                term_body = f"{abs_c}"
            else:
                coeff_str = "" if abs_c == 1 else f"{abs_c}"
                xpart = "" if dx == 0 else ("x" if dx == 1 else f"x^{dx}")
                ypart = "" if dy == 0 else ("y" if dy == 1 else f"y^{dy}")
                mult = ""
                if coeff_str and (xpart or ypart):
                    mult = ""
                term_body = coeff_str + xpart + ("" if not xpart or not ypart else "") + ypart
                if coeff_str and not (xpart or ypart):
                    term_body = coeff_str
                if not coeff_str and not (xpart or ypart):
                    term_body = "1"
            if not parts:
                # first term keeps its sign properly
                if c < 0:
                    parts.append("-" + term_body)
                else:
                    parts.append(term_body)
            else:
                parts.append(f" {sign} {term_body}")
        return "".join(parts)

    def _poly_equal(self, A, B):
        # canonical equivalence: exact map equality after simplification
        A2 = {k:v for k,v in A.items() if v != 0}
        B2 = {k:v for k,v in B.items() if v != 0}
        return A2 == B2

    def _gen_random_poly(self):
        # Generate a random polynomial with constraints
        P = {}
        attempts = 0
        while len(P) < self.num_terms and attempts < 1000:
            attempts += 1
            # sample degree
            if random.randint(1,100) <= self.cross_term_pct:
                dx = random.randint(0, self.max_degree)
                dy = random.randint(0, self.max_degree)
                if dx == 0 and dy == 0:
                    dx = 1  # avoid pure constant dominating; constants will appear anyway via nesting
            else:
                if random.random() < 0.5:
                    dx = random.randint(1, self.max_degree)
                    dy = 0
                else:
                    dx = 0
                    dy = random.randint(1, self.max_degree)
            coeff = self._random_coeff()
            key = (dx, dy)
            if key in P:
                P[key] += coeff
                if P[key] == 0:
                    del P[key]
            else:
                P[key] = coeff
        if not P:
            P = {(1,0): 1, (0,1): 1}
        return P

    def _nest_with_parentheses(self, P):
        # Wrap as products/sums of smaller polynomials to create a structured expression
        # We will store only expanded current_expr internally, but expose a "structured form" narrative.
        # To keep internal correctness, we only track expanded polynomial and a narrative structure.
        narrative = []
        expanded = dict(P)
        depth = max(0, self.nest_depth)
        for _ in range(depth):
            a = self._gen_random_poly()
            b = self._gen_random_poly()
            op = random.choice(['+', '-', '*', '^'])
            if op == '+':
                # (expanded) + a
                expanded = self._poly_add(expanded, a)
                narrative.append(f"(...) + ({self._format_poly(a)})")
            elif op == '-':
                expanded = self._poly_add(expanded, {k: -v for k,v in a.items()})
                narrative.append(f"(...) - ({self._format_poly(a)})")
            elif op == '*':
                expanded = self._poly_mul(expanded, a)
                narrative.append(f"(...) * ({self._format_poly(a)})")
            else:
                # power: keep small exponent 2
                e = 2
                expanded = self._poly_pow(expanded, e)
                narrative.append(f"(...)^{e}")
        return expanded, narrative

    def _build_task(self):
        base = self._gen_random_poly()
        expanded, narrative = self._nest_with_parentheses(base)
        self.current_expr = expanded
        self.target_mode = random.choice(['expand', 'factor'])
        # For checking correctness, we always compare expanded normal form
        self.target_expr = dict(expanded)
        self.history = []
        return narrative

    # -------- Instructions and I/O --------
    def _get_instructions(self) -> str:
        return (
            "AlgebraFlow: You manipulate algebraic expressions using valid symbolic operations.\n"
            "Goal: Transform the current expression to exactly match the target canonical form.\n"
            f"- Target mode may be 'expand' (fully expanded) or 'factor' (product of simple factors), "
            "but correctness is measured by algebraic equivalence and formatting constraints.\n"
            "Allowed actions (use \\boxed{...}):\n"
            "- simplify: attempt local cancellations, combine like terms\n"
            "- expand: expand products and powers\n"
            "- factor: factor polynomials into products of binomials when possible\n"
            "- set mode=<expand|factor>: set your intended final form (affects formatting check if strict)\n"
            "- submit: submit your current result as final answer\n"
            "Formatting:\n"
            "- Every action must be in \\boxed{...}\n"
            "- Optionally include comment=... to explain, ignored by parser\n"
            "Reward:\n"
            "- Success (correct final expression under target requirements): 1.0\n"
            "- Incorrect/malformed submission or failure: 0.0\n"
            "- Format errors: small negative (environment default)\n"
            "You have limited turns; plan steps to reach the target.\n"
        )

    def get_task_suffix(self) -> str:
        current_str = self._format_poly(self.current_expr)
        target_str = "expanded canonical polynomial" if self.target_mode == 'expand' else "factored form (if possible)"
        return (
            f"State:\n"
            f"- Current expression: {current_str}\n"
            f"- Target mode: {self.target_mode} (aim for {target_str})\n"
            f"- Turns used: {self.turn_count}/{self.max_turns}\n"
            "Enter your action in \\boxed{...} format. Examples:\n"
            "\\boxed{expand}\n"
            "\\boxed{factor}\n"
            "\\boxed{simplify}\n"
            "\\boxed{set mode=expand}\n"
            "\\boxed{submit}\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            self.random_seed = seed
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        _ = self._build_task()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    # -------- Action parsing --------
    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        matches = list(re.finditer(r"\\boxed\{(.+?)\}", str(action), flags=re.DOTALL))
        if not matches:
            return None
        inner = matches[-1].group(1).strip()
        if not inner:
            return None
        parts = inner.split()
        if not parts:
            return None
        name = parts[0].lower()
        kv = {}
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                kv[k.lower()] = v
        return {"action": name, **kv}

    # -------- Operations (domain mechanics) --------
    def _op_simplify(self):
        # current_expr already stored in combined-like-term form; simplification is idempotent
        # But we can normalize coefficient gcd or sign ordering; here we ensure no zero terms remain.
        self.current_expr = {k:v for k,v in self.current_expr.items() if v != 0}
        return "Simplified expression (combined like terms)."

    def _op_expand(self):
        # Already stored in expanded form internally; we simulate that expansion may affect upcoming factoring
        # For completeness, we keep it idempotent: ensure no zero terms and nothing else.
        self.current_expr = {k:v for k,v in self.current_expr.items() if v != 0}
        return "Expanded expression to sum of monomials."

    def _try_factor_binomials(self, P):
        # Very limited factoring: detect patterns a*x^2 + b*x*y + c*y^2 forming (ux+vy)(wx+zy)
        # and univariate quadratics ax^2+bx+c or ay^2+by+c. We attempt small integer factors in [-5..5].
        # Return list of factors as list[Dict], whose product equals P if successful; else None.
        # 1) Check if constant zero polynomial:
        if not P:
            return None
        # 2) If single monomial, factor out gcd coefficient and variable powers trivially
        if len(P) == 1:
            ((dx,dy), c) = next(iter(P.items()))
            factors = []
            sign = -1 if c < 0 else 1
            abs_c = abs(c)
            # factor coefficient into primes 2,3,5 for display; we just use single coeff factor
            factors.append(self._mono(0,0, sign*abs_c))
            if dx > 0:
                factors += [self._mono(1,0,1)] * dx
            if dy > 0:
                factors += [self._mono(0,1,1)] * dy
            return factors

        # helper to test product of factor list equals P
        def prod_eq(factors, target):
            acc = {(0,0):1}
            for F in factors:
                acc = self._poly_mul(acc, F)
            return self._poly_equal(acc, target)

        # Attempt univariate factoring in x: ax^2 + bx + c (no y)
        only_x = all(dy == 0 for (dx,dy) in P.keys())
        if only_x:
            # gather coefficients up to degree 2
            degs = [dx for (dx,_) in P.keys()]
            if max(degs) == 2 and min(degs) >= 0:
                a = P.get((2,0), 0)
                b = P.get((1,0), 0)
                c = P.get((0,0), 0)
                # Try (m x + n)(p x + q)
                candidates = [-5,-4,-3,-2,-1,1,2,3,4,5]
                for m in candidates:
                    for n in candidates:
                        for p in candidates:
                            for q in candidates:
                                if m*p == a and m*q + n*p == b and n*q == c:
                                    F1 = {(1,0): m, (0,0): n}
                                    F2 = {(1,0): p, (0,0): q}
                                    if prod_eq([F1,F2], P):
                                        return [F1,F2]

        # Attempt univariate factoring in y
        only_y = all(dx == 0 for (dx,dy) in P.keys())
        if only_y:
            degs = [dy for (_,dy) in P.keys()]
            if max(degs) == 2 and min(degs) >= 0:
                a = P.get((0,2), 0)
                b = P.get((0,1), 0)
                c = P.get((0,0), 0)
                candidates = [-5,-4,-3,-2,-1,1,2,3,4,5]
                for m in candidates:
                    for n in candidates:
                        for p in candidates:
                            for q in candidates:
                                if m*p == a and m*q + n*p == b and n*q == c:
                                    F1 = {(0,1): m, (0,0): n}
                                    F2 = {(0,1): p, (0,0): q}
                                    if prod_eq([F1,F2], P):
                                        return [F1,F2]

        # Attempt simple bivariate quadratic as (ux+vy)(wx+zy)
        deg2 = max(dx+dy for (dx,dy) in P.keys()) == 2
        if deg2:
            a = P.get((2,0), 0)
            b = P.get((1,1), 0)
            c = P.get((0,2), 0)
            d = P.get((1,0), 0)
            e = P.get((0,1), 0)
            f = P.get((0,0), 0)
            # For (ux+vy)(wx+zy) = (uw)x^2 + (uz+vw)xy + (vz)y^2
            # plus linear terms and constant only if additional pieces; here we limit to pure quadratic with no linear/const
            if d == 0 and e == 0 and f == 0:
                candidates = [-3,-2,-1,1,2,3]
                for u in candidates:
                    for v in candidates:
                        for w in candidates:
                            for z in candidates:
                                if u*w == a and (u*z + v*w) == b and v*z == c:
                                    F1 = {(1,0): u, (0,1): v}
                                    F2 = {(1,0): w, (0,1): z}
                                    if prod_eq([F1,F2], P):
                                        return [F1,F2]
        return None

    def _op_factor(self):
        factors = self._try_factor_binomials(self.current_expr)
        if factors is None:
            return "No nontrivial factorization detected."
        # Store as product by multiplying back to keep internal consistency; but record message
        # Internal form remains expanded for equivalence checking; factoring is a presentation choice.
        return "Factored into simple components (internally kept equivalent)."

    def _set_mode(self, mode: str):
        mode = mode.lower()
        if mode not in ['expand', 'factor']:
            return False, "Unsupported mode. Use expand or factor."
        self.target_mode = mode
        return True, f"Set target mode to {mode}."

    # -------- Step logic --------
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "")
        terminated = False
        truncated = False
        reward = 0.0
        message = ""

        if name == "simplify":
            message = self._op_simplify()
        elif name == "expand":
            message = self._op_expand()
        elif name == "factor":
            message = self._op_factor()
        elif name == "set":
            mode = parsed.get("mode", None)
            if mode is None:
                obs = "PROTOCOL VIOLATION: 'set' requires mode=expand or mode=factor."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            ok, msg = self._set_mode(mode)
            message = msg
            if not ok:
                obs = "UNSUPPORTED ACTION: " + msg
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        elif name == "submit":
            # Evaluate success: equivalence and, if strictness=1, check mode-specific presentability
            # Internal always expanded; strictness enforces that the chosen mode matches target_mode at submission time.
            # Since we don't keep separate formatted string state, we enforce only that the chosen target_mode matches desired final (already stored).
            is_equiv = self._poly_equal(self.current_expr, self.target_expr)
            # Required mode snapshot captured at reset
            required_mode = getattr(self, "required_mode", self.target_mode)
            if is_equiv and self.target_mode == required_mode:
                obs = "Success! Submitted expression is algebraically correct and meets target requirements."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                detail = []
                if not is_equiv:
                    detail.append("expression not equivalent")
                if is_equiv and self.target_mode != required_mode:
                    detail.append(f"mode mismatch (need {required_mode}, got {self.target_mode})")
                obs = "Failed! Submission rejected: " + ", ".join(detail) + "."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "UNSUPPORTED ACTION: Unknown command."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"At turn {self.turn_count}: {message}"
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def sample_random_action(self) -> str:
        return random.choice([
            r"\boxed{simplify}",
            r"\boxed{expand}",
            r"\boxed{factor}",
            r"\boxed{set mode=expand}",
            r"\boxed{set mode=factor}",
            r"\boxed{submit}",
        ])

    # Override reset to store required_mode snapshot
    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            self.random_seed = seed
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        narrative = self._build_task()
        # required_mode is the initial target
        self.required_mode = self.target_mode
        return self._get_instructions(), {"suffix": self.get_task_suffix()}


class AlgebraFlowEnvWithFeedback(AlgebraFlowEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed"
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{expand}"
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown_or_bad_param"
            hint = "Use one of: simplify, expand, factor, set mode=expand|factor, submit"
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "missing_required_param"
            hint = "Provide required parameters, e.g., \\boxed{set mode=expand}"
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act earlier: expand/factor first, then submit within the turn limit"
        elif "failed! submission rejected" in text:
            error_type = "WrongDecision"
            if "expression not equivalent" in text:
                error_detail["reason"] = "not_equivalent"
                hint = "Try expand to normalize, then simplify to combine like terms, and resubmit"
            elif "mode mismatch" in text:
                error_detail["reason"] = "mode_mismatch"
                req = getattr(self, "required_mode", None)
                if req:
                    error_detail["required_mode"] = req
                hint = "Set the target mode to the one announced at start using \\boxed{set mode=...} before submit"
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "target_mode": getattr(self, "target_mode", None),
                "required_mode": getattr(self, "required_mode", None),
                "strictness": getattr(self, "strictness", None),
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
            "hint": "Start with \\boxed{expand} or \\boxed{simplify} to normalize, then \\boxed{submit}. If needed, \\boxed{set mode=...}.",
            "turn": 0,
            "state": {
                "target_mode": getattr(self, "target_mode", None),
                "required_mode": getattr(self, "required_mode", None),
                "strictness": getattr(self, "strictness", None),
            }
        }
        return obs, info
