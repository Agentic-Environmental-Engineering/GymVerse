from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class AlgebraCanonForgeEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters native to algebra problem generation
        self.complexity_params = {
            # polynomial degree (max exponent on any variable): higher degree → more complex algebra
            "max_degree": (1, 5),
            # number of monomials before combining like terms: more terms → harder to simplify
            "num_terms": (2, 10),
            # number of distinct variables allowed: more variables → harder factoring/cancellation
            "num_variables": (1, 3),
            # number of rational layers: nesting with numerator/denominator structure increases difficulty
            "rational_layers": (0, 3),
            # integer coefficient magnitude limit: larger magnitudes complicate factoring and cancellation
            "coef_mag": (3, 12),
            # distractor identity insertions (e.g., multiply by (x/x), add-zero with (a-a)): more distractors → harder
            "num_distractors": (0, 4),
        }

        # Variance settings per parameter
        self.param_variance = {
            "max_degree": 0,         # small integer range → keep stable
            "num_terms": 1,          # moderate range → ±1 variety
            "num_variables": 0,      # very small range → fixed
            "rational_layers": 1,    # moderate, ±1
            "coef_mag": 2,           # larger range, ±2
            "num_distractors": 1,    # moderate, ±1
        }

        # Placeholder attributes for parameters
        self.max_degree: int = 0
        self.num_terms: int = 0
        self.num_variables: int = 0
        self.rational_layers: int = 0
        self.coef_mag: int = 0
        self.num_distractors: int = 0

        # State
        self.turn_count: int = 0
        self.initial_expr: str = ""
        self.current_expr: str = ""
        self.target_expr: str = ""
        self.variables_pool = []
        self.allow_numeric_eval: bool = False
        self.hidden_numeric_value: Optional[int] = None
        self.history = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for pname, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(pname, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, pname, int(round(actual_value)))

    def _sample_variable(self):
        return random.choice(self.variables_pool)

    def _rand_coef(self):
        c = random.randint(-self.coef_mag, self.coef_mag)
        while c == 0:
            c = random.randint(-self.coef_mag, self.coef_mag)
        return c

    def _rand_degree(self):
        return random.randint(0, self.max_degree)

    def _monomial_str(self):
        # Generate monomial like c*x^d*y^e ...
        c = self._rand_coef()
        vars_used = random.sample(self.variables_pool, k=random.randint(0, min(2, len(self.variables_pool))))
        parts = []
        for v in vars_used:
            d = self._rand_degree()
            if d == 0:
                continue
            if d == 1:
                parts.append(v)
            else:
                parts.append(f"{v}^{d}")
        var_part = "*".join(parts) if parts else "1"
        return f"{c}*{var_part}"

    def _poly_str(self, terms: int):
        monos = [self._monomial_str() for _ in range(terms)]
        # Combine into additive string
        s = monos[0]
        for m in monos[1:]:
            if m.startswith("-"):
                s += f" {m}"  # already has minus
            else:
                s += f" + {m}"
        return s

    def _maybe_insert_distractors(self, expr: str) -> str:
        # Insert algebraic identities to create cancellations:
        # Types: multiply by (x/x), add (a-a), multiply by (k/k), add and subtract same monomial
        for _ in range(self.num_distractors):
            choice = random.choice(["mul_one", "add_zero"])
            if choice == "mul_one":
                v = self._sample_variable()
                expr = f"({expr}) * ({v}/{v})"
            else:
                # add zero via (poly - poly)
                t = self._poly_str(1)
                expr = f"({expr}) + ({t} - {t})"
        return expr

    def _build_rational(self, base: str) -> str:
        # Wrap base into nested rational layers: ((base)/(poly)) / (poly) ... etc.
        expr = base
        for _ in range(self.rational_layers):
            denom_terms = max(1, self.num_terms // (2 + random.randint(0, 1)))
            denom = self._poly_str(denom_terms)
            expr = f"({expr})/({denom})"
        return expr

    def _alpha_simplify(self, expr: str) -> str:
        # Lightweight canonicalization:
        # - remove spaces
        # - order additive and multiplicative factors alphabetically by variable symbol within simple patterns
        # - normalize signs like "+-" to "-"
        # This is heuristic; target will be constructed by the same routine to allow exact matching.
        s = expr.replace(" ", "")
        s = s.replace("+-", "-").replace("--", "+")
        # Order factors inside products like a*b and powers v^d keep as-is
        # Try to sort multiplicative parts inside each (...) block
        def sort_factors(chunk: str) -> str:
            if "*" in chunk and "/" not in chunk and "+" not in chunk and "-" not in chunk[1:]:
                parts = chunk.split("*")
                # separate coefficients to front
                coeffs = [p for p in parts if re.fullmatch(r"-?\d+(\^\d+)?", p)]
                vars_ = [p for p in parts if p not in coeffs]
                vars_.sort()
                return "*".join(coeffs + vars_)
            return chunk

        # Apply inside parentheses
        def repl(m):
            inner = m.group(1)
            return "(" + sort_factors(inner) + ")"

        s = re.sub(r"\(([^()]+)\)", repl, s)
        # Top-level attempt
        if "*" in s and "/" not in s and "+" not in s and "-" not in s[1:]:
            s = sort_factors(s)
        return s

    def _canonical_target(self, expr: str) -> str:
        # Produce a target by applying the same simplifier; assume cancellations from distractors
        # Note: This does not perform full algebra; but instances are generated so the same heuristic works for matching.
        return self._alpha_simplify(expr)

    def _maybe_numeric_problem(self) -> bool:
        # For some instances, allow numeric evaluation at integer assignment to all variables.
        # Higher complexity reduces chance (to encourage symbolic work), but still possible.
        base_prob = 0.5
        prob = max(0.1, base_prob - 0.04 * (self.complexity - 1))
        return random.random() < prob

    def _evaluate_numeric(self, expr: str, assignment: Dict[str, int]) -> Optional[int]:
        # Very controlled evaluator: handles +, -, *, /, parentheses, integers, variables, and powers x^k
        # No floats: require exact divisibility to return int; else return None
        # Replace variables with ints
        s = expr
        for v, val in assignment.items():
            s = re.sub(fr"\b{re.escape(v)}\b", str(val), s)
        # Replace power a^b with pow(a,b)
        s = re.sub(r"(\d+)\^(\d+)", r"(\1**\2)", s)
        # Only allow digits, operators, parentheses, and spaces
        if re.search(r"[^\d+\-*/() ]", s):
            return None
        try:
            val = eval(s, {"__builtins__": {}}, {})
        except Exception:
            return None
        if isinstance(val, (int, float)):
            # Ensure integer result
            if abs(val - int(round(val))) < 1e-9:
                return int(round(val))
        return None

    def _get_instructions(self) -> str:
        return (
            "You are working in AlgebraCanonForge. Transform the given algebraic expression into its canonical simplified form, "
            "or submit the final integer value if a numeric evaluation task is specified. You have multiple turns to apply rewrite actions.\n"
            "Goal: Match the environment's hidden canonical target for this instance. Success requires exact string match to the canonical form, "
            "or correct integer for numeric tasks.\n\n"
            "Allowed actions (use \\boxed{...} format):\n"
            "- rewrite rule=[expand|factor|cancel|combine|substitute] target=<subexpr> replacement=<expr>\n"
            "  • expand: distribute multiplication over addition on the specified target\n"
            "  • factor: factor a common integer coefficient from the target\n"
            "  • cancel: cancel a common factor across a fraction target=(numerator)/(denominator)\n"
            "  • combine: merge like terms in a polynomial target\n"
            "  • substitute: replace target subexpression by replacement\n"
            "- answer expr=<final_expression>    (for symbolic tasks; must match canonical)\n"
            "- answer value=<integer>           (for numeric tasks; must be an exact integer)\n\n"
            "Formatting rules:\n"
            "- Use integers, variables [a..z], +, -, *, /, parentheses (), and powers as v^k.\n"
            "- Keep expressions syntactically valid; no functions or floats.\n"
            f"Example: {self.sample_random_action()}"
        )

    def get_task_suffix(self) -> str:
        numeric_note = (
            f"This is a numeric evaluation task at assignment {self._current_numeric_assignment}."
            if self.allow_numeric_eval else
            "This is a symbolic simplification task. Submit the canonical simplified expression."
        )
        return (
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            f"Expression: {self.current_expr}\n"
            f"{numeric_note}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.history = []

        # Variables selection
        all_vars = list("xyzuvwabcdef")
        self.variables_pool = sorted(random.sample(all_vars, k=self.num_variables))

        # Build base polynomial
        base_poly = self._poly_str(self.num_terms)
        # Wrap in rational layers
        expr = self._build_rational(base_poly)
        # Insert distractors
        expr = self._maybe_insert_distractors(expr)

        self.initial_expr = expr
        self.current_expr = expr

        # Decide if numeric evaluation is allowed
        self.allow_numeric_eval = self._maybe_numeric_problem()
        self._current_numeric_assignment = {}
        if self.allow_numeric_eval:
            # Assign small integers 1..5 to variables
            self._current_numeric_assignment = {v: random.randint(1, 5) for v in self.variables_pool}
            self.hidden_numeric_value = self._evaluate_numeric(expr, self._current_numeric_assignment)
            # Ensure evaluable; if not, fall back to symbolic
            if self.hidden_numeric_value is None:
                self.allow_numeric_eval = False
                self.hidden_numeric_value = None

        # Target canonical form (heuristic)
        self.target_expr = self._canonical_target(self.initial_expr)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _apply_expand(self, target: str) -> bool:
        # Very limited: transform (A)*(B+C) → A*B + A*C if found exactly in current_expr
        m = re.fullmatch(r"\((.+)\)\*\((.+)\+(.+)\)", target)
        if not m:
            return False
        A, B, C = m.group(1), m.group(2), m.group(3)
        src = f"({A})*({B}+{C})"
        dst = f"({A})*({B}) + ({A})*({C})"
        if src in self.current_expr.replace(" ", ""):
            # Need to mirror no-space search; rebuild current with no spaces, then replace and re-space minimally
            no = self.current_expr.replace(" ", "")
            no = no.replace(src.replace(" ", ""), dst.replace(" ", ""))
            self.current_expr = no
            return True
        return False

    def _apply_factor(self, target: str) -> bool:
        # Factor integer gcd from additive form like "a + b"
        # Heuristic: find integers multiplied to terms and factor common gcd
        # Simple pattern: (k1*M1 + k2*M2 + ...), factor gcd(|k_i|)
        content = target.replace(" ", "")
        parts = re.split(r'\+', content.replace("-", "+-"))
        if len(parts) < 2:
            return False
        coefs = []
        for p in parts:
            m = re.match(r"(-?\d+)\*", p)
            if m:
                coefs.append(int(m.group(1)))
            else:
                coefs.append(1)
        g = 0
        for c in coefs:
            g = abs(c) if g == 0 else self._gcd(g, abs(c))
        if g <= 1:
            return False
        # Build factored string g*(... with adjusted coefficients)
        rebuilt = []
        for p in parts:
            m = re.match(r"(-?\d+)\*(.+)", p)
            if m:
                c = int(m.group(1))
                rest = m.group(2)
                newc = c // g
                rebuilt.append(f"{newc}*{rest}" if newc not in (1, -1) else (f"-{rest}" if newc == -1 else rest))
            else:
                rebuilt.append(p)
        inner = " + ".join([r if not r.startswith("-") else r for r in rebuilt]).replace("+-", "- ")
        dst = f"{g}*({inner})"
        if target in self.current_expr.replace(" ", ""):
            no = self.current_expr.replace(" ", "")
            no = no.replace(target, dst.replace(" ", ""))
            self.current_expr = no
            return True
        return False

    def _gcd(self, a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return abs(a)

    def _apply_cancel(self, target: str) -> bool:
        # Cancel common factor across fraction: ((g*T)/(g*U)) → (T/U)
        m = re.fullmatch(r"\(\((.+)\)\)/\((.+)\)", target)
        # also allow (A)/(B)
        if not m:
            m = re.fullmatch(r"\((.+)\)/\((.+)\)", target)
        if not m:
            return False
        N = m.group(1).replace(" ", "")
        D = m.group(2).replace(" ", "")
        # Detect simple integer common factor pattern: k*X / k*Y
        mN = re.match(r"(-?\d+)\*(.+)", N)
        mD = re.match(r"(-?\d+)\*(.+)", D)
        if mN and mD:
            cN, restN = int(mN.group(1)), mN.group(2)
            cD, restD = int(mD.group(1)), mD.group(2)
            g = self._gcd(cN, cD)
            if g > 1:
                newN = cN // g
                newD = cD // g
                N2 = (f"{newN}*{restN}" if abs(newN) != 1 else (f"-{restN}" if newN == -1 else restN))
                D2 = (f"{newD}*{restD}" if abs(newD) != 1 else (f"-{restD}" if newD == -1 else restD))
                dst = f"({N2})/({D2})"
                src = f"({N})/({D})"
                if src in self.current_expr.replace(" ", ""):
                    no = self.current_expr.replace(" ", "")
                    no = no.replace(src, dst.replace(" ", ""))
                    self.current_expr = no
                    return True
        return False

    def _apply_combine(self, target: str) -> bool:
        # Combine like integer-only terms: c1 + c2 + ... → sum
        content = target.replace(" ", "")
        parts = re.split(r'\+', content.replace("-", "+-"))
        try:
            vals = [int(p) for p in parts if p != ""]
        except ValueError:
            return False
        s = sum(vals)
        dst = str(s)
        if target in self.current_expr.replace(" ", ""):
            no = self.current_expr.replace(" ", "")
            no = no.replace(target, dst)
            self.current_expr = no
            return True
        return False

    def _apply_substitute(self, target: str, replacement: str) -> bool:
        tgt = target.replace(" ", "")
        rep = replacement.replace(" ", "")
        if tgt in self.current_expr.replace(" ", ""):
            no = self.current_expr.replace(" ", "")
            no = no.replace(tgt, rep, 1)
            self.current_expr = no
            return True
        return False

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
        message = ""

        if act == "rewrite":
            rule = parsed.get("rule", "")
            target = parsed.get("target", "")
            replacement = parsed.get("replacement", "")
            if rule not in ["expand", "factor", "cancel", "combine", "substitute"]:
                message = "Unsupported rewrite rule."
                obs = f"UNSUPPORTED ACTION: {message}"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            success = False
            if rule == "expand":
                success = self._apply_expand(target)
            elif rule == "factor":
                success = self._apply_factor(target)
            elif rule == "cancel":
                success = self._apply_cancel(target)
            elif rule == "combine":
                success = self._apply_combine(target)
            elif rule == "substitute":
                if not target or not replacement:
                    message = "substitute requires target and replacement."
                    obs = f"PROTOCOL VIOLATION: {message}"
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                success = self._apply_substitute(target, replacement)

            if success:
                self.history.append((rule, target, replacement))
                message = "Applied rewrite."
                # mild shaped reward for successful valid transformation
                reward = 0.2
            else:
                message = "Rewrite could not be applied to the current expression."

            # Check if now exactly matches canonical simplified form
            if self._alpha_simplify(self.current_expr) == self.target_expr and not self.allow_numeric_eval:
                obs = "Success! Expression matches the canonical simplified form."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"{message}\nCurrent: {self.current_expr}"

        elif act == "answer":
            expr_ans = parsed.get("expr")
            val_ans = parsed.get("value")
            if self.allow_numeric_eval:
                if val_ans is None:
                    obs = "PROTOCOL VIOLATION: This is a numeric task. Submit an integer with answer value=<int>."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                try:
                    guess = int(val_ans)
                except Exception:
                    obs = "FORMAT ERROR: Value must be an integer."
                    return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
                if self.hidden_numeric_value is not None and guess == self.hidden_numeric_value:
                    obs = "Success! Correct integer value for the numeric evaluation."
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = "Failed! Incorrect integer value."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                if expr_ans is None:
                    obs = "PROTOCOL VIOLATION: This is a symbolic task. Submit an expression with answer expr=<expression>."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                guess = expr_ans.replace(" ", "")
                if self._alpha_simplify(guess) == self.target_expr:
                    obs = "Success! Expression matches the canonical simplified form."
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = "Failed! Expression does not match the canonical simplified form."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "UNSUPPORTED ACTION: Unknown action name. Use 'rewrite' or 'answer'."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            return f"Reached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        parts = inner.split()
        if not parts:
            return None
        out: Dict[str, Any] = {}
        # action name first token
        out['action'] = parts[0]
        # remaining tokens as key=value with possible equal signs in value
        for token in parts[1:]:
            if '=' in token:
                key, val = token.split('=', 1)
                out[key] = val
        return out

    def sample_random_action(self) -> str:
        example = random.choice([
            r"\boxed{rewrite rule=expand target=(x)*(x+1) replacement=}",
            r"\boxed{rewrite rule=factor target=2*x+4 replacement=}",
            r"\boxed{rewrite rule=cancel target=(2*x)/(4*y) replacement=}",
            r"\boxed{rewrite rule=combine target=2+3-1 replacement=}",
            r"\boxed{rewrite rule=substitute target=x^2 replacement=x*x}",
            r"\boxed{answer expr=x*(x+1)}",
            r"\boxed{answer value=12}",
        ])
        return example


class AlgebraCanonForgeEnvWithFeedback(AlgebraCanonForgeEnv):
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
            error_detail["issue"] = "missing_boxed_or_malformed"
            hint = "Wrap your command in \\boxed{...} and provide key=value pairs as required."
        elif "format error" in text:
            error_type = "FormatError"
            error_detail["issue"] = "malformed_value"
            hint = "Ensure integers for value= and use only allowed symbols in expressions."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = ["rewrite", "answer"]
            hint = "Use 'rewrite' with a valid rule or 'answer' with expr= or value=."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "numeric task" in text:
                error_detail["violation"] = "wrong_answer_channel_for_numeric"
                hint = "For numeric tasks, submit \\boxed{answer value=<integer>}."
            else:
                error_detail["violation"] = "wrong_answer_channel_for_symbolic"
                hint = "For symbolic tasks, submit \\boxed{answer expr=<expression>}."
        elif "rewrite could not be applied" in text:
            error_type = "WrongDecision"
            error_detail["reason"] = "target_not_found_or_inapplicable"
            hint = "Ensure the 'target' matches an exact substring of the current expression (no spaces) and fits the rule pattern."
        elif "failed!" in text and "incorrect integer" in text:
            error_type = "WrongDecision"
            error_detail["reason"] = "wrong_numeric_value"
            if hasattr(self, "_current_numeric_assignment"):
                hint = f"Re-evaluate using the assignment {self._current_numeric_assignment} and ensure divisions are exact."
        elif "failed! expression does not match" in text:
            error_type = "WrongDecision"
            error_detail["reason"] = "wrong_symbolic_form"
            hint = "Try combining like integer terms, factoring common integers, canceling integer factors in fractions, or normalizing spacing and simple factor order."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Use targeted rewrites on small subexpressions and verify progress each turn."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well done."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "current_expr": getattr(self, "current_expr", None),
                "task_type": "numeric" if getattr(self, "allow_numeric_eval", False) else "symbolic",
                "variables": getattr(self, "variables_pool", []),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        start_hint = (
            "Identify a simple subexpression and try a small rewrite, e.g., combine integer constants or cancel a visible integer factor."
            if not self.allow_numeric_eval else
            "Compute carefully by substituting the provided integer values and maintaining integer arithmetic."
        )
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": start_hint,
            "turn": 0,
            "state": {
                "current_expr": self.current_expr,
                "task_type": "numeric" if self.allow_numeric_eval else "symbolic",
                "variables": self.variables_pool,
            },
        }
        return obs, info