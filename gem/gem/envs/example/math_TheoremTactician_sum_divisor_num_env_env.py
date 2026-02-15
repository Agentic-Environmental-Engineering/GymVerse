from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class TheoremTacticianEnv(Env):
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

        # Evolvable parameters (math-native difficulty controls)
        self.complexity_params = {
            # number of steps reasonably needed; higher means more sub-derivations → harder
            "required_steps": (2, 7),
            # number of candidate tools presented; more tools increase branching → harder
            "toolset_size": (3, 8),
            # numeric magnitude scale for instance parameters; larger values → harder arithmetic
            "magnitude": (5, 50),
            # expression depth (layers of operations); deeper requires sequencing tools → harder
            "expression_depth": (1, 4),
            # reversed: fewer hints make it harder
            "hint_quota": (2, 0),
        }

        # Variance tuned by range size
        self.param_variance = {
            "required_steps": 1,
            "toolset_size": 1,
            "magnitude": 5,
            "expression_depth": 1,
            "hint_quota": 0,
        }

        # Placeholders set in _apply_complexity_params
        self.required_steps: int = 0
        self.toolset_size: int = 0
        self.magnitude: int = 0
        self.expression_depth: int = 0
        self.hint_quota: int = 0

        # Dynamic state
        self.turn_count: int = 0
        self.instance_id: int = 0
        self.instance_text: str = ""
        self.hidden_answer: Optional[int] = None
        self.derived_facts: List[str] = []
        self.available_tools: Dict[str, str] = {}
        self.used_tools: List[str] = []
        self.remaining_hints: int = 0
        self.last_feedback: str = ""
        self.protocol_ok: bool = True  # becomes False on protocol violations

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for pname, (min_v, max_v) in self.complexity_params.items():
            center = min_v + (max_v - min_v) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(pname, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
            lo = min(min_v, max_v)
            hi = max(min_v, max_v)
            actual = max(lo, min(hi, actual))
            setattr(self, pname, int(round(actual)))

    def _get_instructions(self) -> str:
        return (
            "You are Theorem Tactician. Solve the posed math instance by deriving intermediate facts "
            "and finally submitting a single integer as the answer.\n"
            "Goal: produce the correct integer solution using valid math steps. The target is hidden.\n"
            "Protocol:\n"
            "- Use domain tools when relevant (they are listed for this episode).\n"
            "- You may request limited hints.\n"
            "- Derive facts before submitting the final answer.\n"
            "Available actions (use \\boxed{...}):\n"
            "- \\boxed{assume text=...}: Note a given or a safe inference you will use.\n"
            "- \\boxed{derive tool=TOOL text=...}: Apply a listed tool to derive a fact; include brief text on what you computed.\n"
            "- \\boxed{simplify text=...}: Perform arithmetic/algebraic simplification and record the result.\n"
            "- \\boxed{hint}: Consume one hint from your quota.\n"
            "- \\boxed{submit value=INTEGER}: Submit your final integer answer.\n"
            "Rules:\n"
            "- Action must be enclosed in \\boxed{...}. Unknown tools or malformed actions end the episode.\n"
            "- Submit only once; the episode ends on submission.\n"
            "- Keep actions concise; focus on correct sequencing.\n"
            "Scoring:\n"
            "- Correct final submission: reward 1.0.\n"
            "- Incorrect but well-formed submission: reward 0.0.\n"
            "- Format errors or protocol violations: format_error penalty.\n"
            f"Max turns: {self.max_turns}. Use tools wisely.\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        tools_str = "\n".join([f"- {k}: {v}" for k, v in self.available_tools.items()])
        facts_str = "\n".join([f"* {f}" for f in self.derived_facts]) if self.derived_facts else "(none yet)"
        return (
            f"Instance: {self.instance_text}\n"
            f"Tools available ({len(self.available_tools)}):\n{tools_str}\n"
            f"Derived facts: {facts_str}\n"
            f"Remaining hints: {self.remaining_hints}\n"
            f"Turns: {self.turn_count}/{self.max_turns}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.instance_id = random.randint(1000, 9999)
        self.derived_facts = []
        self.used_tools = []
        self.remaining_hints = self.hint_quota
        self.last_feedback = ""
        self.protocol_ok = True

        # Generate instance based on expression_depth/toolset
        # We select a problem archetype from a curated set, guaranteed solvable
        archetypes = []
        # Arithmetic series sum S_n = n/2*(a1 + an)
        archetypes.append(self._gen_arithmetic_series)
        # Quadratic discriminant-based count of integer roots in range (requires discriminant tool)
        archetypes.append(self._gen_quadratic_roots_in_range)
        # Pythagorean triple with scaled magnitude
        archetypes.append(self._gen_scaled_pythagorean_hypotenuse)
        # Combinatorics: choose and simple identities
        archetypes.append(self._gen_combinations_difference)
        # Modular arithmetic linear congruence
        archetypes.append(self._gen_mod_linear)

        # Bias choice by expression depth to vary required tools
        generator = random.choice(archetypes[: 2 + min(self.expression_depth, len(archetypes))])
        self.instance_text, self.hidden_answer, canonical_tools = generator(self.magnitude, self.expression_depth)

        # Tool inventory: include required tools + distractors up to toolset_size
        base_tool_defs = {
            "ARITH_SUM": "Sum arithmetic progression or series totals",
            "QUAD_DISCRIMINANT": "Use b^2 - 4ac to reason about roots",
            "PYTHAGOREAN": "Apply a^2 + b^2 = c^2 in right triangles",
            "COMB_IDENTITIES": "Use nCk identities and arithmetic",
            "MOD_LINEAR": "Solve ax ≡ b (mod m) via gcd and inverses",
            "FACTORIAL": "Compute factorials and simplify ratios",
            "GCD_LCM": "Compute gcd/lcm and use divisibility",
            "POWER_RULES": "Exponent rules for simplification",
        }
        required = set(canonical_tools)
        tools_pool = list(base_tool_defs.items())
        # Start with required
        selected = {k: base_tool_defs[k] for k in required}
        # Fill with distractors
        distractors = [t for t in tools_pool if t[0] not in selected]
        random.shuffle(distractors)
        for k, v in distractors:
            if len(selected) >= self.toolset_size:
                break
            selected[k] = v
        self.available_tools = selected

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        verb = parsed.get("action", "").lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if verb == "assume":
            text = parsed.get("text")
            if not text:
                obs = "PROTOCOL VIOLATION: 'assume' requires text=..."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            self.derived_facts.append(f"Assumed: {text}")
            self.last_feedback = "Assumption recorded."
            obs = f"Recorded assumption. Keep deriving. {self.required_steps - len(self.derived_nonassumptions())} key steps likely remain."

        elif verb == "derive":
            tool = parsed.get("tool")
            text = parsed.get("text")
            if not tool or not text:
                obs = "PROTOCOL VIOLATION: 'derive' requires tool=TOOL and text=..."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            if tool not in self.available_tools:
                obs = f"UNSUPPORTED ACTION: Unknown or unavailable tool '{tool}'."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            self.used_tools.append(tool)
            self.derived_facts.append(f"Derived[{tool}]: {text}")
            self.last_feedback = f"Applied {tool}."
            remaining_est = max(0, self.required_steps - len(self.derived_nonassumptions()))
            obs = f"Derivation noted via {tool}. Estimated remaining key steps: {remaining_est}."

        elif verb == "simplify":
            text = parsed.get("text")
            if not text:
                obs = "PROTOCOL VIOLATION: 'simplify' requires text=..."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            self.derived_facts.append(f"Simplified: {text}")
            self.last_feedback = "Simplification recorded."
            remaining_est = max(0, self.required_steps - len(self.derived_nonassumptions()))
            obs = f"Simplification recorded. Estimated remaining key steps: {remaining_est}."

        elif verb == "hint":
            if self.remaining_hints <= 0:
                obs = "NO HINTS LEFT."
            else:
                self.remaining_hints -= 1
                hint_text = self._produce_hint()
                obs = f"HINT: {hint_text}"

        elif verb == "submit":
            val_str = parsed.get("value")
            if val_str is None:
                obs = "PROTOCOL VIOLATION: 'submit' requires value=INTEGER."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            try:
                guess = int(val_str)
            except ValueError:
                obs = "FORMAT ERROR: Final answer must be an integer."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            if guess == self.hidden_answer:
                obs = f"Success! Correct answer {guess}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Incorrect. Submitted {guess}. The episode ends."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"UNSUPPORTED ACTION: '{verb}' is not recognized."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

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
        if not parts:
            return None
        tokens: Dict[str, Any] = {"action": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                # allow text=... with spaces by capturing remainder if quoted not enforced:
                if v.startswith('"') and not v.endswith('"'):
                    # recombine until closing quote
                    # find index of current part
                    idx = parts.index(part)
                    acc = [v]
                    j = idx + 1
                    while j < len(parts):
                        acc.append(parts[j])
                        if parts[j].endswith('"'):
                            break
                        j += 1
                    v = " ".join(acc)
                v = v.strip()
                if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                    v = v[1:-1]
                tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        options = [
            r'\boxed{assume text="n is a positive integer"}',
            r'\boxed{derive tool=ARITH_SUM text="Computed S_n using n/2(a1+an)"}',
            r'\boxed{simplify text="reduced expression to 45"}',
            r'\boxed{hint}',
            r'\boxed{submit value=42}',
        ]
        return random.choice(options)

    # Helpers

    def derived_nonassumptions(self) -> List[str]:
        return [f for f in self.derived_facts if not f.startswith("Assumed")]

    def _produce_hint(self) -> str:
        # Light, actionable hint without revealing answer
        if "arithmetic progression" in self.instance_text.lower():
            return "Use ARITH_SUM to compute the total; ensure n and endpoints are correct."
        if "discriminant" in self.instance_text.lower() or "quadratic" in self.instance_text.lower():
            return "Apply QUAD_DISCRIMINANT and check whether integer roots fall in the specified range."
        if "right triangle" in self.instance_text.lower():
            return "Use PYTHAGOREAN, then scale carefully; watch integer rounding."
        if "combination" in self.instance_text.lower():
            return "Apply COMB_IDENTITIES and basic arithmetic; avoid overflow by simplifying."
        if "modulo" in self.instance_text.lower():
            return "Solve ax ≡ b (mod m) with MOD_LINEAR; compute gcd and an inverse if it exists."
        return "Identify which listed tool directly targets the structure of the instance."

    # Instance generators (return (text, answer, required_tools))
    def _gen_arithmetic_series(self, magnitude: int, depth: int):
        # Sum of first n terms with given a1 and d
        a1 = random.randint(1, max(2, magnitude // 2))
        d = random.randint(1, max(2, magnitude // 3))
        n = random.randint(4, max(6, 2 + depth * 3))
        an = a1 + (n - 1) * d
        s = n * (a1 + an) // 2
        text = (
            f"Compute the sum of the first {n} terms of an arithmetic progression with first term {a1} "
            f"and common difference {d}. Provide the integer total."
        )
        return text, s, ["ARITH_SUM"]

    def _gen_quadratic_roots_in_range(self, magnitude: int, depth: int):
        # Count integer roots in [L, R] by constructing factorable quadratics with small integer roots
        r1 = random.randint(-magnitude // 2, magnitude // 2)
        r2 = random.randint(-magnitude // 2, magnitude // 2)
        a = random.choice([1, 1, -1])  # mostly monic for solvability clarity
        b = -a * (r1 + r2)
        c = a * r1 * r2
        L = min(r1, r2) - random.randint(0, 2)
        R = max(r1, r2) + random.randint(0, 2)
        # The polynomial ax^2 + bx + c has integer roots at r1 and r2 if a=1 or -1
        roots_in = len([r for r in set([r1, r2]) if L <= r <= R])
        text = (
            f"Consider the quadratic f(x) = {a}x^2 + {b}x + {c}. "
            f"How many integer roots does f(x)=0 have in the interval [{L}, {R}]? "
            f"Answer with an integer count."
        )
        return text, roots_in, ["QUAD_DISCRIMINANT"]

    def _gen_scaled_pythagorean_hypotenuse(self, magnitude: int, depth: int):
        # Construct a scaled primitive triple (3,4,5) or (5,12,13) etc.
        primitives = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
        a0, b0, c0 = random.choice(primitives)
        k = random.randint(1, max(2, magnitude // 10))
        a = a0 * k
        b = b0 * k
        c = c0 * k
        text = (
            f"In a right triangle with legs {a} and {b}, compute the hypotenuse as an integer."
        )
        return text, c, ["PYTHAGOREAN"]

    def _gen_combinations_difference(self, magnitude: int, depth: int):
        # Compute C(n,k) - C(n,k-1) for moderate n,k guaranteeing integer
        n = random.randint(6, max(7, 5 + depth * 3))
        k = random.randint(2, n - 2)
        def nCk(nv, kv):
            from math import comb
            return comb(nv, kv)
        answer = nCk(n, k) - nCk(n, k - 1)
        text = (
            f"Compute the integer value of C({n},{k}) - C({n},{k-1})."
        )
        return text, answer, ["COMB_IDENTITIES"]

    def _gen_mod_linear(self, magnitude: int, depth: int):
        # Solve ax ≡ b (mod m), ask for smallest nonnegative solution
        # Ensure solvable by choosing a,b,m with gcd(a,m) | b
        m = random.randint(7, max(9, magnitude))
        a = random.randint(2, max(4, magnitude // 2))
        g = random.randint(1, 5)
        m = max(m, a + 3)
        # Build b as multiple of gcd
        from math import gcd
        g_actual = gcd(a, m)
        # if not divisible, adjust b to be divisible
        b = random.randint(0, m - 1)
        b = b - (b % g_actual)
        # Find minimal solution
        # Reduce: a' x ≡ b' (mod m') with m' = m/g, a' = a/g, b' = b/g
        a1 = a // g_actual
        m1 = m // g_actual
        b1 = (b // g_actual) % m1
        # Modular inverse of a1 mod m1
        inv = self._modinv(a1 % m1, m1)
        x0 = (inv * b1) % m1 if inv is not None else 0
        # Lift to modulus m: solutions x = x0 + t*m1; choose smallest nonnegative
        solution = x0
        text = (
            f"Find the smallest nonnegative integer x solving {a}*x ≡ {b} (mod {m})."
        )
        return text, solution, ["MOD_LINEAR"]

    def _egcd(self, a, b):
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = self._egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)

    def _modinv(self, a, m):
        g, x, _ = self._egcd(a, m)
        if g != 1:
            return None
        return x % m


class TheoremTacticianEnvWithFeedback(TheoremTacticianEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text or "format error" in text:
            error_type = "FormatError"
            if "missing" in text or "requires" in text:
                error_detail["issue"] = "malformed_parameters"
            else:
                error_detail["issue"] = "boxed_or_parse_failure"
            hint = 'Use \\boxed{...} and include required fields like tool= or value=.'

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command_or_tool"
            hint = "Use actions: assume, derive, simplify, hint, submit. Tools must be from the listed inventory."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "missing_required_fields"
            hint = "Provide all required parameters, e.g., derive tool=TOOL text=\"...\"."

        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["turn_limit"] = self.max_turns
            hint = "Plan your derivations in fewer steps and submit before the turn limit."

        elif "incorrect. submitted" in text:
            error_type = "WrongDecision"
            error_detail["got"] = self._extract_number_from_text(obs)
            error_detail["expected_form"] = "integer"
            hint = "Re-examine which tool applies; consider using a hint or checking arithmetic."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "used_tools": list(self.used_tools),
                "remaining_hints": self.remaining_hints,
                "derived_facts_count": len(self.derived_facts),
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
            "hint": "Start by identifying the matching tool (e.g., ARITH_SUM, QUAD_DISCRIMINANT) and derive a key fact.",
            "turn": 0,
            "state": {
                "used_tools": [],
                "remaining_hints": self.remaining_hints,
                "derived_facts_count": 0,
            },
        }
        return obs, info

    def _extract_number_from_text(self, s: str) -> Optional[int]:
        m = re.search(r"(-?\d+)", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None