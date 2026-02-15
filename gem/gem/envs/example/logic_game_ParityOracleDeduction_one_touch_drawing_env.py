from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class ParityOracleDeductionEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters (logic-native):
        self.complexity_params = {
            # Number of propositional atoms A_i (larger = harder search)
            "num_atoms": (4, 12),
            # Number of binary relations R_j between atoms (implications/iff/xor) (more = harder)
            "num_relations": (3, 18),
            # Number of parity constraints over subsets (more and larger = harder)
            "num_parity": (0, 6),
            # Max size of subsets used in parity constraints (larger = harder)
            "max_parity_span": (2, 6),
            # Global quota constraints: number of "True-budget" caps/enforcements (more = harder)
            "num_global_quota": (0, 3),
            # Clue noise level (0: direct statements; higher: more indirection/alias text; harder)
            "obfuscation_level": (0, 3),
        }

        # Variance tuned to ranges
        self.param_variance = {
            "num_atoms": 1,
            "num_relations": 2,
            "num_parity": 1,
            "max_parity_span": 1,
            "num_global_quota": 1,
            "obfuscation_level": 0,
        }

        # Placeholders set during _apply_complexity_params
        self.num_atoms: int = 0
        self.num_relations: int = 0
        self.num_parity: int = 0
        self.max_parity_span: int = 0
        self.num_global_quota: int = 0
        self.obfuscation_level: int = 0

        # State
        self.turn_count: int = 0
        self.instance_seed: Optional[int] = None

        # Instance details
        self.atom_names: List[str] = []
        self.truth_assignment: Dict[str, bool] = {}
        self.relations: List[Tuple[str, str, str]] = []  # (type, X, Y) where type in {"IMP","IFF","XOR"}
        self.parity_constraints: List[Tuple[List[str], int]] = []  # ([atoms], parity 0/1)
        self.global_quota: List[Tuple[List[str], int]] = []  # ([atoms], exactly k true)
        self.story_clauses: List[str] = []  # Rendered puzzle statements

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
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _pick_names(self, n: int) -> List[str]:
        base_pool = [
            "Alpha", "Bravo", "Cedar", "Delta", "Echo", "Fjord", "Garnet", "Helix", "Indigo", "Jasper",
            "Kappa", "Lumen", "Mosaic", "Nimbus", "Onyx", "Pico", "Quanta", "Rivet", "Sable", "Topaz",
            "Umber", "Vivid", "Wisp", "Xenon", "Yarrow", "Zircon"
        ]
        random.shuffle(base_pool)
        chosen = base_pool[:n]
        # Ensure uniqueness if n > pool
        if len(chosen) < n:
            for i in range(n - len(chosen)):
                chosen.append(f"Atom{i+1}")
        return chosen

    def _generate_consistent_instance(self):
        n = self.num_atoms
        atoms = self._pick_names(n)
        self.atom_names = atoms

        # Sample a ground-truth satisfying assignment
        if not self.enable_param_randomization and self.complexity <= 1:
            # Make the lowest complexity deterministic and easy for scripted minimal-path tests
            self.truth_assignment = {a: False for a in atoms}
        else:
            def sample_assign() -> Dict[str, bool]:
                return {a: (random.random() < 0.5) for a in atoms}

            # Allow trivial uniform assignments only at complexity 1 to keep early curriculum easy
            if self.complexity <= 1:
                self.truth_assignment = sample_assign()
            else:
                while True:
                    assign = sample_assign()
                    # Avoid completely uniform assignments for higher complexity
                    if n == 1 or any(assign[a] != assign[atoms[0]] for a in atoms[1:]):
                        self.truth_assignment = assign
                        break

        # Build relations consistent with assignment
        assign = self.truth_assignment
        self.relations = []
        rel_types = ["IMP", "IFF", "XOR"]
        # Ensure connectivity-like structure: pick pairs and set type consistent with assignment
        pairs = []
        for _ in range(self.num_relations):
            x, y = random.sample(atoms, 2)
            pairs.append((x, y))
        for (x, y) in pairs:
            t = random.choice(rel_types)
            xv, yv = assign[x], assign[y]
            if t == "IMP":
                # x -> y must be true; if xv and not yv, flip to maintain truth
                if xv and not yv:
                    # convert to something true given xv,yv; choose y -> x instead
                    t = "IMP"
                    x, y = y, x
                    xv, yv = yv, xv
                # Now implication true under assignment
                self.relations.append((t, x, y))
            elif t == "IFF":
                # x <-> y must be true; if not equal, swap to XOR or pick different pair
                if xv == yv:
                    self.relations.append((t, x, y))
                else:
                    # if differing, use XOR instead
                    self.relations.append(("XOR", x, y))
            else:  # XOR
                if xv != yv:
                    self.relations.append(("XOR", x, y))
                else:
                    # if equal, use IFF instead
                    self.relations.append(("IFF", x, y))

        # Parity constraints: pick random subsets, enforce parity per assignment
        self.parity_constraints = []
        for _ in range(self.num_parity):
            span = random.randint(2, min(self.max_parity_span, n))
            subset = sorted(random.sample(atoms, span))
            parity = sum(1 for a in subset if assign[a]) % 2
            self.parity_constraints.append((subset, parity))

        # Global quota constraints: exactly-k-true among a subset
        self.global_quota = []
        used_subsets: Set[str] = set()
        for _ in range(self.num_global_quota):
            span = random.randint(2, max(2, min(n, self.max_parity_span + 1)))
            subset = sorted(random.sample(atoms, span))
            key = ",".join(subset)
            if key in used_subsets:
                continue
            k = sum(1 for a in subset if assign[a])
            self.global_quota.append((subset, k))
            used_subsets.add(key)

        # Render story clauses
        self.story_clauses = self._render_clauses()

    def _render_atom(self, a: str) -> str:
        # Add obfuscation by aliasing and indirection
        if self.obfuscation_level == 0:
            return a
        elif self.obfuscation_level == 1:
            return f"Proposition[{a}]"
        elif self.obfuscation_level == 2:
            return f"Claim<{a}>"
        else:
            return f"Signal({a})"

    def _render_rel(self, kind: str, x: str, y: str) -> str:
        X = self._render_atom(x)
        Y = self._render_atom(y)
        if kind == "IMP":
            if self.obfuscation_level <= 1:
                return f"If {X} then {Y}."
            elif self.obfuscation_level == 2:
                return f"{X} implies {Y}."
            else:
                return f"Whenever {X} holds, {Y} must hold."
        elif kind == "IFF":
            if self.obfuscation_level <= 1:
                return f"{X} if and only if {Y}."
            elif self.obfuscation_level == 2:
                return f"{X} exactly when {Y}."
            else:
                return f"{X} is equivalent to {Y}."
        else:  # XOR
            if self.obfuscation_level <= 1:
                return f"Exactly one of {X} and {Y} is true."
            elif self.obfuscation_level == 2:
                return f"{X} and {Y} differ in truth."
            else:
                return f"An exclusive alternative holds between {X} and {Y}."

    def _render_parity(self, subset: List[str], parity: int) -> str:
        names = ", ".join(self._render_atom(a) for a in subset)
        if self.obfuscation_level == 0:
            return f"The number of true among [{names}] has parity {parity}."
        elif self.obfuscation_level == 1:
            return f"Among [{names}], an {'odd' if parity==1 else 'even'} count are true."
        elif self.obfuscation_level == 2:
            return f"[{names}] evaluates to an {'odd' if parity==1 else 'even'} tally of truths."
        else:
            return f"Truth-count over [{names}] is {'odd' if parity==1 else 'even'}."

    def _render_quota(self, subset: List[str], k: int) -> str:
        names = ", ".join(self._render_atom(a) for a in subset)
        if self.obfuscation_level <= 1:
            return f"Exactly {k} of [{names}] are true."
        elif self.obfuscation_level == 2:
            return f"The set [{names}] contains exactly {k} truths."
        else:
            return f"Truth quota: [{names}] must have cardinality {k} of truths."

    def _render_clauses(self) -> List[str]:
        text = []
        for (t, x, y) in self.relations:
            text.append(self._render_rel(t, x, y))
        for (subset, p) in self.parity_constraints:
            text.append(self._render_parity(subset, p))
        for (subset, k) in self.global_quota:
            text.append(self._render_quota(subset, k))
        return text

    def _get_instructions(self) -> str:
        atoms_list = ", ".join(self.atom_names) if self.atom_names else "(not initialized)"
        rules = "\n".join(f"- {cl}" for cl in self.story_clauses) if self.story_clauses else "- No constraints."
        action_fmt = (
            "Submit a complete truth assignment covering every proposition exactly once.\n"
            "Format: \\boxed{assign A=True, B=False, ...}\n"
            "Separate pairs by commas. Use exact names as listed. Booleans must be True or False."
        )
        return (
            "Logic Quest: Parity Oracle Deduction\n"
            "Goal: Provide a complete truth assignment for all propositions so that ALL constraints hold.\n"
            "You have multiple turns to refine your submission. Only a fully consistent assignment yields success.\n"
            "\n"
            f"Propositions: {atoms_list}\n"
            "Constraints:\n"
            f"{rules}\n"
            "\n"
            f"{action_fmt}\n"
            "Example:\n"
            r"\boxed{assign Alpha=True, Bravo=False, Cedar=True}"
        )

    def get_task_suffix(self) -> str:
        remaining = self.max_turns - self.turn_count
        atoms_list = ", ".join(self.atom_names)
        return (
            f"Turns left: {remaining}\n"
            f"Propositions: {atoms_list}\n"
            "Enter your action as \\boxed{assign Name=True/False, ...} covering ALL propositions."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self.instance_seed = seed if seed is not None else random.randint(0, 10**9)
        self.turn_count = 0
        self._apply_complexity_params()
        self._generate_consistent_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _eval_assignment(self, cand: Dict[str, bool]) -> Tuple[bool, List[str]]:
        errors = []

        # Coverage
        if set(cand.keys()) != set(self.atom_names):
            missing = [a for a in self.atom_names if a not in cand]
            extra = [a for a in cand if a not in self.atom_names]
            if missing:
                errors.append(f"CoverageError: missing assignments for {', '.join(missing)}")
            if extra:
                errors.append(f"CoverageError: unknown propositions {', '.join(extra)}")

        # Relation checks
        for (t, x, y) in self.relations:
            xv = cand.get(x, None)
            yv = cand.get(y, None)
            if xv is None or yv is None:
                continue
            ok = True
            if t == "IMP":
                ok = (not xv) or yv
            elif t == "IFF":
                ok = (xv == yv)
            elif t == "XOR":
                ok = (xv != yv)
            if not ok:
                errors.append(f"RelationViolation: {t} between {x} and {y} fails")

        # Parity checks
        for (subset, p) in self.parity_constraints:
            vals = [cand.get(a, None) for a in subset]
            if any(v is None for v in vals):
                continue
            parity = sum(1 for v in vals if v) % 2
            if parity != p:
                names = ", ".join(subset)
                expect = "odd" if p == 1 else "even"
                errors.append(f"ParityViolation: [{names}] expected {expect} count")

        # Quota checks
        for (subset, k) in self.global_quota:
            vals = [cand.get(a, None) for a in subset]
            if any(v is None for v in vals):
                continue
            cnt = sum(1 for v in vals if v)
            if cnt != k:
                names = ", ".join(subset)
                errors.append(f"QuotaViolation: [{names}] expected exactly {k} true")

        return (len(errors) == 0), errors

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None

        # Support multiple boxed segments; the last boxed command takes precedence.
        matches = list(re.finditer(r"\\boxed\{(.*?)\}", action, flags=re.DOTALL))
        if not matches:
            return None
        inner = matches[-1].group(1).strip()
        if not inner.lower().startswith("assign"):
            return {"action": "unsupported", "raw": inner}

        # Expect: assign A=True, B=False, ...
        rest = inner[len("assign"):].strip()
        if not rest:
            return {"action": "assign", "pairs": {}}

        # Split by commas
        pairs = {}
        for chunk in rest.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                # malformed pair
                return {"action": "malformed", "chunk": chunk}
            name, val = chunk.split("=", 1)
            name = name.strip()
            val = val.strip()
            if val not in ["True", "False"]:
                return {"action": "malformed", "chunk": chunk}
            pairs[name] = True if val == "True" else False
        return {"action": "assign", "pairs": pairs}

    def sample_random_action(self) -> str:
        # Provide a random complete assignment guess
        if not self.atom_names:
            return r"\boxed{assign A=True}"
        pairs = []
        for a in self.atom_names:
            v = "True" if random.random() < 0.5 else "False"
            pairs.append(f"{a}={v}")
        inside = "assign " + ", ".join(pairs)
        return f"\\boxed{{{inside}}}"

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} and include 'assign' with name=value pairs."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("action") == "unsupported":
            obs = "UNSUPPORTED ACTION: Only 'assign' is accepted. Use \\boxed{assign Name=True, ...}"
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("action") == "malformed":
            bad = parsed.get("chunk", "")
            obs = f"MALFORMED ASSIGNMENT: Problem near '{bad}'. Use Name=True/False with commas."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("action") == "assign":
            pairs = parsed.get("pairs", {})
            # Basic validation
            if not pairs:
                obs = "EMPTY ASSIGNMENT: Provide all propositions with True/False values."
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

            # Evaluate
            ok, errors = self._eval_assignment(pairs)
            if ok:
                obs = "Success! Your assignment satisfies all constraints."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                # Provide compact feedback listing distinct categories first
                categories = {}
                for e in errors:
                    cat = e.split(":")[0]
                    categories.setdefault(cat, 0)
                    categories[cat] += 1
                summary_parts = [f"{k}={v}" for k, v in categories.items()]
                obs = "Inconsistent assignment. Violations: " + "; ".join(errors[:6])
                obs += f" | Summary: {'; '.join(summary_parts)}"
                # Continue episode unless out of turns
                if self.turn_count >= self.max_turns:
                    obs += f" | Reached max turns ({self.max_turns})."
                    return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        # Fallback
        obs = "UNEXPECTED: Parsing reached unknown branch."
        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}


class ParityOracleDeductionEnvWithFeedback(ParityOracleDeductionEnv):
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
            error_detail["issue"] = "missing_boxed_or_assign"
            hint = "Wrap your command like \\boxed{assign Name=True, Other=False}."

        elif "empty assignment" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "empty_assignment"
            hint = "Include every proposition exactly once with True/False values."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["action"] = "non-assign"
            hint = "Use only 'assign' with full coverage, e.g., \\boxed{assign A=True, B=False}."

        elif "malformed assignment" in text:
            error_type = "FormatError"
            error_detail["issue"] = "malformed_pair"
            hint = "Pairs must be Name=True or Name=False, separated by commas."

        elif "inconsistent assignment" in text:
            error_type = "WrongDecision"
            # Extract categories summary
            summary_match = re.search(r"summary:\s*([^\|]+)$", obs, flags=re.IGNORECASE)
            if summary_match:
                error_detail["summary"] = summary_match.group(1).strip()
            # Domain hints
            if "relationviolation" in text:
                hint = "Check implications/iff/xor pairs: try flipping one endpoint where constraints focus."
            elif "parityviolation" in text:
                hint = "Adjust the parity subsets: toggle a minimal set to match odd/even requirements."
            elif "quotaviolation" in text:
                hint = "For quota sets, ensure exactly k truths; count and adjust within that subset."
            elif "coverageerror" in text:
                hint = "Ensure every listed proposition appears exactly once with True/False."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan adjustments systematically: address coverage, then binary relations, then parity/quota."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_atoms": getattr(self, "num_atoms", None),
                "num_relations": getattr(self, "num_relations", None),
                "num_parity": getattr(self, "num_parity", None),
                "num_global_quota": getattr(self, "num_global_quota", None),
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
            "hint": "Start by proposing any complete assignment. Then refine to satisfy relations, parity, and quotas.",
            "turn": 0,
            "state": {
                "num_atoms": getattr(self, "num_atoms", None),
                "num_relations": getattr(self, "num_relations", None),
                "num_parity": getattr(self, "num_parity", None),
                "num_global_quota": getattr(self, "num_global_quota", None),
            },
        }
        return obs, info
