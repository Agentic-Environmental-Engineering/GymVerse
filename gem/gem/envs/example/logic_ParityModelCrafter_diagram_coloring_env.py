from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set

class ParityModelCrafterEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 25,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 25

        # Evolvable parameters
        self.complexity_params = {
            # Number of propositional atoms: more atoms = larger assignment space = harder
            'num_atoms': (3, 12),
            # Number of constraints: more constraints = tighter consistency = harder
            'num_constraints': (3, 30),
            # Required number of distinct labels used by assignment (resource count): higher K = harder to meet exactly
            # We use categories beyond classical True/False by introducing 3-valued logic labels: T, F, U (unknown)
            # Higher K means you must use more distinct labels among {T, F, U}
            'required_labels': (2, 3),
            # Proportion (%) of parity-type constraints (XOR/equivalence). Higher parity share = more global coupling = harder
            'parity_pct': (20, 70),
            # Noise/exceptions rate (%) allowing "soft" hints count; lower hints = harder (REVERSED: easy high -> hard low)
            'hint_tokens': (2, 0),
        }

        # Variance per parameter
        self.param_variance = {
            'num_atoms': 1,           # medium range integer
            'num_constraints': 2,     # larger range integer
            'required_labels': 0,     # tiny range, keep fixed per level
            'parity_pct': 5,          # percentage small jitter
            'hint_tokens': 0,         # deterministic per level
        }

        # Placeholders
        self.num_atoms: int = 0
        self.num_constraints: int = 0
        self.required_labels: int = 0
        self.parity_pct: int = 0
        self.hint_tokens: int = 0

        # State
        self.turn_count: int = 0
        self.atoms: List[str] = []
        # Constraints are tuples: (type, a, b, neg_a, neg_b)
        # Supported types: 'imp' (a -> b), 'xor' (a XOR b), 'eqv' (a <-> b), 'nb' (NOT BOTH)
        self.constraints: List[Tuple[str, str, str, bool, bool]] = []
        # Solution assignment over three-valued set: {T, F, U}
        self.solution: Dict[str, str] = {}
        self.active: bool = True

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (lo, hi) in self.complexity_params.items():
            center = lo + (hi - lo) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
            # clamp
            minv, maxv = (hi, lo) if lo > hi else (lo, hi)
            val = max(minv, min(maxv, val))
            setattr(self, name, int(round(val)))

    def _get_instructions(self) -> str:
        base = []
        base.append("You are constructing a three-valued logical model using exactly K distinct labels.")
        base.append("Atoms can be assigned one of: T (true), F (false), U (undetermined).")
        base.append("Constraints types:")
        base.append("- imp: a -> b means if a=T then b must be T; negations may appear on either side.")
        base.append("- eqv: a <-> b means a and b must match exactly (T with T, F with F, U with U).")
        base.append("- xor: a XOR b means exactly one of a,b is T; U counts as non-T.")
        base.append("- nb: NOT BOTH means a and b cannot both be T.")
        base.append("You must output a complete assignment for all atoms and use exactly K distinct labels among {T,F,U}.")
        base.append("Action format:")
        base.append(r"- Use \boxed{assign A=T B=F C=U ...}")
        base.append("Rules:")
        base.append("- Provide all atoms in one line; missing atoms cause failure.")
        base.append("- Only labels T,F,U are valid; extra keys are ignored.")
        base.append(f"- K is given below; using fewer or more distinct labels than K fails.")
        example = self.sample_random_action()
        base.append("Example action:")
        base.append(example)
        return "\n".join(base)

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Current instance:")
        lines.append(f"- Atoms: {', '.join(self.atoms)}")
        cdesc = []
        for (t, a, b, na, nb) in self.constraints:
            def lit(x, neg):
                return ("¬" + x) if neg else x
            if t == 'imp':
                cdesc.append(f"{lit(a,na)} -> {lit(b,nb)}")
            elif t == 'eqv':
                cdesc.append(f"{lit(a,na)} <-> {lit(b,nb)}")
            elif t == 'xor':
                cdesc.append(f"{lit(a,na)} XOR {lit(b,nb)}")
            elif t == 'nb':
                cdesc.append(f"NOT_BOTH({lit(a,na)}, {lit(b,nb)})")
        lines.append(f"- Constraints ({len(self.constraints)}): " + "; ".join(cdesc))
        lines.append(f"- K (distinct labels required): {self.required_labels}")
        if self.hint_tokens > 0:
            # Provide up to hint_tokens random hints derived from a known satisfying solution
            hints = []
            reveal_keys = random.sample(self.atoms, k=min(self.hint_tokens, len(self.atoms)))
            for k in reveal_keys:
                hints.append(f"{k}={self.solution.get(k,'U')}")
            if hints:
                lines.append(f"- Hints: " + ", ".join(hints))
        lines.append("Enter your assignment as \\boxed{assign A=T B=F ...}")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.active = True

        # Generate atoms
        self.atoms = [f"P{i+1}" for i in range(self.num_atoms)]

        # Generate a hidden satisfying assignment first to ensure feasibility
        # Choose label set to exactly K distinct labels among T,F,U
        labels_pool = ['T', 'F', 'U']
        k = max(1, min(3, self.required_labels))
        # Guarantee we can use exactly K labels
        chosen = labels_pool[:k]
        # Assign labels ensuring all chosen appear
        self.solution = {}
        # Seed ensure at least one of each chosen label appears
        forced_positions = random.sample(self.atoms, k=min(k, len(self.atoms)))
        for idx, a in enumerate(forced_positions):
            self.solution[a] = chosen[idx % k]
        for a in self.atoms:
            if a not in self.solution:
                self.solution[a] = random.choice(chosen)

        # Build constraints consistent with the solution
        self.constraints = []
        target = self.num_constraints
        parity_share = self.parity_pct / 100.0
        parity_targets = int(round(target * parity_share))
        others = target - parity_targets

        # Helper to sample literals respecting solution
        def sample_pair() -> Tuple[str, str, bool, bool]:
            a, b = random.sample(self.atoms, 2)
            na = random.choice([False, True])
            nb = random.choice([False, True])
            return a, b, na, nb

        # Define literal evaluation under three-valued logic
        def eval_lit(atom: str, neg: bool, assign: Dict[str, str]) -> str:
            base = assign[atom]  # 'T','F','U'
            if neg:
                if base == 'T': return 'F'
                if base == 'F': return 'T'
                return 'U'
            return base

        def check_imp(a, b, na, nb, assign):
            va = eval_lit(a, na, assign)
            vb = eval_lit(b, nb, assign)
            # a -> b: If a is T then b must be T. Otherwise OK.
            if va == 'T':
                return vb == 'T'
            return True

        def check_eqv(a, b, na, nb, assign):
            va = eval_lit(a, na, assign)
            vb = eval_lit(b, nb, assign)
            return va == vb

        def check_xor(a, b, na, nb, assign):
            va = eval_lit(a, na, assign)
            vb = eval_lit(b, nb, assign)
            # exactly one is T
            return (va == 'T') ^ (vb == 'T')

        def check_nb(a, b, na, nb, assign):
            va = eval_lit(a, na, assign)
            vb = eval_lit(b, nb, assign)
            return not (va == 'T' and vb == 'T')

        # Build parity constraints first (xor/eqv)
        attempts = 0
        while len(self.constraints) < parity_targets and attempts < target * 10:
            attempts += 1
            ttype = random.choice(['xor', 'eqv'])
            a, b, na, nb = sample_pair()
            ok = (check_xor(a,b,na,nb,self.solution) if ttype=='xor'
                  else check_eqv(a,b,na,nb,self.solution))
            if ok:
                self.constraints.append((ttype, a, b, na, nb))

        # Then other constraints (imp/nb)
        attempts = 0
        while len(self.constraints) < target and attempts < target * 20:
            attempts += 1
            ttype = random.choice(['imp', 'nb'])
            a, b, na, nb = sample_pair()
            if ttype == 'imp':
                ok = check_imp(a,b,na,nb,self.solution)
            else:
                ok = check_nb(a,b,na,nb,self.solution)
            if ok:
                self.constraints.append((ttype, a, b, na, nb))

        # If under-filled due to randomness, backfill with easy implications that hold
        attempts = 0
        while len(self.constraints) < target and attempts < target * 50:
            attempts += 1
            a, b, na, nb = sample_pair()
            if check_imp(a,b,na,nb,self.solution):
                self.constraints.append(('imp', a, b, na, nb))

        # Final feasibility check
        if not self._validate_assignment(self.solution):
            # As a fallback (rare), regenerate with trivial constraints
            self.constraints = []
            for i in range(min(self.num_constraints, max(0, self.num_atoms - 1))):
                a = self.atoms[i]
                b = self.atoms[(i+1) % len(self.atoms)]
                self.constraints.append(('imp', a, b, False, False))

        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if not self.active:
            return "Episode already ended.", 0.0, True, False, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if parsed is None:
            obs = "INVALID ACTION FORMAT: Use \\boxed{assign A=T B=F ...}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get('action') != 'assign':
            obs = "UNSUPPORTED ACTION: Only 'assign' is allowed."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        assignment = parsed.get('assignment', {})
        # Check completeness
        missing = [a for a in self.atoms if a not in assignment]
        extra = [k for k in assignment.keys() if k not in self.atoms]
        invalid_labels = {k: v for k, v in assignment.items() if v not in ('T','F','U')}

        if missing:
            obs = f"Failed! Missing atoms: {', '.join(missing)}."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        if invalid_labels:
            bads = ", ".join([f"{k}={v}" for k,v in invalid_labels.items()])
            obs = f"Failed! Invalid labels used: {bads}. Use only T,F,U."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        if extra:
            # Extra keys are ignored but note them; does not immediately fail
            pass

        # Check label-count resource requirement
        used: Set[str] = set(assignment.values())
        if len(used) != self.required_labels:
            obs = f"Failed! Resource mismatch: used {len(used)} distinct labels, required {self.required_labels}."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Check constraints satisfaction
        ok, report = self._check_constraints(assignment)
        if not ok:
            obs = "Failed! Constraint violations: " + "; ".join(report[:5])
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.active = False
        obs = "Success! Valid assignment satisfies all constraints and uses exactly K labels."
        return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

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
        if len(parts) == 0:
            return None
        action_name = parts[0].lower()
        if action_name != 'assign':
            return {'action': action_name}
        assign: Dict[str,str] = {}
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip().upper()
                if k:
                    assign[k] = v
        return {'action': 'assign', 'assignment': assign}

    def sample_random_action(self) -> str:
        if not self.atoms:
            # generic example
            return r"\boxed{assign P1=T P2=F P3=U}"
        # Build a random valid-format assignment; not necessarily correct
        labels = ['T','F','U']
        # Try to match required_labels by sampling a palette
        palette = random.sample(labels, k=min(self.required_labels or 2, 3))
        assignment = []
        # Ensure palette appears
        atoms_copy = self.atoms[:]
        random.shuffle(atoms_copy)
        for i, a in enumerate(atoms_copy):
            lab = palette[i % len(palette)]
            assignment.append(f"{a}={lab}")
        return r"\boxed{assign " + " ".join(assignment) + "}"

    def _validate_assignment(self, assign: Dict[str,str]) -> bool:
        # Check constraints for internal solution validation and K labels
        ok, _ = self._check_constraints(assign)
        used = set(assign.values())
        return ok and len(used) == self.required_labels

    def _check_constraints(self, assignment: Dict[str,str]) -> Tuple[bool, List[str]]:
        def eval_lit(atom: str, neg: bool) -> str:
            val = assignment[atom]
            if neg:
                if val == 'T': return 'F'
                if val == 'F': return 'T'
                return 'U'
            return val
        violations = []
        for (t, a, b, na, nb) in self.constraints:
            va = eval_lit(a, na)
            vb = eval_lit(b, nb)
            if t == 'imp':
                if va == 'T' and vb != 'T':
                    violations.append(f"imp violated: ({'¬' if na else ''}{a})->({'¬' if nb else ''}{b})")
            elif t == 'eqv':
                if va != vb:
                    violations.append(f"eqv violated: ({'¬' if na else ''}{a})<->({'¬' if nb else ''}{b})")
            elif t == 'xor':
                if ((va == 'T') ^ (vb == 'T')) is False:
                    violations.append(f"xor violated: ({'¬' if na else ''}{a}) XOR ({'¬' if nb else ''}{b})")
            elif t == 'nb':
                if va == 'T' and vb == 'T':
                    violations.append(f"nb violated: NOT_BOTH({a},{b})")
        return (len(violations) == 0, violations)


class ParityModelCrafterEnvWithFeedback(ParityModelCrafterEnv):
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
            error_detail["issue"] = "missing_boxed_or_structure"
            hint = "Wrap your assignment in \\boxed{...} and start with 'assign'."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_action"
            hint = "Use only: \\boxed{assign A=T B=F ...}"
        elif "missing atoms" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "incomplete_assignment"
            hint = "Provide every listed atom exactly once with a label T/F/U."
        elif "invalid labels used" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "bad_label_symbol"
            hint = "Use only uppercase T, F, or U for labels."
        elif "resource mismatch" in text:
            error_type = "WrongDecision"
            error_detail["violation"] = "wrong_distinct_label_count"
            error_detail["required"] = self.required_labels
            hint = f"Adjust labels so the set of distinct labels has size {self.required_labels}."
        elif "constraint violations" in text:
            error_type = "WrongDecision"
            error_detail["violation"] = "unsatisfied_constraints"
            # Provide a light, actionable hint
            hint = "Review constraints causing failure; try flipping labels involved in violated relations."
        elif "reached max turns" in text or truncated:
            error_type = "Timeout"
            error_detail["outcome"] = "time_limit"
            hint = "Submit a complete assignment earlier; start by satisfying EQV/XOR constraints."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "atoms": self.atoms,
                "num_constraints": len(self.constraints),
                "required_labels": self.required_labels,
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
            "hint": "Start by planning which labels you will use to meet K, then satisfy XOR/EQV first.",
            "turn": 0,
            "state": {
                "atoms": self.atoms,
                "num_constraints": len(self.constraints),
                "required_labels": self.required_labels,
            },
        }
        return obs, info