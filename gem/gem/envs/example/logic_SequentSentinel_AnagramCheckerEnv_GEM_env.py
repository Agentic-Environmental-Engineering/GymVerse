from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class SequentSentinelEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 24,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 24

        # Evolvable parameters
        self.complexity_params = {
            # Number of propositional variables; more atoms -> exponential model space -> harder
            'num_atoms': (2, 8),
            # Number of premises; more premises -> tighter constraints and more reasoning -> harder
            'num_premises': (1, 8),
            # Max internal operator count per generated formula; deeper structure -> harder
            'max_ops_per_formula': (1, 5),
            # Percentage of premises that may use implication; richer connectives -> harder
            'implies_share_pct': (0, 70),
            # Percentage of disjunction-heavy structure; more branching -> harder
            'or_share_pct': (0, 60),
            # Rate of negation occurrence (percent); polarity flips increase reasoning load -> harder
            'negation_rate_pct': (0, 50),
        }

        # Parameter variance (adds variety within each complexity)
        self.param_variance = {
            'num_atoms': 0,               # small range
            'num_premises': 1,            # medium discrete range
            'max_ops_per_formula': 1,     # medium discrete range
            'implies_share_pct': 7,       # ~10% of 0-70
            'or_share_pct': 6,            # ~10% of 0-60
            'negation_rate_pct': 5,       # ~10% of 0-50
        }

        # Placeholder attributes (set in _apply_complexity_params)
        self.num_atoms: int = 0
        self.num_premises: int = 0
        self.max_ops_per_formula: int = 0
        self.implies_share_pct: int = 0
        self.or_share_pct: int = 0
        self.negation_rate_pct: int = 0

        # Other state
        self.turn_count: int = 0
        self.atoms: List[str] = []
        self.premises = []        # list of formula trees
        self.premise_texts = []   # stringified forms
        self.conclusion = None
        self.conclusion_text = ""
        self.is_entailed: Optional[bool] = None
        self._cached_models: Optional[List[Dict[str, bool]]] = None

        self.reset()

    # ----- Utility: complexity application -----
    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    # ----- Internal formula helpers -----
    def _rand_literal(self):
        var = random.choice(self.atoms)
        lit = ('var', var)
        # Negate with probability per negation rate
        if random.randint(1, 100) <= self.negation_rate_pct:
            return ('not', lit)
        return lit

    def _rand_formula(self, remaining_ops: int):
        # Generate random formula tree with up to remaining_ops internal nodes.
        if remaining_ops <= 0:
            return self._rand_literal()

        # Bias operation choice by shares
        ops = []
        # Ensure and/or always available
        ops.extend(['and', 'or'])
        # Add 'or' weight by or_share_pct
        if random.randint(1, 100) <= self.or_share_pct:
            ops.append('or')
        # Add implication based on implies_share_pct
        if random.randint(1, 100) <= self.implies_share_pct:
            ops.append('imp')
        # Allow occasional biconditional at higher complexity (derived from implies share)
        allow_iff = (self.implies_share_pct >= 40 and self.max_ops_per_formula >= 3 and self.num_atoms >= 5)
        if allow_iff and random.random() < 0.25:
            ops.append('iff')
        # Allow isolated negation as a node sometimes
        if random.randint(1, 100) <= max(10, self.negation_rate_pct // 2):
            ops.append('not')

        op = random.choice(ops)

        if op == 'not':
            sub = self._rand_formula(remaining_ops - 1)
            return ('not', sub)

        if op in ('and', 'or'):
            # Choose fan-in 2 or 3 sparingly
            fanin = 2 if remaining_ops <= 1 else random.choice([2, 2, 3])
            subs = []
            # Distribute remaining_ops - 1 among children
            budget = remaining_ops - 1
            for i in range(fanin):
                child_budget = budget // fanin
                # Randomly tweak
                if budget > 0 and random.random() < 0.3:
                    tweak = random.randint(0, max(0, budget - child_budget))
                else:
                    tweak = 0
                subs.append(self._rand_formula(max(0, child_budget - tweak)))
            return (op, subs)

        if op in ('imp', 'iff'):
            left = self._rand_formula(remaining_ops - 1)
            right = self._rand_formula(max(0, remaining_ops - 2))
            return (op, left, right)

        # Fallback
        return self._rand_literal()

    def _formula_to_str(self, f):
        t = f[0]
        if t == 'var':
            return f[1]
        if t == 'not':
            return f"~({self._formula_to_str(f[1])})"
        if t == 'and':
            return "(" + " & ".join(self._formula_to_str(x) for x in f[1]) + ")"
        if t == 'or':
            return "(" + " | ".join(self._formula_to_str(x) for x in f[1]) + ")"
        if t == 'imp':
            return "(" + self._formula_to_str(f[1]) + " -> " + self._formula_to_str(f[2]) + ")"
        if t == 'iff':
            return "(" + self._formula_to_str(f[1]) + " <-> " + self._formula_to_str(f[2]) + ")"
        return "?"

    def _eval_formula(self, f, assign: Dict[str, bool]) -> bool:
        t = f[0]
        if t == 'var':
            return bool(assign[f[1]])
        if t == 'not':
            return not self._eval_formula(f[1], assign)
        if t == 'and':
            for x in f[1]:
                if not self._eval_formula(x, assign):
                    return False
            return True
        if t == 'or':
            for x in f[1]:
                if self._eval_formula(x, assign):
                    return True
            return False
        if t == 'imp':
            return (not self._eval_formula(f[1], assign)) or self._eval_formula(f[2], assign)
        if t == 'iff':
            return self._eval_formula(f[1], assign) == self._eval_formula(f[2], assign)
        return False

    def _all_assignments(self) -> List[Dict[str, bool]]:
        # Enumerate all assignments for atoms
        n = len(self.atoms)
        res = []
        for mask in range(1 << n):
            a = {}
            for i, v in enumerate(self.atoms):
                a[v] = bool((mask >> i) & 1)
            res.append(a)
        return res

    def _premises_satisfied(self, assign: Dict[str, bool]) -> bool:
        for p in self.premises:
            if not self._eval_formula(p, assign):
                return False
        return True

    def _models(self) -> List[Dict[str, bool]]:
        # Cache models of premises
        if self._cached_models is None:
            ms = []
            for a in self._all_assignments():
                if self._premises_satisfied(a):
                    ms.append(a)
            self._cached_models = ms
        return self._cached_models

    def _check_entailment(self) -> bool:
        models = self._models()
        if len(models) == 0:
            # Vacuous truth: unsatisfiable premises entail any conclusion
            return True
        for a in models:
            if not self._eval_formula(self.conclusion, a):
                return False
        return True

    def _find_counterexample(self) -> Optional[Dict[str, bool]]:
        models = self._models()
        for a in models:
            if not self._eval_formula(self.conclusion, a):
                return a
        return None

    def _parse_values(self, s: str) -> Optional[Dict[str, bool]]:
        # Parse "A=1,B=0,..." into dict; must cover all atoms
        try:
            items = [x.strip() for x in s.split(',') if x.strip()]
            assign = {}
            for it in items:
                if '=' not in it:
                    return None
                k, v = it.split('=', 1)
                k = k.strip()
                v = v.strip()
                if k not in self.atoms:
                    return None
                if v not in ('0', '1'):
                    return None
                assign[k] = (v == '1')
            if set(assign.keys()) != set(self.atoms):
                return None
            return assign
        except Exception:
            return None

    # ----- Instance generation -----
    def _generate_instance(self):
        # Atoms
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.atoms = list(letters[:self.num_atoms])

        # Generate premises; ensure satisfiable by retrying if needed
        def gen_premises():
            ps = []
            for _ in range(self.num_premises):
                ops = random.randint(0, self.max_ops_per_formula)
                ps.append(self._rand_formula(ops))
            return ps

        # Try multiple times to get satisfiable premises
        for _ in range(40):
            ps = gen_premises()
            self.premises = ps
            self.premise_texts = [self._formula_to_str(p) for p in self.premises]
            self._cached_models = None
            if len(self._models()) > 0:
                break
        else:
            # As a fallback, use simple trivially satisfiable premises (all variables or)
            self.premises = [('or', [('var', v) for v in self.atoms])]
            self.premise_texts = [self._formula_to_str(self.premises)]
            self._cached_models = None

        # Conclusion generation: half valid, half invalid if possible
        want_valid = random.random() < 0.5
        if want_valid:
            # Guaranteed entailed: conjunction of all premises (any model of P satisfies it)
            if len(self.premises) == 1:
                self.conclusion = self.premises[0]
            else:
                self.conclusion = ('and', list(self.premises))
        else:
            # Guaranteed not entailed: pick a model of P and choose a formula false in that model
            m = random.choice(self._models())
            # Try to build a false literal under m
            false_lits = []
            for v in self.atoms:
                val = m[v]
                # literal false under m means choose var if val False, or ~var if val True
                if val:
                    false_lits.append(('not', ('var', v)))
                else:
                    false_lits.append(('var', v))
            if false_lits:
                base = random.choice(false_lits)
            else:
                base = ('not', ('var', self.atoms[0]))
            # Optionally wrap with some ops to keep complexity alignment
            wrap_ops = random.randint(0, max(0, self.max_ops_per_formula - 1))
            f = base
            for _ in range(wrap_ops):
                # Wrap with neutral-ish context that preserves falsity under m: (f & T) or (f & literal true under m)
                if random.random() < 0.5:
                    f = ('and', [f, ('var', random.choice([v for v in self.atoms if m[v]] or [self.atoms[0]]))])
                else:
                    f = ('and', [f, ('not', ('var', random.choice([v for v in self.atoms if not m[v]] or [self.atoms[0]])))])
            self.conclusion = f

        self.conclusion_text = self._formula_to_str(self.conclusion)
        self.is_entailed = self._check_entailment()

    # ----- Interface -----
    def _get_instructions(self) -> str:
        return (
            "You are analyzing a propositional logic sequent: do the hidden premises entail the hidden conclusion?\n"
            "Your goal: decide whether Premises |= Conclusion. Submit your final decision using a commit action.\n"
            "\n"
            "Available actions (use \\boxed{...}):\n"
            "- list_atoms: reveal the set of propositional variables.\n"
            "- show_premise idx=<k>: reveal a specific premise (1-indexed).\n"
            "- show_premises: reveal all premises at once.\n"
            "- models_count: return the number of assignments that satisfy all premises.\n"
            "- check_assignment values=A=1,B=0,...: check if an assignment satisfies all premises.\n"
            "- eval_premise idx=<k> values=A=1,B=0,...: evaluate a premise under an assignment.\n"
            "- eval_conclusion values=A=1,B=0,...: evaluate the conclusion under an assignment.\n"
            "- find_counterexample: if exists, reveal an assignment that satisfies premises but falsifies the conclusion.\n"
            "- commit answer=<yes|no>: final decision; 'yes' means entailment holds, 'no' means it does not.\n"
            "\n"
            "Rules:\n"
            "- Actions must be inside \\boxed{...}.\n"
            "- For actions requiring 'values', provide all atoms exactly once using 0/1.\n"
            "- Unsupported or malformed actions terminate the episode with a penalty.\n"
            "\n"
            "Example actions:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = self.max_turns - self.turn_count
        known = 0  # we do not track partial reveals explicitly here
        return (
            f"Turns left: {remaining}. Premises are hidden; the conclusion is hidden.\n"
            "Decide whether Premises |= Conclusion. Enter actions in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.atoms = []
        self.premises = []
        self.premise_texts = []
        self.conclusion = None
        self.conclusion_text = ""
        self.is_entailed = None
        self._cached_models = None

        self._generate_instance()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} and supported action names."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get('action', '').lower()

        def proto_violation(msg: str):
            return f"PROTOCOL VIOLATION: {msg}", LanguageGameReward.format_error_reward, True, False

        def unsupported(msg: str):
            return f"UNSUPPORTED ACTION: {msg}", LanguageGameReward.format_error_reward, True, False

        reward = 0.0
        obs = ""

        if name == 'list_atoms':
            obs = "OK: atoms = {" + ", ".join(self.atoms) + "}"

        elif name == 'show_premise':
            if 'idx' not in parsed:
                o, r, t, tr = proto_violation("missing parameter 'idx'.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            try:
                k = int(parsed['idx'])
            except Exception:
                o, r, t, tr = proto_violation("parameter 'idx' must be integer.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            if k < 1 or k > len(self.premises):
                o, r, t, tr = proto_violation("idx out of range.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            obs = f"OK: premise[{k}] = {self.premise_texts[k-1]}"

        elif name == 'show_premises':
            lines = [f"{i+1}. {self.premise_texts[i]}" for i in range(len(self.premises))]
            block = "\n".join(lines) if lines else "(none)"
            obs = "OK: premises:\n" + block

        elif name == 'models_count':
            cnt = len(self._models())
            obs = f"OK: models_count = {cnt}"

        elif name == 'check_assignment':
            values = parsed.get('values', None)
            if values is None:
                o, r, t, tr = proto_violation("missing parameter 'values'.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            assign = self._parse_values(values)
            if assign is None:
                o, r, t, tr = proto_violation("invalid 'values' format or missing atoms.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            ok = self._premises_satisfied(assign)
            obs = "OK: assignment satisfies premises" if ok else "OK: assignment does NOT satisfy premises"

        elif name == 'eval_premise':
            if 'idx' not in parsed or 'values' not in parsed:
                o, r, t, tr = proto_violation("need 'idx' and 'values'.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            try:
                k = int(parsed['idx'])
            except Exception:
                o, r, t, tr = proto_violation("parameter 'idx' must be integer.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            if k < 1 or k > len(self.premises):
                o, r, t, tr = proto_violation("idx out of range.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            assign = self._parse_values(parsed['values'])
            if assign is None:
                o, r, t, tr = proto_violation("invalid 'values' format or missing atoms.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            val = self._eval_formula(self.premises[k-1], assign)
            obs = f"OK: premise[{k}] under assignment = {int(val)}"

        elif name == 'eval_conclusion':
            values = parsed.get('values', None)
            if values is None:
                o, r, t, tr = proto_violation("missing parameter 'values'.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            assign = self._parse_values(values)
            if assign is None:
                o, r, t, tr = proto_violation("invalid 'values' format or missing atoms.")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            val = self._eval_formula(self.conclusion, assign)
            obs = f"OK: conclusion under assignment = {int(val)}"

        elif name == 'find_counterexample':
            ce = self._find_counterexample()
            if ce is None:
                obs = "OK: no counterexample exists (entailment holds)."
            else:
                parts = [f"{k}={'1' if v else '0'}" for k, v in ce.items()]
                obs = "OK: counterexample values=" + ",".join(parts)

        elif name == 'commit':
            ans = parsed.get('answer', '').strip().lower()
            if ans in ('yes', 'true', 'entailed'):
                claim = True
            elif ans in ('no', 'false', 'not_entailed', 'not-entailed'):
                claim = False
            else:
                o, r, t, tr = proto_violation("answer must be yes/no (or true/false, entailed/not_entailed).")
                return o, r, t, tr, {"suffix": self.get_task_suffix()}
            if claim == self.is_entailed:
                obs = f"SUCCESS: Correct. Entailment is {int(self.is_entailed)}. Conclusion: {self.conclusion_text}"
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"FAILED: Wrong decision. Entailment is {int(self.is_entailed)}. Conclusion: {self.conclusion_text}"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            o, r, t, tr = unsupported("unknown action name.")
            return o, r, t, tr, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs_timeout = f"TIMEOUT: Reached max turns ({self.max_turns})."
            return obs_timeout, 0.0, True, True, {"suffix": self.get_task_suffix()}

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
        tokens: Dict[str, Any] = {}
        tokens['action'] = parts[0]
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                tokens[key.strip()] = value.strip()
            else:
                # support flag-like tokens (ignored here)
                pass
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.4:
            return r'\boxed{list_atoms}'
        if random.random() < 0.5:
            k = random.randint(1, max(1, self.num_premises))
            return rf'\boxed{{show_premise idx={k}}}'
        # Build a random assignment string
        vals = ",".join(f"{v}={random.choice(['0','1'])}" for v in self.atoms)
        if random.random() < 0.5:
            return rf'\boxed{{check_assignment values={vals}}}'
        else:
            return rf'\boxed{{eval_conclusion values={vals}}}'


class SequentSentinelEnvWithFeedback(SequentSentinelEnv):
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
            error_detail["issue"] = "boxed_format_or_parse"
            hint = 'Use \\boxed{action key=value} and ensure spacing and parameters.'

        elif text.startswith("protocol violation"):
            error_type = "ProtocolViolation"
            if "missing parameter 'idx'" in text:
                error_detail["violation"] = "missing_idx"
                hint = "Provide idx=<k> for premise operations (1-indexed)."
            elif "parameter 'idx' must be integer" in text:
                error_detail["violation"] = "idx_not_int"
                hint = "Use a numeric idx like idx=1."
            elif "idx out of range" in text:
                error_detail["violation"] = "idx_range"
                hint = "Use an idx between 1 and number of premises."
            elif "missing parameter 'values'" in text:
                error_detail["violation"] = "missing_values"
                hint = "Provide values=A=0,B=1,... including all atoms."
            elif "invalid 'values' format" in text:
                error_detail["violation"] = "bad_values"
                hint = "Use comma-separated pairs for all atoms: A=0,B=1,... with 0/1."
            elif "answer must be yes/no" in text:
                error_detail["violation"] = "bad_answer"
                hint = "Use commit answer=yes or commit answer=no."
            else:
                error_detail["violation"] = "other"
                hint = "Check parameters and required format."

        elif text.startswith("unsupported action"):
            error_type = "UnsupportedAction"
            error_detail["action"] = "unknown"
            hint = "Use listed actions like list_atoms, show_premise, models_count, commit."

        elif text.startswith("failed: wrong decision"):
            error_type = "WrongDecision"
            # Extract expected from text
            expected = None
            if "entailment is" in text:
                try:
                    # ... is 1 or 0
                    idx = text.index("entailment is")
                    flag = text[idx:].split()[2].strip('.')
                    expected = "yes" if flag == '1' else "no"
                except Exception:
                    expected = None
            error_detail["expected"] = expected
            error_detail["got"] = "yes" if expected == "no" else "no"
            hint = "Try finding a counterexample or counting models; if a model satisfies premises but falsifies the conclusion, entailment is no."

        elif text.startswith("timeout"):
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Use show_premises or find_counterexample earlier to reach a decision within the turn limit."

        elif text.startswith("success"):
            error_type = "OK"
            error_detail["outcome"] = "success"

        else:
            error_type = "OK"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_atoms": getattr(self, "num_atoms", None),
                "num_premises": getattr(self, "num_premises", None),
                "entailed": int(self.is_entailed) if self.is_entailed is not None else None,
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
            "hint": "Start with list_atoms or show_premises, then try find_counterexample.",
            "turn": 0,
            "state": {
                "num_atoms": getattr(self, "num_atoms", None),
                "num_premises": getattr(self, "num_premises", None),
                "entailed": int(self.is_entailed) if self.is_entailed is not None else None,
            },
        }
        return obs, info