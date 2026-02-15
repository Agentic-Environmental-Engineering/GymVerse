from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class SequentSmithEnv(Env):
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
            'num_atoms': (2, 6),              # Number of propositional symbols: more atoms → larger search space
            'premise_count': (2, 6),          # More premises → more combinatorics/longer proofs
            'max_depth': (2, 5),              # Formula nesting depth: deeper → harder
            'allow_contradictions': (0, 1),   # Whether instances can include contradictory premises; enabling adds tricky cases
            'require_proof_probability': (80, 40),  # REVERSED: percent chance that the task asks for a proof vs countermodel; lower → more countermodels needed (harder for many LLMs)
            'domain_size': (0, 2),            # 0 means propositional only; 1-2 adds unary predicate with tiny finite domain (harder)
            'negation_density': (1, 3),       # More negations introduce complexity
        }
        # Variance settings
        self.param_variance = {
            'num_atoms': 1,
            'premise_count': 1,
            'max_depth': 0,
            'allow_contradictions': 0,
            'require_proof_probability': 5,
            'domain_size': 0,
            'negation_density': 1,
        }

        # Placeholder attributes assigned by _apply_complexity_params
        self.num_atoms = 0
        self.premise_count = 0
        self.max_depth = 0
        self.allow_contradictions = 0
        self.require_proof_probability = 0
        self.domain_size = 0
        self.negation_density = 0

        # Episode state
        self.turn_count = 0
        self.premises: List[str] = []
        self.conclusion: str = ""
        self.target_type: str = ""  # 'proof' or 'countermodel'
        self.assignment: Dict[str, bool] = {}  # for propositional
        self.domain_elems: List[str] = []      # for predicate domain
        self.predicate_truth: Dict[Tuple[str, str], bool] = {}  # (pred, elem) -> truth
        self.terminated = False

        # Tracking progress (optional, for shaped rewards)
        self.claimed_steps: List[str] = []
        self.proposed_model: Dict[str, str] = {}  # parsing countermodel proposal
        self.proven_lines: List[str] = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for pname, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            actual = center
            if self.enable_param_randomization:
                var = self.param_variance.get(pname, 0)
                if var > 0:
                    actual = center + random.uniform(-var, var)
            # clamp, support reversed
            low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual = max(low, min(high, actual))
            setattr(self, pname, int(round(actual)))

    def _atoms(self) -> List[str]:
        base = ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
        return base[:max(1, self.num_atoms)]

    def _rnd_atom(self) -> str:
        return random.choice(self._atoms())

    def _rnd_pred(self) -> str:
        return 'A'  # single unary predicate symbol A(x) to keep parsing simple

    def _rnd_var(self) -> str:
        return random.choice(['x', 'y', 'z'])

    def _rnd_const(self) -> str:
        # domain elements named a, b, c
        return random.choice(['a', 'b', 'c'][:max(1, self.domain_size)])

    def _rnd_connective(self) -> str:
        return random.choice(['∧', '∨', '→'])

    def _gen_prop_formula(self, depth: int) -> str:
        if depth <= 0 or random.random() < 0.3:
            f = self._rnd_atom()
        else:
            if random.random() < (0.3 + 0.1 * self.negation_density):
                f = f'¬({self._gen_prop_formula(depth-1)})'
            else:
                left = self._gen_prop_formula(depth - 1)
                right = self._gen_prop_formula(depth - 1)
                f = f'({left} {self._rnd_connective()} {right})'
        return f

    def _gen_pred_formula(self, depth: int) -> str:
        # Allow ∀x, ∃x with unary predicate A(x) and constants
        if depth <= 0 or random.random() < 0.25:
            if random.random() < 0.5:
                term = self._rnd_var()
            else:
                term = self._rnd_const()
            return f'{self._rnd_pred()}({term})'
        roll = random.random()
        if roll < 0.25 + 0.1 * self.negation_density:
            return f'¬({self._gen_pred_formula(depth-1)})'
        elif roll < 0.6:
            left = self._gen_pred_formula(depth - 1)
            right = self._gen_pred_formula(depth - 1)
            return f'({left} {self._rnd_connective()} {right})'
        else:
            quant = random.choice(['∀', '∃'])
            var = self._rnd_var()
            inner = self._gen_pred_formula(depth - 1)
            return f'{quant}{var}.({inner})'

    def _instantiate_predicate_truth(self):
        self.domain_elems = ['a', 'b', 'c'][:max(1, self.domain_size)]
        self.predicate_truth = {}
        for e in self.domain_elems:
            self.predicate_truth[(self._rnd_pred(), e)] = random.choice([True, False])

    def _sample_instance(self):
        use_pred = self.domain_size > 0
        gen = self._gen_pred_formula if use_pred else self._gen_prop_formula

        self.premises = []
        for _ in range(self.premise_count):
            self.premises.append(gen(self.max_depth))

        # Optionally introduce contradictions in propositional mode
        if not use_pred and self.allow_contradictions and len(self.premises) >= 2:
            # Make second premise the negation of first atom with decent probability
            if random.random() < 0.7:
                atom = self._rnd_atom()
                self.premises[0] = atom
                self.premises[1] = f'¬({atom})'

        self.conclusion = gen(self.max_depth)

        # Choose whether the task is proof (valid) or countermodel (invalid)
        # We'll bias sampling by require_proof_probability, then ensure feasibility by constructing evaluation
        want_proof = random.randint(1,100) <= self.require_proof_probability
        if use_pred:
            self._instantiate_predicate_truth()
            # For evaluation in predicate case, we restrict to ground evaluation and simple quantifiers
            # We cannot guarantee arbitrary validity; we will pick target based on actual truth under a sampled model
            # Build an assignment-like object for atoms unused; not used in predicate
        else:
            # Propositional: sample a truth assignment to atoms
            self.assignment = {a: random.choice([True, False]) for a in self._atoms()}

        # Determine semantic status and set target_type accordingly
        is_valid = self._evaluate_validity()

        # Adjust if desire conflicts and would make unsolvable: flip target while keeping same instance
        # If want_proof but instance invalid → target is countermodel; if want_countermodel but valid → target proof
        if want_proof:
            self.target_type = 'proof' if is_valid else 'countermodel'
        else:
            self.target_type = 'countermodel' if not is_valid else 'proof'

    def _eval_prop(self, formula: str, assign: Dict[str, bool]) -> bool:
        # Evaluate propositional formula with connectives ∧, ∨, →, negation ¬
        def parse_tokens(s: str) -> List[str]:
            s = s.replace('(', ' ( ').replace(')', ' ) ')
            s = s.replace('¬', ' ¬ ').replace('∧', ' ∧ ').replace('∨', ' ∨ ').replace('→', ' → ')
            toks = [t for t in s.split() if t]
            return toks

        def precedence(op: str) -> int:
            return {'¬': 3, '∧': 2, '∨': 1, '→': 0}.get(op, -1)

        def to_rpn(tokens: List[str]) -> List[str]:
            out, stack = [], []
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if re.fullmatch(r'[A-Z]', t):
                    out.append(t)
                elif t == '¬':
                    stack.append(t)
                elif t in ('∧', '∨', '→'):
                    while stack and stack[-1] != '(' and precedence(stack[-1]) >= precedence(t):
                        out.append(stack.pop())
                    stack.append(t)
                elif t == '(':
                    stack.append(t)
                elif t == ')':
                    while stack and stack[-1] != '(':
                        out.append(stack.pop())
                    if stack and stack[-1] == '(':
                        stack.pop()
                i += 1
            while stack:
                out.append(stack.pop())
            return out

        def eval_rpn(rpn: List[str]) -> bool:
            st: List[bool] = []
            for t in rpn:
                if re.fullmatch(r'[A-Z]', t):
                    st.append(assign.get(t, False))
                elif t == '¬':
                    a = st.pop()
                    st.append(not a)
                elif t in ('∧', '∨', '→'):
                    b = st.pop()
                    a = st.pop()
                    if t == '∧':
                        st.append(a and b)
                    elif t == '∨':
                        st.append(a or b)
                    else:
                        st.append((not a) or b)
            return st[-1] if st else False

        return eval_rpn(to_rpn(parse_tokens(formula)))

    def _eval_predicate(self, formula: str, env: Dict[str, str]) -> bool:
        # Very small evaluator for A(t), boolean connectives, and ∀x./∃x. over finite domain
        # env maps variables to concrete domain elements like 'a'
        def strip_outer(s: str) -> str:
            s = s.strip()
            if s.startswith('(') and s.endswith(')'):
                return s[1:-1]
            return s

        s = formula.strip()

        # Quantifiers
        m = re.match(r'^(∀|∃)([a-z])\.\((.*)\)$', s)
        if m:
            quant, var, inner = m.group(1), m.group(2), m.group(3)
            if quant == '∀':
                for d in self.domain_elems:
                    env2 = dict(env)
                    env2[var] = d
                    if not self._eval_predicate(inner, env2):
                        return False
                return True
            else:
                for d in self.domain_elems:
                    env2 = dict(env)
                    env2[var] = d
                    if self._eval_predicate(inner, env2):
                        return True
                return False

        # Negation
        if s.startswith('¬(') and s.endswith(')'):
            inner = s[2:-1]
            return not self._eval_predicate(inner, dict(env))

        # Binary connectives
        # Try split at top-level by →, ∨, ∧ (right-associative for →)
        def split_top(expr: str, op: str) -> Optional[Tuple[str, str]]:
            depth = 0
            for i in range(len(expr) - 1, -1, -1) if op == '→' else range(len(expr)):
                ch = expr[i]
                if ch == ')':
                    depth += 1
                elif ch == '(':
                    depth -= 1
                elif depth == 0 and expr[i:i+1] == op:
                    # op is single char for ∧, ∨; arrow is one char here
                    left = expr[:i].strip()
                    right = expr[i+1:].strip()
                    return left, right
            return None

        inner = strip_outer(s)
        for op in ['→', '∨', '∧']:
            res = split_top(inner, op)
            if res:
                left, right = res
                if op == '∧':
                    return self._eval_predicate(left, dict(env)) and self._eval_predicate(right, dict(env))
                if op == '∨':
                    return self._eval_predicate(left, dict(env)) or self._eval_predicate(right, dict(env))
                if op == '→':
                    return (not self._eval_predicate(left, dict(env))) or self._eval_predicate(right, dict(env))

        # Atomic predicate A(term)
        m2 = re.match(r'^([A-Z])\(([a-z])\)$', inner)
        if m2:
            pred, var = m2.group(1), m2.group(2)
            elem = env.get(var, None)
            if elem is None:
                # maybe var is actually a constant
                elem = var
            return self.predicate_truth.get((pred, elem), False)

        # Constant predicate A(a)
        m3 = re.match(r'^([A-Z])\(([a-z])\)$', inner)
        if m3:
            pred, const = m3.group(1), m3.group(2)
            return self.predicate_truth.get((pred, const), False)

        # Constant term variant like A(a) already handled; fallback:
        return False

    def _evaluate_validity(self) -> bool:
        # Semantic check: premises |= conclusion
        # For propositional: sample multiple randomized assignments and require that in all cases where premises are true, conclusion true
        use_pred = self.domain_size > 0
        if not use_pred:
            atoms = self._atoms()
            # Monte Carlo check across multiple assignments; include current assignment
            trials = 20
            valid = True
            for i in range(trials):
                if i == 0:
                    assign = dict(self.assignment)
                else:
                    assign = {a: random.choice([True, False]) for a in atoms}
                prem_true = all(self._eval_prop(p, assign) for p in self.premises)
                if prem_true:
                    concl_true = self._eval_prop(self.conclusion, assign)
                    if not concl_true:
                        valid = False
                        break
            return valid
        else:
            # For predicate, test under current sampled model and slight perturbations
            variants = [dict(self.predicate_truth)]
            # Perturb one element truth to see if counterexample arises
            for _ in range(3):
                v = dict(self.predicate_truth)
                if self.domain_elems:
                    e = random.choice(self.domain_elems)
                    key = (self._rnd_pred(), e)
                    v[key] = not v.get(key, False)
                variants.append(v)
            for varworld in variants:
                self_world = dict(self.predicate_truth)
                self.predicate_truth = varworld
                # Check: for all variable envs empty
                env = {}
                prem_true = all(self._eval_predicate(p, env) for p in self.premises)
                if prem_true and not self._eval_predicate(self.conclusion, env):
                    self.predicate_truth = self_world
                    return False
            self.predicate_truth = variants[0]
            return True

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are in SequentSmith: craft a proof or a countermodel for a sequent.")
        lines.append("Goal: Given premises Γ and conclusion C, either:")
        lines.append("- PROVE: Provide a valid derivation outline showing Γ ⊢ C.")
        lines.append("- COUNTERMODEL: Provide a truth assignment (and predicate interpretation if any) where all premises are true and C is false.")
        lines.append("")
        lines.append("Allowed actions:")
        lines.append("- claim line=...: Add a labeled proof step in your derivation (free-form but concise).")
        lines.append("- propose_model P=true Q=false ...: Propose propositional truth values.")
        lines.append("- set_pred A(a)=true A(b)=false ...: Set unary predicate truth on domain elements.")
        lines.append("- submit type=proof|countermodel: Submit your final answer for evaluation.")
        lines.append("")
        lines.append("Rules:")
        lines.append("- For proof: you may outline steps informally; we judge only the final submit by semantic validity.")
        lines.append("- For countermodel: you must provide a total assignment for all atoms; if predicates are used, also give all A(c) for each domain element shown.")
        lines.append("- You may call propose_model and set_pred multiple times; latest values overwrite previous ones.")
        lines.append("- Use only one submit; the episode ends on submit or timeout.")
        lines.append("")
        lines.append("Format actions inside \\boxed{...}. Examples:")
        lines.append(self.sample_random_action())
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        state_desc = []
        state_desc.append(f"Turn: {self.turn_count}/{self.max_turns}")
        state_desc.append(f"Target task: {self.target_type.upper()} the sequent.")
        state_desc.append("Premises:")
        for i, p in enumerate(self.premises, 1):
            state_desc.append(f"  {i}. {p}")
        state_desc.append(f"Conclusion: {self.conclusion}")
        if self.domain_size > 0:
            state_desc.append(f"Domain elements: {', '.join(self.domain_elems) if self.domain_elems else '(to be revealed at submit)'}")
        if self.claimed_steps:
            state_desc.append(f"Proof steps claimed: {len(self.claimed_steps)}")
        if self.proposed_model:
            state_desc.append(f"Model so far: {self.proposed_model}")
        state_desc.append("Enter your action in \\boxed{...} format.")
        return "\n".join(state_desc)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.claimed_steps = []
        self.proposed_model = {}
        self.proven_lines = []
        self.assignment = {}
        self.domain_elems = []
        self.predicate_truth = {}
        self.terminated = False

        self._sample_instance()
        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated:
            return "Episode already ended.", 0.0, True, False, {"suffix": self.get_task_suffix()}

        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get('action', '').lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if name == 'claim':
            line = parsed.get('line', None)
            if not line:
                obs = "Protocol violation: 'claim' requires line=... text."
                terminated = False
            else:
                self.claimed_steps.append(line)
                obs = f"Recorded proof line #{len(self.claimed_steps)}."
        elif name == 'propose_model':
            # parse atom assignments: P=true Q=false
            entries = {k:v for k,v in parsed.items() if k != 'action'}
            # accept booleans and store as strings; validation at submit
            for k, v in entries.items():
                self.proposed_model[k] = v
            obs = f"Model updated with {len(entries)} entries."
        elif name == 'set_pred':
            # set predicate truths A(a)=true etc.
            entries = {k:v for k,v in parsed.items() if k != 'action'}
            for k, v in entries.items():
                self.proposed_model[k] = v
            obs = f"Predicate interpretation updated with {len(entries)} entries."
        elif name == 'submit':
            typ = parsed.get('type', '').lower()
            if typ not in ('proof', 'countermodel'):
                obs = "Protocol violation: submit requires type=proof or type=countermodel."
            else:
                success, detail = self._evaluate_submission(typ)
                if success:
                    obs = f"Success: {typ} accepted. {detail}"
                    self.terminated = True
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"Failed: {typ} rejected. {detail}"
                    self.terminated = True
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "UNSUPPORTED ACTION: Use claim, propose_model, set_pred, or submit."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _evaluate_submission(self, typ: str) -> Tuple[bool, str]:
        use_pred = self.domain_size > 0

        if typ == 'proof':
            # Must be semantically valid: Γ |= C
            valid = self._evaluate_validity()
            if valid:
                return True, "Semantically valid entailment established."
            else:
                return False, "Premises do not semantically entail the conclusion."
        else:
            # countermodel: need total model making premises true and conclusion false
            # Build assignment for propositional atoms
            if not use_pred:
                atoms = self._atoms()
                assign: Dict[str, bool] = {}
                for a in atoms:
                    sval = self.proposed_model.get(a, None)
                    if sval is None:
                        return False, f"Missing assignment for atom {a}."
                    if sval.lower() not in ('true', 'false'):
                        return False, f"Invalid truth value for {a}: {sval}."
                    assign[a] = (sval.lower() == 'true')
                prem_true = all(self._eval_prop(p, assign) for p in self.premises)
                concl_false = not self._eval_prop(self.conclusion, assign)
                if prem_true and concl_false:
                    return True, f"Valid countermodel provided: {assign}."
                else:
                    return False, "Your model does not satisfy all premises true and conclusion false."
            else:
                # predicate case: need all A(e) specified
                # ensure domain revealed
                if not self.domain_elems:
                    self._instantiate_predicate_truth()
                truth: Dict[Tuple[str, str], bool] = {}
                for e in self.domain_elems:
                    key = f"{self._rnd_pred()}({e})"
                    sval = self.proposed_model.get(key, None)
                    if sval is None:
                        return False, f"Missing predicate truth for {key}."
                    if sval.lower() not in ('true', 'false'):
                        return False, f"Invalid truth for {key}: {sval}."
                    truth[(self._rnd_pred(), e)] = (sval.lower() == 'true')
                # Temporarily set, evaluate
                saved = dict(self.predicate_truth)
                self.predicate_truth = truth
                env = {}
                prem_true = all(self._eval_predicate(p, env) for p in self.premises)
                concl_false = not self._eval_predicate(self.conclusion, env)
                self.predicate_truth = saved
                if prem_true and concl_false:
                    return True, "Valid countermodel interpretation provided over the finite domain."
                else:
                    return False, "Interpretation fails: premises must be all true and conclusion false."

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
        name = parts[0].lower()
        tokens: Dict[str, Any] = {'action': name}
        # For claim, capture everything after 'line=' as a single value if present
        # For others, parse k=v pairs
        if name == 'claim':
            # allow "claim line=This is a step with spaces"
            mline = re.search(r'line=(.*)', inner, flags=re.DOTALL)
            if mline:
                tokens['line'] = mline.group(1).strip()
            return tokens
        # generic k=v parsing
        for p in parts[1:]:
            if '=' in p:
                k, v = p.split('=', 1)
                tokens[k] = v
            else:
                # ignore bare tokens
                pass
        return tokens

    def sample_random_action(self) -> str:
        choices = []
        choices.append(r"\boxed{claim line=Assume premises and aim to derive the conclusion.}")
        atoms = self._atoms()
        if atoms:
            ex = " ".join([f"{a}={'true' if random.random()<0.5 else 'false'}" for a in atoms])
            choices.append(rf"\boxed{{propose_model {ex}}}")
        if self.domain_size > 0:
            elems = ['a', 'b', 'c'][:self.domain_size]
            ex2 = " ".join([f"A({e})={'true' if random.random()<0.5 else 'false'}" for e in elems])
            choices.append(rf"\boxed{{set_pred {ex2}}}")
        choices.append(r"\boxed{submit type=proof}")
        choices.append(r"\boxed{submit type=countermodel}")
        return random.choice(choices)


class SequentSmithEnvWithFeedback(SequentSmithEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Wrap your action in \\boxed{...} and include a supported command."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["claim", "propose_model", "set_pred", "submit"]
            hint = "Use one of: claim, propose_model, set_pred, submit."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "line=" in text:
                error_detail["violation"] = "claim_missing_line"
                hint = "For claim, include line=... with your proof step."
            elif "submit requires" in text:
                error_detail["violation"] = "submit_missing_type"
                hint = "Use submit type=proof or submit type=countermodel."
            else:
                error_detail["violation"] = "unspecified"
                hint = "Check the command's required parameters."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan earlier: propose the model completely and then submit."
        elif "failed:" in text:
            error_type = "WrongDecision"
            if "missing assignment for atom" in text:
                m = re.search(r"missing assignment for atom ([A-Z])", text)
                miss = m.group(1) if m else None
                error_detail["missing_atom"] = miss
                hint = f"Provide all atoms with true/false using propose_model, e.g., \\boxed{{propose_model {miss}=true ...}}"
            elif "missing predicate truth" in text:
                m = re.search(r"missing predicate truth for ([a-z]\([a-z]\))", text)
                miss = m.group(1) if m else None
                error_detail["missing_predicate"] = miss
                hint = f"Set all domain predicate values with set_pred, e.g., \\boxed{{set_pred {miss}=true ...}}"
            else:
                hint = "For a countermodel, ensure all premises evaluate to true and the conclusion to false under your assignments."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["target"] = getattr(self, "target_type", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "If tasked with countermodel, first propose_model for all atoms. If predicates appear, also use set_pred. Submit when ready.",
            "turn": 0,
            "target": getattr(self, "target_type", None),
        }
        return obs, info