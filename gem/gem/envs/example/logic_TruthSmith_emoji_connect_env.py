from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class TruthSmithEnv(Env):
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

        # Evolvable parameters
        self.complexity_params = {
            # Number of distinct propositional variables: more vars → larger truth space → harder
            'num_vars': (2, 6),
            # Maximum formula depth (nesting of connectives): deeper trees → harder parsing/evaluation
            'max_depth': (2, 6),
            # Number of connectives allowed in the palette: more operators → more complex semantics
            'connective_variety': (3, 6),
            # REVERSED: number of hints allowed (each hint reveals truth under a specific assignment)
            'hint_budget': (3, 0),
        }
        self.param_variance = {
            'num_vars': 1,              # medium discrete range
            'max_depth': 1,             # medium discrete range
            'connective_variety': 1,    # medium discrete range
            'hint_budget': 1,           # medium discrete range
        }

        # Placeholders
        self.num_vars: int = 0
        self.max_depth: int = 0
        self.connective_variety: int = 0
        self.hint_budget: int = 0

        # Domain state
        self.turn_count: int = 0
        self.vars: List[str] = []
        self.target_property: str = ""  # one of {'TAUTOLOGY','CONTRADICTION','SATISFIABLE'}
        self.formula_tree = None
        self.formula_str: str = ""
        self.truth_table_complete: bool = False
        self.success_submitted: bool = False
        self.used_hints: int = 0

        # latent ground-truth
        self.is_tautology: bool = False
        self.is_contradiction: bool = False
        self.is_satisfiable: bool = False
        self.counterexample: Optional[Dict[str, bool]] = None  # for tautology: assignment making False
        self.witness: Optional[Dict[str, bool]] = None         # for satisfiable: assignment making True
        self.contra_witness: Optional[Dict[str, bool]] = None  # for contradiction: proof via none; optional

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
                    low = min(min_val, max_val)
                    high = max(min_val, max_val)
                    if min_val > max_val:
                        # reversed, but clamp still via low/high
                        pass
                    actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _random_connectives(self) -> List[str]:
        palette_all = ['NOT', 'AND', 'OR', 'IMPLIES', 'IFF', 'XOR']
        k = max(1, min(len(palette_all), self.connective_variety))
        return palette_all[:k]

    def _gen_formula(self):
        vars_list = [chr(ord('p') + i) for i in range(self.num_vars)]
        self.vars = vars_list

        connectives = self._random_connectives()

        def gen_atom():
            v = random.choice(vars_list)
            return ('VAR', v)

        def gen_unary(child):
            return ('NOT', child)

        def gen_binary(op, left, right):
            return (op, left, right)

        def random_op():
            candidates = [c for c in connectives if c != 'NOT']
            if not candidates:
                candidates = ['AND', 'OR']
            return random.choice(candidates)

        def build(depth):
            if depth <= 0:
                node = gen_atom()
                if 'NOT' in connectives and random.random() < 0.3:
                    node = gen_unary(node)
                return node
            choice = random.random()
            if 'NOT' in connectives and choice < 0.2:
                return gen_unary(build(depth - 1))
            else:
                op = random_op()
                left = build(depth - 1 if random.random() < 0.8 else max(0, depth - 2))
                right = build(depth - 1 if random.random() < 0.8 else max(0, depth - 2))
                return gen_binary(op, left, right)

        depth = random.randint(max(1, self.max_depth - 1), self.max_depth)
        tree = build(depth)

        def to_str(node):
            t = node[0]
            if t == 'VAR':
                return node[1]
            if t == 'NOT':
                return f"¬({to_str(node[1])})"
            if t in ('AND', 'OR', 'IMPLIES', 'IFF', 'XOR'):
                a = to_str(node[1])
                b = to_str(node[2])
                sym = {'AND': '∧', 'OR': '∨', 'IMPLIES': '→', 'IFF': '↔', 'XOR': '⊕'}[t]
                return f"({a} {sym} {b})"
            return "?"
        self.formula_tree = tree
        self.formula_str = to_str(tree)

    def _eval(self, node, assign: Dict[str, bool]) -> bool:
        t = node[0]
        if t == 'VAR':
            return bool(assign[node[1]])
        if t == 'NOT':
            return not self._eval(node[1], assign)
        if t == 'AND':
            return self._eval(node[1], assign) and self._eval(node[2], assign)
        if t == 'OR':
            return self._eval(node[1], assign) or self._eval(node[2], assign)
        if t == 'IMPLIES':
            a = self._eval(node[1], assign)
            b = self._eval(node[2], assign)
            return (not a) or b
        if t == 'IFF':
            a = self._eval(node[1], assign)
            b = self._eval(node[2], assign)
            return a == b
        if t == 'XOR':
            a = self._eval(node[1], assign)
            b = self._eval(node[2], assign)
            return (a and not b) or (b and not a)
        return False

    def _analyze_truth(self):
        all_assignments = []
        n = len(self.vars)
        for mask in range(1 << n):
            assign = {}
            for i, var in enumerate(self.vars):
                assign[var] = bool((mask >> i) & 1)
            all_assignments.append(assign)

        truths = []
        for a in all_assignments:
            truths.append(self._eval(self.formula_tree, a))

        self.is_tautology = all(truths)
        self.is_contradiction = not any(truths)
        self.is_satisfiable = any(truths)

        self.counterexample = None
        self.witness = None
        if not self.is_tautology:
            for a, val in zip(all_assignments, truths):
                if not val:
                    self.counterexample = dict(a)
                    break
        if self.is_satisfiable:
            for a, val in zip(all_assignments, truths):
                if val:
                    self.witness = dict(a)
                    break
        self.contra_witness = None  # not needed explicitly

        # pick a target property consistent with complexity to ensure feasibility:
        # ensure tasks remain solvable and balanced; rotate among properties
        choices = ['TAUTOLOGY', 'CONTRADICTION', 'SATISFIABLE']
        # prefer properties that are non-trivial based on instance
        feasible = []
        if self.is_tautology:
            feasible.append('TAUTOLOGY')
            feasible.append('SATISFIABLE')
        if self.is_contradiction:
            feasible.append('CONTRADICTION')
        if self.is_satisfiable and not self.is_tautology:
            feasible.append('SATISFIABLE')
        if not feasible:
            feasible = choices
        self.target_property = random.choice(feasible)

    def _hint_text(self) -> str:
        if self.used_hints >= self.hint_budget:
            return "No hints remaining."
        # reveal evaluation for a random assignment not previously revealed
        assign = {v: bool(random.getrandbits(1)) for v in self.vars}
        val = self._eval(self.formula_tree, assign)
        self.used_hints += 1
        asg = ", ".join(f"{k}={'T' if assign[k] else 'F'}" for k in self.vars)
        return f"Hint {self.used_hints}/{self.hint_budget}: Under [{asg}], formula evaluates to {'T' if val else 'F'}."

    def _get_instructions(self) -> str:
        palette_desc = "Available logical operators: ¬ (NOT), ∧ (AND), ∨ (OR), → (IMPLIES), ↔ (IFF), ⊕ (XOR)."
        return (
            "You are the TruthSmith.\n"
            "Goal: verify the stated property of the given propositional formula using truth-functional reasoning.\n"
            f"{palette_desc}\n"
            "You can:\n"
            "- propose an assignment as evidence with: \\boxed{assign p=T q=F ...}\n"
            "  Use T/F. This is useful to show satisfiable (witness T) or refute tautology (make it F).\n"
            "- request a hint (limited) with: \\boxed{hint}\n"
            "- submit your final decision with: \\boxed{final label=TAUTOLOGY|CONTRADICTION|SATISFIABLE}\n"
            "Scoring: Only the final submission matters. Correct = 1.0, otherwise 0.0. Format errors terminate with format-error reward.\n"
            "You may take multiple turns to reason before submitting final.\n"
        )

    def get_task_suffix(self) -> str:
        hint_info = f"Hints used: {self.used_hints}/{self.hint_budget}"
        vars_info = f"Variables: {', '.join(self.vars)}"
        return (
            f"Task: Determine if the formula satisfies the target property.\n"
            f"Target property: {self.target_property}\n"
            f"Formula: {self.formula_str}\n"
            f"{vars_info}\n"
            f"{hint_info}\n"
            "Enter action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.used_hints = 0
        self.success_submitted = False
        self.truth_table_complete = False

        # regenerate instance until non-trivial and feasible
        attempts = 0
        while True:
            attempts += 1
            self._gen_formula()
            self._analyze_truth()
            # ensure solvability and avoid degenerate: if connective variety too low, regenerate a few times
            if attempts > 5 or True:
                break

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _format_assign(self, d: Dict[str, bool]) -> str:
        return ", ".join(f"{k}={'T' if d[k] else 'F'}" for k in self.vars)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} enclosing a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get('action', '').lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        if name == 'hint':
            if self.used_hints >= self.hint_budget:
                obs = "No hints remaining."
            else:
                obs = self._hint_text()
        elif name == 'assign':
            assign: Dict[str, bool] = {}
            # parse values for all known vars if provided
            provided_keys = [k for k in parsed.keys() if k in self.vars]
            if not provided_keys:
                obs = "Protocol violation: No variables provided in assignment."
            else:
                ok = True
                for v in self.vars:
                    if v in parsed:
                        val = parsed[v].strip().upper()
                        if val not in ('T', 'F'):
                            ok = False
                            break
                        assign[v] = (val == 'T')
                    else:
                        # if not all provided, fill randomly to make evaluation meaningful
                        assign[v] = bool(random.getrandbits(1))
                if not ok:
                    obs = "Protocol violation: Assignment values must be T or F."
                else:
                    val = self._eval(self.formula_tree, assign)
                    role_text = []
                    if self.target_property == 'TAUTOLOGY' and val is False:
                        role_text.append("This assignment refutes TAUTOLOGY (formula = F).")
                    if self.target_property == 'SATISFIABLE' and val is True:
                        role_text.append("This assignment witnesses SATISFIABLE (formula = T).")
                    if self.target_property == 'CONTRADICTION' and val is True:
                        role_text.append("This assignment shows non-contradiction (formula = T).")
                    if not role_text:
                        role_text.append("Assignment evaluated.")
                    obs = f"Assignment [{self._format_assign(assign)}] -> value={ 'T' if val else 'F' }. " + " ".join(role_text)
        elif name == 'final':
            label = parsed.get('label', '').upper()
            if label not in ('TAUTOLOGY', 'CONTRADICTION', 'SATISFIABLE'):
                obs = "UnsupportedAction: label must be one of TAUTOLOGY, CONTRADICTION, SATISFIABLE."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            correct_label = 'TAUTOLOGY' if self.is_tautology else ('CONTRADICTION' if self.is_contradiction else 'SATISFIABLE')
            if label == correct_label:
                obs = f"Success! Correct: {label}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Your label {label} is incorrect."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "UnsupportedAction: Use 'assign', 'hint', or 'final'."
            terminated = False

        if self.turn_count >= self.max_turns:
            return f"Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

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
                k, v = part.split('=', 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.4:
            # random assignment for available vars
            picks = {}
            for v in self.vars:
                if random.random() < 0.6:
                    picks[v] = 'T' if random.random() < 0.5 else 'F'
            if not picks:
                v = random.choice(self.vars)
                picks[v] = 'T'
            parts = " ".join([f"{k}={picks[k]}" for k in picks])
            return rf"\boxed{{assign {parts}}}"
        elif random.random() < 0.7:
            return r"\boxed{hint}"
        else:
            label = random.choice(['TAUTOLOGY', 'CONTRADICTION', 'SATISFIABLE'])
            return rf"\boxed{{final label={label}}}"


class TruthSmithEnvWithFeedback(TruthSmithEnv):
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
            error_detail["issue"] = "missing_boxed_or_parse_failed"
            hint = "Wrap your command in \\boxed{...} and use one of: assign, hint, final."
        elif "unsupportedaction" in text:
            error_type = "UnsupportedAction"
            if "label must be one of" in text:
                error_detail["issue"] = "invalid_label"
                hint = "Use final with label=TAUTOLOGY|CONTRADICTION|SATISFIABLE."
            else:
                error_detail["issue"] = "unknown_action"
                hint = "Allowed actions: assign, hint, final."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "no variables provided" in text:
                error_detail["violation"] = "empty_assignment"
                hint = "Include at least one variable like assign p=T."
            elif "must be t or f" in text:
                error_detail["violation"] = "invalid_truth_symbol"
                hint = "Use T or F only, e.g., q=F."
        elif "failed!" in text and "incorrect" in text:
            error_type = "WrongDecision"
            correct_label = 'TAUTOLOGY' if getattr(self, 'is_tautology', False) else ('CONTRADICTION' if getattr(self, 'is_contradiction', False) else 'SATISFIABLE')
            error_detail["expected"] = correct_label
            # Try to extract user's answer
            m = re.search(r"your label ([a-z]+)", text)
            if m:
                error_detail["got"] = m.group(1).upper()
            else:
                error_detail["got"] = "UNKNOWN"
            if self.feedback_level >= 2:
                if correct_label == 'TAUTOLOGY' and self.counterexample is not None:
                    hint = "Try to find an assignment that makes the formula false to refute TAUTOLOGY."
                elif correct_label == 'CONTRADICTION':
                    hint = "Check if any assignment can make the formula true; contradictions are never true."
                else:
                    hint = "Search for an assignment that makes the formula true to witness satisfiability."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Submit a final decision earlier next time."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
        else:
            # Ongoing, normal step
            error_type = "OK"
            error_detail["outcome"] = "progress"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "target_property": getattr(self, "target_property", None),
                "hints_used": getattr(self, "used_hints", None),
                "hint_budget": getattr(self, "hint_budget", None),
                "num_vars": getattr(self, "num_vars", None),
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
            "hint": "Start by requesting a hint or testing an assignment with assign p=T q=F ...",
            "turn": 0,
            "state": {
                "target_property": getattr(self, "target_property", None),
                "hints_used": 0,
                "hint_budget": getattr(self, "hint_budget", None),
                "num_vars": getattr(self, "num_vars", None),
            },
        }
        return obs, info