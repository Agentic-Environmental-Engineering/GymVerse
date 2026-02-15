from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class ClauseWardenEnv(Env):
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

        self.complexity_params = {
            # Number of propositional atoms in the knowledge base: more atoms = larger state space = harder
            'num_atoms': (4, 14),
            # REVERSED: number of initially known base facts: fewer base facts = less information = harder
            'num_base_facts': (3, 1),
            # Maximum antecedent size in rules: longer conjuncts require tracking more prerequisites = harder
            'max_antecedent': (1, 4),
            # Target derivation chain length: longer chain from facts to target = deeper reasoning = harder
            'target_chain_length': (1, 7),
            # Number of additional random rules (not part of the target chain): more structure = more reasoning = harder
            'num_extra_rules': (3, 24),
            # Number of distractor rules that look plausible but are inert (rarely fire): more confusion = harder
            'num_distractor_rules': (0, 12),
        }

        self.param_variance = {
            'num_atoms': 1,
            'num_base_facts': 0,
            'max_antecedent': 0,
            'target_chain_length': 1,
            'num_extra_rules': 3,
            'num_distractor_rules': 2,
        }

        self.num_atoms: int = 0
        self.num_base_facts: int = 0
        self.max_antecedent: int = 0
        self.target_chain_length: int = 0
        self.num_extra_rules: int = 0
        self.num_distractor_rules: int = 0

        self.turn_count: int = 0
        self.atoms: list = []
        self.rules: list = []
        self.base_facts: set = set()
        self.assumptions: set = set()
        self.derived_facts: set = set()
        self.has_run_chain: bool = False
        self.target_atom: str = ""
        self.ground_truth_entails: bool = False
        self._instance_description: str = ""

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
                    if min_val > max_val:
                        actual_value = max(max_val, min(min_val, actual_value))
                    else:
                        actual_value = max(min_val, min(max_val, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are the Clause Warden. A hidden Horn logic knowledge base defines implications among atoms.\n"
            "Your goal: determine whether the target atom is entailed by the base facts and rules.\n"
            "You may reveal structure, make temporary TRUE assumptions, run forward chaining, and finally submit YES or NO.\n"
            "\n"
            "Available actions:\n"
            "- reveal_atoms: list the atom names.\n"
            "- show_clauses: show all Horn rules as '(a & b & ...) -> c'.\n"
            "- assume atom=<name> value=TRUE: add a temporary TRUE assumption for use in chaining.\n"
            "- retract atom=<name>: remove a previous assumption.\n"
            "- forward_chain: compute derived facts from base facts and assumptions.\n"
            "- query atom=<name>: check if the atom is currently derived (requires forward_chain first).\n"
            "- show_workspace: display base facts, assumptions, and derived count.\n"
            "- submit answer=YES|NO reasoning=<optional_text>: final answer if target is entailed (YES) or not (NO).\n"
            "\n"
            "Rules:\n"
            "- Assumptions only add TRUE; there are no negations.\n"
            "- forward_chain uses base facts and your assumptions to compute current derived facts.\n"
            "- The final ground truth is evaluated only on base facts (assumptions are ignored for grading).\n"
            "- Actions must be in \\boxed{...}. Example: \\boxed{assume atom=a1 value=TRUE}\n"
            "- Submissions must be in \\boxed{submit answer=YES} or \\boxed{submit answer=NO} format.\n"
            "\n"
            "Example action:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        ws_status = "ready" if self.has_run_chain else "needs_forward_chain"
        return (
            f"Turn {self.turn_count}/{self.max_turns}\n"
            f"Target atom: {self.target_atom}\n"
            f"Known base facts: {sorted(self.base_facts)}\n"
            f"Assumptions: {sorted(self.assumptions)}\n"
            f"Derived facts count: {len(self.derived_facts)} (workspace {ws_status})\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.assumptions = set()
        self.derived_facts = set()
        self.has_run_chain = False

        self.atoms = [f"a{i+1}" for i in range(self.num_atoms)]

        if self.num_base_facts > len(self.atoms):
            self.num_base_facts = max(1, len(self.atoms) // 3)

        self.base_facts = set(random.sample(self.atoms, self.num_base_facts))

        chain_len = min(self.target_chain_length, max(1, len(self.atoms) - len(self.base_facts)))
        entails_target = random.choice([True, False])

        self.rules = []

        available_true = set(self.base_facts)
        chain_atoms = []

        if entails_target:
            current = random.choice(list(available_true))
            for step in range(chain_len):
                k = random.randint(1, max(1, self.max_antecedent))
                antecedents_pool = sorted(list(available_true))
                if current not in antecedents_pool:
                    antecedents_pool.append(current)
                k = min(k, max(1, len(set(antecedents_pool))))
                antecedents = set(random.sample(antecedents_pool, k))
                antecedents.add(current)
                antecedents = sorted(list(antecedents))

                unused_atoms = [x for x in self.atoms if x not in available_true and x not in chain_atoms]
                if not unused_atoms:
                    break
                consequent = random.choice(unused_atoms)
                self.rules.append((antecedents, consequent))
                chain_atoms.append(consequent)
                available_true.add(consequent)
                current = consequent

            if chain_atoms:
                self.target_atom = chain_atoms[-1]
            else:
                pool = [x for x in self.atoms if x not in self.base_facts]
                self.target_atom = random.choice(pool) if pool else random.choice(self.atoms)
        else:
            pool = [x for x in self.atoms if x not in self.base_facts]
            self.target_atom = random.choice(pool) if pool else random.choice(self.atoms)

        added = 0
        attempts = 0
        while added < self.num_extra_rules and attempts < self.num_extra_rules * 5:
            attempts += 1
            k = random.randint(1, max(1, self.max_antecedent))
            antecedents = sorted(random.sample(self.atoms, k))
            consequent_pool = [x for x in self.atoms]
            if not entails_target:
                consequent_pool = [x for x in consequent_pool if x != self.target_atom]
                if not consequent_pool:
                    consequent_pool = [x for x in self.atoms]
            consequent = random.choice(consequent_pool)
            rule = (antecedents, consequent)
            if rule not in self.rules:
                self.rules.append(rule)
                added += 1

        inert_added = 0
        attempts = 0
        while inert_added < self.num_distractor_rules and attempts < self.num_distractor_rules * 5:
            attempts += 1
            k = random.randint(1, max(1, self.max_antecedent))
            antecedents = sorted(random.sample(self.atoms, k))
            if random.random() < 0.7:
                antecedents = sorted(list(set(antecedents + [random.choice(self.atoms)])))
            consequent_pool = [x for x in self.atoms if x not in self.base_facts]
            if not consequent_pool:
                consequent_pool = [x for x in self.atoms]
            consequent = random.choice(consequent_pool)
            rule = (antecedents, consequent)
            if rule not in self.rules:
                self.rules.append(rule)
                inert_added += 1

        def closure(facts: set) -> set:
            derived = set(facts)
            changed = True
            while changed:
                changed = False
                for ants, cons in self.rules:
                    if cons not in derived and set(ants).issubset(derived):
                        derived.add(cons)
                        changed = True
            return derived

        gt_derived = closure(self.base_facts)
        self.ground_truth_entails = self.target_atom in gt_derived

        self._instance_description = (
            f"Instance: {len(self.atoms)} atoms, {len(self.rules)} rules, {len(self.base_facts)} base facts.\n"
            f"Target: {self.target_atom} (ground truth entailment: {'YES' if self.ground_truth_entails else 'NO'} hidden)"
        )

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get('action', '').strip()

        if name == 'reveal_atoms':
            obs = f"Atoms: {', '.join(sorted(self.atoms))}"
            reward = 0.0

        elif name == 'show_clauses':
            if not self.rules:
                obs = "Rules: none"
            else:
                clauses = [f"({' & '.join(ants)}) -> {cons}" for ants, cons in self.rules]
                obs = "Rules:\n" + "\n".join(clauses)
            reward = 0.0

        elif name == 'assume':
            atom = parsed.get('atom', '')
            val = parsed.get('value', '').upper()
            if atom not in self.atoms:
                obs = f"PROTOCOL VIOLATION: unknown_atom '{atom}'."
                reward = 0.0
            elif val != 'TRUE':
                obs = "PROTOCOL VIOLATION: assumption_value_invalid (only TRUE is allowed)."
                reward = 0.0
            else:
                self.assumptions.add(atom)
                obs = f"Assumed TRUE: {atom}"
                reward = 0.0

        elif name == 'retract':
            atom = parsed.get('atom', '')
            if atom not in self.atoms:
                obs = f"PROTOCOL VIOLATION: unknown_atom '{atom}'."
                reward = 0.0
            else:
                if atom in self.assumptions:
                    self.assumptions.remove(atom)
                    obs = f"Retracted assumption: {atom}"
                else:
                    obs = f"No assumption to retract for: {atom}"
                reward = 0.0

        elif name == 'forward_chain':
            def closure(facts: set) -> set:
                derived = set(facts)
                changed = True
                while changed:
                    changed = False
                    for ants, cons in self.rules:
                        if cons not in derived and set(ants).issubset(derived):
                            derived.add(cons)
                            changed = True
                return derived

            before = set(self.derived_facts)
            current_base = set(self.base_facts) | set(self.assumptions)
            self.derived_facts = closure(current_base)
            self.has_run_chain = True
            new = len(self.derived_facts - before)
            obs = f"Forward chaining completed. Derived facts count: {len(self.derived_facts)} (+{new})."
            reward = 0.0

        elif name == 'query':
            atom = parsed.get('atom', '')
            if atom not in self.atoms:
                obs = f"PROTOCOL VIOLATION: unknown_atom '{atom}'."
                reward = 0.0
            elif not self.has_run_chain:
                obs = "PROTOCOL VIOLATION: run forward_chain before query."
                reward = 0.0
            else:
                status = "DERIVED" if atom in self.derived_facts else "NOT DERIVED"
                obs = f"Query: {atom} is {status} under current workspace."
                reward = 0.0

        elif name == 'show_workspace':
            obs = (
                f"Workspace\n"
                f"- Base facts: {sorted(self.base_facts)}\n"
                f"- Assumptions: {sorted(self.assumptions)}\n"
                f"- Derived count: {len(self.derived_facts)}"
            )
            reward = 0.0

        elif name == 'submit':
            ans = parsed.get('answer', '').upper()
            if ans not in ('YES', 'NO'):
                obs = "PROTOCOL VIOLATION: submission_format_invalid (answer must be YES or NO)."
                reward = 0.0
            else:
                correct = (ans == 'YES' and self.ground_truth_entails) or (ans == 'NO' and not self.ground_truth_entails)
                if correct:
                    obs = f"Success! The target atom '{self.target_atom}' entailment is {ans}."
                    return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"Failed! The correct entailment for '{self.target_atom}' is {'YES' if self.ground_truth_entails else 'NO'}."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"UNSUPPORTED_ACTION: '{name}'. Episode terminated."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"{obs}\n{self._instance_description}"
        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

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
                tokens[key] = value
        return tokens

    def sample_random_action(self) -> str:
        options = [
            r'\boxed{reveal_atoms}',
            r'\boxed{show_clauses}',
            r'\boxed{assume atom=a1 value=TRUE}',
            r'\boxed{forward_chain}',
            r'\boxed{query atom=a2}',
            r'\boxed{show_workspace}',
            r'\boxed{submit answer=YES}',
        ]
        return random.choice(options)


class ClauseWardenEnvWithFeedback(ClauseWardenEnv):
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
            if self.feedback_level >= 2:
                hint = "Use \\boxed{...} and include a valid action name and parameters."

        elif "unsupported_action" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported_action:\s*'([^']+)'", obs, flags=re.IGNORECASE)
            if m:
                error_detail["action"] = m.group(1)
            if self.feedback_level >= 2:
                hint = "Choose one of: reveal_atoms, show_clauses, assume, retract, forward_chain, query, show_workspace, submit."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "unknown_atom" in text:
                error_detail["violation"] = "unknown_atom"
                if self.feedback_level >= 2:
                    hint = "List atoms first with reveal_atoms, then reference a valid atom name."
            elif "assumption_value_invalid" in text:
                error_detail["violation"] = "assumption_value_invalid"
                if self.feedback_level >= 2:
                    hint = "Only TRUE is permitted for assumptions: use \\boxed{assume atom=<name> value=TRUE}."
            elif "submission_format_invalid" in text:
                error_detail["violation"] = "submission_format_invalid"
                if self.feedback_level >= 2:
                    hint = "Submit with \\boxed{submit answer=YES} or \\boxed{submit answer=NO}."
            elif "run forward_chain before query" in text:
                error_detail["violation"] = "query_before_chain"
                if self.feedback_level >= 2:
                    hint = "Call \\boxed{forward_chain} before using \\boxed{query atom=<name>}."

        elif "failed!" in text:
            error_type = "WrongDecision"
            m = re.search(r"correct entailment.*is\s(yes|no)", text)
            if m:
                error_detail["expected"] = m.group(1).upper()
            m2 = re.search(r"submit answer=(YES|NO)", action)
            if m2:
                error_detail["got"] = m2.group(1)
            if self.feedback_level >= 2:
                hint = "Reveal rules and base facts, run forward_chain, then query the target atom to verify before submitting."

        elif "reached max turns" in text:
            error_type = "Timeout"
            if self.feedback_level >= 2:
                hint = "Act decisively. A typical flow: reveal_atoms → show_clauses → forward_chain → query atom=<target> → submit."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "target": getattr(self, "target_atom", None),
                "has_run_chain": getattr(self, "has_run_chain", False),
                "assumptions": sorted(getattr(self, "assumptions", [])),
                "derived_count": len(getattr(self, "derived_facts", set())),
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
            "hint": "Start by reveal_atoms and show_clauses, then forward_chain.",
            "turn": 0,
            "state": {
                "target": getattr(self, "target_atom", None),
                "has_run_chain": False,
                "assumptions": [],
                "derived_count": 0,
            },
        }
        return obs, info