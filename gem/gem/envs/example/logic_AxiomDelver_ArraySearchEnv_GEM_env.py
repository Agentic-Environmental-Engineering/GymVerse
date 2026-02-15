from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AxiomDelverEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 28,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 28

        # Evolvable parameters
        self.complexity_params = {
            # Number of propositional variables: larger set -> more combinatorial structure -> harder
            'num_vars': (3, 14),
            # Number of clauses in CNF: more clauses to evaluate -> harder
            'num_clauses': (4, 24),
            # Minimum clause width: larger minimum width can make structure richer but keep range tight
            'clause_width_min': (1, 2),
            # Maximum clause width: broader clauses -> more literals per clause -> harder to reason
            'clause_width_max': (2, 5),
            # Percentage chance a literal is negated when generating general clauses (0-100): mid-high makes reasoning less trivial
            'negation_rate': (30, 70),
            # REVERSED: budget for variable truth queries; fewer available -> harder
            'max_value_queries': (4, 0),
            # REVERSED: probe detail mode: 2=rich details, 1=limited counts, 0=only T/F; less detail -> harder
            'probe_detail_mode': (2, 0),
            # REVERSED: minimum number of violated clauses when the assignment does NOT satisfy the CNF;
            # fewer violations -> harder (harder to find a counterexample)
            'min_violations_if_unsatisfied': (3, 1),
        }

        # Variance for randomization within complexity level
        self.param_variance = {
            'num_vars': 1,                     # medium discrete range
            'num_clauses': 2,                  # medium-large discrete range
            'clause_width_min': 0,             # tiny range, fix at interpolated value
            'clause_width_max': 1,             # small range, limited jitter
            'negation_rate': 5,                # percent jitter
            'max_value_queries': 1,            # small integer jitter
            'probe_detail_mode': 0,            # keep stable per level
            'min_violations_if_unsatisfied': 0 # keep stable per level
        }

        # Placeholder attributes
        self.num_vars: int = 0
        self.num_clauses: int = 0
        self.clause_width_min: int = 0
        self.clause_width_max: int = 0
        self.negation_rate: int = 0
        self.max_value_queries: int = 0
        self.probe_detail_mode: int = 0
        self.min_violations_if_unsatisfied: int = 0

        # State
        self.turn_count: int = 0
        self.remaining_value_queries: int = 0
        self.assignment: Dict[int, bool] = {}
        self.clauses: List[List[tuple]] = []
        self.is_model: bool = False
        self._clause_length_counts: Dict[int, int] = {}
        self._pos_literal_count: int = 0
        self._neg_literal_count: int = 0

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
            # Clamp to range regardless of reversed or normal ranges
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are Axiom Delver. A hidden truth assignment over propositional variables x1..xN is fixed. "
            "A CNF formula (AND of OR-clauses) is generated. Your task is to determine whether the hidden assignment "
            "satisfies all clauses (i.e., the CNF evaluates to True under the assignment).\n\n"
            "Actions (use integers where required; indices start at 1):\n"
            "- summary: Get high-level info (variables, clauses, clause width range, remaining variable queries).\n"
            "- profile: Get distribution info (clause length histogram, counts of positive/negative literals).\n"
            "- view clause=k: Reveal the structure of clause k (lists literals with variable indices and negations).\n"
            "- probe clause=k: Evaluate clause k under the hidden assignment. Detail depends on probe_detail_mode.\n"
            "- value var=j: Reveal the truth value of variable xj (limited budget).\n"
            "- report verdict=true|false: Submit your final decision. true means 'assignment satisfies CNF'.\n"
            "- help: Show these instructions again.\n\n"
            "Formatting:\n"
            "- All actions must be within \\boxed{...}\n"
            "- With parameters: \\boxed{action key=value}\n"
            "- Without parameters: \\boxed{action}\n\n"
            "Example action:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        turns_left = self.max_turns - self.turn_count
        return (
            f"State: vars={self.num_vars}, clauses={self.num_clauses}, "
            f"width_range={self.clause_width_min}-{self.clause_width_max}, "
            f"probe_detail_mode={self.probe_detail_mode}, "
            f"remaining_value_queries={self.remaining_value_queries}, "
            f"turn={self.turn_count}, turns_left={turns_left}.\n"
            "Enter your action in \\boxed{...} format. Actions: summary | profile | "
            "view clause=k | probe clause=k | value var=j | report verdict=true|false | help"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.remaining_value_queries = int(self.max_value_queries)
        self.assignment = {i: bool(random.getrandbits(1)) for i in range(1, self.num_vars + 1)}
        self._generate_formula()
        obs = self._get_instructions()
        return obs, {"suffix": self.get_task_suffix()}

    def _generate_formula(self):
        # Decide satisfiability under the hidden assignment
        self.is_model = bool(random.getrandbits(1))

        self.clauses = []
        self._clause_length_counts = {}
        self._pos_literal_count = 0
        self._neg_literal_count = 0

        # Helper to build a clause guaranteed satisfied
        def make_satisfied_clause():
            length = random.randint(self.clause_width_min, self.clause_width_max)
            vars_pool = list(range(1, self.num_vars + 1))
            random.shuffle(vars_pool)
            chosen_vars = vars_pool[:max(1, min(length, self.num_vars))]
            # Random signs
            clause = []
            for v in chosen_vars:
                if random.randint(1, 100) <= self.negation_rate:
                    clause.append((v, True))
                else:
                    clause.append((v, False))
            # Ensure at least one literal is True under assignment
            witness_idx = random.randrange(len(clause))
            v, neg = clause[witness_idx]
            desired_neg = (not self.assignment[v])  # True if var False, False if var True
            clause[witness_idx] = (v, desired_neg)
            return clause

        # Helper to build a clause guaranteed violated
        def make_violated_clause():
            length = random.randint(self.clause_width_min, self.clause_width_max)
            vars_pool = list(range(1, self.num_vars + 1))
            random.shuffle(vars_pool)
            chosen_vars = vars_pool[:max(1, min(length, self.num_vars))]
            # All literals evaluate False under assignment: neg == assignment[var]
            clause = []
            for v in chosen_vars:
                neg = self.assignment[v]
                clause.append((v, neg))
            return clause

        if self.is_model:
            # All clauses must be satisfied
            for _ in range(self.num_clauses):
                c = make_satisfied_clause()
                self.clauses.append(c)
        else:
            # Ensure at least min_violations_if_unsatisfied violated clauses
            target_violations = max(1, int(self.min_violations_if_unsatisfied))
            remaining = self.num_clauses
            # Place required violated clauses
            for _ in range(min(target_violations, remaining)):
                c = make_violated_clause()
                self.clauses.append(c)
                remaining -= 1
            # Fill the rest with random, could be satisfied or not
            for _ in range(remaining):
                # Mix: bias toward satisfied to avoid trivial detection
                if random.random() < 0.6:
                    c = make_satisfied_clause()
                else:
                    c = make_violated_clause()
                self.clauses.append(c)
            # Shuffle clauses so violated ones aren't clustered
            random.shuffle(self.clauses)
            # If by chance all clauses became satisfied, fix by forcing one violated
            if all(self._eval_clause(cl) for cl in self.clauses):
                idx_to_flip = random.randrange(len(self.clauses))
                self.clauses[idx_to_flip] = make_violated_clause()

        # Compute profiles
        for cl in self.clauses:
            L = len(cl)
            self._clause_length_counts[L] = self._clause_length_counts.get(L, 0) + 1
            for (_, neg) in cl:
                if neg:
                    self._neg_literal_count += 1
                else:
                    self._pos_literal_count += 1

    def _eval_clause(self, clause: List[tuple]) -> bool:
        for (v, neg) in clause:
            val = self.assignment[v]
            lit_val = (not val) if neg else val
            if lit_val:
                return True
        return False

    def _format_clause(self, clause: List[tuple]) -> str:
        parts = []
        for (v, neg) in clause:
            if neg:
                parts.append(f"¬x{v}")
            else:
                parts.append(f"x{v}")
        return " ∨ ".join(parts)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
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

        if name == 'help':
            obs = self._get_instructions()

        elif name == 'summary':
            obs = (
                f"SUMMARY: variables={self.num_vars}, clauses={self.num_clauses}, "
                f"width_range={self.clause_width_min}-{self.clause_width_max}, "
                f"remaining_value_queries={self.remaining_value_queries}, turn={self.turn_count}"
            )

        elif name == 'profile':
            obs = (
                f"PROFILE: clause_length_counts={self._clause_length_counts}, "
                f"positive_literals={self._pos_literal_count}, negative_literals={self._neg_literal_count}"
            )

        elif name == 'view':
            if 'clause' not in parsed:
                obs = "MISSING PARAMETER: 'clause'"
            else:
                try:
                    idx = int(parsed['clause'])
                except ValueError:
                    idx = -1
                if idx < 1 or idx > self.num_clauses:
                    obs = f"INVALID PARAMETER: clause index out of range (1..{self.num_clauses})"
                else:
                    clause = self.clauses[idx - 1]
                    obs = f"CLAUSE {idx} STRUCTURE: ({self._format_clause(clause)})"

        elif name == 'probe':
            if 'clause' not in parsed:
                obs = "MISSING PARAMETER: 'clause'"
            else:
                try:
                    idx = int(parsed['clause'])
                except ValueError:
                    idx = -1
                if idx < 1 or idx > self.num_clauses:
                    obs = f"INVALID PARAMETER: clause index out of range (1..{self.num_clauses})"
                else:
                    clause = self.clauses[idx - 1]
                    val = self._eval_clause(clause)
                    if self.probe_detail_mode >= 2:
                        satisfied_positions = []
                        for j, (v, neg) in enumerate(clause, start=1):
                            lit_val = (not self.assignment[v]) if neg else self.assignment[v]
                            if lit_val:
                                satisfied_positions.append(j)
                        first_true = satisfied_positions[0] if satisfied_positions else None
                        obs = (
                            f"CLAUSE {idx} EVAL: {val}; total_literals={len(clause)}, "
                            f"satisfied_count={len(satisfied_positions)}, "
                            f"first_true_literal_pos={first_true}"
                        )
                    elif self.probe_detail_mode == 1:
                        count_true = 0
                        for (v, neg) in clause:
                            if ((not self.assignment[v]) if neg else self.assignment[v]):
                                count_true += 1
                        obs = f"CLAUSE {idx} EVAL: {val}; satisfied_count={count_true}"
                    else:
                        obs = f"CLAUSE {idx} EVAL: {val}"

        elif name == 'value':
            if 'var' not in parsed:
                obs = "MISSING PARAMETER: 'var'"
            else:
                try:
                    v = int(parsed['var'])
                except ValueError:
                    v = -1
                if v < 1 or v > self.num_vars:
                    obs = f"INVALID PARAMETER: var index out of range (1..{self.num_vars})"
                else:
                    if self.remaining_value_queries <= 0:
                        obs = "NO VARIABLE QUERIES LEFT"
                    else:
                        val = self.assignment[v]
                        self.remaining_value_queries -= 1
                        obs = f"VALUE x{v}={val}; remaining_value_queries={self.remaining_value_queries}"

        elif name == 'report':
            # Accept 'verdict' or 'answer'
            verdict_raw = parsed.get('verdict', parsed.get('answer', None))
            if verdict_raw is None:
                obs = "MISSING PARAMETER: 'verdict'"
            else:
                verdict_str = str(verdict_raw).strip().lower()
                if verdict_str in ['true', 'yes', '1']:
                    guess = True
                elif verdict_str in ['false', 'no', '0']:
                    guess = False
                else:
                    obs = "INVALID PARAMETER: verdict must be true|false|yes|no|1|0"
                    guess = None
                if guess is not None:
                    if guess == self.is_model:
                        obs = "Success! Your verdict was correct. The formula is satisfied by the hidden assignment."
                        reward = 1.0
                    else:
                        obs = (
                            "Failed! Your verdict was incorrect. "
                            f"Correct answer: {'true' if self.is_model else 'false'}."
                        )
                        reward = 0.0
                    terminated = True

        else:
            obs = f"UNKNOWN ACTION: {name}"

        if not terminated and self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})."

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
        if not parts:
            return None
        tokens: Dict[str, Any] = {'action': parts[0]}
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        # Provide a plausible example
        choices = [
            r"\boxed{summary}",
            r"\boxed{profile}",
            r"\boxed{view clause=1}",
            r"\boxed{probe clause=1}",
            r"\boxed{value var=1}",
            r"\boxed{report verdict=true}"
        ]
        return random.choice(choices)


class AxiomDelverEnvWithFeedback(AxiomDelverEnv):
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
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{summary}"

        elif text.startswith("unknown action"):
            error_type = "UnsupportedAction"
            error_detail["action"] = obs.split(":")[-1].strip()
            hint = "Use one of: summary, profile, view clause=k, probe clause=k, value var=j, report verdict=true|false, help"

        elif text.startswith("missing parameter"):
            error_type = "ProtocolViolation"
            error_detail["issue"] = "missing_parameter"
            if "'clause'" in obs:
                hint = "Provide a clause index: \\boxed{probe clause=1} or \\boxed{view clause=1}"
            elif "'var'" in obs:
                hint = "Provide a variable index: \\boxed{value var=1}"
            elif "'verdict'" in obs:
                hint = "Use true|false (or yes/no/1/0): \\boxed{report verdict=true}"

        elif text.startswith("invalid parameter"):
            error_type = "ProtocolViolation"
            error_detail["issue"] = "invalid_parameter_range_or_type"
            if "clause index out of range" in obs:
                error_detail["param"] = "clause"
                hint = f"Choose clause in 1..{self.num_clauses}"
            elif "var index out of range" in obs:
                error_detail["param"] = "var"
                hint = f"Choose variable in 1..{self.num_vars}"
            elif "verdict must be" in obs:
                error_detail["param"] = "verdict"
                hint = "Use true|false|yes|no|1|0"

        elif "no variable queries left" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "exhausted_query_budget"
            hint = "Switch to probe clause=k or view clause=k to gather info without using variable queries"

        elif text.startswith("failed!"):
            error_type = "WrongDecision"
            got = None
            if "correct answer:" in obs.lower():
                # best-effort parse
                got = "wrong"
                correct = "true" if self.is_model else "false"
            else:
                correct = "unknown"
            error_detail["expected"] = correct
            error_detail["got"] = got
            hint = "Probe multiple clauses. If any clause evaluates False, the verdict should be false."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Make a report sooner after gathering enough evidence"

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Good job!"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["remaining_value_queries"] = getattr(self, "remaining_value_queries", None)
            diagnostic["clauses"] = getattr(self, "num_clauses", None)
            diagnostic["vars"] = getattr(self, "num_vars", None)
            diagnostic["probe_detail_mode"] = getattr(self, "probe_detail_mode", None)
            diagnostic["turns_left"] = self.max_turns - self.turn_count
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{summary}, then \\boxed{view clause=1} or \\boxed{probe clause=1}.",
            "turn": 0,
            "remaining_value_queries": self.remaining_value_queries,
            "vars": self.num_vars,
            "clauses": self.num_clauses,
            "probe_detail_mode": self.probe_detail_mode,
            "turns_left": self.max_turns,
        }
        return obs, info