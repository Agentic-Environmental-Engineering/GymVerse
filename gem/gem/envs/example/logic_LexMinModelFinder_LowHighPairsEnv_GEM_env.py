from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class LexMinModelFinderEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            # Number of variables: larger = more search and reasoning branches = harder
            "num_vars": (3, 12),
            # Clauses per variable percentage: higher density = more constraints to parse = harder
            # Actual num_clauses = round(num_vars * clauses_per_var_pct / 100)
            "clauses_per_var_pct": (120, 220),
            # Max clause width allowed during generation: higher width can increase uncertainty; slightly harder
            "max_width": (3, 4),
            # REVERSED: Reveal budget as percentage of clauses: less reveal = less direct info = harder
            "reveal_percent": (80, 10),
            # REVERSED: Prefix-query margin: allowed prefix queries = num_vars + margin; smaller = tighter info budget = harder
            "prefix_query_margin": (4, 0),
            # REVERSED: Assignment test limit: fewer tests = harder
            "test_limit": (10, 3),
        }

        # Parameter variance
        self.param_variance = {
            "num_vars": 1,                 # discrete range (3..12) → ±1
            "clauses_per_var_pct": 10,     # range (120..220) → ±10 (~10%)
            "max_width": 0,                # small range → fixed
            "reveal_percent": 5,           # range (10..80) → ±5
            "prefix_query_margin": 1,      # small integer margin → ±1
            "test_limit": 1,               # medium integer → ±1
        }

        # Placeholder attributes
        self.num_vars: int = 0
        self.clauses_per_var_pct: int = 0
        self.max_width: int = 0
        self.reveal_percent: int = 0
        self.prefix_query_margin: int = 0
        self.test_limit: int = 0

        # Derived and stateful
        self.num_clauses: int = 0
        self.variables: List[str] = []
        self.clauses: List[List[int]] = []
        self.hidden_witness: List[int] = []
        self.target_model: List[int] = []
        self.turn_count: int = 0

        self.reveal_budget_remaining: int = 0
        self.revealed_indices: Set[int] = set()
        self.prefix_queries_used: int = 0
        self.prefix_query_limit: int = 0
        self.test_used: int = 0

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
                    lo = min(min_val, max_val)
                    hi = max(min_val, max_val)
                    if min_val > max_val:
                        # reversed ranges clamp between max and min
                        actual_value = max(max_val, min(min_val, actual_value))
                    else:
                        actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Logic Task: Find the lexicographically smallest satisfying assignment for a hidden CNF over variables v1..vn.\n"
            "Lex order compares bitstrings b1..bn where 0<1 at each position from v1 to vn.\n"
            "\n"
            "Goal:\n"
            "- Submit the bitstring (e.g., 0101) that is the lexicographically smallest model of the hidden CNF.\n"
            "\n"
            "Available actions (use \\boxed{...}):\n"
            "- vars                         → List variable names.\n"
            "- counts                       → Show n (variables) and m (clauses).\n"
            "- show_clause idx=K            → Reveal clause K (1-indexed); consumes reveal budget.\n"
            "- reveal_all                   → Reveal all unrevealed clauses; consumes reveal budget equal to remaining clauses.\n"
            "- check_prefix prefix=01?0     → Ask if some completion of the prefix satisfies CNF ('?' means unknown). Uses limited budget.\n"
            "- test assign=0101             → Check if full assignment satisfies; uses limited tests.\n"
            "- submit answer=0101           → Submit your final answer; ends the episode.\n"
            "- help                         → Show these instructions.\n"
            "\n"
            "Rules:\n"
            "- Indices start at 1. Prefix length must be ≤ n and contain only 0,1,?.\n"
            "- Budgets: reveals, prefix checks, and tests are limited. Use counts/suffix to see remaining.\n"
            "- Actions must be inside \\boxed{...}. Unknown actions are ignored with a protocol message.\n"
            "\n"
            "Format:\n"
            "- With parameters: \\boxed{action key=value}\n"
            "- Without parameters: \\boxed{action}\n"
            "\n"
            "Example:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        remaining_turns = max(0, self.max_turns - self.turn_count)
        rev_shown = sorted(list(self.revealed_indices))
        shown_preview = ", ".join(str(i) for i in rev_shown[:10])
        if len(rev_shown) > 10:
            shown_preview += ", ..."
        return (
            f"State:\n"
            f"- Turns used: {self.turn_count}/{self.max_turns} (remaining {remaining_turns})\n"
            f"- Variables: n={self.num_vars} named v1..v{self.num_vars}\n"
            f"- Clauses: m={self.num_clauses}\n"
            f"- Reveal budget remaining: {self.reveal_budget_remaining}; revealed indices: [{shown_preview if shown_preview else ''}]\n"
            f"- Prefix checks used/limit: {self.prefix_queries_used}/{self.prefix_query_limit}\n"
            f"- Tests used/limit: {self.test_used}/{self.test_limit}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.variables = [f"v{i}" for i in range(1, self.num_vars + 1)]
        self.num_clauses = max(self.num_vars, int(round(self.num_vars * self.clauses_per_var_pct / 100.0)))
        self.revealed_indices = set()
        self.reveal_budget_remaining = max(0, min(self.num_clauses, int(round(self.num_clauses * self.reveal_percent / 100.0))))
        self.prefix_queries_used = 0
        self.prefix_query_limit = self.num_vars + max(0, self.prefix_query_margin)
        self.test_used = 0

        # Generate satisfiable CNF by sampling a witness and forcing each clause to be satisfied by it.
        self.hidden_witness = [random.randint(0, 1) for _ in range(self.num_vars)]
        self.clauses = []
        for _ in range(self.num_clauses):
            width = random.randint(2, max(2, self.max_width))
            vars_in_clause = random.sample(range(1, self.num_vars + 1), min(width, self.num_vars))
            # Initial random signs
            signs = [random.choice([True, False]) for _ in vars_in_clause]  # True means positive literal
            clause = []
            for v, pos in zip(vars_in_clause, signs):
                lit = v if pos else -v
                clause.append(lit)
            # Ensure clause is satisfied by hidden_witness
            if not self._clause_satisfied_by_assignment(clause, self.hidden_witness):
                # Flip one literal so it becomes satisfied
                j = random.choice(range(len(clause)))
                vj = abs(clause[j])
                # Set literal polarity to match witness so literal becomes True under witness
                desired_positive = True if self.hidden_witness[vj - 1] == 1 else False
                clause[j] = vj if desired_positive else -vj
            # Remove accidental tautologies (v OR ¬v) by re-sampling literal polarities if needed
            if self._is_tautology(clause):
                # Regenerate this clause differently
                vset = vars_in_clause
                clause = []
                for v in vset:
                    # Choose polarity aligned with witness with high chance to avoid contradictions
                    if random.random() < 0.7:
                        lit = v if self.hidden_witness[v - 1] == 1 else -v
                    else:
                        lit = v if random.choice([True, False]) else -v
                    clause.append(lit)
                if not self._clause_satisfied_by_assignment(clause, self.hidden_witness):
                    # Force satisfaction
                    j = random.choice(range(len(clause)))
                    vj = abs(clause[j])
                    desired_positive = True if self.hidden_witness[vj - 1] == 1 else False
                    clause[j] = vj if desired_positive else -vj
            self.clauses.append(self._dedup_clause(clause))

        # Compute lexicographically smallest satisfying assignment
        self.target_model = self._find_lex_smallest_model()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "").strip().lower()

        if name == "help":
            obs = self._get_instructions()
            # Check timeout after providing help
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif name == "vars":
            obs = "Variables: " + ", ".join(self.variables)
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif name == "counts":
            obs = f"Counts: n={self.num_vars}, m={self.num_clauses}"
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif name == "show_clause":
            idx_str = parsed.get("idx", None)
            if idx_str is None or not idx_str.isdigit():
                obs = "PROTOCOL VIOLATION: 'idx' missing or not an integer."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            idx = int(idx_str)
            if idx < 1 or idx > self.num_clauses:
                obs = "PROTOCOL VIOLATION: clause index out of range."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if self.reveal_budget_remaining <= 0:
                obs = "PROTOCOL VIOLATION: reveal budget exhausted."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            self.reveal_budget_remaining -= 1
            self.revealed_indices.add(idx)
            clause_str = self._format_clause(self.clauses[idx - 1])
            obs = f"Clause {idx}: {clause_str}"
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif name == "reveal_all":
            remaining = self.num_clauses - len(self.revealed_indices)
            if remaining <= 0:
                obs = "All clauses already revealed."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if self.reveal_budget_remaining < remaining:
                obs = "PROTOCOL VIOLATION: insufficient reveal budget for reveal_all."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            self.reveal_budget_remaining -= remaining
            unrevealed = [i for i in range(1, self.num_clauses + 1) if i not in self.revealed_indices]
            self.revealed_indices.update(unrevealed)
            listed = "; ".join([f"{i}: {self._format_clause(self.clauses[i - 1])}" for i in unrevealed[:10]])
            if len(unrevealed) > 10:
                listed += "; ..."
            obs = f"Revealed {remaining} clauses: {listed}"
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif name == "check_prefix":
            prefix = parsed.get("prefix", None)
            if prefix is None:
                obs = "PROTOCOL VIOLATION: 'prefix' missing."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if any(ch not in "01?" for ch in prefix):
                obs = "PROTOCOL VIOLATION: prefix must contain only 0,1,?."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if len(prefix) > self.num_vars:
                obs = "PROTOCOL VIOLATION: prefix length exceeds number of variables."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if self.prefix_queries_used >= self.prefix_query_limit:
                obs = "PROTOCOL VIOLATION: prefix check limit exhausted."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            self.prefix_queries_used += 1
            consistent = self._is_satisfiable_with_prefix(prefix)
            obs = f"Prefix consistent: {'yes' if consistent else 'no'}"
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif name == "test":
            assign = parsed.get("assign", None)
            if assign is None:
                obs = "PROTOCOL VIOLATION: 'assign' missing."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if len(assign) != self.num_vars or any(ch not in "01" for ch in assign):
                obs = "PROTOCOL VIOLATION: assign must be a 0/1 string of length n."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            if self.test_used >= self.test_limit:
                obs = "PROTOCOL VIOLATION: test limit exhausted."
                if self.turn_count >= self.max_turns:
                    return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            self.test_used += 1
            bits = [int(c) for c in assign]
            ok, unsat_idx = self._evaluate_full_assignment(bits)
            if ok:
                obs = "Assignment satisfies: yes"
            else:
                preview = ", ".join(str(i) for i in unsat_idx[:10])
                if len(unsat_idx) > 10:
                    preview += ", ..."
                obs = f"Assignment satisfies: no; unsatisfied_count={len(unsat_idx)}; examples=[{preview}]"
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif name == "submit":
            ans = parsed.get("answer", None)
            if ans is None:
                obs = "PROTOCOL VIOLATION: 'answer' missing."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if len(ans) != self.num_vars or any(ch not in "01" for ch in ans):
                obs = "Failed: answer must be a 0/1 string of length n. Reference answer is not revealed."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            ans_bits = [int(c) for c in ans]
            target_str = "".join(str(b) for b in self.target_model)
            if ans_bits == self.target_model:
                obs = f"Success! Correct lexicographically smallest model submitted: {target_str}"
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed: incorrect answer. Reference answer: {target_str}"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "UNSUPPORTED ACTION: use one of [vars, counts, show_clause, reveal_all, check_prefix, test, submit, help]."
            if self.turn_count >= self.max_turns:
                return f"{obs}\nReached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
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
        if not parts:
            return None
        tokens: Dict[str, Any] = {"action": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        choices = ["vars", "counts", "show_clause", "check_prefix", "test"]
        a = random.choice(choices)
        if a == "show_clause":
            idx = random.randint(1, max(1, self.num_clauses or 1))
            return rf"\boxed{ {f'show_clause idx={idx}'} }".replace(" ", "")
        elif a == "check_prefix":
            length = random.randint(0, max(0, self.num_vars - 1))
            prefix = "".join(random.choice(["0", "1", "?"]) for _ in range(length))
            return rf"\boxed{{check_prefix prefix={prefix}}}"
        elif a == "test":
            if self.num_vars <= 0:
                return r"\boxed{vars}"
            assign = "".join(str(random.randint(0, 1)) for _ in range(self.num_vars))
            return rf"\boxed{{test assign={assign}}}"
        elif a == "counts":
            return r"\boxed{counts}"
        else:
            return r"\boxed{vars}"

    # ---------- Logic helpers ----------
    def _format_clause(self, clause: List[int]) -> str:
        lits = []
        for lit in clause:
            v = abs(lit)
            if lit > 0:
                lits.append(f"v{v}")
            else:
                lits.append(f"¬v{v}")
        return "(" + " OR ".join(lits) + ")"

    def _is_tautology(self, clause: List[int]) -> bool:
        s = set(clause)
        for lit in clause:
            if -lit in s:
                return True
        return False

    def _dedup_clause(self, clause: List[int]) -> List[int]:
        # remove duplicate literals while keeping order
        seen = set()
        out = []
        for lit in clause:
            if lit not in seen:
                out.append(lit)
                seen.add(lit)
        return out

    def _clause_satisfied_by_assignment(self, clause: List[int], assign: List[int]) -> bool:
        for lit in clause:
            idx = abs(lit) - 1
            val = assign[idx]
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                return True
        return False

    def _evaluate_full_assignment(self, bits: List[int]) -> Tuple[bool, List[int]]:
        unsat = []
        for i, clause in enumerate(self.clauses, start=1):
            if not self._clause_satisfied_by_assignment(clause, bits):
                unsat.append(i)
        return (len(unsat) == 0), unsat

    def _is_clause_false_under_partial(self, clause: List[int], partial: List[Optional[int]]) -> bool:
        # Clause false if all literals are assigned and all evaluate to False
        all_false = True
        for lit in clause:
            idx = abs(lit) - 1
            v = partial[idx]
            if v is None:
                all_false = False
                # literal could still be True later
            else:
                lit_true = (lit > 0 and v == 1) or (lit < 0 and v == 0)
                if lit_true:
                    return False
        return all_false

    def _is_clause_true_under_partial(self, clause: List[int], partial: List[Optional[int]]) -> bool:
        for lit in clause:
            idx = abs(lit) - 1
            v = partial[idx]
            if v is not None:
                lit_true = (lit > 0 and v == 1) or (lit < 0 and v == 0)
                if lit_true:
                    return True
        return False

    def _is_satisfiable_with_prefix(self, prefix: str) -> bool:
        # Build partial assignment from prefix string of 0/1/?; remaining None
        partial: List[Optional[int]] = [None] * self.num_vars
        for i, ch in enumerate(prefix):
            if ch == "?":
                partial[i] = None
            else:
                partial[i] = int(ch)
        # Simple backtracking with early clause pruning
        return self._sat_backtrack(partial)

    def _sat_backtrack(self, partial: List[Optional[int]]) -> bool:
        # Clause-level pruning
        for clause in self.clauses:
            if self._is_clause_true_under_partial(clause, partial):
                continue
            if self._is_clause_false_under_partial(clause, partial):
                return False
        # Find next unassigned
        try:
            idx = partial.index(None)
        except ValueError:
            # Fully assigned and no false clauses -> satisfiable
            return True
        # Try 0 then 1 for lexicographic preference
        for val in [0, 1]:
            partial[idx] = val
            if self._sat_backtrack(partial):
                partial[idx] = None
                return True
            partial[idx] = None
        return False

    def _find_lex_smallest_model(self) -> List[int]:
        # Greedy: for i in 1..n, try setting bit to 0 if exists completion; else 1
        model: List[int] = []
        for i in range(self.num_vars):
            prefix = "".join(str(b) for b in model) + "0"
            if self._is_satisfiable_with_prefix(prefix):
                model.append(0)
            else:
                # must be 1
                prefix1 = "".join(str(b) for b in model) + "1"
                if not self._is_satisfiable_with_prefix(prefix1):
                    # Safety fallback: in rare case of inconsistency, use witness’ bit
                    model.append(self.hidden_witness[i])
                else:
                    model.append(1)
        return model


class LexMinModelFinderEnvWithFeedback(LexMinModelFinderEnv):
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
            hint = "Wrap your command inside \\boxed{...} and follow the listed actions."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["vars", "counts", "show_clause", "reveal_all", "check_prefix", "test", "submit", "help"]
            hint = "Use one of the supported actions. Start with \\boxed{counts} or \\boxed{vars}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "index out of range" in text:
                error_detail["violation"] = "clause_index_out_of_range"
                hint = "Use \\boxed{counts} to get m, then choose 1 ≤ idx ≤ m."
            elif "reveal budget exhausted" in text:
                error_detail["violation"] = "reveal_budget_exhausted"
                hint = "Use \\boxed{check_prefix prefix=...} to reason without reveals."
            elif "insufficient reveal budget" in text:
                error_detail["violation"] = "reveal_all_insufficient_budget"
                hint = "Reveal clauses individually with \\boxed{show_clause idx=k}."
            elif "prefix check limit exhausted" in text:
                error_detail["violation"] = "prefix_limit_exhausted"
                hint = "Submit or use remaining tests with \\boxed{test assign=...} judiciously."
            elif "prefix must contain only" in text:
                error_detail["violation"] = "bad_prefix_chars"
                hint = "Prefix can only contain 0,1,? and be at most n characters."
            elif "prefix length exceeds" in text:
                error_detail["violation"] = "prefix_too_long"
                hint = "Keep prefix length ≤ n. Use \\boxed{counts} to see n."
            elif "'idx' missing" in text:
                error_detail["violation"] = "missing_idx"
                hint = "Provide idx like \\boxed{show_clause idx=3}."
            elif "'prefix' missing" in text:
                error_detail["violation"] = "missing_prefix"
                hint = "Provide prefix like \\boxed{check_prefix prefix=01?}."
            elif "'assign' missing" in text:
                error_detail["violation"] = "missing_assign"
                hint = "Provide a full bitstring: \\boxed{test assign=...}."
            elif "assign must be a 0/1 string" in text:
                error_detail["violation"] = "bad_assign_format"
                hint = "Ensure assign is length n and only 0/1."
            elif "test limit exhausted" in text:
                error_detail["violation"] = "test_limit_exhausted"
                hint = "Use \\boxed{check_prefix} to narrow before submitting."
            else:
                error_detail["violation"] = "general"

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns"
            hint = "Plan with counts and budgets; prioritize check_prefix to construct the model in ≤ n queries."

        elif "failed: incorrect answer" in text:
            error_type = "WrongDecision"
            # Extract expected and maybe got
            expected = None
            got = None
            m_exp = re.search(r"reference answer:\s*([01]+)", obs, flags=re.IGNORECASE)
            if m_exp:
                expected = m_exp.group(1)
            m_got = re.search(r"submit.*answer=([01]+)", obs, flags=re.IGNORECASE)
            if m_got:
                got = m_got.group(1)
            error_detail["expected"] = expected
            if got:
                error_detail["got"] = got
            hint = "Build the lex-min model: for i=1..n, try prefix with 0 using \\boxed{check_prefix}; if yes keep 0 else set 1."

        elif "failed: answer must be a 0/1 string" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "bad_submit_format"
            hint = "Use \\boxed{submit answer=...} with a 0/1 string of length n."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "n": getattr(self, "num_vars", None),
                "m": getattr(self, "num_clauses", None),
                "reveal_remaining": getattr(self, "reveal_budget_remaining", None),
                "prefix_used": getattr(self, "prefix_queries_used", None),
                "prefix_limit": getattr(self, "prefix_query_limit", None),
                "test_used": getattr(self, "test_used", None),
                "test_limit": getattr(self, "test_limit", None),
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
            "hint": "Use \\boxed{counts} then construct the lex-min model via \\boxed{check_prefix}.",
            "turn": 0,
        }
        return obs, info