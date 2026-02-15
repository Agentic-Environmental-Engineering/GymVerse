from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class TableauVoyagerEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        self.complexity_params = {
            # Number of variables in the CNF: more variables = larger search space = harder
            "num_vars": (3, 10),
            # Number of clauses: more constraints = more interactions = harder
            "num_clauses": (4, 24),
            # Maximum clause width (number of literals per clause): larger width = more combinatorial complexity
            "clause_max_width": (2, 5),
            # REVERSED: per-query literal reveal limit in query_clause partial mode: fewer reveals = less information = harder
            "reveal_limit": (3, 1),
            # Probability (percent) that the instance is UNSAT by injecting a direct contradiction (v and ¬v unit clauses): higher = more difficult decision
            "unsat_probability": (0, 50),
        }

        self.param_variance = {
            "num_vars": 1,
            "num_clauses": 2,
            "clause_max_width": 0,
            "reveal_limit": 0,
            "unsat_probability": 5,
        }

        self.variables: list = []
        self.clauses: list = []
        self.occ_counts: Dict[str, Dict[str, int]] = {}
        self.buffer: Dict[str, bool] = {}
        self.store: Dict[str, bool] = {}
        self.turn_count: int = 0
        self.is_unsat: bool = False
        self.sat_model: Dict[str, bool] = {}
        self.unsat_pivot_var: Optional[str] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                else:
                    actual_value = center_value
            else:
                actual_value = center_value

            if min_val > max_val:
                lo, hi = max_val, min_val
            else:
                lo, hi = min_val, max_val

            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are exploring a hidden CNF formula. Variables are named v1, v2, ..., vn.\n"
            "Your goal is to determine satisfiability and, if SAT, submit a full satisfying assignment.\n"
            "Alternatively, if the formula is UNSAT, submit that verdict.\n"
            "\n"
            "Available actions (use \\boxed{...}):\n"
            "- query_vars: reveal variable count and names.\n"
            "- query_clauses: reveal total clause count.\n"
            "- query_clause idx=<i> [show=length|partial|all] [limit=<k>]: inspect a clause.\n"
            "  • idx is 1-based. show=partial reveals up to 'reveal_limit' literals; show=all reveals the full clause.\n"
            "- query_is_unit idx=<i>: check if clause i is unit.\n"
            "- query_is_pure var=<v>: check if variable appears with only one polarity.\n"
            "- query_occurrences var=<v>: count positive/negative occurrences of variable.\n"
            "- set var=<v> value=<T|F>: set buffer assignment.\n"
            "- unset var=<v>: remove buffer assignment.\n"
            "- reset_buffer: clear the buffer.\n"
            "- commit var=<v> value=<T|F>: commit a permanent assignment to the store (cannot conflict).\n"
            "- commit_unit idx=<i>: commit the unit clause's implied assignment.\n"
            "- commit_pure var=<v>: commit pure literal assignment.\n"
            "- show_buffer: inspect your working buffer.\n"
            "- show_store: inspect committed assignments.\n"
            "- check_conflict: check if current assignments falsify any clause.\n"
            "- submit_assignment v1=<T|F> v2=<T|F> ...: final submission of a full assignment.\n"
            "- submit_unsat: final submission declaring the formula UNSAT.\n"
            "\n"
            "Protocol:\n"
            "- Clause indices are 1-based.\n"
            "- Committing a value conflicting with already committed value is a protocol violation and ends the episode.\n"
            "- Final submissions end the episode and are judged for correctness.\n"
            "\n"
            "Actions must be in \\boxed{...} format. Example:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        var_list = ", ".join(self.variables) if self.variables else "(hidden)"
        buf_items = ", ".join([f"{k}={'T' if v else 'F'}" for k, v in sorted(self.buffer.items())]) or "(empty)"
        store_items = ", ".join([f"{k}={'T' if v else 'F'}" for k, v in sorted(self.store.items())]) or "(empty)"
        base = [
            f"Turn: {self.turn_count}/{self.max_turns}",
            f"Variables: {var_list}",
            f"Buffer: {buf_items}",
            f"Store: {store_items}",
            "Enter your action in \\boxed{...} format."
        ]
        return "\n".join(base)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.buffer = {}
        self.store = {}
        self.variables = [f"v{i+1}" for i in range(self.num_vars)]
        self.sat_model = {v: bool(random.getrandbits(1)) for v in self.variables}
        self.clauses = []
        self.occ_counts = {v: {"pos": 0, "neg": 0} for v in self.variables}
        self.is_unsat = False
        self.unsat_pivot_var = None

        for _ in range(self.num_clauses):
            width = random.randint(1, self.clause_max_width)
            vars_chosen = random.sample(self.variables, k=min(width, len(self.variables)))
            pivot = random.choice(vars_chosen)
            pivot_sign = self.sat_model[pivot]  # ensure clause is satisfied by sat_model
            clause = [(pivot, pivot_sign)]
            for v in vars_chosen:
                if v == pivot:
                    continue
                sign = bool(random.getrandbits(1))
                clause.append((v, sign))
            # Deduplicate literals per var keeping one occurrence (prefer pivot satisfaction)
            seen = {}
            for var, sign in clause:
                if var not in seen:
                    seen[var] = sign
            normalized_clause = [(var, seen[var]) for var in seen.keys()]
            self.clauses.append(normalized_clause)

        if random.randint(1, 100) <= self.unsat_probability and len(self.variables) > 0:
            self.is_unsat = True
            self.unsat_pivot_var = random.choice(self.variables)
            self.clauses.append([(self.unsat_pivot_var, True)])   # v
            self.clauses.append([(self.unsat_pivot_var, False)])  # ¬v

        # Build occurrence counts
        self.occ_counts = {v: {"pos": 0, "neg": 0} for v in self.variables}
        for clause in self.clauses:
            for var, sign in clause:
                if sign:
                    self.occ_counts[var]["pos"] += 1
                else:
                    self.occ_counts[var]["neg"] += 1

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "").strip()

        supported = {
            "query_vars", "query_clauses", "query_clause", "query_is_unit", "query_is_pure",
            "query_occurrences", "set", "unset", "reset_buffer", "commit", "commit_unit", "commit_pure",
            "show_buffer", "show_store", "check_conflict", "submit_assignment", "submit_unsat"
        }
        if name not in supported:
            obs = f"UNSUPPORTED ACTION: {name}"
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        def var_exists(v: str) -> bool:
            return v in self.variables

        def clause_index_ok(i: int) -> bool:
            return 1 <= i <= len(self.clauses)

        reward = 0.0
        obs = ""

        if name == "query_vars":
            obs = f"INFO: {len(self.variables)} variables -> {', '.join(self.variables)}"

        elif name == "query_clauses":
            obs = f"INFO: {len(self.clauses)} clauses total"

        elif name == "query_clause":
            idx_str = parsed.get("idx")
            show = parsed.get("show", "partial")
            limit_str = parsed.get("limit")
            if not idx_str or not idx_str.isdigit():
                obs = "PROTOCOL VIOLATION: missing or invalid idx for query_clause"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            idx = int(idx_str)
            if not clause_index_ok(idx):
                obs = f"PROTOCOL VIOLATION: clause index out of range (1..{len(self.clauses)})"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            clause = self.clauses[idx - 1]
            if show == "length":
                obs = f"INFO: clause #{idx} length = {len(clause)}"
            elif show == "all":
                lits = ", ".join([f"{var}" if sign else f"¬{var}" for var, sign in clause])
                obs = f"INFO: clause #{idx} = ({lits})"
            else:
                limit = self.reveal_limit
                if limit_str and limit_str.isdigit():
                    limit = max(1, min(int(limit_str), len(clause)))
                reveal = random.sample(clause, k=min(limit, len(clause)))
                lits = ", ".join([f"{var}" if sign else f"¬{var}" for var, sign in reveal])
                obs = f"INFO: clause #{idx} partial = ({lits})"

        elif name == "query_is_unit":
            idx_str = parsed.get("idx")
            if not idx_str or not idx_str.isdigit():
                obs = "PROTOCOL VIOLATION: missing or invalid idx for query_is_unit"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            idx = int(idx_str)
            if not clause_index_ok(idx):
                obs = f"PROTOCOL VIOLATION: clause index out of range (1..{len(self.clauses)})"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            clause = self.clauses[idx - 1]
            obs = f"INFO: clause #{idx} is {'UNIT' if len(clause) == 1 else 'NON-UNIT'}"

        elif name == "query_is_pure":
            v = parsed.get("var")
            if not v or not var_exists(v):
                obs = "PROTOCOL VIOLATION: missing or unknown var for query_is_pure"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            counts = self.occ_counts[v]
            is_pure = (counts["pos"] == 0) ^ (counts["neg"] == 0)
            obs = f"INFO: {v} is {'PURE' if is_pure else 'NOT PURE'} (pos={counts['pos']}, neg={counts['neg']})"

        elif name == "query_occurrences":
            v = parsed.get("var")
            if not v or not var_exists(v):
                obs = "PROTOCOL VIOLATION: missing or unknown var for query_occurrences"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            counts = self.occ_counts[v]
            obs = f"INFO: {v} occurrences -> pos={counts['pos']}, neg={counts['neg']}"

        elif name == "set":
            v = parsed.get("var")
            val = parsed.get("value")
            if not v or not var_exists(v) or val not in {"T", "F", "True", "False"}:
                obs = "PROTOCOL VIOLATION: set requires var=<v> and value=<T|F>"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            b = True if val in {"T", "True"} else False
            self.buffer[v] = b
            obs = f"STATE: buffer set {v}={'T' if b else 'F'}"

        elif name == "unset":
            v = parsed.get("var")
            if not v or not var_exists(v):
                obs = "PROTOCOL VIOLATION: unset requires a known var=<v>"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.buffer.pop(v, None)
            obs = f"STATE: buffer unset {v}"

        elif name == "reset_buffer":
            self.buffer.clear()
            obs = "STATE: buffer cleared"

        elif name == "commit":
            v = parsed.get("var")
            val = parsed.get("value")
            if not v or not var_exists(v) or val not in {"T", "F", "True", "False"}:
                obs = "PROTOCOL VIOLATION: commit requires var=<v> and value=<T|F>"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            b = True if val in {"T", "True"} else False
            if v in self.store and self.store[v] != b:
                obs = f"PROTOCOL VIOLATION: commit conflict for {v} (already {'T' if self.store[v] else 'F'})"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.store[v] = b
            obs = f"STATE: committed {v}={'T' if b else 'F'}"

        elif name == "commit_unit":
            idx_str = parsed.get("idx")
            if not idx_str or not idx_str.isdigit():
                obs = "PROTOCOL VIOLATION: missing or invalid idx for commit_unit"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            idx = int(idx_str)
            if not clause_index_ok(idx):
                obs = f"PROTOCOL VIOLATION: clause index out of range (1..{len(self.clauses)})"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            clause = self.clauses[idx - 1]
            if len(clause) != 1:
                obs = "PROTOCOL VIOLATION: clause is not unit"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            v, sign = clause[0]
            if v in self.store and self.store[v] != sign:
                obs = f"PROTOCOL VIOLATION: commit_unit conflict for {v} (already {'T' if self.store[v] else 'F'})"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.store[v] = sign
            obs = f"STATE: committed from unit {v}={'T' if sign else 'F'}"

        elif name == "commit_pure":
            v = parsed.get("var")
            if not v or not var_exists(v):
                obs = "PROTOCOL VIOLATION: missing or unknown var for commit_pure"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            counts = self.occ_counts[v]
            if counts["pos"] > 0 and counts["neg"] > 0:
                obs = "PROTOCOL VIOLATION: variable is not pure"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            sign = True if counts["pos"] > 0 else False
            if v in self.store and self.store[v] != sign:
                obs = f"PROTOCOL VIOLATION: commit_pure conflict for {v} (already {'T' if self.store[v] else 'F'})"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.store[v] = sign
            obs = f"STATE: committed pure {v}={'T' if sign else 'F'}"

        elif name == "show_buffer":
            buf_items = ", ".join([f"{k}={'T' if v else 'F'}" for k, v in sorted(self.buffer.items())]) or "(empty)"
            obs = f"STATE: buffer -> {buf_items}"

        elif name == "show_store":
            store_items = ", ".join([f"{k}={'T' if v else 'F'}" for k, v in sorted(self.store.items())]) or "(empty)"
            obs = f"STATE: store -> {store_items}"

        elif name == "check_conflict":
            assignment = dict(self.store)
            for k, v in self.buffer.items():
                if k in assignment and assignment[k] != v:
                    obs = f"INFO: assignment conflict on {k} (store={'T' if assignment[k] else 'F'}, buffer={'T' if v else 'F'})"
                    break
                assignment[k] = v
            else:
                falsified_idx = None
                for i, clause in enumerate(self.clauses, start=1):
                    evals = []
                    fully_assigned = True
                    for var, sign in clause:
                        if var not in assignment:
                            fully_assigned = False
                            evals.append(None)
                        else:
                            evals.append(assignment[var] if sign else (not assignment[var]))
                    if fully_assigned and all(val is False for val in evals):
                        falsified_idx = i
                        break
                if falsified_idx is not None:
                    obs = f"INFO: conflict — clause #{falsified_idx} falsified by current assignments"
                else:
                    obs = "INFO: no immediate clause conflicts detected"

        elif name == "submit_assignment":
            # Enforce full assignment submission
            submission = {}
            for v in self.variables:
                if v in parsed:
                    val = parsed[v]
                    if val not in {"T", "F", "True", "False"}:
                        obs = f"PROTOCOL VIOLATION: invalid value for {v} in submission"
                        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                    submission[v] = True if val in {"T", "True"} else False
            if len(submission) != len(self.variables):
                obs = "PROTOCOL VIOLATION: submit_assignment requires all variables"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            def clause_sat(clause, assign):
                for var, sign in clause:
                    val = assign[var]
                    lit = val if sign else (not val)
                    if lit:
                        return True
                return False

            if self.is_unsat:
                obs = "FAILED: formula is UNSAT; no assignment can satisfy all clauses"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            ok = all(clause_sat(c, submission) for c in self.clauses)
            if ok:
                obs = "SUCCESS: assignment satisfies all clauses"
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "FAILED: assignment does not satisfy all clauses"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        elif name == "submit_unsat":
            if self.is_unsat:
                obs = "SUCCESS: correctly declared UNSAT"
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "FAILED: formula is SAT; UNSAT declaration is incorrect"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"TIMEOUT: Reached max turns ({self.max_turns})"
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

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
        tokens: Dict[str, Any] = {}
        tokens["action"] = parts[0]
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                tokens[key] = value
        return tokens

    def sample_random_action(self) -> str:
        return r"\boxed{query_vars}"


class TableauVoyagerEnvWithFeedback(TableauVoyagerEnv):
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
            error_detail["issue"] = "boxed_format_missing"
            hint = "Use \\boxed{...} and include a supported action name and parameters."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported action:\s*(\S+)", obs, flags=re.IGNORECASE)
            error_detail["action"] = m.group(1) if m else None
            hint = "Check the instruction list of supported actions and match names exactly."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "clause index" in text:
                error_detail["violation"] = "invalid_clause_index"
                hint = "Use indices between 1 and the number of clauses revealed by query_clauses."
            elif "missing or invalid idx" in text:
                error_detail["violation"] = "missing_or_invalid_idx"
                hint = "Provide idx=<i> with a positive integer."
            elif "unknown var" in text or "requires var" in text:
                error_detail["violation"] = "unknown_or_missing_var"
                hint = "Query variables with query_vars and use those names exactly."
            elif "commit conflict" in text:
                error_detail["violation"] = "commit_conflict"
                hint = "Inspect store with show_store and avoid committing a contradictory value."
            elif "not pure" in text:
                error_detail["violation"] = "not_pure_commit"
                hint = "Check purity with query_is_pure var=<v> before using commit_pure."
            elif "requires all variables" in text:
                error_detail["violation"] = "incomplete_submission"
                hint = "Submit a full assignment specifying every variable (e.g., v1=T v2=F ...)."
            elif "invalid value for" in text:
                error_detail["violation"] = "invalid_value_in_submission"
                hint = "Use T or F for each variable in submit_assignment."
            else:
                error_detail["violation"] = "general_protocol_error"
                hint = "Re-read action formats and parameter requirements in the instructions."

        elif "timeout" in text or truncated:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan queries first (query_vars, query_clauses), then focus on unit/pure commits and submit before the turn limit."

        elif "failed:" in text:
            error_type = "WrongDecision"
            if "unsat; no assignment can satisfy" in text:
                error_detail["context"] = "assignment_in_unsat_instance"
                hint = "Declare UNSAT when you detect contradictory unit clauses."
            elif "formula is sat" in text and "unsat declaration" in text:
                error_detail["context"] = "unsat_declared_on_sat_instance"
                hint = "Build a satisfying assignment via clause inspection and pure/unit steps."
            else:
                error_detail["context"] = "assignment_not_satisfying"
                hint = "Use check_conflict and query_clause to find violated clauses; adjust assignments accordingly."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["unsat_hint_var"] = getattr(self, "unsat_pivot_var", None)
            diagnostic["is_unsat"] = getattr(self, "is_unsat", None)
            diagnostic["variables"] = list(getattr(self, "variables", []))
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with query_vars and query_clauses; then inspect individual clauses using query_clause.",
            "turn": 0,
            "is_unsat": getattr(self, "is_unsat", None),
            "unsat_hint_var": getattr(self, "unsat_pivot_var", None),
            "variables": list(getattr(self, "variables", [])),
        }
        return obs, info