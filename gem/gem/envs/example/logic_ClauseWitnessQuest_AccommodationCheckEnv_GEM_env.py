from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class ClauseWitnessQuestEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            # Number of variables: more variables increase hypothesis space → harder
            "num_variables": (4, 20),
            # Number of clauses: more clauses require more checks → harder
            "num_clauses": (3, 20),
            # Maximum clause length (min is fixed at 2). Larger clauses change structure; modest range.
            "max_clause_len": (3, 6),
            # REVERSED: limit on variable-reveal queries; fewer allowed → harder
            "var_query_limit": (6, 2),
        }

        # Parameter variance
        self.param_variance = {
            "num_variables": 2,    # medium range → ±2
            "num_clauses": 2,      # medium range → ±2
            "max_clause_len": 0,   # small range → fixed at level-determined value
            "var_query_limit": 1,  # medium small → ±1 but will be clamped
        }

        # Placeholder attributes
        self.num_variables: int = 0
        self.num_clauses: int = 0
        self.max_clause_len: int = 0
        self.var_query_limit: int = 0

        # State
        self.turn_count: int = 0
        self.variables: List[str] = []
        self.assignment: List[bool] = []
        self.clauses: List[List[Tuple[int, bool]]] = []  # list of (var_idx, is_negated)
        self.tested_clauses: Set[int] = set()
        self.var_revealed: Dict[int, bool] = {}
        self.var_queries_used: int = 0
        self.global_satisfied: bool = False

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
            # Clamp to proper range, supporting reversed params
            low = min(self.complexity_params[param_name][0], self.complexity_params[param_name][1])
            high = max(self.complexity_params[param_name][0], self.complexity_params[param_name][1])
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

        # Safety clamps for feasibility
        self.num_variables = max(2, self.num_variables)
        self.num_clauses = max(1, self.num_clauses)
        self.max_clause_len = max(2, min(self.max_clause_len, self.num_variables))
        self.var_query_limit = max(0, self.var_query_limit)

    def _get_instructions(self) -> str:
        return (
            "You are in ClauseWitnessQuest.\n"
            "A CNF formula over hidden boolean assignment is fixed. Each clause is a disjunction of literals.\n"
            "Goal: Decide whether every clause has at least one true literal under the hidden assignment.\n"
            "Submit a final report: yes (every clause is satisfied) or no (at least one clause is unsatisfied).\n"
            "\n"
            "Available actions:\n"
            "- get_counts: Reveal number of variables and clauses.\n"
            "- get_formula: Reveal the clause structure (variables and negations) without the hidden assignment.\n"
            "- test_clause id=<k>: Ask whether clause k (1-indexed) is satisfied.\n"
            "- ask_var id=<i> or ask_var var=x<i>: Reveal the truth value of variable xi (limited uses per episode).\n"
            "- report answer=<yes|no>: Submit your final decision and end the episode.\n"
            "\n"
            "Rules:\n"
            "- Actions must follow the \\boxed{...} format exactly.\n"
            "- Indices are 1-indexed: clauses in [1..M], variables in [1..N].\n"
            "- Variable reveal queries are limited; exceeding the limit is a protocol violation ending the episode.\n"
            "- Information queries do not change the hidden assignment.\n"
            "\n"
            "Action format:\n"
            "- \\boxed{get_counts}\n"
            "- \\boxed{get_formula}\n"
            "- \\boxed{test_clause id=3}\n"
            "- \\boxed{ask_var id=2} or \\boxed{ask_var var=x2}\n"
            "- \\boxed{report answer=yes}\n"
            "\n"
            "Example:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        tested = len(self.tested_clauses)
        suffix = (
            f"CNF summary: N={self.num_variables} variables, M={self.num_clauses} clauses. "
            f"Variable query usage: {self.var_queries_used}/{self.var_query_limit}. "
            f"Tested clauses so far: {tested}/{self.num_clauses}. "
            f"Turn {self.turn_count}/{self.max_turns}. "
            "Enter your action in \\boxed{...} format."
        )
        return suffix

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.variables = [f"x{i}" for i in range(1, self.num_variables + 1)]
        self.assignment = [random.choice([True, False]) for _ in range(self.num_variables)]
        self.clauses = []
        self.tested_clauses = set()
        self.var_revealed = {}
        self.var_queries_used = 0

        for _ in range(self.num_clauses):
            k = random.randint(2, max(2, self.max_clause_len))
            k = min(k, self.num_variables)
            var_idxs = random.sample(range(self.num_variables), k)
            lits = []
            for v in var_idxs:
                is_neg = random.choice([False, True])
                lits.append((v, is_neg))
            self.clauses.append(lits)

        clause_results = []
        for lits in self.clauses:
            sat = False
            for (vidx, is_neg) in lits:
                val = self.assignment[vidx]
                lit_val = (not val) if is_neg else val
                if lit_val:
                    sat = True
                    break
            clause_results.append(sat)
        self.global_satisfied = all(clause_results)

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

        if name == "get_counts":
            obs = f"COUNTS: variables={self.num_variables}, clauses={self.num_clauses}."
            reward = 0.0

        elif name == "get_formula":
            parts = []
            for idx, clause in enumerate(self.clauses, start=1):
                lit_strs = []
                for (vidx, is_neg) in clause:
                    vname = self.variables[vidx]
                    lit_strs.append(("¬" if is_neg else "") + vname)
                parts.append(f"{idx}: (" + " OR ".join(lit_strs) + ")")
            obs = "FORMULA: " + "; ".join(parts)
            reward = 0.0

        elif name == "test_clause":
            cid_raw = parsed.get("id", None)
            if cid_raw is None:
                obs = "PARAMETER ERROR: 'id' is required for test_clause."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            try:
                cid = int(cid_raw)
            except:
                obs = "PARAMETER ERROR: 'id' must be an integer."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            if cid < 1 or cid > self.num_clauses:
                obs = f"PARAMETER ERROR: clause id out of range [1..{self.num_clauses}]."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

            idx = cid - 1
            lits = self.clauses[idx]
            sat = False
            for (vidx, is_neg) in lits:
                val = self.assignment[vidx]
                lit_val = (not val) if is_neg else val
                if lit_val:
                    sat = True
                    break
            self.tested_clauses.add(cid)
            obs = f"CLAUSE {cid} RESULT: " + ("SATISFIED" if sat else "UNSATISFIED")
            reward = 0.0

        elif name == "ask_var":
            target = parsed.get("id", None)
            if target is None:
                target = parsed.get("var", None)
            if target is None:
                obs = "PARAMETER ERROR: provide 'id' or 'var=x<i>' for ask_var."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

            if isinstance(target, str) and target.lower().startswith("x"):
                target_num = target[1:]
            else:
                target_num = target
            try:
                vid = int(target_num)
            except:
                obs = "PARAMETER ERROR: variable index must be integer or var=x<i>."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

            if vid < 1 or vid > self.num_variables:
                obs = f"PARAMETER ERROR: variable id out of range [1..{self.num_variables}]."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

            if self.var_queries_used >= self.var_query_limit:
                obs = "PROTOCOL VIOLATION: variable query limit exceeded."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            val = self.assignment[vid - 1]
            self.var_queries_used += 1
            self.var_revealed[vid] = val
            obs = f"VAR x{vid} = " + ("True" if val else "False") + f" (used {self.var_queries_used}/{self.var_query_limit})"
            reward = 0.0

        elif name == "report":
            ans = parsed.get("answer", "").strip().lower()
            if ans not in ("yes", "no"):
                obs = "PARAMETER ERROR: report requires answer=yes or answer=no."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            guess = (ans == "yes")
            if guess == self.global_satisfied:
                obs = "SUCCESS: your report is correct. Episode ends."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "FAILURE: incorrect report. Episode ends."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "UNSUPPORTED ACTION: use get_counts, get_formula, test_clause, ask_var, or report."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            return f"TIMEOUT: reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}

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
                tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        if self.num_clauses > 0 and self.num_variables > 0:
            choice = random.choice(["get_counts", "get_formula", "test_clause", "ask_var", "report"])
        else:
            choice = "get_counts"
        if choice == "get_counts":
            return r"\boxed{get_counts}"
        if choice == "get_formula":
            return r"\boxed{get_formula}"
        if choice == "test_clause":
            cid = random.randint(1, max(1, self.num_clauses))
            return rf"\boxed{{test_clause id={cid}}}"
        if choice == "ask_var":
            vid = random.randint(1, max(1, self.num_variables))
            if random.random() < 0.5:
                return rf"\boxed{{ask_var id={vid}}}"
            else:
                return rf"\boxed{{ask_var var=x{vid}}}"
        if choice == "report":
            return r"\boxed{report answer=yes}"
        return r"\boxed{get_counts}"


class ClauseWitnessQuestEnvWithFeedback(ClauseWitnessQuestEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed"
            hint = "Wrap actions exactly like \\boxed{test_clause id=1}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["get_counts", "get_formula", "test_clause", "ask_var", "report"]
            hint = "Use one of the supported actions and required parameters."

        elif "parameter error" in text:
            error_type = "ProtocolViolation"
            if "clause id out of range" in text:
                error_detail["violation"] = "clause_index_range"
                hint = f"Choose an id between 1 and {self.num_clauses}."
            elif "variable id out of range" in text:
                error_detail["violation"] = "variable_index_range"
                hint = f"Choose an id between 1 and {self.num_variables}."
            elif "answer" in text and "report" in text:
                error_detail["violation"] = "report_missing_or_bad_answer"
                hint = "Use report answer=yes or report answer=no."
            else:
                error_detail["violation"] = "bad_or_missing_parameter"
                hint = "Check required keys like id=<int> or var=x<int>."

        elif "protocol violation" in text and "limit exceeded" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "var_query_limit_exceeded"
            error_detail["used"] = self.var_queries_used
            error_detail["limit"] = self.var_query_limit
            hint = "Stop asking variables; use test_clause to check satisfaction directly."

        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "max_turns_reached"
            hint = "Act sooner: test a subset of clauses and report when confident."

        elif "failure: incorrect report" in text:
            error_type = "WrongDecision"
            parsed = self._parse_action(action)
            user_ans = None
            if parsed and parsed.get("action", "").lower() == "report":
                user_ans = parsed.get("answer", None)
            error_detail["expected"] = "yes" if self.global_satisfied else "no"
            error_detail["got"] = user_ans
            hint = "Find any unsatisfied clause to justify 'no', or verify all tested clauses before 'yes'."

        elif "success: your report is correct" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        else:
            error_type = "OK"
            if "clause" in text and "result" in text:
                error_detail["outcome"] = "clause_checked"
                hint = "Continue checking more clauses or report if you've found one unsatisfied."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "N": self.num_variables,
                "M": self.num_clauses,
                "tested_clauses": len(self.tested_clauses),
                "var_queries_used": self.var_queries_used,
                "var_query_limit": self.var_query_limit,
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
            "hint": "You may start with \\boxed{get_counts} or directly \\boxed{test_clause id=1}.",
            "turn": 0,
            "state": {
                "N": self.num_variables,
                "M": self.num_clauses,
                "tested_clauses": 0,
                "var_queries_used": 0,
                "var_query_limit": self.var_query_limit,
            },
        }
        return obs, info