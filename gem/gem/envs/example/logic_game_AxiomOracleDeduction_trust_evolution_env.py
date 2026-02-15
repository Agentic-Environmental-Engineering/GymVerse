from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class AxiomOracleDeductionEnv(Env):
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

        # Evolvable parameters
        self.complexity_params = {
            # Number of boolean variables in the world: larger space is harder
            "num_vars": (3, 8),
            # Number of clauses in the hidden CNF: more clauses = more structured constraint = harder to infer
            "num_clauses": (2, 8),
            # Literals per clause (k in k-CNF): higher k allows more nuanced constraints = harder
            "clause_size": (2, 4),
            # Max allowed ASK queries before penalties dominate; larger budget is easier (REVERSED)
            "ask_budget": (9, 4),  # REVERSED: fewer queries = harder
            # Points per informative ASK; higher = easier guidance (REVERSED)
            "ask_reward": (0.3, 0.1),  # REVERSED: less shaping = harder
            # Success target confidence: number of SAT checks auto-verified for a guess; higher threshold is harder
            "auto_verify_checks": (3, 10),
        }

        # Variance settings
        self.param_variance = {
            "num_vars": 0,            # limited small range – deterministic
            "num_clauses": 1,         # discrete moderate
            "clause_size": 0,         # small range – fixed
            "ask_budget": 1,          # discrete moderate (REVERSED inherently handled in clamp)
            "ask_reward": 0.05,       # small continuous variation
            "auto_verify_checks": 1,  # discrete moderate
        }

        # Placeholder attributes set by _apply_complexity_params
        self.num_vars: int = 0
        self.num_clauses: int = 0
        self.clause_size: int = 0
        self.ask_budget: int = 0
        self.ask_reward: float = 0.0
        self.auto_verify_checks: int = 0

        # State
        self.turn_count: int = 0
        self.hidden_cnf: List[List[int]] = []
        self.truth_table: Dict[Tuple[int, ...], bool] = {}
        self.history: List[str] = []
        self.ask_used: int = 0
        self.score: float = 0.0
        self.seed: Optional[int] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                variance = self.param_variance.get(name, 0)
                if variance > 0:
                    center = center + random.uniform(-variance, variance)
            # Clamp with reversed support
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            if name == "ask_reward":
                actual = max(lo, min(hi, center))
                setattr(self, name, float(actual))
            else:
                actual = int(round(max(lo, min(hi, center))))
                setattr(self, name, actual)

    def _get_instructions(self) -> str:
        return (
            "You are consulting an oracle guarding a hidden logical rule expressed as a CNF formula over boolean variables v1..vN.\n"
            "Goal: Identify a satisfying assignment (a full True/False for all variables) that the oracle will accept as consistent with the hidden CNF.\n"
            "You can:\n"
            "- ASK: Test a partial or full assignment. The oracle returns an entailment-style verdict:\n"
            "  • SAT if at least one extension of your partial assignment satisfies the hidden CNF.\n"
            "  • UNSAT if your partial assignment cannot be extended to satisfy the hidden CNF.\n"
            "  Optionally, ASK can also request a counterexample (one violating clause) if UNSAT is certain.\n"
            "- GUESS: Propose a full assignment; if correct (satisfies hidden CNF), you win immediately.\n"
            "\n"
            "Constraints:\n"
            f"- You have a limited ASK budget. Overusing ASK yields no extra reward beyond the cap.\n"
            "- Each ASK that tightens knowledge (i.e., changes the feasibility verdict relative to prior knowledge) yields a small reward.\n"
            "- A correct GUESS ends the episode with reward 1.0; an incorrect GUESS ends with 0.0.\n"
            "- The episode also ends on timeout.\n"
            "\n"
            "Action format (use \\boxed{...}):\n"
            "- ASK with a partial assignment: \\boxed{ASK v1=T v2=F v3=U}\n"
            "  • Use T for True, F for False, U to omit/unknown. You may specify any subset; unspecified variables are treated as U.\n"
            "  • Optional flag: ce=yes to request a counterexample on UNSAT (e.g., \\boxed{ASK v1=T ce=yes}).\n"
            "- GUESS with a full assignment: \\boxed{GUESS v1=T v2=F v3=T ...}\n"
            "\n"
            "Example actions:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        var_list = " ".join([f"v{i+1}" for i in range(self.num_vars)])
        return (
            f"State:\n"
            f"- Variables: {var_list}\n"
            f"- Turns: {self.turn_count}/{self.max_turns}\n"
            f"- ASK used: {self.ask_used}/{self.ask_budget}\n"
            f"- Score: {round(self.score, 3)}\n"
            f"- History:\n" + ("\n".join([f"  {h}" for h in self.history[-8:]]) if self.history else "  (empty)") +
            "\nEnter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
            self.seed = seed

        self._apply_complexity_params()
        self.turn_count = 0
        self.ask_used = 0
        self.history = []
        self.score = 0.0

        # Generate a satisfiable random CNF with given parameters
        self.hidden_cnf = self._generate_satisfiable_cnf(self.num_vars, self.num_clauses, self.clause_size)
        # Precompute truth table feasibility map for quick ASK evaluation
        self.truth_table = self._precompute_truths(self.num_vars, self.hidden_cnf)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with ASK or GUESS and variable settings."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "").upper()
        if name not in ["ASK", "GUESS"]:
            obs = "UNSUPPORTED ACTION: Use ASK or GUESS."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if name == "ASK":
            assign = self._extract_assignment(parsed)
            if assign is None:
                obs = "PROTOCOL VIOLATION: Invalid variable specification for ASK."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            ce_flag = parsed.get("ce", "no").lower() == "yes"
            sat = self._ask_sat(assign)
            info_line = f"ASK {self._fmt_assign(assign)} -> {'SAT' if sat else 'UNSAT'}"
            if ce_flag and not sat:
                clause = self._find_violated_clause(assign)
                if clause is not None:
                    info_line += f" | counterexample_clause={self._format_clause(clause)}"
            self.history.append(info_line)

            reward = 0.0
            if self.ask_used < self.ask_budget:
                # Reward informative ASK: if it excludes previously possible configurations or confirms feasibility after ambiguity
                informative = self._is_informative(assign)
                if informative:
                    reward = float(self.ask_reward)
                self.ask_used += 1
            obs = f"At turn {self.turn_count}, processed ASK. Result: {'SAT' if sat else 'UNSAT'}."
            # Continue episode
            if self.turn_count >= self.max_turns:
                truncated = True
                terminated = True
                obs = f"Reached max turns ({self.max_turns})."
            self.score += reward
            return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

        if name == "GUESS":
            assign = self._extract_assignment(parsed, require_full=True)
            if assign is None:
                obs = "PROTOCOL VIOLATION: GUESS requires a full assignment for all variables."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            correct = self._full_sat(assign)
            self.history.append(f"GUESS {self._fmt_assign(assign)} -> {'CORRECT' if correct else 'INCORRECT'}")
            if correct:
                obs = "Success! Your full assignment satisfies the hidden CNF."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Failed! Your full assignment does not satisfy the hidden CNF."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        obs = "UNSUPPORTED ACTION: Use ASK or GUESS."
        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

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
        tokens["action"] = parts[0]
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.6:
            # ASK with some variables unspecified
            pieces = ["ASK"]
            for i in range(1, self.num_vars + 1):
                if random.random() < 0.5:
                    val = random.choice(["T", "F", "U"])
                    pieces.append(f"v{i}={val}")
            if len(pieces) == 1:
                pieces.append("v1=U")
            if random.random() < 0.3:
                pieces.append("ce=yes")
            return "\\boxed{" + " ".join(pieces) + "}"
        else:
            # Full GUESS
            pieces = ["GUESS"]
            for i in range(1, self.num_vars + 1):
                val = random.choice(["T", "F"])
                pieces.append(f"v{i}={val}")
            return "\\boxed{" + " ".join(pieces) + "}"

    # ---------- Internal logic helpers ----------
    def _generate_satisfiable_cnf(self, n: int, m: int, k: int) -> List[List[int]]:
        # Choose a random satisfying assignment first
        model = [random.choice([True, False]) for _ in range(n)]
        clauses: List[List[int]] = []
        trials = 0
        while len(clauses) < m and trials < m * 50:
            trials += 1
            lits = set()
            while len(lits) < k:
                var = random.randint(1, n)
                sign = random.choice([True, False])
                lit = var if sign else -var
                lits.add(lit if abs(lit) not in [abs(x) for x in lits] else lit)
                # ensure no duplicate variable with different sign within a clause
                # rebuild lits to avoid conflicting var entries
                cleaned = {}
                for L in lits:
                    cleaned[abs(L)] = L
                lits = set(cleaned.values())
            clause = list(lits)
            if self._satisfies_clause(model, clause):
                clauses.append(clause)
        if len(clauses) < m:
            # Fallback: if not enough satisfying clauses created, pad with unit clauses aligned with model
            for var_idx in range(1, n + 1):
                if len(clauses) >= m:
                    break
                lit = var_idx if model[var_idx - 1] else -var_idx
                clauses.append([lit])
        return clauses

    def _satisfies_clause(self, model: List[bool], clause: List[int]) -> bool:
        for lit in clause:
            var = abs(lit) - 1
            if (lit > 0 and model[var]) or (lit < 0 and not model[var]):
                return True
        return False

    def _precompute_truths(self, n: int, cnf: List[List[int]]) -> Dict[Tuple[int, ...], bool]:
        table: Dict[Tuple[int, ...], bool] = {}
        for mask in range(1 << n):
            asg = tuple(1 if (mask >> i) & 1 else 0 for i in range(n))
            ok = True
            for clause in cnf:
                if not self._satisfies_clause([bool(b) for b in asg], clause):
                    ok = False
                    break
            table[asg] = ok
        return table

    def _ask_sat(self, partial: Dict[int, Optional[bool]]) -> bool:
        for asg, ok in self.truth_table.items():
            if ok and self._compatible(asg, partial):
                return True
        return False

    def _is_informative(self, partial: Dict[int, Optional[bool]]) -> bool:
        # Informative if it rules out or confirms feasibility relative to previous ASK about the same fixed subset
        # Strategy: if we haven't asked the exact same mask and values before, count as informative when it yields UNSAT
        mask_vals = tuple((i, partial.get(i, None)) for i in range(1, self.num_vars + 1))
        key = f"ASK_KEY:{mask_vals}"
        seen_same = any(h.startswith(key) for h in self.history)
        sat = self._ask_sat(partial)
        if not seen_same:
            self.history.append(f"{key}")
            return not sat or random.random() < 0.25  # small chance to reward novel SAT checks
        return False

    def _compatible(self, asg: Tuple[int, ...], partial: Dict[int, Optional[bool]]) -> bool:
        for i in range(1, self.num_vars + 1):
            pv = partial.get(i, None)
            if pv is None:
                continue
            if pv != bool(asg[i - 1]):
                return False
        return True

    def _full_sat(self, full_assign: Dict[int, bool]) -> bool:
        asg = tuple(1 if full_assign[i] else 0 for i in range(1, self.num_vars + 1))
        return self.truth_table.get(asg, False)

    def _extract_assignment(self, tokens: Dict[str, Any], require_full: bool = False) -> Optional[Dict[int, Optional[bool]]]:
        result: Dict[int, Optional[bool]] = {}
        for i in range(1, self.num_vars + 1):
            key = f"v{i}"
            if key in tokens:
                val = tokens[key].upper()
                if val == "T":
                    result[i] = True
                elif val == "F":
                    result[i] = False
                elif val == "U" and not require_full:
                    result[i] = None
                else:
                    return None
            else:
                if require_full:
                    return None
                result[i] = None
        return result

    def _find_violated_clause(self, partial: Dict[int, Optional[bool]]) -> Optional[List[int]]:
        # Find a clause that is impossible to satisfy given partial (all literals falsified under any extension)
        # We check if every literal is currently falsified by partial; if some literal is undecided, clause might still be satisfiable.
        for clause in self.hidden_cnf:
            all_false = True
            undecided = False
            for lit in clause:
                var = abs(lit)
                pv = partial.get(var, None)
                if pv is None:
                    undecided = True
                else:
                    sat_lit = (pv and lit > 0) or ((not pv) and lit < 0)
                    if sat_lit:
                        all_false = False
                        break
            if all_false and not undecided:
                return clause
        return None

    def _format_clause(self, clause: List[int]) -> str:
        return "(" + " OR ".join([f"v{abs(l)}" + ("" if l > 0 else "̄") for l in clause]) + ")"

    def _fmt_assign(self, partial: Dict[int, Optional[bool]]) -> str:
        parts = []
        for i in range(1, self.num_vars + 1):
            v = partial.get(i, None)
            if v is True:
                parts.append(f"v{i}=T")
            elif v is False:
                parts.append(f"v{i}=F")
            else:
                parts.append(f"v{i}=U")
        return " ".join(parts)


class AxiomOracleDeductionEnvWithFeedback(AxiomOracleDeductionEnv):
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
            error_detail["issue"] = "boxed_or_syntax"
            hint = "Use \\boxed{ASK v1=T v2=U} or \\boxed{GUESS v1=T v2=F ...}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["ASK", "GUESS"]
            hint = "Only two actions are supported: ASK for partial checks, GUESS for full assignment."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "ask" in text:
                error_detail["violation"] = "invalid_variable_in_ask"
                hint = "Use T/F/U for ASK; e.g., v1=T v2=U. U means unspecified."
            else:
                error_detail["violation"] = "guess_requires_full"
                hint = "Provide every variable in GUESS using T/F (no U)."

        elif "failed! your full assignment" in text:
            error_type = "WrongDecision"
            error_detail["guess_status"] = "incorrect"
            hint = "Use ASK to narrow possibilities: try fixing a variable and observe SAT/UNSAT to guide next steps."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Balance ASK and GUESS. After a few informative ASKs, commit to a GUESS."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["ask_used"] = getattr(self, "ask_used", None)
            diagnostic["ask_budget"] = getattr(self, "ask_budget", None)
            diagnostic["num_vars"] = getattr(self, "num_vars", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Begin with an ASK that sets one variable to T or F (others U) to test feasibility.",
            "turn": 0,
            "ask_used": 0,
            "ask_budget": self.ask_budget,
            "num_vars": self.num_vars,
        }
        return obs, info