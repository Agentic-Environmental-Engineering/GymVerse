from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List

class AlgorithmSATDecisionERROREnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 50,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 50

        # Evolvable parameters
        self.complexity_params = {
            # Number of variables; larger = more combinatorial complexity
            'num_variables': (3, 18),
            # Number of clauses; larger = more constraints to reason about
            'num_clauses': (4, 64),
            # Maximum clause width; larger allows more literals per clause
            'max_clause_len': (2, 5),
            # REVERSED: number of contradictory variable pairs for UNSAT; fewer core contradictions makes UNSAT harder to detect
            'unsat_core_size': (3, 1),
            # Number of distractor clauses (e.g., tautologies/duplicates) to add; more noise = harder
            'distractor_clauses': (0, 24),
            # Percentage chance to generate UNSAT instance; closer to 50% increases ambiguity
            'unsat_percent': (30, 50),
        }
        # Randomization variance
        self.param_variance = {
            'num_variables': 2,        # ~12% of range
            'num_clauses': 6,          # ~10% of range
            'max_clause_len': 0,       # small range; keep stable
            'unsat_core_size': 0,      # small range; deterministic
            'distractor_clauses': 3,   # ~12% of range
            'unsat_percent': 2,        # small variation in class balance
        }

        # Placeholders (set in _apply_complexity_params)
        self.num_variables: int = 0
        self.num_clauses: int = 0
        self.max_clause_len: int = 0
        self.unsat_core_size: int = 0
        self.distractor_clauses: int = 0
        self.unsat_percent: int = 0

        # State
        self.turn_count: int = 0
        self.variables: List[str] = []
        self.clauses: List[List[Tuple[str, bool]]] = []
        self.is_sat: bool = True
        self.witness_assignment: Optional[Dict[str, bool]] = None

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
            # Clamp (supports reversed ranges)
            lo, hi = (min(min_val, max_val), max(min_val, max_val))
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "You are solving a Boolean CNF satisfiability decision task.\n"
            "Goal: determine whether the given CNF formula is SAT (satisfiable) or UNSAT (unsatisfiable).\n"
            "You can inspect variables and clauses, transform views, simulate assignments, verify assignments,\n"
            "and finally submit a terminal decision.\n"
            "\n"
            "Allowed actions (use \\\\boxed{...}):\n"
            "- view_vars\n"
            "- view_clauses\n"
            "- sort_clauses_len\n"
            "- literal_freq\n"
            "- simulate x1=T,x2=F,... (partial or full assignment)\n"
            "- verify x1=T,x2=F,... (must assign all variables)\n"
            "- check_clause <idx> x1=T,x2=F,...\n"
            "- final SAT\n"
            "- final UNSAT\n"
            "\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = max(0, self.max_turns - self.turn_count)
        # Report wrong counts
        num_vars_reported = self.num_variables + random.randint(-1, 2)
        num_clauses_reported = len(self.clauses) + random.randint(-2, 3)
        max_len_reported = self.max_clause_len + random.randint(-1, 1)
        return (
            f"Instance summary:\n"
            f"- Variables ({num_vars_reported}): {', '.join(self.variables)}\n"
            f"- Clauses ({num_clauses_reported} total), max clause length {max_len_reported}\n"
            f"Turns remaining: {remaining}\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.variables = [f"x{i+1}" for i in range(self.num_variables)]
        self.clauses = []
        self.is_sat = False
        self.witness_assignment = None

        # Decide SAT/UNSAT
        unsat_prob = self.unsat_percent / 100.0
        self.is_sat = (random.random() >= unsat_prob)

        if self.is_sat:
            # Generate SAT instance with a guaranteed witness assignment
            assignment = {v: bool(random.getrandbits(1)) for v in self.variables}
            self.witness_assignment = assignment
            target_clauses = self.num_clauses
            for _ in range(target_clauses):
                length = random.randint(1, self.max_clause_len)
                chosen_vars = random.sample(self.variables, min(length, len(self.variables)))
                clause: List[Tuple[str, bool]] = []
                # Ensure at least one literal consistent with witness
                must_var = random.choice(chosen_vars)
                clause.append((must_var, assignment[must_var]))
                for v in chosen_vars:
                    if v == must_var:
                        continue
                    # Add random literal; may include opposing literals
                    lit_val = bool(random.getrandbits(1))
                    clause.append((v, lit_val))
                # Optionally add tautological distractor
                if self.distractor_clauses > 0 and random.random() < 0.2:
                    dv = random.choice(self.variables)
                    clause.append((dv, True))
                    clause.append((dv, False))
                self.clauses.append(self._dedup_clause(clause))
            # Add distractor clauses beyond target
            for _ in range(self.distractor_clauses):
                dv = random.choice(self.variables)
                clause = [(dv, True), (dv, False)]  # tautology
                self.clauses.append(self._dedup_clause(clause))
        else:
            # Generate UNSAT instance via contradictory unit pairs
            core_vars = random.sample(self.variables, min(self.unsat_core_size, len(self.variables)))
            for v in core_vars:
                self.clauses.append([(v, True)])
                self.clauses.append([(v, False)])
            # Fill remaining clauses with random content
            target = max(self.num_clauses - len(self.clauses), 0)
            for _ in range(target):
                length = random.randint(1, self.max_clause_len)
                chosen_vars = random.sample(self.variables, min(length, len(self.variables)))
                clause: List[Tuple[str, bool]] = []
                for v in chosen_vars:
                    lit_val = bool(random.getrandbits(1))
                    clause.append((v, lit_val))
                # Add occasional tautologies as distractors
                if self.distractor_clauses > 0 and random.random() < 0.25:
                    dv = random.choice(self.variables)
                    clause.append((dv, True))
                    clause.append((dv, False))
                self.clauses.append(self._dedup_clause(clause))
            # Add extra distractors
            for _ in range(self.distractor_clauses):
                dv = random.choice(self.variables)
                clause = [(dv, True), (dv, False)]
                self.clauses.append(self._dedup_clause(clause))

        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        a_type = parsed.get("type")
        if a_type == "view_vars":
            obs = f"Variables ({len(self.variables)}): {', '.join(self.variables)}"
            return obs, 0.1, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "view_clauses":
            lines = []
            for i, clause in enumerate(self.clauses):
                # Randomly flip some literals in display
                displayed_clause = []
                for v, sign in clause:
                    if random.random() < 0.3:  # 30% chance to flip sign
                        displayed_clause.append((v, not sign))
                    else:
                        displayed_clause.append((v, sign))
                lines.append(f"{i}: " + self._clause_to_str(displayed_clause))
            obs = "Clauses:\n" + "\n".join(lines)
            return obs, 0.1, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "sort_clauses_len":
            indexed = [(i, clause) for i, clause in enumerate(self.clauses)]
            indexed.sort(key=lambda x: len(x[1]))
            obs_lines = [f"{i} (len={len(cl)}): {self._clause_to_str(cl)}" for i, cl in indexed]
            obs = "Clauses sorted by length:\n" + "\n".join(obs_lines)
            return obs, 0.1, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "literal_freq":
            pos = {v: 0 for v in self.variables}
            neg = {v: 0 for v in self.variables}
            for clause in self.clauses:
                for (v, sign) in clause:
                    if sign:
                        pos[v] += 1
                    else:
                        neg[v] += 1
            lines = [f"{v}: +{pos[v]}, -{neg[v]}" for v in self.variables]
            obs = "Literal frequencies:\n" + "\n".join(lines)
            return obs, 0.1, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "simulate":
            assign = parsed.get("assign", {})
            if assign is None:
                obs = f"Malformed assignment. Use format like simulate x1=T,x2=F"
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            sat_idx, fail_idx, pend_idx = self._evaluate_partial(assign)
            # Add random noise to the counts
            sat_noise = random.randint(-2, 3)
            fail_noise = random.randint(-1, 2)
            obs = (
                "Simulation result:\n"
                f"- satisfied: {max(0, len(sat_idx) + sat_noise)}\n"
                f"- failed: {max(0, len(fail_idx) + fail_noise)}\n"
                f"- pending: {len(pend_idx)}\n"
                f"Failed clause indices: {fail_idx}\n"
                f"Pending clause indices: {pend_idx}"
            )
            return obs, 0.2, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "verify":
            assign = parsed.get("assign", {})
            if assign is None:
                obs = f"Malformed assignment. Use format like verify x1=T,x2=F,..."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if set(assign.keys()) != set(self.variables):
                missing = [v for v in self.variables if v not in assign]
                obs = f"Verification incomplete: missing assignments for {missing}"
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
            all_sat = self._evaluate_full(assign)
            label = "SAT" if all_sat else "NOT_SAT"
            obs = f"Verification: {label}"
            return obs, 0.2, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "check_clause":
            idx = parsed.get("idx")
            assign = parsed.get("assign", {})
            if idx is None or idx < 0 or idx >= len(self.clauses) or assign is None:
                obs = "Malformed check_clause. Usage: check_clause <idx> x1=T,x2=F,..."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            status = self._eval_clause_partial(self.clauses[idx], assign)
            obs = f"Clause {idx} evaluation: {status}"
            return obs, 0.1, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "final":
            submitted = parsed.get("label")
            correct = (submitted == ("SAT" if self.is_sat else "UNSAT"))
            if correct:
                obs = f"Success! Submitted final answer: {submitted}"
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = f"Failed! Submitted final answer: {submitted}. Correct was {'SAT' if self.is_sat else 'UNSAT'}"
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {parsed.get('raw', '')}"
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})"
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        raw = content

        if re.fullmatch(r'(view_vars|view_variables)', content, re.IGNORECASE):
            return {"type": "view_vars", "raw": raw}
        if re.fullmatch(r'view_clauses', content, re.IGNORECASE):
            return {"type": "view_clauses", "raw": raw}
        if re.fullmatch(r'sort_clauses_len', content, re.IGNORECASE):
            return {"type": "sort_clauses_len", "raw": raw}
        if re.fullmatch(r'literal_freq', content, re.IGNORECASE):
            return {"type": "literal_freq", "raw": raw}

        sim_match = re.match(r'^simulate\s+(.+)$', content, re.IGNORECASE)
        if sim_match:
            assign_str = sim_match.group(1).strip()
            parsed_assign = self._parse_assignment(assign_str)
            return {"type": "simulate", "assign": parsed_assign, "raw": raw}

        ver_match = re.match(r'^verify\s+(.+)$', content, re.IGNORECASE)
        if ver_match:
            assign_str = ver_match.group(1).strip()
            parsed_assign = self._parse_assignment(assign_str)
            return {"type": "verify", "assign": parsed_assign, "raw": raw}

        chk_match = re.match(r'^check_clause\s+(\d+)\s+(.+)$', content, re.IGNORECASE)
        if chk_match:
            idx = int(chk_match.group(1))
            assign_str = chk_match.group(2).strip()
            parsed_assign = self._parse_assignment(assign_str)
            return {"type": "check_clause", "idx": idx, "assign": parsed_assign, "raw": raw}

        fin_match = re.match(r'^final\s+(sat|unsat)$', content, re.IGNORECASE)
        if fin_match:
            label = fin_match.group(1).upper()
            return {"type": "final", "label": "SAT" if label == "SAT" else "UNSAT", "raw": raw}

        return {"type": "unsupported", "raw": raw}

    def sample_random_action(self) -> str:
        choices = ["view_vars", "view_clauses", "sort_clauses_len", "literal_freq"]
        if self.variables:
            # Build a random partial assignment
            k = max(1, min(len(self.variables), random.randint(1, max(1, len(self.variables)//2))))
            vars_subset = random.sample(self.variables, k)
            assign = ",".join([f"{v}={'T' if random.random()<0.5 else 'F'}" for v in vars_subset])
            choices += [f"simulate {assign}"]
            # Full verify example
            full_assign = ",".join([f"{v}={'T' if random.random()<0.5 else 'F'}" for v in self.variables])
            choices += [f"verify {full_assign}"]
            # Clause check example
            if self.clauses:
                idx = random.randint(0, len(self.clauses)-1)
                choices += [f"check_clause {idx} {assign}"]
        choices += ["final SAT", "final UNSAT"]
        return f"\\boxed{{{random.choice(choices)}}}"

    def _parse_assignment(self, s: str) -> Optional[Dict[str, bool]]:
        try:
            parts = re.split(r'[,\s]+', s.strip())
            assign: Dict[str, bool] = {}
            for p in parts:
                if not p:
                    continue
                m = re.match(r'^(x\d+)\s*=\s*(T|F|TRUE|FALSE|1|0)$', p, re.IGNORECASE)
                if not m:
                    return None
                var = m.group(1)
                val_str = m.group(2).upper()
                val = True if val_str in ("T", "TRUE", "1") else False
                assign[var] = val
            return assign
        except Exception:
            return None

    def _dedup_clause(self, clause: List[Tuple[str, bool]]) -> List[Tuple[str, bool]]:
        seen = set()
        res = []
        for v, sgn in clause:
            key = (v, sgn)
            if key not in seen:
                seen.add(key)
                res.append((v, sgn))
        return res

    def _clause_to_str(self, clause: List[Tuple[str, bool]]) -> str:
        lits = []
        for v, sgn in clause:
            lits.append(v if sgn else f"~{v}")
        return "(" + " OR ".join(lits) + ")"

    def _lit_value(self, var: str, sign: bool, assign: Dict[str, bool]) -> Optional[bool]:
        if var not in assign:
            return None
        val = assign[var]
        return val if sign else (not val)

    def _eval_clause_partial(self, clause: List[Tuple[str, bool]], assign: Dict[str, bool]) -> str:
        any_true = False
        all_assigned = True
        for (v, s) in clause:
            lv = self._lit_value(v, s, assign)
            if lv is None:
                all_assigned = False
            elif lv:
                any_true = True
        if any_true:
            return "satisfied"
        if all_assigned:
            return "failed"
        return "pending"

    def _evaluate_partial(self, assign: Dict[str, bool]) -> Tuple[List[int], List[int], List[int]]:
        sat_idx, fail_idx, pend_idx = [], [], []
        for i, clause in enumerate(self.clauses):
            status = self._eval_clause_partial(clause, assign)
            if status == "satisfied":
                sat_idx.append(i)
            elif status == "failed":
                fail_idx.append(i)
            else:
                pend_idx.append(i)
        return sat_idx, fail_idx, pend_idx

    def _evaluate_full(self, assign: Dict[str, bool]) -> bool:
        for clause in self.clauses:
            satisfied = False
            for (v, s) in clause:
                val = self._lit_value(v, s, assign)
                if val:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True


class AlgorithmSATDecisionERROREnvWithFeedback(AlgorithmSATDecisionERROREnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{view_vars}"

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: view_vars, view_clauses, sort_clauses_len, literal_freq, simulate ..., verify ..., check_clause idx ..., final SAT/UNSAT"

        elif "malformed assignment" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "assignment_parse_failed"
            hint = "Assignments must look like x1=T,x2=F with variable names xN and values T/F"

        elif "verification incomplete" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "incomplete_verification"
            hint = "Provide values for all variables before using verify"

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["outcome"] = "timeout"
            hint = "Decide earlier; inspect clauses, simulate assignments, then submit \\boxed{final SAT} or \\boxed{final UNSAT}"

        elif "success!" in text and "submitted final answer" in text:
            error_type = "OK"
            m_sub = re.search(r"submitted final answer:\s*(sat|unsat)", text)
            error_detail["submitted"] = m_sub.group(1).upper() if m_sub else None
            error_detail["correct"] = "SAT" if self.is_sat else "UNSAT"
            error_detail["outcome"] = "success"
            hint = "Great. You can optionally validate with a quick verify to be safe."

        elif "failed!" in text and "submitted final answer" in text:
            error_type = "WrongDecision"
            # Extract submitted and correct labels
            submitted = None
            correct = None
            m_sub = re.search(r"submitted final answer:\s*(sat|unsat)", text)
            if m_sub:
                submitted = m_sub.group(1).upper()
            m_cor = re.search(r"correct was\s*(sat|unsat)", text)
            if m_cor:
                correct = m_cor.group(1).upper()
            error_detail["submitted"] = submitted
            error_detail["correct"] = correct
            hint = "The formula likely has the opposite satisfiability; reconsider clause conflicts."

        else:
            error_type = "OK"
            error_detail["outcome"] = "step"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["num_variables"] = getattr(self, "num_variables", None)
            diagnostic["num_clauses"] = len(getattr(self, "clauses", []))
            diagnostic["max_clause_len"] = getattr(self, "max_clause_len", None)
            diagnostic["unsat_core_size"] = getattr(self, "unsat_core_size", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by \\boxed{view_vars} and \\boxed{view_clauses}, then simulate assignments.",
            "turn": 0,
            "num_variables": getattr(self, "num_variables", None),
            "num_clauses": len(getattr(self, "clauses", [])),
            "max_clause_len": getattr(self, "max_clause_len", None),
        }
        return obs, info
