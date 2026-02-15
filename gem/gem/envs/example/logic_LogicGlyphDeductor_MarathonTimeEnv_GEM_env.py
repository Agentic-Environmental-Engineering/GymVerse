from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class LogicGlyphDeductorEnv(Env):
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
            # Number of boolean variables in the puzzle: more variables = more state to infer = harder
            'num_vars': (3, 12),
            # Number of additional redundant pairwise constraints: more clutter to parse = harder
            'noise_constraints': (0, 6),
            # REVERSED: number of allowed direct peeks at variable truth: fewer peeks = harder
            'peek_allowance': (6, 0),
        }

        # Variance settings for parameter randomization
        self.param_variance = {
            'num_vars': 1,           # ±1 around level-based center
            'noise_constraints': 1,  # ±1 around level-based center
            'peek_allowance': 1,     # ±1 around level-based center
        }

        # Placeholder attributes (set in _apply_complexity_params)
        self.num_vars: int = 0
        self.noise_constraints: int = 0
        self.peek_allowance: int = 0

        # Other domain-specific state
        self.turn_count: int = 0
        self.variables: List[str] = []
        self.assignment: Dict[str, bool] = {}
        self.constraints: List[str] = []
        self._pair_constraints_set: Set[Tuple[str, str, str]] = set()
        self.peeks_used: int = 0
        self.history: List[str] = []

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
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        var_list = ", ".join(self.variables)
        constraints_text = "\n".join([f"- {c}" for c in self.constraints])
        return (
            "You are playing Logic Glyph Deductor.\n"
            "Goal: Deduce the full truth assignment of the boolean variables and submit it exactly.\n"
            "Hidden World: A set of variables has a secret True/False assignment consistent with the given constraints.\n"
            "\n"
            f"Variables: {var_list}\n"
            "Given constraints (all are true statements about the hidden world):\n"
            f"{constraints_text if constraints_text else '- (none)'}\n"
            "\n"
            "Available functions:\n"
            "- list_constraints: Reprint the constraints.\n"
            "- query_equal a=VAR b=VAR: Returns whether VAR a and VAR b have equal truth values.\n"
            "- query_diff a=VAR b=VAR: Returns whether VAR a and VAR b have different truth values.\n"
            "- query_count subset=A,B,C: Returns how many in the subset are True (subset must have >=2 vars).\n"
            "- query_parity subset=A,B,C: Returns whether the number of True vars in the subset is odd (subset >=2).\n"
            f"- peek var=VAR: Reveal the exact truth of one variable (uses one peek; peeks available: {max(0, self.peek_allowance - self.peeks_used)}).\n"
            "- submit assignment=A=T,B=F,...: Submit your final full assignment for all variables.\n"
            "\n"
            "Rules:\n"
            "- Use the \\boxed{...} format for all actions.\n"
            "- Variable names must be from the provided set.\n"
            "- Subsets must contain at least two distinct variables.\n"
            "- submit must include every variable exactly once with T or F.\n"
            "- Invalid format or protocol violations terminate the episode.\n"
            "\n"
            "Action format examples:\n"
            "- \\boxed{list_constraints}\n"
            "- \\boxed{query_equal a=A b=B}\n"
            "- \\boxed{query_count subset=A,B,C}\n"
            "- \\boxed{peek var=A}\n"
            "- \\boxed{submit assignment=A=T,B=F,C=T}\n"
            "\n"
            f"Example action: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        peeks_left = max(0, self.peek_allowance - self.peeks_used)
        history_tail = "\n".join(self.history[-5:]) if self.history else "(no actions yet)"
        return (
            f"State:\n"
            f"- Turn: {self.turn_count}/{self.max_turns}\n"
            f"- Variables: {', '.join(self.variables)}\n"
            f"- Peeks left: {peeks_left}\n"
            f"- Recent actions:\n{history_tail}\n"
            "Enter your next action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.peeks_used = 0
        self.history = []
        self._pair_constraints_set = set()

        # Generate variables
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # Avoid ambiguous letters? Keep simple given small N.
        self.variables = [letters[i] for i in range(self.num_vars)]

        # Generate hidden assignment
        self.assignment = {v: bool(random.getrandbits(1)) for v in self.variables}

        # Build a solvable constraint set:
        # - One root fact
        # - A random spanning tree of equal/diff constraints that make all vars deducible
        self.constraints = []
        if self.variables:
            root = self.variables[0]
            root_val = self.assignment[root]
            self.constraints.append(f"{root} is {'True' if root_val else 'False'}")
            # Tree edges
            parents = {root: None}
            for v in self.variables[1:]:
                parent = random.choice(list(parents.keys()))
                parents[v] = parent
                same = (self.assignment[v] == self.assignment[parent])
                rel = "equals" if same else "differs from"
                self.constraints.append(f"{v} {rel} {parent}")
                key = tuple(sorted((v, parent)) + [rel])
                self._pair_constraints_set.add(key)

        # Add noise (redundant) constraints consistent with assignment
        attempts = 0
        added = 0
        while added < self.noise_constraints and attempts < self.noise_constraints * 10:
            attempts += 1
            a, b = random.sample(self.variables, 2)
            rel = "equals" if (self.assignment[a] == self.assignment[b]) else "differs from"
            key = tuple(sorted((a, b)) + [rel])
            if key not in self._pair_constraints_set:
                self.constraints.append(f"{a} {rel} {b}")
                self._pair_constraints_set.add(key)
                added += 1

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "").strip()
        name_lower = name.lower()

        # Supported actions
        supported = {
            "list_constraints",
            "query_equal",
            "query_diff",
            "query_count",
            "query_parity",
            "peek",
            "submit",
        }

        if name_lower not in supported:
            obs = f"UNSUPPORTED ACTION: '{name}'."
            self.history.append(f"Unsupported action: {name}")
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        def valid_var(v: str) -> bool:
            return v in self.variables

        def parse_subset(s: str) -> Optional[List[str]]:
            items = [x.strip() for x in s.split(",") if x.strip()]
            # ensure unique and valid
            if len(items) < 2:
                return None
            if any(not valid_var(x) for x in items):
                return None
            if len(set(items)) != len(items):
                return None
            return items

        # Execute actions
        if name_lower == "list_constraints":
            constraints_text = "\n".join([f"{i+1}) {c}" for i, c in enumerate(self.constraints)]) or "(none)"
            obs = f"CONSTRAINTS:\n{constraints_text}"
            self.history.append("list_constraints")
            # turn limit check after processing
            if self.turn_count >= self.max_turns:
                return f"TIMEOUT: Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if name_lower in ("query_equal", "query_diff"):
            a = parsed.get("a")
            b = parsed.get("b")
            if not a or not b or not valid_var(a) or not valid_var(b) or a == b:
                obs = "PROTOCOL VIOLATION: query_equal/query_diff requires distinct variables a and b from the provided set."
                self.history.append(f"{name_lower} INVALID")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            equal = (self.assignment[a] == self.assignment[b])
            if name_lower == "query_equal":
                obs = f"EQUAL: {a} and {b} are equal: {str(equal)}"
                self.history.append(f"query_equal {a},{b} -> {equal}")
            else:
                diff = not equal
                obs = f"DIFF: {a} and {b} differ: {str(diff)}"
                self.history.append(f"query_diff {a},{b} -> {diff}")
            if self.turn_count >= self.max_turns:
                return f"TIMEOUT: Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if name_lower in ("query_count", "query_parity"):
            subset_raw = parsed.get("subset")
            if not subset_raw:
                obs = "PROTOCOL VIOLATION: query_count/query_parity requires subset=A,B,... with at least two distinct variables."
                self.history.append(f"{name_lower} INVALID")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            items = parse_subset(subset_raw)
            if not items:
                obs = "PROTOCOL VIOLATION: subset must contain >=2 distinct valid variables."
                self.history.append(f"{name_lower} INVALID subset")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            count_true = sum(1 for v in items if self.assignment[v])
            if name_lower == "query_count":
                obs = f"COUNT: Among [{', '.join(items)}], count_true = {count_true}"
                self.history.append(f"query_count [{','.join(items)}] -> {count_true}")
            else:
                parity = (count_true % 2 == 1)
                obs = f"PARITY: Among [{', '.join(items)}], parity_is_odd = {str(parity)}"
                self.history.append(f"query_parity [{','.join(items)}] -> {parity}")
            if self.turn_count >= self.max_turns:
                return f"TIMEOUT: Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if name_lower == "peek":
            var = parsed.get("var")
            if not var or not valid_var(var):
                obs = "PROTOCOL VIOLATION: peek requires var=VAR with a valid variable name."
                self.history.append("peek INVALID")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.peeks_used >= self.peek_allowance:
                obs = "PROTOCOL VIOLATION: No peeks remaining."
                self.history.append("peek DENIED")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            value = self.assignment[var]
            self.peeks_used += 1
            peeks_left = max(0, self.peek_allowance - self.peeks_used)
            obs = f"PEEK: {var} is {str(value)}. Peeks left: {peeks_left}"
            self.history.append(f"peek {var} -> {value}")
            if self.turn_count >= self.max_turns:
                return f"TIMEOUT: Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if name_lower == "submit":
            assign_str = parsed.get("assignment")
            if not assign_str:
                obs = "PROTOCOL VIOLATION: submit requires assignment=A=T,B=F,... including all variables exactly once."
                self.history.append("submit INVALID")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            # Parse assignment mapping
            proposed: Dict[str, bool] = {}
            ok = True
            for part in assign_str.split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    ok = False
                    break
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip().lower()
                if not valid_var(k):
                    ok = False
                    break
                if v in ("t", "true", "1"):
                    proposed[k] = True
                elif v in ("f", "false", "0"):
                    proposed[k] = False
                else:
                    ok = False
                    break
            if not ok:
                obs = "PROTOCOL VIOLATION: Invalid assignment format. Use A=T or A=F for each variable."
                self.history.append("submit INVALID FORMAT")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if set(proposed.keys()) != set(self.variables):
                obs = "PROTOCOL VIOLATION: Assignment must include every variable exactly once."
                self.history.append("submit MISSING/EXTRA VARS")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            correct = all(proposed[v] == self.assignment[v] for v in self.variables)
            if correct:
                obs = "Success! Your assignment matches the hidden truth assignment."
                self.history.append("submit SUCCESS")
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Failed! Submission incorrect."
                self.history.append("submit FAIL")
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Should not reach here
        obs = "UNSUPPORTED ACTION: Internal error."
        self.history.append("internal error")
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
        if not parts:
            return None
        tokens: Dict[str, Any] = {}
        tokens["action"] = parts[0]
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                tokens[key.strip()] = value.strip()
        return tokens

    def sample_random_action(self) -> str:
        if not self.variables:
            return r"\boxed{list_constraints}"
        choices = ["list_constraints", "query_equal", "query_diff", "query_count", "query_parity", "peek"]
        act = random.choice(choices)
        if act == "list_constraints":
            return r"\boxed{list_constraints}"
        if act in ("query_equal", "query_diff"):
            a, b = random.sample(self.variables, 2)
            return rf"\boxed{{{act} a={a} b={b}}}"
        if act in ("query_count", "query_parity"):
            k = random.randint(2, min(4, len(self.variables)))
            subset = ",".join(random.sample(self.variables, k))
            return rf"\boxed{{{act} subset={subset}}}"
        if act == "peek":
            v = random.choice(self.variables)
            return rf"\boxed{{peek var={v}}}"
        return r"\boxed{list_constraints}"


class LogicGlyphDeductorEnvWithFeedback(LogicGlyphDeductorEnv):
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
            hint = "Wrap your command in \\boxed{...} and include required parameters."

        elif text.startswith("unsupported action"):
            error_type = "UnsupportedAction"
            error_detail["allowed"] = [
                "list_constraints", "query_equal", "query_diff",
                "query_count", "query_parity", "peek", "submit"
            ]
            hint = "Use one of the supported actions and match the parameter names."

        elif text.startswith("protocol violation"):
            error_type = "ProtocolViolation"
            if "requires distinct variables a and b" in text:
                error_detail["violation"] = "bad_pair_params"
                hint = "Provide a and b as two different valid variable names, e.g., a=A b=B."
            elif "subset must contain >=2" in text:
                error_detail["violation"] = "bad_subset"
                hint = "Use subset=A,B,... with at least two distinct valid variables."
            elif "peek requires var" in text:
                error_detail["violation"] = "bad_peek_param"
                hint = "Use peek var=V where V is one of the listed variables."
            elif "no peeks remaining" in text:
                error_detail["violation"] = "peek_limit"
                hint = "You have no peeks left. Use query_equal/diff or parity/count to deduce values."
            elif "submit requires assignment" in text:
                error_detail["violation"] = "missing_assignment_arg"
                hint = "Include assignment=A=T,B=F,... covering all variables."
            elif "invalid assignment format" in text:
                error_detail["violation"] = "bad_assignment_value"
                hint = "Use T or F for each variable, for example A=T."
            elif "must include every variable" in text:
                error_detail["violation"] = "incomplete_assignment"
                hint = "Include each variable exactly once in the assignment."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Check parameter names and values against the instructions."

        elif text.startswith("failed! submission incorrect"):
            error_type = "WrongDecision"
            # Avoid revealing the true assignment; provide process hint
            error_detail["outcome"] = "incorrect_submission"
            hint = "Revisit constraints, compare variables via query_equal/query_diff, and use query_count or query_parity to validate groups. Submit only when confident."

        elif text.startswith("timeout"):
            error_type = "Timeout"
            error_detail["outcome"] = "turn_limit_reached"
            hint = "Plan queries: first map relations using query_equal/diff, then use counts/parity, and reserve time for submit."

        elif text.startswith("success!"):
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        else:
            error_type = "OK"
            error_detail["outcome"] = "step_ok"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["peeks_left"] = max(0, self.peek_allowance - self.peeks_used)
            diagnostic["variables"] = list(self.variables)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by reading constraints (list_constraints). Then sketch relationships with query_equal/query_diff.",
            "turn": 0,
            "peeks_left": max(0, self.peek_allowance - self.peeks_used),
            "variables": list(self.variables),
        }
        return obs, info