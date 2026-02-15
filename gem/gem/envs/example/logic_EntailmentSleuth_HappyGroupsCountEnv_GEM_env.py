from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class EntailmentSleuthEnv(Env):
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
            # Number of propositional variables: more variables -> larger valuation space -> harder
            'num_vars': (3, 9),
            # Number of premises: more premises -> more structure to analyze -> harder
            'num_premises': (1, 6),
            # Average number of binary ops (AND/OR) per premise: more ops -> deeper/wider trees -> harder
            'avg_ops_per_premise': (1, 5),
            # Number of binary ops in the conclusion: more ops -> more complex target -> harder
            'ops_in_conclusion': (1, 6),
            # Percentage chance to negate a variable occurrence: higher -> more intricate logic -> harder
            'negation_density': (20, 60),
        }

        # Variance settings
        self.param_variance = {
            'num_vars': 1,
            'num_premises': 1,
            'avg_ops_per_premise': 1,
            'ops_in_conclusion': 1,
            'negation_density': 5,
        }

        # Placeholder attributes populated by _apply_complexity_params
        self.num_vars: int = 0
        self.num_premises: int = 0
        self.avg_ops_per_premise: int = 0
        self.ops_in_conclusion: int = 0
        self.negation_density: int = 0

        # State
        self.turn_count: int = 0
        self.vars: List[str] = []
        self.premises: List[Any] = []
        self.conclusion: Any = None
        self.is_valid: bool = False
        self.known_vars_shown: bool = False
        self.revealed_premise_indices: Set[int] = set()
        self.revealed_conclusion: bool = False
        self.last_eval_summary: Optional[str] = None
        self.last_eval_counterexample: Optional[bool] = None
        self.last_eval_assignment: Optional[Dict[str, bool]] = None
        self.last_submission_guess: Optional[str] = None

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
                    low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _random_var_names(self, count: int) -> List[str]:
        base = ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
        if count <= len(base):
            return base[:count]
        # Fallback to numbered names if needed
        extra = [f'x{i}' for i in range(count - len(base))]
        return base + extra

    def _rand_literal(self, var: str) -> Any:
        if random.randint(1, 100) <= self.negation_density:
            return ('not', var)
        return var

    def _rand_formula(self, variables: List[str], target_ops: int) -> Any:
        # Build a random binary tree with target_ops internal nodes; leaves are literals.
        # If target_ops == 0 -> single literal.
        def rand_leaf():
            return self._rand_literal(random.choice(variables))

        if target_ops <= 0:
            return rand_leaf()

        # Start with a pool of leaves
        leaves = [rand_leaf() for _ in range(target_ops + 1)]
        ops_used = 0
        nodes = leaves[:]
        while ops_used < target_ops:
            # Pick two nodes to combine
            if len(nodes) < 2:
                nodes.append(rand_leaf())
            a = nodes.pop(random.randrange(len(nodes)))
            b = nodes.pop(random.randrange(len(nodes)))
            op = random.choice(['and', 'or'])
            nodes.append((op, a, b))
            ops_used += 1
        assert len(nodes) >= 1
        # If more than one node remains, combine them arbitrarily
        while len(nodes) > 1:
            a = nodes.pop()
            b = nodes.pop()
            op = random.choice(['and', 'or'])
            nodes.append((op, a, b))
        return nodes[0]

    def _eval_formula(self, node: Any, assignment: Dict[str, bool]) -> bool:
        if isinstance(node, str):
            return assignment[node]
        if isinstance(node, tuple):
            if node[0] == 'not':
                return not self._eval_formula(node[1], assignment)
            if node[0] == 'and':
                return self._eval_formula(node[1], assignment) and self._eval_formula(node[2], assignment)
            if node[0] == 'or':
                return self._eval_formula(node[1], assignment) or self._eval_formula(node[2], assignment)
        return False

    def _formula_str(self, node: Any) -> str:
        if isinstance(node, str):
            return node
        op = node[0]
        if op == 'not':
            inner = self._formula_str(node[1])
            # Avoid double parens for variables
            if isinstance(node[1], str):
                return f"NOT {inner}"
            return f"NOT ({inner})"
        left = self._formula_str(node[1])
        right = self._formula_str(node[2])
        if op == 'and':
            return f"({left} AND {right})"
        if op == 'or':
            return f"({left} OR {right})"
        return "?"

    def _all_assignments(self, variables: List[str]) -> List[Dict[str, bool]]:
        n = len(variables)
        res = []
        for mask in range(1 << n):
            assignment = {}
            for i, v in enumerate(variables):
                assignment[v] = bool((mask >> i) & 1)
            res.append(assignment)
        return res

    def _compute_entailment(self) -> bool:
        # Γ |= φ iff for all assignments, if all premises true then conclusion true
        for assignment in self._all_assignments(self.vars):
            gamma_true = all(self._eval_formula(p, assignment) for p in self.premises) if self.premises else True
            if gamma_true:
                phi_true = self._eval_formula(self.conclusion, assignment)
                if not phi_true:
                    return False
        return True

    def _get_instructions(self) -> str:
        lines = []
        lines.append("You are solving a propositional entailment mystery: decide if Γ entails φ (Γ |= φ).")
        lines.append("You may reveal structure and test valuations before submitting your final decision.")
        lines.append("")
        lines.append("Available actions (use \\boxed{...}):")
        lines.append("- vars: reveal the list of propositional variables.")
        lines.append("- premise index=<i>: reveal the i-th premise (1-based).")
        lines.append("- premises: reveal all premises.")
        lines.append("- conclusion: reveal the conclusion.")
        lines.append("- eval assignment=v1=TRUE,v2=F,...: evaluate Γ and φ under a full assignment.")
        lines.append("  Accepted booleans: TRUE/FALSE, T/F, 1/0 (case-insensitive). Must provide ALL variables.")
        lines.append("- submit answer=valid|invalid: submit your final judgment about Γ |= φ.")
        lines.append("- help: reprint these instructions.")
        lines.append("")
        lines.append("Important:")
        lines.append("- Each action must be enclosed exactly as \\boxed{action_name ...}.")
        lines.append("- Invalid or malformed actions terminate the episode.")
        lines.append("- Rewards: intermediate steps yield 0.0; correct submit yields 1.0; incorrect yields 0.0;")
        lines.append("  format errors yield a special negative format reward.")
        lines.append("")
        lines.append("Example actions:")
        lines.append(rf"- {r'\boxed{vars}'}")
        lines.append(rf"- {r'\boxed{premise index=1}'}")
        lines.append(rf"- {r'\boxed{eval assignment=p=T,q=F,r=T}'}")
        lines.append(rf"- {r'\boxed{submit answer=valid}'}")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        revealed_premises = sorted(list(self.revealed_premise_indices))
        vars_status = "known" if self.known_vars_shown else "hidden"
        conc_status = "revealed" if self.revealed_conclusion else "hidden"
        last_eval = self.last_eval_summary if self.last_eval_summary else "none"
        lines = []
        lines.append(f"Turn {self.turn_count}/{self.max_turns}")
        lines.append(f"Premises: {self.num_premises} total; revealed indices: {revealed_premises if revealed_premises else 'none'}")
        lines.append(f"Conclusion: {conc_status}")
        lines.append(f"Variables: {vars_status}")
        lines.append(f"Last eval: {last_eval}")
        lines.append("Enter your action in \\boxed{...} format.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.known_vars_shown = False
        self.revealed_premise_indices = set()
        self.revealed_conclusion = False
        self.last_eval_summary = None
        self.last_eval_counterexample = None
        self.last_eval_assignment = None
        self.last_submission_guess = None

        self.vars = self._random_var_names(self.num_vars)
        # Generate premises
        self.premises = []
        for _ in range(self.num_premises):
            ops = max(0, int(round(random.gauss(self.avg_ops_per_premise, 0.8))))
            ops = max(0, min(ops, max(1, self.avg_ops_per_premise * 2)))
            self.premises.append(self._rand_formula(self.vars, ops))
        # Generate conclusion
        conc_ops = self.ops_in_conclusion
        self.conclusion = self._rand_formula(self.vars, conc_ops)

        self.is_valid = self._compute_entailment()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

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
        tokens: Dict[str, Any] = {'action': parts[0].lower()}
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                tokens[k.lower()] = v
            else:
                # allow bare flags like 'conclusion' if we used a different verb, but we keep simple
                pass
        return tokens

    def _parse_bool(self, s: str) -> Optional[bool]:
        s = s.strip().lower()
        if s in ('t', 'true', '1'):
            return True
        if s in ('f', 'false', '0'):
            return False
        return None

    def _parse_assignment(self, text: str) -> Tuple[Optional[Dict[str, bool]], str]:
        if text is None:
            return None, "MISSING PARAMETER: 'assignment' required for 'eval'."
        pairs = [p for p in text.split(',') if p.strip() != ""]
        if not pairs:
            return None, "MALFORMED ASSIGNMENT: expected comma-separated name=bool."
        assign: Dict[str, bool] = {}
        for pair in pairs:
            if '=' not in pair:
                return None, "MALFORMED ASSIGNMENT: expected comma-separated name=bool."
            name, val = pair.split('=', 1)
            name = name.strip()
            if name == "":
                return None, "MALFORMED ASSIGNMENT: empty variable name."
            if name not in self.vars:
                return None, f"UNKNOWN VARIABLE: '{name}' is not in current variable set."
            b = self._parse_bool(val)
            if b is None:
                return None, f"MALFORMED ASSIGNMENT: value for '{name}' must be boolean."
            assign[name] = b
        missing = [v for v in self.vars if v not in assign]
        if missing:
            return None, f"INCOMPLETE ASSIGNMENT: provide values for all variables: {missing}."
        return assign, ""

    def _reveal_premise_text(self, idx: int) -> str:
        f = self.premises[idx]
        return self._formula_str(f)

    def _reveal_premises_text(self) -> str:
        lines = []
        for i, f in enumerate(self.premises, start=1):
            lines.append(f"Premise {i}: {self._formula_str(f)}")
        return "\n".join(lines) if lines else "(no premises)"

    def _conclusion_text(self) -> str:
        return self._formula_str(self.conclusion)

    def sample_random_action(self) -> str:
        choices = []
        choices.append(r'\boxed{vars}')
        if self.num_premises > 0:
            choices.append(r'\boxed{premise index=1}')
            choices.append(r'\boxed{premises}')
        choices.append(r'\boxed{conclusion}')
        # Build a random assignment example
        if self.vars:
            parts = []
            for v in self.vars:
                parts.append(f"{v}={'T' if random.random()>0.5 else 'F'}")
            choices.append(rf"\boxed{{eval assignment={','.join(parts)}}}")
        choices.append(r'\boxed{submit answer=valid}')
        return random.choice(choices)

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

        if name == 'vars':
            self.known_vars_shown = True
            obs = "Variables: " + ", ".join(self.vars)
        elif name == 'premise':
            if 'index' not in parsed:
                obs = "MISSING PARAMETER: 'index' required for 'premise'."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            try:
                idx = int(parsed['index'])
            except ValueError:
                obs = "MALFORMED PARAMETER: 'index' must be an integer."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if not (1 <= idx <= self.num_premises):
                obs = f"OUT-OF-RANGE INDEX: valid range is 1..{self.num_premises}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.revealed_premise_indices.add(idx)
            obs = f"Premise {idx}: {self._reveal_premise_text(idx-1)}"
        elif name == 'premises':
            self.revealed_premise_indices = set(range(1, self.num_premises + 1))
            obs = self._reveal_premises_text()
        elif name == 'conclusion':
            self.revealed_conclusion = True
            obs = "Conclusion: " + self._conclusion_text()
        elif name == 'eval':
            assignment_raw = parsed.get('assignment')
            assign, err = self._parse_assignment(assignment_raw)
            if assign is None:
                obs = err
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            gamma_true = all(self._eval_formula(p, assign) for p in self.premises) if self.premises else True
            phi_true = self._eval_formula(self.conclusion, assign)
            counterexample = (gamma_true and not phi_true)
            self.last_eval_assignment = assign
            self.last_eval_counterexample = counterexample
            self.last_eval_summary = f"Γ={gamma_true}, φ={phi_true}, counterexample={counterexample}"
            obs = f"Evaluation: Γ={gamma_true}, φ={phi_true}. " \
                  f"{'Counterexample found.' if counterexample else 'No counterexample under this assignment.'}"
        elif name == 'help':
            obs = self._get_instructions()
        elif name == 'submit':
            ans = parsed.get('answer', '').strip().lower()
            if ans in ('valid', 'entailed', 'yes'):
                guess = True
            elif ans in ('invalid', 'not_entailed', 'no'):
                guess = False
            else:
                obs = "MALFORMED PARAMETER: 'answer' must be 'valid' or 'invalid'."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.last_submission_guess = 'valid' if guess else 'invalid'
            if guess == self.is_valid:
                obs = "Success: your answer 'valid' matches ground truth (Γ |= φ)." if guess else \
                      "Success: your answer 'invalid' matches ground truth (Γ ⊭ φ)."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                truth = 'valid' if self.is_valid else 'invalid'
                obs = f"Incorrect submission: ground truth is {truth}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"UNSUPPORTED ACTION: '{name}'. Use one of: vars, premise, premises, conclusion, eval, submit, help."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"Reached max turns ({self.max_turns})"
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}


class EntailmentSleuthEnvWithFeedback(EntailmentSleuthEnv):
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
            hint = "Wrap actions as \\boxed{action_name ...} exactly."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_function"
            error_detail["allowed"] = ["vars", "premise", "premises", "conclusion", "eval", "submit", "help"]
            hint = "Use one of the supported actions listed in the instructions."
        elif ("missing parameter" in text or
              "malformed parameter" in text or
              "malformed assignment" in text or
              "unknown variable" in text or
              "out-of-range index" in text or
              "incomplete assignment" in text):
            error_type = "ProtocolViolation"
            if "index" in text and ("missing parameter" in text or "malformed parameter" in text):
                error_detail["violation"] = "premise_index_param"
                hint = "Specify the premise index like \\boxed{premise index=2}."
            elif "out-of-range index" in text:
                error_detail["violation"] = "index_out_of_range"
                hint = "Use an index within the displayed total number of premises."
            elif "unknown variable" in text:
                error_detail["violation"] = "unknown_variable_in_assignment"
                hint = "Call \\boxed{vars} and ensure assignment uses only listed variables."
            elif "incomplete assignment" in text:
                error_detail["violation"] = "incomplete_assignment"
                missing = obs[obs.find('['):] if '[' in obs else ""
                error_detail["missing"] = missing
                hint = "Provide all variables: e.g., \\boxed{eval assignment=p=T,q=F,...}."
            elif "malformed assignment" in text:
                error_detail["violation"] = "malformed_assignment"
                hint = "Use name=bool pairs separated by commas, e.g., \\boxed{eval assignment=p=T,q=F}."
            elif "missing parameter" in text and "assignment" in text:
                error_detail["violation"] = "missing_assignment_param"
                hint = "Provide the assignment, e.g., \\boxed{eval assignment=p=T,q=F}."
            else:
                hint = "Follow the parameter formats shown in the instructions."
        elif "incorrect submission" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = "valid" if self.is_valid else "invalid"
            error_detail["got"] = getattr(self, "last_submission_guess", None)
            hint = "Search for a counterexample via eval; if none exist, submit 'valid'. If you find one, submit 'invalid'."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act earlier: list variables, reveal formulas, test key assignments, then submit."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_vars": getattr(self, "num_vars", None),
                "num_premises": getattr(self, "num_premises", None),
                "revealed_premises": sorted(list(getattr(self, "revealed_premise_indices", set()))),
                "conclusion_revealed": getattr(self, "revealed_conclusion", False),
                "last_eval_counterexample": getattr(self, "last_eval_counterexample", None),
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
            "hint": "Begin with \\boxed{vars}, then \\boxed{conclusion} or \\boxed{premises}, and try an \\boxed{eval ...}.",
            "turn": 0,
            "state": {
                "num_vars": self.num_vars,
                "num_premises": self.num_premises,
                "revealed_premises": [],
                "conclusion_revealed": False,
                "last_eval_counterexample": None,
            }
        }
        return obs, info