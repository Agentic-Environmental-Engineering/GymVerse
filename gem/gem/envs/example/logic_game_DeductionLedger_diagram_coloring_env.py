from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class DeductionLedgerEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 25,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 25

        # Evolvable parameters
        self.complexity_params = {
            # Number of suspects (items). More suspects = larger search space and more complex deductions.
            "num_suspects": (3, 8),
            # Number of roles (labels). Each role is unique and assigned to exactly one suspect.
            "num_roles": (3, 8),
            # Number of core clues. More clues = more constraints to parse and satisfy. Harder due to combinatorial pruning complexity.
            "num_core_clues": (3, 10),
            # Number of decoy (redundant but consistent) clues. More decoys = more text to sift through, harder reasoning load.
            "num_decoy_clues": (0, 5),
            # REVERSED: number of hints shown (counts basic structure hints). Fewer hints = harder.
            "num_hints": (2, 0),
        }

        self.param_variance = {
            "num_suspects": 1,       # medium range → ±1
            "num_roles": 1,          # medium range → ±1
            "num_core_clues": 1,     # medium range → ±1
            "num_decoy_clues": 1,    # medium range → ±1
            "num_hints": 0,          # tiny range (3 values) → fix at center of chosen level
        }

        # Placeholders set by _apply_complexity_params
        self.num_suspects: int = 0
        self.num_roles: int = 0
        self.num_core_clues: int = 0
        self.num_decoy_clues: int = 0
        self.num_hints: int = 0

        # State
        self.turn_count: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

        # Puzzle content
        self.suspects: List[str] = []
        self.roles: List[str] = []
        self.solution: Dict[str, str] = {}
        self.clues_text: List[str] = []
        self.core_constraints: List[Dict[str, Any]] = []
        self.decoy_constraints: List[Dict[str, Any]] = []
        self.history: List[str] = []

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    actual_value = center_value + random.uniform(-var, var)
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, name, int(round(actual_value)))

        # Ensure feasibility: roles cannot exceed suspects; also ensure at least 2 of each
        self.num_suspects = max(2, self.num_suspects)
        self.num_roles = max(2, min(self.num_roles, self.num_suspects))
        # Ensure enough clues to determine uniqueness at higher levels; enforce minimum
        self.num_core_clues = max(2, self.num_core_clues)
        self.num_decoy_clues = max(0, self.num_decoy_clues)
        self.num_hints = max(0, self.num_hints)

    def _generate_names(self, n: int, prefix: str) -> List[str]:
        return [f"{prefix}{i+1}" for i in range(n)]

    def _random_solution(self):
        roles_shuffled = self.roles[:]
        random.shuffle(roles_shuffled)
        return {s: r for s, r in zip(self.suspects, roles_shuffled)}

    def _make_constraint_text(self, c: Dict[str, Any]) -> str:
        t = c["type"]
        if t == "not_pair":
            return f"Clue: {c['suspect']} is not the {c['role']}."
        if t == "must_pair":
            return f"Clue: Exactly one suspect is the {c['role']}."
        if t == "implication":
            return f"Clue: If {c['suspect']} is the {c['role_if']}, then {c['suspect_then']} is not the {c['role_then']}."
        if t == "either_or":
            return f"Clue: Either {c['suspect_a']} is the {c['role_a']} or {c['suspect_b']} is the {c['role_b']}, but not both."
        if t == "exclusive":
            return f"Clue: No two suspects can share the same role."
        return "Clue: [unrecognized]"

    def _evaluate_constraint(self, assignment: Dict[str, str], c: Dict[str, Any]) -> bool:
        t = c["type"]
        if t == "not_pair":
            s, r = c["suspect"], c["role"]
            if s in assignment and assignment[s] == r:
                return False
            return True
        if t == "must_pair":
            # Redundant with uniqueness but adds clarity: exactly one suspect must have this role
            count = sum(1 for s in assignment if assignment.get(s) == c["role"])
            # If assignment incomplete, allow consistency so far (<=1), full check deferred to final completeness
            if len(assignment) < len(self.suspects):
                return count <= 1
            return count == 1
        if t == "implication":
            si, ri = c["suspect"], c["role_if"]
            st, rt = c["suspect_then"], c["role_then"]
            if si in assignment and assignment[si] == ri:
                if st in assignment and assignment[st] == rt:
                    return False
            return True
        if t == "either_or":
            sa, ra = c["suspect_a"], c["role_a"]
            sb, rb = c["suspect_b"], c["role_b"]
            a_true = (sa in assignment and assignment[sa] == ra)
            b_true = (sb in assignment and assignment[sb] == rb)
            # If incomplete, allow if not both true and at least one can still be true
            if len(assignment) < len(self.suspects):
                if a_true and b_true:
                    return False
                # If none true yet, keep consistency (cannot invalidate yet)
                return True
            return (a_true != b_true)
        if t == "exclusive":
            # Enforce bijection properties incrementally
            # No two suspects share same role
            seen = set()
            for s in assignment:
                r = assignment[s]
                if r in seen:
                    return False
                seen.add(r)
            # And each suspect has at most one role (assignment format already enforces)
            return True
        return True

    def _generate_constraints(self):
        self.core_constraints = []
        self.decoy_constraints = []

        # Always include exclusivity as a global constraint (core)
        self.core_constraints.append({"type": "exclusive"})

        # Derive constraints from a hidden solution to ensure solvable
        # Strategy: create not_pair constraints for some false matches; add implication and either_or for structure
        remaining_clues = self.num_core_clues - 1  # considering 'exclusive' already added

        # Create a set of false pairs to forbid
        false_pairs = []
        for s in self.suspects:
            wrong_roles = [r for r in self.roles if r != self.solution[s]]
            random.shuffle(wrong_roles)
            take = min(len(wrong_roles), max(1, remaining_clues // max(1, len(self.suspects))))
            for r in wrong_roles[:take]:
                false_pairs.append({"type": "not_pair", "suspect": s, "role": r})
        random.shuffle(false_pairs)
        while remaining_clues > 0 and false_pairs:
            self.core_constraints.append(false_pairs.pop())
            remaining_clues -= 1

        # If still need core clues, add structural ones
        if remaining_clues > 0 and len(self.suspects) >= 2:
            # add an implication if possible
            s_list = self.suspects[:]
            random.shuffle(s_list)
            s1, s2 = s_list[0], s_list[1]
            # pick a role that s1 has, and a role s2 doesn't have
            role_if = self.solution[s1]
            role_then = random.choice([r for r in self.roles if r != self.solution[s2]])
            self.core_constraints.append({
                "type": "implication",
                "suspect": s1,
                "role_if": role_if,
                "suspect_then": s2,
                "role_then": role_then
            })
            remaining_clues -= 1

        if remaining_clues > 0 and len(self.suspects) >= 2:
            # add either_or using two different suspects and roles aligned with solution to keep solvable
            s_list = self.suspects[:]
            random.shuffle(s_list)
            sa, sb = s_list[0], s_list[1]
            ra = self.solution[sa]
            # choose rb so that exactly one can be true (pick rb not equal to solution[sb])
            rb = random.choice([r for r in self.roles if r != self.solution[sb]])
            self.core_constraints.append({
                "type": "either_or",
                "suspect_a": sa,
                "role_a": ra,
                "suspect_b": sb,
                "role_b": rb
            })
            remaining_clues -= 1

        # Decoy constraints: consistent facts that don't reduce solution space further
        # e.g., "Exactly one suspect is RoleX" or additional not_pair that are already implied
        decoys: List[Dict[str, Any]] = []
        # must_pair for some roles (already true by bijection)
        for r in self.roles:
            decoys.append({"type": "must_pair", "role": r})
        # extra not_pair consistent with core (avoid contradicting solution)
        for s in self.suspects:
            for r in self.roles:
                if r != self.solution[s]:
                    decoys.append({"type": "not_pair", "suspect": s, "role": r})
        random.shuffle(decoys)
        self.decoy_constraints = decoys[: self.num_decoy_clues]

        # Compose text
        self.clues_text = []
        for c in self.core_constraints + self.decoy_constraints:
            self.clues_text.append(self._make_constraint_text(c))

    def _check_assignment_valid(self, assignment: Dict[str, str]) -> Tuple[bool, List[str]]:
        errors = []
        # Basic well-formedness: suspects subset and roles from known set
        if set(assignment.keys()) != set(self.suspects):
            errors.append("Assignment must provide a role for every suspect exactly once.")
        used_roles = list(assignment.values())
        if any(r not in self.roles for r in used_roles):
            errors.append("Unknown role used in assignment.")
        # Bijection: unique roles
        if len(set(used_roles)) != len(used_roles):
            errors.append("Two or more suspects share the same role.")

        # Constraint checks
        all_constraints = self.core_constraints + self.decoy_constraints
        for c in all_constraints:
            if not self._evaluate_constraint(assignment, c):
                errors.append(f"Violated: {self._make_constraint_text(c)}")

        return (len(errors) == 0), errors

    def _format_state_description(self) -> str:
        lines = []
        lines.append("Puzzle: Deduction Ledger")
        lines.append(f"- Suspects: {', '.join(self.suspects)}")
        lines.append(f"- Roles (each used exactly once): {', '.join(self.roles)}")
        if self.num_hints > 0:
            # Provide generic hints
            base_hints = [
                "Hint: Each role is assigned to exactly one suspect.",
                "Hint: Use clues to eliminate impossible suspect-role pairs.",
                "Hint: Look for implications and exclusivity to prune options.",
            ]
            random.shuffle(base_hints)
            lines.extend(base_hints[: self.num_hints])
        lines.append("Clues:")
        for i, t in enumerate(self.clues_text, 1):
            lines.append(f"  {i}. {t}")
        lines.append("Action format: Use \\boxed{propose S1=RoleX S2=RoleY ...} to submit a complete assignment.")
        lines.append("You may also query structure: \\boxed{query list_suspects} or \\boxed{query list_roles}.")
        return "\n".join(lines)

    def _get_instructions(self) -> str:
        return (
            "You are investigating a logic puzzle. Assign each suspect a unique role so that all clues are satisfied.\n"
            "- Goal: Provide one complete mapping from suspects to roles that satisfies every clue and uses each role exactly once.\n"
            "- Actions:\n"
            "  • Submit a solution: \\boxed{propose S1=RoleA S2=RoleB ...}\n"
            "  • Query suspects: \\boxed{query list_suspects}\n"
            "  • Query roles: \\boxed{query list_roles}\n"
            "- Rules:\n"
            "  • Exactly one role per suspect; each role used exactly once.\n"
            "  • Your proposal must mention every suspect exactly once in key=value pairs.\n"
            f"- You have up to {self.max_turns} turns; only a correct complete proposal yields reward 1.0.\n"
            "Clues: See the task suffix for the numbered clues provided.\n"
            "Action format: Use \\boxed{propose S1=RoleX S2=RoleY ...} or \\boxed{query list_roles}.\n"
        )

    def get_task_suffix(self) -> str:
        return self._format_state_description()

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.terminated = False
        self.truncated = False
        self.history = []

        # Generate puzzle
        self.suspects = self._generate_names(self.num_suspects, "S")
        self.roles = self._generate_names(self.num_roles, "Role")
        # Ensure bijection (same count)
        if len(self.roles) < len(self.suspects):
            # pad roles if needed to maintain bijection (due to clamping)
            needed = len(self.suspects) - len(self.roles)
            extra = [f"RoleX{i+1}" for i in range(needed)]
            self.roles = self.roles + extra
        elif len(self.roles) > len(self.suspects):
            self.roles = self.roles[: len(self.suspects)]

        self.solution = self._random_solution()
        self._generate_constraints()

        obs = self._get_instructions()
        info = {"suffix": self.get_task_suffix()}
        return obs, info

    def evolve(self, delta: int = 1, step_success_rate: Optional[float] = None, **_) -> None:
        """
        Adjust difficulty. If step_success_rate is provided, increase when high, decrease when low.
        Otherwise, apply delta.
        """
        if step_success_rate is not None:
            if step_success_rate >= 0.8:
                self.complexity += 1
            elif step_success_rate <= 0.2:
                self.complexity -= 1
        else:
            self.complexity += delta
        self.complexity = max(1, min(10, self.complexity))
        self._apply_complexity_params()

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            return "Episode already ended.", 0.0, True, self.truncated, {"suffix": self.get_task_suffix()}

        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            self.history.append(obs)
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "")
        if act == "query":
            sub = parsed.get("sub", "")
            if sub == "list_suspects":
                obs = "Suspects: " + ", ".join(self.suspects)
            elif sub == "list_roles":
                obs = "Roles: " + ", ".join(self.roles)
            else:
                obs = "Unsupported query. Use \\boxed{query list_suspects} or \\boxed{query list_roles}."
            self.history.append(obs)
            if self.turn_count >= self.max_turns:
                return f"{obs}\nTIMEOUT: Reached max turns ({self.max_turns}).", 0.0, True, True, {"suffix": self.get_task_suffix()}
            # Unsupported queries should end the step as UnsupportedAction
            if sub not in ("list_suspects", "list_roles"):
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        elif act == "propose":
            # Build assignment from kv pairs
            assignment: Dict[str, str] = {}
            for k, v in parsed.items():
                if k in ("action", "sub"):
                    continue
                # keys should be suspect names, values role names
                assignment[k] = v
            # Validate completeness and correctness
            ok, errors = self._check_assignment_valid(assignment)
            if not ok:
                obs = "FAILED: Proposal violates constraints:\n- " + "\n- ".join(errors)
                self.history.append(obs)
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # Check equality with hidden solution (any satisfying solution is okay; uniqueness ensured by design)
            # Given constraints are derived from a specific solution, satisfying them implies matching solution due to bijection.
            if assignment == self.solution:
                obs = "SUCCESS: Your assignment satisfies all clues and uses each role exactly once."
                self.history.append(obs)
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "FAILED: Proposal does not match the hidden solution."
                self.history.append(obs)
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "Unsupported action. Use 'propose' or 'query'."
            self.history.append(obs)
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        # Support multiple boxed segments; last boxed command wins
        matches = list(re.finditer(r"\\boxed\{(.*?)\}", action, flags=re.DOTALL))
        if not matches:
            return None
        inner = matches[-1].group(1).strip()
        if not inner:
            return None

        parts = inner.split()
        if not parts:
            return None

        tokens: Dict[str, Any] = {}
        cmd_raw = parts[0]
        cmd = cmd_raw.lower()
        # Enforce lowercase commands; uppercase variants are treated as unsupported actions
        if cmd_raw != cmd:
            return {"action": "unsupported"}

        if cmd == "propose":
            tokens["action"] = "propose"
            # Expect pairs like S1=Role2
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    tokens[k] = v
            # Minimal guard: ensure at least one kv
            if len(tokens.keys()) <= 1:
                return None
            return tokens

        if cmd == "query":
            tokens["action"] = "query"
            # support 'query list_suspects' or 'query list_roles'
            if len(parts) >= 2:
                sub = parts[1].strip().lower()
                if sub in ("list_suspects", "list_roles"):
                    tokens["sub"] = sub
                else:
                    tokens["sub"] = sub  # will be rejected later
            else:
                tokens["sub"] = ""
            return tokens

        return {"action": "unsupported"}

    def sample_random_action(self) -> str:
        # 50% query, 50% random propose
        if random.random() < 0.5:
            return r"\boxed{query list_suspects}"
        # random full bijection proposal
        roles = self.roles[:]
        random.shuffle(roles)
        pairs = [f"{s}={r}" for s, r in zip(self.suspects, roles)]
        return r"\boxed{propose " + " ".join(pairs) + "}"


class DeductionLedgerEnvWithFeedback(DeductionLedgerEnv):
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
            error_detail["issue"] = "missing_boxed_or_empty"
            hint = "Wrap your command in \\boxed{...}. For example: \\boxed{query list_roles}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use 'propose' to submit, or 'query list_suspects'/'query list_roles' to inspect."
        elif "unsupported query" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_query"
            hint = "Valid queries: \\boxed{query list_suspects} or \\boxed{query list_roles}."
        elif "failed: proposal violates constraints" in text or "failed: proposal does not match the hidden solution" in text:
            error_type = "WrongDecision"
            # Extract first violated line to give focused advice
            lines = obs.splitlines()
            details = []
            for ln in lines:
                if ln.startswith("- Violated:") or ln.lower().startswith(" - violated:") or "violated:" in ln.lower():
                    details.append(ln.strip("- ").strip())
            if details:
                error_detail["violations"] = details[:3]
            else:
                error_detail["violations"] = ["constraint_violation"]
            hint = "Check exclusivity (unique roles) and eliminate pairs explicitly forbidden by clues."
        elif "timeout: reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act sooner: query once to see options, then propose a complete assignment."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["suspects"] = getattr(self, "suspects", None)
            diagnostic["roles"] = getattr(self, "roles", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by querying suspects or roles, then propose a full mapping.",
            "turn": 0,
            "suspects": getattr(self, "suspects", None),
            "roles": getattr(self, "roles", None),
        }
        return obs, info
