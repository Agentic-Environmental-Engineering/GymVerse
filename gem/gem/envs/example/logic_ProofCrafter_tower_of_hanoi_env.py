from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class ProofCrafterEnv(Env):
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

        # Evolvable parameters
        self.complexity_params = {
            # number of distinct atomic symbols available; more symbols → larger search space
            "num_atoms": (2, 6),
            # number of given premises; more premises can increase branching and distractors → harder
            "num_premises": (2, 6),
            # number of distractor premises (irrelevant ones) → more noise makes it harder
            "num_distractors": (0, 4),
            # target proof depth (minimum steps required); deeper → harder
            "target_min_steps": (2, 6),
            # allowable rule set size; more rules → more choices → harder
            "rule_set_size": (3, 6),
        }

        # Variance settings
        self.param_variance = {
            "num_atoms": 1,
            "num_premises": 1,
            "num_distractors": 1,
            "target_min_steps": 1,
            "rule_set_size": 1,
        }

        # Placeholder evolvable attributes
        self.num_atoms: int = 0
        self.num_premises: int = 0
        self.num_distractors: int = 0
        self.target_min_steps: int = 0
        self.rule_set_size: int = 0

        # Fixed rule catalog (subset chosen per episode)
        self.rule_catalog = ["AND_INTRO", "AND_ELIM_L", "AND_ELIM_R", "MP", "OR_INTRO", "CHAIN"]

        # State
        self.turn_count: int = 0
        self.available_atoms: List[str] = []
        self.premises: List[str] = []
        self.distractors: List[str] = []
        self.known: Set[str] = set()
        self.target: str = ""
        self.rule_set: Set[str] = set()
        self.history: List[str] = []
        self.required_min_steps: int = 0
        self.done: bool = False

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

    def _random_atom(self) -> str:
        return random.choice(self.available_atoms)

    def _make_and(self, a: str, b: str) -> str:
        return f"({a}&{b})"

    def _make_or(self, a: str, b: str) -> str:
        return f"({a}∨{b})"

    def _make_imp(self, a: str, b: str) -> str:
        return f"({a}->{b})"

    def _gen_chain(self, length: int) -> List[str]:
        # Generate a -> b -> c -> ... implication chain with distinct atoms
        chain_atoms = random.sample(self.available_atoms, k=min(length + 1, len(self.available_atoms)))
        prems = []
        for i in range(len(chain_atoms) - 1):
            prems.append(self._make_imp(chain_atoms[i], chain_atoms[i + 1]))
        return prems

    def _pick_rule_set(self) -> Set[str]:
        # Always include MP; fill others up to rule_set_size
        rules = set(["MP"])
        others = [r for r in self.rule_catalog if r != "MP"]
        random.shuffle(others)
        for r in others:
            if len(rules) >= self.rule_set_size:
                break
            rules.add(r)
        return rules

    def _gen_premises_and_target(self):
        # Ensure solvable target with at least target_min_steps using the selected rule set
        # Strategy:
        # - If CHAIN present and enough atoms, build a chain of length target_min_steps, target is last atom
        # - Else if AND_INTRO and AND_ELIM available, create need to build conjunction then eliminate
        # - Else rely on MP tree
        rules = self.rule_set
        n_steps = self.target_min_steps
        atoms = self.available_atoms[:]

        base_prems: List[str] = []
        target = ""

        if "CHAIN" in rules and n_steps >= 3 and len(atoms) >= n_steps + 1:
            chain_prems = self._gen_chain(n_steps)
            base_prems.extend(chain_prems)
            target = re.findall(r"->\((.*?)\)", self._make_imp("X", re.findall(r"\((.*?)\)->", chain_prems[-1])[0]))  # placeholder
            # Actually compute target as last atom
            last_atom = re.findall(r"\)->\((.*?)\)", "dummy")
            target_atom = chain_prems[-1].split("->")[-1].strip("()")
            target = target_atom
            # Add starting atom premise
            start_atom = chain_prems[0].split("->")[0].strip("()")
            base_prems.append(start_atom)
            needed_steps = len(chain_prems)  # chain length applications + 0 for start atom
            if needed_steps < n_steps:
                # Add an extra MP step: (target->Z) and want Z
                if "MP" in rules:
                    z = random.choice([a for a in atoms if a != target])
                    base_prems.append(self._make_imp(target, z))
                    target = z
                    needed_steps += 1
        elif "AND_INTRO" in rules and "AND_ELIM_L" in rules and "MP" in rules and n_steps >= 3:
            a, b, c = random.sample(atoms, 3) if len(atoms) >= 3 else (atoms[0], atoms[0], atoms[-1])
            # We will need to construct (a&b), use MP with (a&b->c) to derive c
            base_prems.append(a)
            base_prems.append(b)
            base_prems.append(self._make_imp(self._make_and(a, b), c))
            target = c
        else:
            # Pure MP chain with implications to reach target
            # Build simple chain length n_steps: p0, (p0->p1), (p1->p2)... target=pN
            length = max(2, n_steps)
            chain = self._gen_chain(min(length, max(2, len(atoms) - 1)))
            base_prems.extend(chain)
            start_atom = chain[0].split("->")[0].strip("()")
            base_prems.append(start_atom)
            target = chain[-1].split("->")[-1].strip("()")

        # Trim or pad premises to match num_premises using implications built from atoms
        core = list(dict.fromkeys(base_prems))  # remove dup while preserving order
        while len(core) < self.num_premises:
            a = self._random_atom()
            b = self._random_atom()
            if a == b:
                continue
            candidate = self._make_imp(a, b)
            if candidate not in core:
                core.append(candidate)
        if len(core) > self.num_premises:
            core = core[: self.num_premises]

        # Distractors: random implications or ors not necessary for proof
        distractors: List[str] = []
        while len(distractors) < self.num_distractors:
            a = self._random_atom()
            b = self._random_atom()
            if random.random() < 0.5:
                cand = self._make_or(a, b)
            else:
                cand = self._make_imp(a, b)
            if cand not in core and cand not in distractors and cand != target:
                distractors.append(cand)

        return core, distractors, target

    def _get_instructions(self) -> str:
        return (
            "You are in ProofCrafter, a constructive propositional logic task.\n"
            "Goal: derive the target proposition using allowed inference rules from given premises.\n"
            "Rules available this episode:\n"
            f"- {', '.join(sorted(self.rule_set))}\n"
            "Rule semantics:\n"
            "- MP: Modus Ponens. From X and (X->Y), derive Y.\n"
            "- AND_INTRO: From X and Y, derive (X&Y).\n"
            "- AND_ELIM_L: From (X&Y), derive X.\n"
            "- AND_ELIM_R: From (X&Y), derive Y.\n"
            "- OR_INTRO: From X, derive (X∨Y) for any Y.\n"
            "- CHAIN: From (X->Y) and (Y->Z), derive (X->Z).\n"
            "Action format: use \\boxed{...} with a single command per turn.\n"
            "Commands:\n"
            "- use_premise idx=k            (adds premise k to known)\n"
            "- apply rule=RULE args=A,B     (apply RULE using known statements A,B or single A)\n"
            "- args syntax: refer by exact formula or atom shown in Known/Premises; commas separate two args.\n"
            "- You may also use syntactic constructors in args like (A&B), (A->B), (A∨B).\n"
            "- End when you have derived the Target.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append("Premises (index:start at 1):")
        for i, p in enumerate(self.premises, start=1):
            lines.append(f"  {i}. {p}")
        if self.distractors:
            lines.append("Distractors:")
            for d in self.distractors:
                lines.append(f"  - {d}")
        lines.append("Known:")
        if self.known:
            for k in sorted(self.known):
                lines.append(f"  • {k}")
        else:
            lines.append("  • (empty)")
        lines.append(f"Target: {self.target}")
        lines.append("Enter your action in \\boxed{...} format.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.done = False

        # Setup atoms
        self.available_atoms = [chr(ord("P") + i) for i in range(self.num_atoms)]

        # Pick rule set
        self.rule_set = self._pick_rule_set()

        # Generate premises and target
        self.premises, self.distractors, self.target = self._gen_premises_and_target()
        self.required_min_steps = self.target_min_steps

        # Initialize known as empty
        self.known = set()
        self.history = []

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _parse_formula(self, s: str) -> str:
        # Lightweight sanity trim; trust strings as-is for matching
        return s.strip()

    def _parse_args(self, s: str) -> List[str]:
        parts = [p.strip() for p in s.split(",")]
        return [self._parse_formula(p) for p in parts if p]

    def _apply_rule(self, rule: str, args: List[str]) -> Tuple[bool, str, str]:
        # Returns (success, derived, message)
        rule = rule.upper()
        known = set(self.known).union(set(self.premises))  # allow using raw premises by name too
        if rule not in self.rule_set:
            return False, "", f"RULE_NOT_AVAILABLE: {rule}"

        def has(stmt: str) -> bool:
            return stmt in self.known

        # Implementation of rules
        if rule == "MP":
            if len(args) != 2:
                return False, "", "ARG_ERROR: MP needs two args: X, (X->Y)"
            X, impl = args
            if not has(X) or not has(impl):
                return False, "", "MISSING_KNOWN: required statements not in Known"
            m = re.match(r"^\((.+)->(.+)\)$", impl)
            if not m:
                return False, "", "TYPE_ERROR: second arg must be (X->Y)"
            X2, Y = m.group(1), m.group(2)
            if X2 != X:
                return False, "", "MISMATCH: antecedent does not match"
            return True, Y, f"Derived {Y} via MP"
        elif rule == "AND_INTRO":
            if len(args) != 2:
                return False, "", "ARG_ERROR: AND_INTRO needs two args X,Y"
            X, Y = args
            if not has(X) or not has(Y):
                return False, "", "MISSING_KNOWN: X or Y not in Known"
            return True, f"({X}&{Y})", f"Derived ({X}&{Y}) via AND_INTRO"
        elif rule == "AND_ELIM_L":
            if len(args) != 1:
                return False, "", "ARG_ERROR: AND_ELIM_L needs one arg (X&Y)"
            C = args[0]
            m = re.match(r"^\((.+)&(.+)\)$", C)
            if not m or not has(C):
                return False, "", "TYPE_ERROR: need known (X&Y)"
            X = m.group(1)
            return True, X, f"Derived {X} via AND_ELIM_L"
        elif rule == "AND_ELIM_R":
            if len(args) != 1:
                return False, "", "ARG_ERROR: AND_ELIM_R needs one arg (X&Y)"
            C = args[0]
            m = re.match(r"^\((.+)&(.+)\)$", C)
            if not m or not has(C):
                return False, "", "TYPE_ERROR: need known (X&Y)"
            Y = m.group(2)
            return True, Y, f"Derived {Y} via AND_ELIM_R"
        elif rule == "OR_INTRO":
            if len(args) != 2:
                return False, "", "ARG_ERROR: OR_INTRO needs two args X,Y"
            X, Y = args
            if not has(X):
                return False, "", "MISSING_KNOWN: X not in Known"
            return True, f"({X}∨{Y})", f"Derived ({X}∨{Y}) via OR_INTRO"
        elif rule == "CHAIN":
            if len(args) != 2:
                return False, "", "ARG_ERROR: CHAIN needs two args (X->Y),(Y->Z)"
            A, B = args
            if not has(A) or not has(B):
                return False, "", "MISSING_KNOWN: need both implications known"
            m1 = re.match(r"^\((.+)->(.+)\)$", A)
            m2 = re.match(r"^\((.+)->(.+)\)$", B)
            if not m1 or not m2:
                return False, "", "TYPE_ERROR: both args must be implications"
            X, Y = m1.group(1), m1.group(2)
            Y2, Z = m2.group(1), m2.group(2)
            if Y != Y2:
                return False, "", "MISMATCH: middle terms differ"
            return True, f"({X}->{Z})", f"Derived ({X}->{Z}) via CHAIN"
        else:
            return False, "", f"UNKNOWN_RULE: {rule}"

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if getattr(self, "done", False):
            return "EPISODE_ALREADY_FINISHED", 0.0, True, False, {"suffix": self.get_task_suffix()}
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("action", "").lower()
        msg = ""
        success = False
        terminated = False
        reward = 0.0

        if cmd == "use_premise":
            idx_s = parsed.get("idx", None)
            try:
                idx = int(str(idx_s).strip("{}"))
            except Exception:
                obs = "PROTOCOL VIOLATION: use_premise requires idx=k where k is an integer."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if idx < 1 or idx > len(self.premises):
                obs = f"PROTOCOL VIOLATION: premise index {idx} is out of range."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            p = self.premises[idx - 1]
            if p in self.known:
                msg = f"No-op: premise already known: {p}"
            else:
                self.known.add(p)
                self.history.append(f"use_premise {idx}")
                msg = f"Added premise {idx}: {p} to Known."
        elif cmd == "apply":
            rule = parsed.get("rule", "")
            args_s = parsed.get("args", "")
            if not rule:
                obs = "PROTOCOL VIOLATION: apply requires rule=RULE."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            args_list = self._parse_args(args_s) if args_s else []
            ok, derived, message = self._apply_rule(rule, args_list)
            if ok:
                if derived in self.known:
                    msg = f"No-op: already known {derived}."
                else:
                    self.known.add(derived)
                    self.history.append(f"apply {rule} {args_s}")
                    msg = message
                success = True
            else:
                msg = f"RULE APPLICATION FAILED: {message}"
                return msg, 0.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = "UNSUPPORTED ACTION: Use 'use_premise' or 'apply'."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.target in self.known:
            obs = f"Success! Derived target: {self.target}. {msg}"
            self.done = True
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        effective_max = self.max_turns if self.max_turns <= 5 else max(self.max_turns, self.target_min_steps * 2 + 6)
        if self.turn_count > effective_max:
            self.done = True
            obs = f"TIMEOUT: Reached max turns ({effective_max}) without deriving target."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"Step processed. {msg}"
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        matches = re.findall(r"\\boxed\{(.+?)\}", action, flags=re.DOTALL)
        if not matches:
            return None
        inner = matches[-1].strip()
        parts = inner.split()
        if not parts:
            return None
        tokens: Dict[str, Any] = {"action": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k] = v.strip().strip("}")
        return tokens

    def sample_random_action(self) -> str:
        if random.random() < 0.5 and self.premises:
            idx = random.randint(1, len(self.premises))
            return rf"\boxed{{use_premise idx={idx}}}"
        else:
            rule = random.choice(list(self.rule_set))
            # crude random args
            if rule in ["AND_ELIM_L", "AND_ELIM_R"]:
                return rf"\boxed{{apply rule={rule} args=(P&Q)}}"
            elif rule in ["MP", "CHAIN"]:
                return rf"\boxed{{apply rule={rule} args=P,(P->Q)}}"
            elif rule == "AND_INTRO":
                return rf"\boxed{{apply rule=AND_INTRO args=P,Q}}"
            elif rule == "OR_INTRO":
                return rf"\boxed{{apply rule=OR_INTRO args=P,Q}}"
            else:
                return rf"\boxed{{use_premise idx=1}}"


class ProofCrafterEnvWithFeedback(ProofCrafterEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            detail["issue"] = "missing_boxed_or_malformed"
            hint = "Wrap your command in \\boxed{...} and follow 'use_premise' or 'apply' syntax."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            detail["allowed"] = ["use_premise", "apply"]
            hint = "Use 'use_premise idx=k' or 'apply rule=RULE args=...'."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "idx" in text:
                detail["violation"] = "bad_index_or_missing_idx"
                hint = "Provide a valid integer index within the premise range: use_premise idx=K."
            elif "rule" in text:
                detail["violation"] = "missing_rule_param"
                hint = "Specify the rule: apply rule=MP args=X,(X->Y)."
            else:
                detail["violation"] = "general_protocol"
                hint = "Follow the command signatures exactly."
        elif "rule application failed" in text:
            error_type = "WrongDecision"
            if "missing_known" in text:
                detail["reason"] = "premise_or_derived_not_known"
                hint = "Add required premises via use_premise or derive intermediates before applying the rule."
            elif "type_error" in text:
                detail["reason"] = "bad_argument_shape"
                hint = "Check the rule’s required argument forms, e.g., MP needs X and (X->Y)."
            elif "mismatch" in text:
                detail["reason"] = "antecedent_or_chain_mismatch"
                hint = "Ensure antecedents match exactly and chain middle terms align."
            elif "rule_not_available" in text:
                error_type = "UnsupportedAction"
                detail["reason"] = "rule_not_in_set"
                hint = f"Allowed rules: {', '.join(sorted(self.rule_set))}."
            else:
                detail["reason"] = "other_failure"
                hint = "Re-examine the known set and the rule’s preconditions."
        elif "timeout" in text:
            error_type = "Timeout"
            detail["limit"] = self.max_turns
            hint = "Plan a shorter derivation: prioritize premises that lead toward the target."
        elif "success" in text:
            error_type = "OK"
            detail["outcome"] = "success"
            hint = None
        else:
            error_type = "OK"
            detail["note"] = "progress"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["known_size"] = len(self.known)
            diagnostic["target"] = self.target
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Begin by adding a relevant premise with use_premise idx=1, then apply MP if possible.",
            "turn": 0,
            "known_size": 0,
            "target": self.target,
        }
        return obs, info
