from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class HornTrailEntailerEnv(Env):
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
            'num_vars': (4, 14),            # Number of propositional atoms: larger KB = harder search
            'num_rules': (3, 28),           # Number of Horn rules: more edges = harder reasoning
            'num_base_facts': (1, 5),       # Base facts: fewer facts = harder (scarcer starting info)
            'premise_size_max': (1, 4),     # Premise width: wider premises = harder (more dependencies)
            'target_chain_depth': (1, 8),   # Depth to target: deeper chains = harder (longer derivation)
        }
        self.param_variance = {
            'num_vars': 2,           # ±2 around level interpolation
            'num_rules': 3,          # ±3
            'num_base_facts': 1,     # ±1
            'premise_size_max': 1,   # ±1
            'target_chain_depth': 1, # ±1
        }

        # Placeholders (set in _apply_complexity_params)
        self.num_vars: int = 0
        self.num_rules: int = 0
        self.num_base_facts: int = 0
        self.premise_size_max: int = 0
        self.target_chain_depth: int = 0

        # Domain state
        self.turn_count: int = 0
        self.variables: List[str] = []
        self.facts: Set[str] = set()
        self.rules: List[Tuple[Set[str], str]] = []
        self.target: str = ""
        self._reachable: Set[str] = set()
        self._depth_map: Dict[str, int] = {}
        self._entailed: bool = False
        self._blocked_var: Optional[str] = None

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

    def _build_kb(self):
        n = max(4, self.num_vars)
        self.variables = [f"V{i}" for i in range(1, n + 1)]
        self.target = self.variables[-1]
        # Ensure base facts at early indices
        base_pool = self.variables[:max(2, n // 3)]
        k = max(1, min(self.num_base_facts, len(base_pool)))
        self.facts = set(random.sample(base_pool, k))

        # Determine chain depth feasibly
        d = max(1, min(self.target_chain_depth, n - 2))
        chain_vars_needed = max(0, d - 1)
        eligible_for_chain = [v for v in self.variables if v not in self.facts and v != self.target]
        if len(eligible_for_chain) < chain_vars_needed:
            chain_vars_needed = len(eligible_for_chain)
            d = chain_vars_needed + 1

        chain_vars = random.sample(eligible_for_chain, chain_vars_needed)
        levels = []
        levels.append(list(self.facts))
        for i in range(chain_vars_needed):
            levels.append([chain_vars[i]])
        final_level = levels[-1] if levels else list(self.facts)

        self.rules = []
        # Decide entailed or not
        entailed_flag = random.choice([True, False])
        self._blocked_var = None

        def choose_premises(candidate_pool: List[str], max_width: int) -> Set[str]:
            width = random.randint(1, max(1, max_width))
            width = min(width, len(candidate_pool)) if candidate_pool else 1
            chosen = set(random.sample(candidate_pool, max(1, width))) if candidate_pool else set()
            return chosen

        # Construct chain rules
        if d == 1:
            # direct rule to target from base facts
            premises = choose_premises(levels[0], self.premise_size_max)
            if not entailed_flag:
                self._blocked_var = self._pick_blocked_var(exclude=set(self.facts | set(chain_vars) | {self.target}))
                if self._blocked_var:
                    premises = set(premises)
                    premises.add(self._blocked_var)
            self.rules.append((premises, self.target))
        else:
            # build chain intermediates then final to target
            # level 1 to first chain var
            premises = choose_premises(levels[0], self.premise_size_max)
            if not entailed_flag:
                # block at a random step
                block_step = random.randint(1, d)  # 1..d (dth means final rule)
            else:
                block_step = None
            if block_step == 1:
                self._blocked_var = self._pick_blocked_var(exclude=set(self.facts | set(chain_vars) | {self.target}))
                if self._blocked_var:
                    premises = set(premises)
                    premises.add(self._blocked_var)
            self.rules.append((set(premises), chain_vars[0]))
            # middle steps
            for i in range(1, d - 1):
                pool = list(set(levels[0] + chain_vars[:i]))
                premises = choose_premises(pool, self.premise_size_max)
                if block_step == (i + 1):
                    if self._blocked_var is None:
                        self._blocked_var = self._pick_blocked_var(exclude=set(self.facts | set(chain_vars) | {self.target}))
                    if self._blocked_var:
                        premises = set(premises)
                        premises.add(self._blocked_var)
                self.rules.append((set(premises), chain_vars[i]))
            # final to target
            pool = list(set(levels[0] + chain_vars))
            premises = choose_premises(pool, self.premise_size_max)
            if block_step == d:
                if self._blocked_var is None:
                    self._blocked_var = self._pick_blocked_var(exclude=set(self.facts | set(chain_vars) | {self.target}))
                if self._blocked_var:
                    premises = set(premises)
                    premises.add(self._blocked_var)
            self.rules.append((set(premises), self.target))

        # Fill distractor rules
        total_needed = max(len(self.rules), self.num_rules)
        attempts = 0
        while len(self.rules) < total_needed and attempts < total_needed * 3:
            attempts += 1
            head_candidates = [v for v in self.variables if v != self.target and v != self._blocked_var]
            head = random.choice(head_candidates)
            head_idx = int(head[1:])
            lower_pool = [v for v in self.variables if int(v[1:]) < head_idx]
            lower_pool = [v for v in lower_pool if v != self._blocked_var]
            if not lower_pool:
                continue
            premises = set(random.sample(lower_pool, k=random.randint(1, min(self.premise_size_max, len(lower_pool)))))
            # avoid duplicate exact rule
            if any(rp == premises and rh == head for rp, rh in self.rules):
                continue
            self.rules.append((premises, head))

        # Compute forward-chaining closure
        self._reachable, self._depth_map = self._forward_chain(self.facts, self.rules)
        self._entailed = self.target in self._reachable

        # Ensure solvability and consistency
        # If a blocked var accidentally got derived, remove rules generating it and recompute
        if self._blocked_var and self._blocked_var in self._reachable:
            self.rules = [r for r in self.rules if r[1] != self._blocked_var]
            self._reachable, self._depth_map = self._forward_chain(self.facts, self.rules)
            self._entailed = self.target in self._reachable

    def _pick_blocked_var(self, exclude: Set[str]) -> Optional[str]:
        candidates = [v for v in self.variables if v not in exclude]
        if not candidates:
            return None
        return random.choice(candidates)

    def _forward_chain(self, facts: Set[str], rules: List[Tuple[Set[str], str]]) -> Tuple[Set[str], Dict[str, int]]:
        reachable = set(facts)
        depth_map = {f: 0 for f in facts}
        changed = True
        # We assume acyclicity by construction (heads have higher index than premises indices),
        # but forward chaining is safe either way.
        while changed:
            changed = False
            for premises, head in rules:
                if head in reachable:
                    continue
                if premises and premises.issubset(reachable):
                    reachable.add(head)
                    premise_depths = [depth_map.get(p, 0) for p in premises]
                    depth_map[head] = (max(premise_depths) + 1) if premise_depths else 1
                    changed = True
        return reachable, depth_map

    def _format_rules(self) -> str:
        lines = []
        for premises, head in self.rules:
            if premises:
                p_str = " & ".join(sorted(premises))
                lines.append(f"{p_str} -> {head}")
            else:
                lines.append(f"TRUE -> {head}")
        return "\n".join(lines)

    def _get_instructions(self) -> str:
        return (
            "Logic Task: Decide entailment in a Horn knowledge base.\n"
            "Goal: Determine whether the target atom is logically entailed by the hidden base facts and Horn rules.\n"
            "\n"
            "Available actions (use \\boxed{...} format):\n"
            "- show_kb: Reveal the full KB (facts and rules).\n"
            "- summary_kb: Get global summary (counts and derived data like reachable atoms count).\n"
            "- query_premises head=V#: List premises of all rules whose head is the given atom.\n"
            "- query_reachable atom=V#: Check if a specific atom is derivable via forward chaining.\n"
            "- submit answer=yes|no: Final answer whether the TARGET atom is entailed.\n"
            "\n"
            "Rules:\n"
            "- Actions must be in \\boxed{...}. Unknown actions or malformed parameters terminate the episode.\n"
            "- Non-terminal actions return information and zero reward.\n"
            "- Final submission yields success (1.0) if correct, 0.0 if incorrect.\n"
            "- Format errors (missing \\boxed) yield an immediate penalty and termination.\n"
            "\n"
            f"Example action:\n{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        turns_left = max(0, self.max_turns - self.turn_count)
        return (
            f"Target atom: {self.target}\n"
            f"Turns remaining: {turns_left}\n"
            "Enter an action in \\boxed{...} format. Supported: show_kb, summary_kb, "
            "query_premises head=V#, query_reachable atom=V#, submit answer=yes|no."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self.variables = []
        self.facts = set()
        self.rules = []
        self.target = ""
        self._reachable = set()
        self._depth_map = {}
        self._blocked_var = None
        self._build_kb()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get('action', '')
        reward = 0.0
        obs = ""

        if name == 'show_kb':
            facts_str = ", ".join(sorted(self.facts))
            rules_str = self._format_rules()
            obs = (
                "KB DISCLOSURE:\n"
                f"Facts: {facts_str if facts_str else '(none)'}\n"
                "Rules:\n"
                f"{rules_str if rules_str else '(none)'}\n"
            )

        elif name == 'summary_kb':
            reachable_count = len(self._reachable)
            target_depth = self._depth_map.get(self.target, None)
            depth_info = f"target_depth={target_depth}" if target_depth is not None else "target_depth=unreachable"
            obs = (
                "KB SUMMARY:\n"
                f"num_vars={len(self.variables)}, num_rules={len(self.rules)}, num_facts={len(self.facts)}\n"
                f"reachable_atoms_count={reachable_count}, {depth_info}\n"
            )

        elif name == 'query_premises':
            head = parsed.get('head', None)
            if head is None or head not in self.variables:
                obs = f"MALFORMED PARAMETERS: 'head' missing or invalid for query_premises."
                terminated = True
                reward = 0.0
            else:
                matching = [prem for prem, h in self.rules if h == head]
                if not matching:
                    obs = f"No rules with head={head}."
                else:
                    lines = []
                    for idx, prem in enumerate(matching, 1):
                        p_str = " & ".join(sorted(prem)) if prem else "(none)"
                        lines.append(f"Rule {idx}: {p_str} -> {head}")
                    obs = "PREMISES LIST:\n" + "\n".join(lines) + "\n"

        elif name == 'query_reachable':
            atom = parsed.get('atom', None)
            if atom is None or atom not in self.variables:
                obs = f"MALFORMED PARAMETERS: 'atom' missing or invalid for query_reachable."
                terminated = True
                reward = 0.0
            else:
                is_reachable = atom in self._reachable
                depth = self._depth_map.get(atom, None)
                depth_str = f"depth={depth}" if depth is not None else "depth=unreachable"
                obs = f"REACHABILITY: atom={atom}, reachable={'yes' if is_reachable else 'no'}, {depth_str}\n"

        elif name == 'submit':
            ans = parsed.get('answer', None)
            if ans is None:
                obs = "MALFORMED PARAMETERS: 'answer' missing for submit."
                terminated = True
                reward = 0.0
            else:
                ans_norm = str(ans).strip().lower()
                if ans_norm in ['yes', 'true', 'y', '1']:
                    guess = True
                elif ans_norm in ['no', 'false', 'n', '0']:
                    guess = False
                else:
                    obs = "MALFORMED PARAMETERS: 'answer' must be yes|no."
                    terminated = True
                    reward = 0.0
                    guess = None
                if guess is not None:
                    correct = self._entailed
                    if guess == correct:
                        obs = "VERIFICATION: Success! The answer matches entailment."
                        reward = 1.0
                    else:
                        obs = "VERIFICATION: Failed. The answer does not match entailment."
                        reward = 0.0
                    terminated = True

        else:
            obs = f"UNSUPPORTED ACTION: '{name}'."
            terminated = True
            reward = 0.0

        if not terminated and self.turn_count >= self.max_turns:
            obs = f"TIMEOUT: Reached max turns ({self.max_turns})."
            terminated = True
            truncated = True

        if not terminated and not truncated and obs.strip() == "":
            obs = f"At turn {self.turn_count}, no-op."

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
        tokens: Dict[str, Any] = {}
        tokens['action'] = parts[0]
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                tokens[key] = value
        return tokens

    def sample_random_action(self) -> str:
        choices = [
            r'\boxed{summary_kb}',
            r'\boxed{show_kb}',
            rf'\boxed{{query_premises head={self.target}}}',
            rf'\boxed{{query_reachable atom={self.target}}}',
            r'\boxed{submit answer=yes}',
            r'\boxed{submit answer=no}',
        ]
        return random.choice(choices)


class HornTrailEntailerEnvWithFeedback(HornTrailEntailerEnv):
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
            hint = "Wrap actions in \\boxed{...} and use supported command names."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported action: '(.+?)'", obs)
            bad = m.group(1) if m else None
            error_detail["action_name"] = bad
            hint = "Use one of: show_kb, summary_kb, query_premises head=V#, query_reachable atom=V#, submit answer=yes|no."

        elif "malformed parameters" in text:
            error_type = "ProtocolViolation"
            if "query_premises" in text:
                error_detail["violation"] = "missing_or_invalid_head"
                hint = "Provide a valid atom name: \\boxed{query_premises head=V#}."
            elif "query_reachable" in text:
                error_detail["violation"] = "missing_or_invalid_atom"
                hint = "Provide a valid atom name: \\boxed{query_reachable atom=V#}."
            elif "submit" in text:
                error_detail["violation"] = "invalid_answer_value"
                hint = "Use \\boxed{submit answer=yes} or \\boxed{submit answer=no}."
            else:
                error_detail["violation"] = "invalid_parameters"
                hint = "Check parameter names and values per action specification."

        elif "timeout: reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan fewer information steps. Start with summary_kb, then query_reachable atom=TARGET, then submit."

        elif "verification: failed" in text:
            error_type = "WrongDecision"
            error_detail["expected"] = "yes" if self._entailed else "no"
            m = re.search(r"submit answer=(yes|no)", action.lower())
            error_detail["got"] = m.group(1) if m else None
            hint = "Use \\boxed{query_reachable atom=TARGET} to verify before submitting."

        elif "verification: success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["target"] = getattr(self, "target", None)
            diagnostic["kb_size"] = {
                "vars": len(getattr(self, "variables", [])),
                "rules": len(getattr(self, "rules", [])),
                "facts": len(getattr(self, "facts", set())),
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
            "hint": f"Start with \\boxed{{summary_kb}}. Then try \\boxed{{query_reachable atom={self.target}}} before submitting.",
            "turn": 0,
            "target": self.target,
        }
        return obs, info