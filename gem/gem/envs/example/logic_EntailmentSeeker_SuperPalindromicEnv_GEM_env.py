from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class EntailmentSeekerEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 28,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 28

        # Evolvable parameters
        self.complexity_params = {
            # Number of propositional variables p1..pN. Larger N increases search space for minimal entailed variable.
            'num_vars': (5, 40),
            # Total number of Horn rules (including noise). More rules deepen and widen inference, increasing cognitive load.
            'num_rules': (6, 60),
            # Max body length in rules. Longer bodies increase combinatorial preconditions → harder reasoning.
            'max_body_len': (1, 3),
            # REVERSED: number of initial facts (true atoms). Fewer facts → less immediate entailment, harder discovery.
            'num_facts': (4, 1),
            # Number of noise rules that never fire. More noise = more irrelevant structure to sift through → harder.
            'noise_rules': (0, 15),
        }

        # Variance settings for parameter randomization
        self.param_variance = {
            'num_vars': 3,       # ±3 over a 36 range (~8%)
            'num_rules': 5,      # ±5 over a 54 range (~9%)
            'max_body_len': 0,   # very small range → fixed
            'num_facts': 1,      # small integer jitter
            'noise_rules': 3,    # ±3 over ~16 values (~19%) but clamped
        }

        # Placeholder attributes
        self.num_vars: int = 0
        self.num_rules: int = 0
        self.max_body_len: int = 0
        self.num_facts: int = 0
        self.noise_rules: int = 0

        # Domain state
        self.turn_count: int = 0
        self.variables: List[str] = []
        self.rules: List[Tuple[List[int], int]] = []  # (body_indices, head_index), indices are 1-based
        self.facts: Set[int] = set()
        self.entailed: Set[int] = set()
        self.candidate: int = 1
        self.min_true_idx: Optional[int] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (lo, hi) in self.complexity_params.items():
            center = lo + (hi - lo) * normalized
            val = center
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
            # Clamp supporting reversed ranges
            minv, maxv = (hi, lo) if lo > hi else (lo, hi)
            val = max(minv, min(maxv, val))
            setattr(self, name, int(round(val)))

        # Safety guards for feasibility
        self.num_vars = max(3, self.num_vars)
        self.max_body_len = max(1, min(3, self.max_body_len))
        self.num_facts = max(1, min(self.num_vars - 1, self.num_facts))
        self.num_rules = max(self.num_facts + 2, self.num_rules)  # enough room for structure
        self.noise_rules = max(0, min(self.num_rules - 1, self.noise_rules))  # keep at least 1 real rule

    def _get_instructions(self) -> str:
        return (
            "You are EntailmentSeeker. A hidden Horn clause knowledge base over variables p1..pN is generated.\n"
            "Horn rules have the form (a ∧ b ∧ ...) -> c. The least-model semantics determines which variables are entailed True.\n"
            "Your objective: submit the smallest index i such that pi is entailed True by the instance.\n"
            "\n"
            "Available actions:\n"
            "- peek: Reveal overview counts (N variables, number of rules). No parameters.\n"
            "- show_rule id=<k>: Reveal the k-th rule (1-based indexing).\n"
            "- check var=<i>: Query whether variable pi is entailed True.\n"
            "- set i=<i>: Set your working candidate index to i.\n"
            "- next [k=<steps>]: Move your candidate forward by k (default 1), clamped to [1..N].\n"
            "- prev [k=<steps>]: Move your candidate backward by k (default 1), clamped to [1..N].\n"
            "- submit [i=<i>]: Submit your final answer. If i is omitted, submits current candidate.\n"
            "\n"
            "Rules:\n"
            "- Use only the supported actions. Unknown actions or invalid parameters terminate the episode with a penalty.\n"
            "- Indices are 1-based and must be within bounds.\n"
            "- Non-terminal actions give zero reward; only the final submit yields success/failure reward.\n"
            "- You have a limited number of turns.\n"
            "\n"
            "Action format: wrap exactly one action in \\boxed{...}\n"
            "Examples:\n"
            f"{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        known_n = len(self.variables) if self.variables else 0
        n_disp = known_n if known_n > 0 else "hidden"
        return (
            f"State: candidate=p{self.candidate}, turns={self.turn_count}/{self.max_turns}, N={n_disp}.\n"
            "Enter your next action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.variables = [f"p{i}" for i in range(1, self.num_vars + 1)]
        self.rules = []
        self.facts = set()
        self.entailed = set()
        self.candidate = 1
        self.min_true_idx = None

        # Generate initial facts
        all_indices = list(range(1, self.num_vars + 1))
        random.shuffle(all_indices)
        self.facts = set(all_indices[: self.num_facts])

        # Build real rules to expand entailment from facts
        real_rules_target = max(1, self.num_rules - self.noise_rules)
        reachable = set(self.facts)
        used_pairs = set()
        # Create chains from reachable set
        for _ in range(real_rules_target):
            # ensure body has at least one reachable atom to propagate entailment
            body_len = random.randint(1, self.max_body_len)
            if len(reachable) == 0:
                body = [random.randint(1, self.num_vars)]
            else:
                body_seed_count = min(len(reachable), body_len)
                body = random.sample(sorted(reachable), body_seed_count)
                while len(body) < body_len:
                    extra = random.randint(1, self.num_vars)
                    if extra not in body:
                        body.append(extra)
            head = random.randint(1, self.num_vars)
            # avoid trivial body -> same head duplicate
            if head in body:
                # try to move head off body
                candidates = [i for i in range(1, self.num_vars + 1) if i not in body]
                if candidates:
                    head = random.choice(candidates)
            key = (tuple(sorted(body)), head)
            if key in used_pairs:
                continue
            used_pairs.add(key)
            self.rules.append((body, head))
            # probabilistically mark head as reachable to encourage expansions
            if random.random() < 0.7:
                reachable.add(head)

        # Add noise rules that never fire: include at least one "dead" literal not in closure of facts under real rules
        # We'll synthesize dead anchors from variables unlikely to be reachable (random pick outside current reachable)
        dead_pool = [i for i in range(1, self.num_vars + 1) if i not in reachable]
        for _ in range(self.noise_rules):
            body_len = random.randint(1, self.max_body_len)
            body = []
            if dead_pool:
                body.append(random.choice(dead_pool))
            # fill remaining with random others (can include reachable, still never fires due to dead literal)
            while len(body) < body_len:
                x = random.randint(1, self.num_vars)
                if x not in body:
                    body.append(x)
            head = random.randint(1, self.num_vars)
            # also avoid head in body
            if head in body:
                alt = [i for i in range(1, self.num_vars + 1) if i not in body]
                if alt:
                    head = random.choice(alt)
            key = (tuple(sorted(body)), head)
            if key in used_pairs:
                continue
            used_pairs.add(key)
            self.rules.append((body, head))

        # Shuffle rules to hide structure
        random.shuffle(self.rules)
        # Ensure non-empty rule list
        if not self.rules:
            # fallback single rule from a fact to another
            f = next(iter(self.facts))
            head = random.randint(1, self.num_vars)
            if head == f:
                head = (head % self.num_vars) + 1
            self.rules.append(([f], head))

        # Compute least model
        self.entailed = self._compute_least_model(self.facts, self.rules)
        # Guarantee at least one entailed var (facts ensure this)
        if len(self.entailed) == 0:
            # fail-safe: inject a fact
            v = random.randint(1, self.num_vars)
            self.facts.add(v)
            self.entailed = self._compute_least_model(self.facts, self.rules)
        self.min_true_idx = min(self.entailed) if self.entailed else None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _compute_least_model(self, facts: Set[int], rules: List[Tuple[List[int], int]]) -> Set[int]:
        true_set = set(facts)
        changed = True
        # Forward chaining until fixpoint
        while changed:
            changed = False
            for body, head in rules:
                if all(b in true_set for b in body):
                    if head not in true_set:
                        true_set.add(head)
                        changed = True
        return true_set

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action and valid parameters."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "").lower()

        # Action handlers
        if name == "peek":
            obs = f"OVERVIEW: N={self.num_vars}, rules={len(self.rules)}"
            # proceed

        elif name == "show_rule":
            rid_str = parsed.get("id")
            if not rid_str or not rid_str.isdigit():
                obs = "INVALID PARAMETER: 'id' must be a positive integer."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            rid = int(rid_str)
            if rid < 1 or rid > len(self.rules):
                obs = f"INVALID PARAMETER: rule id out of range (1..{len(self.rules)})."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            body, head = self.rules[rid - 1]
            body_str = " ∧ ".join(f"p{b}" for b in body) if body else "TRUE"
            obs = f"RULE[{rid}]: ({body_str}) -> p{head}"

        elif name == "check":
            var_str = parsed.get("var") or parsed.get("i") or parsed.get("idx")
            if not var_str or not var_str.isdigit():
                obs = "INVALID PARAMETER: 'var' must be a positive integer."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            i = int(var_str)
            if i < 1 or i > self.num_vars:
                obs = f"INVALID PARAMETER: variable index out of range (1..{self.num_vars})."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            val = i in self.entailed
            obs = f"ENTAILMENT: p{i} -> {str(val)}"

        elif name == "set":
            i_str = parsed.get("i") or parsed.get("var") or parsed.get("idx")
            if not i_str or not i_str.isdigit():
                obs = "INVALID PARAMETER: 'i' must be a positive integer."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            i = int(i_str)
            if i < 1 or i > self.num_vars:
                obs = f"INVALID PARAMETER: candidate out of range (1..{self.num_vars})."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            self.candidate = i
            obs = f"CANDIDATE SET: p{self.candidate}"

        elif name == "next":
            k_str = parsed.get("k")
            if k_str is not None and (not k_str.isdigit()):
                obs = "INVALID PARAMETER: 'k' must be a positive integer if provided."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            k = int(k_str) if k_str is not None else 1
            self.candidate = min(self.num_vars, max(1, self.candidate + (k if k >= 0 else 0)))
            obs = f"CANDIDATE NEXT: p{self.candidate}"

        elif name == "prev":
            k_str = parsed.get("k")
            if k_str is not None and (not k_str.isdigit()):
                obs = "INVALID PARAMETER: 'k' must be a positive integer if provided."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            k = int(k_str) if k_str is not None else 1
            self.candidate = min(self.num_vars, max(1, self.candidate - (k if k >= 0 else 0)))
            obs = f"CANDIDATE PREV: p{self.candidate}"

        elif name == "submit":
            # get index; default to current candidate
            i_str = parsed.get("i") or parsed.get("var") or parsed.get("idx")
            ans = self.candidate if (i_str is None) else (int(i_str) if i_str.isdigit() else None)
            if ans is None:
                obs = "INVALID PARAMETER: 'i' must be a positive integer if provided."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            if ans < 1 or ans > self.num_vars:
                obs = f"INVALID PARAMETER: submission out of range (1..{self.num_vars})."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

            # Evaluate success
            if self.min_true_idx is not None and ans == self.min_true_idx:
                obs = f"FINAL: Success. Minimal entailed variable is p{ans}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                # wrong decision
                obs = f"FINAL: Incorrect. p{ans} is not the minimal entailed variable."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "UNSUPPORTED ACTION: Use one of {peek, show_rule, check, set, next, prev, submit}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        # Turn/timeout check after performing non-terminal action
        if self.turn_count >= self.max_turns:
            obs_timeout = f"Reached max turns ({self.max_turns})."
            return obs_timeout, 0.0, True, True, {"suffix": self.get_task_suffix()}

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
            if '=' in part:
                key, value = part.split('=', 1)
                tokens[key.strip()] = value.strip()
        return tokens

    def sample_random_action(self) -> str:
        choices = []
        # Always valid within default bounds
        choices.append(r"\boxed{peek}")
        if self.num_vars >= 3:
            choices.append(r"\boxed{check var=1}")
            choices.append(r"\boxed{set i=2}")
            choices.append(r"\boxed{next k=2}")
            choices.append(r"\boxed{prev}")
            choices.append(r"\boxed{show_rule id=1}")
            choices.append(r"\boxed{submit i=1}")
        return random.choice(choices)


class EntailmentSeekerEnvWithFeedback(EntailmentSeekerEnv):
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
            hint = "Wrap exactly one action in \\boxed{...} and include required parameters."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = ["peek", "show_rule", "check", "set", "next", "prev", "submit"]
            hint = "Use a supported action. Try \\boxed{peek} to see overview."

        elif "invalid parameter" in text:
            error_type = "ProtocolViolation"
            # Extract clue
            if "out of range" in text:
                error_detail["violation"] = "index_out_of_range"
            else:
                error_detail["violation"] = "bad_or_missing_parameter"
            hint = "Ensure indices are positive integers within valid bounds and parameters match action requirements."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan ahead. Use check var=1, then increment candidate until you find the first entailed variable and submit."

        elif "final: success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        elif "final: incorrect" in text:
            error_type = "WrongDecision"
            error_detail["got_candidate"] = getattr(self, "candidate", None)
            # Provide expected at feedback level >=1 for debugging (not in obs)
            error_detail["expected_min_index"] = getattr(self, "min_true_idx", None)
            hint = "Find the smallest i with check var=i returning True. Start from i=1 and move up."

        else:
            error_type = "OK"
            # Non-terminal informative step
            if "rule[" in text:
                error_detail["note"] = "rule_revealed"
            elif "entailment:" in text:
                error_detail["note"] = "queried_entailment"
            elif "candidate" in text:
                error_detail["note"] = "candidate_adjusted"
            else:
                error_detail["note"] = "progress"

            hint = "Systematically check var=i from i=1 upward. Use next/prev and submit once you find the first True."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["domain_state"] = {
                "candidate": getattr(self, "candidate", None),
                "N": getattr(self, "num_vars", None),
                "rules": len(getattr(self, "rules", [])),
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
            "hint": "Begin with \\boxed{peek} to learn N and rule count, then \\boxed{check var=1}.",
            "turn": 0,
            "domain_state": {"candidate": self.candidate, "N": self.num_vars, "rules": len(self.rules)},
        }
        return obs, info