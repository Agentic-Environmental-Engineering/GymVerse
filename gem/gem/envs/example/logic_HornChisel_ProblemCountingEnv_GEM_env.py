from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class HornChiselEnv(Env):
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
            # Size of predicate vocabulary. More predicates expand state space and reasoning branches → harder.
            "num_predicates": (6, 24),
            # Number of Horn rules. More rules increases inference possibilities and search complexity → harder.
            "num_rules": (5, 40),
            # Average antecedent size of rules. Larger bodies require combining more premises → harder.
            "avg_body_size": (1, 3),
            # REVERSED: number of base facts. Fewer initial facts makes derivations longer/sparser → harder.
            "num_facts": (6, 2),
            # Desired minimal chain length for target derivation when entailed. Deeper chains → harder.
            "target_chain_length": (1, 6),
            # REVERSED: proportion (in %) of tasks where the target is entailed. Less often entailed reduces guessability.
            "entailed_ratio": (70, 40),
        }

        self.param_variance = {
            "num_predicates": 2,
            "num_rules": 3,
            "avg_body_size": 0,
            "num_facts": 0,
            "target_chain_length": 1,
            "entailed_ratio": 3,
        }

        self.num_predicates: int = 0
        self.num_rules: int = 0
        self.avg_body_size: int = 0
        self.num_facts: int = 0
        self.target_chain_length: int = 0
        self.entailed_ratio: int = 0

        self.turn_count: int = 0
        self.preds: List[str] = []
        self.base_facts: Set[str] = set()
        self.rules: List[Tuple[List[str], str]] = []
        self.query: str = ""
        self.target_is_entailed: bool = False
        self.min_depths: Dict[str, int] = {}
        self.last_submission: Optional[str] = None

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
        return (
            "You are in HornChisel: a propositional Horn-logic inference challenge.\n"
            "- Hidden world: a set of definite Horn rules (A & B -> C) and base facts.\n"
            "- Query: decide if a specific predicate is logically entailed by the knowledge base.\n"
            "- Goal: submit the final answer 'yes' if the query is entailed, otherwise 'no'.\n"
            "\n"
            "Available actions (use \\boxed{...}):\n"
            "- reveal_theory                 → show all base facts and rules.\n"
            "- forward_chain                 → return the full closure of entailed predicates.\n"
            "- ask_entails var=X             → answer 'yes' or 'no' if X is entailed.\n"
            "- ask_depth var=X               → minimal derivation depth of X, or 'not derivable'.\n"
            "- help                          → brief reminder of actions.\n"
            "- submit answer=yes|no          → final answer for the QUERY ONLY.\n"
            "\n"
            "Rules:\n"
            "- Use only supported actions with required parameters.\n"
            "- Invalid format or unsupported actions end the episode with a penalty.\n"
            "- The episode times out if you exceed the maximum turns.\n"
            "\n"
            "Action format:\n"
            "- With parameters: \\boxed{action_name key=value}\n"
            "- Without parameters: \\boxed{action_name}\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        remaining = self.max_turns - self.turn_count
        return (
            f"QUERY: Is '{self.query}' entailed? "
            f"Turns used: {self.turn_count}/{self.max_turns} (remaining {remaining}).\n"
            "Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.last_submission = None

        self._generate_task()
        self._compute_closure_and_depths()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        reward = 0.0

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed.get("action", "")
        name_lower = name.lower()

        if name_lower == "reveal_theory":
            rules_str = ", ".join([self._rule_to_str(b, h) for (b, h) in self.rules]) if self.rules else "(none)"
            facts_str = ", ".join(sorted(self.base_facts)) if self.base_facts else "(none)"
            obs = f"THEORY: facts = {{{facts_str}}}; rules = [{rules_str}]."
        elif name_lower == "forward_chain":
            closure = sorted([p for p, d in self.min_depths.items() if d >= 0])
            obs = f"CLOSURE: {{{', '.join(closure)}}}."
        elif name_lower == "ask_entails":
            var = parsed.get("var")
            if not var:
                obs = "PROTOCOL VIOLATION: missing parameter 'var' for action 'ask_entails'."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            ent = "yes" if self.min_depths.get(var, -1) >= 0 else "no"
            obs = f"ENTAILED[{var}]: {ent}."
        elif name_lower == "ask_depth":
            var = parsed.get("var")
            if not var:
                obs = "PROTOCOL VIOLATION: missing parameter 'var' for action 'ask_depth'."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            d = self.min_depths.get(var, -1)
            obs = f"DEPTH[{var}]: {d if d >= 0 else 'not derivable'}."
        elif name_lower == "help":
            obs = (
                "HELP: actions are reveal_theory, forward_chain, ask_entails var=X, ask_depth var=X, submit answer=yes|no."
            )
        elif name_lower == "submit":
            ans = parsed.get("answer", "").lower()
            if ans not in ("yes", "no"):
                obs = "PROTOCOL VIOLATION: submission must be 'yes' or 'no'."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
            self.last_submission = ans
            correct = (ans == ("yes" if self.target_is_entailed else "no"))
            if correct:
                obs = f"SUCCESS: correct final answer ({ans})."
                reward = 1.0
            else:
                exp = "yes" if self.target_is_entailed else "no"
                obs = f"FAILURE: incorrect final answer (got={ans}, expected={exp})."
                reward = 0.0
            return obs, reward, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"UNSUPPORTED ACTION: '{name}'."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"TIMEOUT: reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

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
        if not parts:
            return None
        tokens: Dict[str, Any] = {"action": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        choices = ["reveal_theory", "forward_chain", "ask_entails", "ask_depth", "submit"]
        act = random.choice(choices)
        if act == "reveal_theory":
            return r"\boxed{reveal_theory}"
        if act == "forward_chain":
            return r"\boxed{forward_chain}"
        if act == "ask_entails":
            var = f"p{random.randint(1, max(1, self.num_predicates))}"
            return rf"\boxed{{ask_entails var={var}}}"
        if act == "ask_depth":
            var = f"p{random.randint(1, max(1, self.num_predicates))}"
            return rf"\boxed{{ask_depth var={var}}}"
        ans = random.choice(["yes", "no"])
        return rf"\boxed{{submit answer={ans}}}"

    def _rule_to_str(self, body: List[str], head: str) -> str:
        if not body:
            return f"(-> {head})"
        return f"({' & '.join(body)} -> {head})"

    def _generate_task(self):
        n = max(3, self.num_predicates)
        self.preds = [f"p{i}" for i in range(1, n + 1)]
        all_set = set(self.preds)

        fcount = max(1, min(self.num_facts, n - 1))
        self.base_facts = set(random.sample(self.preds, fcount))

        indices_sorted = list(range(n))
        indices_sorted.sort()
        target_entailed_flag = random.random() < (self.entailed_ratio / 100.0)

        self.rules = []
        used_heads = set()

        def pick_body_size():
            # Allow body sizes 1..max(3,avg_body_size)
            max_b = max(1, self.avg_body_size)
            # slight randomness around avg
            candidates = [max(1, min(3, self.avg_body_size + d)) for d in [-1, 0, 1]]
            return random.choice(candidates)

        # Ensure query selection
        # Choose a query not in base facts initially
        candidates_for_query = list(all_set - self.base_facts) if len(all_set - self.base_facts) > 0 else list(all_set)
        self.query = random.choice(candidates_for_query)

        # Build a derivation chain if entailed is desired
        if target_entailed_flag:
            # Chain length cannot exceed available non-fact intermediates
            L = max(1, self.target_chain_length)
            nonfacts = list(all_set - self.base_facts - {self.query})
            # Ensure enough intermediates
            if len(nonfacts) < max(0, L - 1):
                L = min(L, len(nonfacts) + 1)
            # Choose intermediate sequence distinct and try to make acyclic by order
            sequence = []
            if L > 1:
                sequence = random.sample(nonfacts, L - 1)
            chain_nodes = sequence + [self.query]

            prev = random.choice(list(self.base_facts))
            for node in chain_nodes:
                bsize = pick_body_size()
                supporters = []
                if bsize > 1:
                    supporters = random.sample(list(self.base_facts), min(len(self.base_facts), bsize - 1))
                    if prev in supporters:
                        supporters.remove(prev)
                body = [prev] + supporters
                body = list(dict.fromkeys(body))
                if len(body) > bsize:
                    body = body[:bsize]
                if node not in body:
                    self.rules.append((body, node))
                    used_heads.add(node)
                prev = node
        else:
            # Not entailed: forbid any rule that would have head == query; ensure query not in base facts
            if self.query in self.base_facts:
                rem = random.choice(list(self.base_facts))
                if rem != self.query:
                    self.base_facts.remove(self.query)
                    self.base_facts.add(rem)

        # Fill remaining rules randomly with acyc-style preference
        def head_candidates():
            return list(all_set - used_heads)

        attempts = 0
        while len(self.rules) < self.num_rules and attempts < self.num_rules * 5:
            attempts += 1
            head = random.choice(self.preds)
            if not target_entailed_flag and head == self.query:
                continue
            bsz = pick_body_size()
            # pick body from predicates excluding head
            body = set(random.sample(self.preds, min(bsz, n)))
            if head in body:
                continue
            body = list(body)
            # avoid duplicates
            if any(set(b) == set(body) and h == head for (b, h) in self.rules):
                continue
            self.rules.append((body, head))
            used_heads.add(head)

        # Compute closure to finalize target truth
        self._compute_closure_and_depths()
        self.target_is_entailed = self.min_depths.get(self.query, -1) >= 0
        # If the randomly filled rules broke our intent for entailed vs not-entailed, adjust minimally
        if target_entailed_flag and not self.target_is_entailed:
            # Add a direct rule from some fact to query
            src = random.choice(list(self.base_facts))
            self.rules.append(([src], self.query))
            self._compute_closure_and_depths()
            self.target_is_entailed = True
        if not target_entailed_flag and self.target_is_entailed:
            # Remove any rules that derive query
            self.rules = [(b, h) for (b, h) in self.rules if h != self.query]
            self._compute_closure_and_depths()
            self.target_is_entailed = False

    def _compute_closure_and_depths(self):
        depths: Dict[str, int] = {p: -1 for p in self.preds}
        for f in self.base_facts:
            depths[f] = 0
        changed = True
        # iterative update for minimal derivation depth
        while changed:
            changed = False
            for body, head in self.rules:
                if depths[head] == 0:
                    continue
                if all(depths.get(b, -1) >= 0 for b in body):
                    d = 1 + max(depths[b] for b in body) if body else 0
                    if depths[head] == -1 or d < depths[head]:
                        depths[head] = d
                        changed = True
        self.min_depths = depths


class HornChiselEnvWithFeedback(HornChiselEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_or_syntax"
            hint = "Wrap your command in \\boxed{...} and include required parameters."
        elif text.startswith("unsupported action"):
            error_type = "UnsupportedAction"
            error_detail["allowed_actions"] = [
                "reveal_theory", "forward_chain", "ask_entails var=X", "ask_depth var=X", "submit answer=yes|no", "help"
            ]
            hint = "Use one of the supported actions; try \\boxed{help} to see options."
        elif text.startswith("protocol violation"):
            error_type = "ProtocolViolation"
            if "missing parameter 'var'" in text:
                error_detail["violation"] = "missing_var_parameter"
                hint = "Add var=X, e.g., \\boxed{ask_depth var=p3}."
            elif "submission must be 'yes' or 'no'" in text:
                error_detail["violation"] = "invalid_submission_value"
                hint = "Submit as \\boxed{submit answer=yes} or \\boxed{submit answer=no}."
            else:
                error_detail["violation"] = "other_protocol_violation"
                hint = "Check the action format and required parameters; try \\boxed{help}."
        elif text.startswith("failure"):
            error_type = "WrongDecision"
            expected = "yes" if getattr(self, "target_is_entailed", False) else "no"
            got = getattr(self, "last_submission", None)
            error_detail["expected"] = expected
            error_detail["got"] = got
            error_detail["query"] = self.query
            hint = "Use \\boxed{forward_chain} to see all entailed predicates, then submit accordingly."
        elif text.startswith("timeout"):
            error_type = "Timeout"
            error_detail["max_turns"] = self.max_turns
            hint = "Act sooner; a typical strategy is reveal_theory → forward_chain → submit."
        elif text.startswith("success"):
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            error_detail["turn"] = getattr(self, "turn_count", None)
            error_detail["query"] = getattr(self, "query", None)
            error_detail["num_rules"] = len(getattr(self, "rules", []))
            error_detail["num_facts"] = len(getattr(self, "base_facts", set()))
            diagnostic["error_detail"] = error_detail
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {
                "outcome": "episode_start",
                "turn": 0,
                "query": getattr(self, "query", None),
                "num_rules": len(getattr(self, "rules", [])),
                "num_facts": len(getattr(self, "base_facts", set())),
            },
            "hint": "Start with \\boxed{reveal_theory} or \\boxed{forward_chain}, then decide and \\boxed{submit}.",
        }
        return obs, info