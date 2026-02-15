from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class PrimeWeaveMatchingEnv(Env):
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
            # Number of items per side; larger = more combinations and deeper search
            "n_items": (3, 8),
            # Domain range for sampling integers; larger span increases variability and harder reasoning
            "value_span": (10, 60),
            # Predicate family selector; higher index uses stricter/more intricate predicates
            # 1: sum is prime, 2: gcd=1, 3: sum is a perfect square, 4: |a-b| is prime
            # Higher = generally sparser admissible edges → harder to find perfect matchings
            "predicate_level": (1, 4),
            # Allowed mistakes (rejections) before hard failure; fewer allowed errors = harder (REVERSED)
            "error_budget": (3, 1),
            # Hint budget for listing possible partners; fewer hints = harder (REVERSED)
            "hint_budget": (2, 0),
        }

        # Variance settings
        self.param_variance = {
            "n_items": 1,           # medium discrete range
            "value_span": 5,        # large range → ~±5 variation
            "predicate_level": 0,   # small discrete set; keep fixed per level
            "error_budget": 0,      # keep deterministic across level
            "hint_budget": 0,       # keep deterministic across level
        }

        # Placeholders
        self.n_items: int = 0
        self.value_span: int = 0
        self.predicate_level: int = 0
        self.error_budget: int = 0
        self.hint_budget: int = 0

        # State
        self.turn_count: int = 0
        self.left: List[int] = []
        self.right: List[int] = []
        self.unmatched_left: Set[int] = set()
        self.unmatched_right: Set[int] = set()
        self.pairs: List[Tuple[int, int]] = []
        self.errors_made: int = 0
        self.hints_used: int = 0
        self.predicate_name: str = ""
        self._instance_seed: Optional[int] = None

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

    def _is_prime(self, x: int) -> bool:
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        d = 3
        while d * d <= x:
            if x % d == 0:
                return False
            d += 2
        return True

    def _is_square(self, x: int) -> bool:
        if x < 0:
            return False
        r = int(x ** 0.5)
        return r * r == x

    def _gcd(self, a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return abs(a)

    def _predicate(self, a: int, b: int) -> bool:
        if self.predicate_level == 1:
            return self._is_prime(a + b)
        elif self.predicate_level == 2:
            return self._gcd(a, b) == 1
        elif self.predicate_level == 3:
            return self._is_square(a + b)
        else:
            return self._is_prime(abs(a - b))

    def _predicate_label(self) -> str:
        if self.predicate_level == 1:
            return "sum_is_prime"
        elif self.predicate_level == 2:
            return "gcd_is_one"
        elif self.predicate_level == 3:
            return "sum_is_square"
        else:
            return "abs_diff_is_prime"

    def _generate_instance(self):
        # Ensure feasibility by sampling until a perfect matching exists
        # Use bounded retries to avoid infinite loops, adjust sampling if needed
        tries = 0
        while True:
            tries += 1
            base_min = 2
            base_max = base_min + self.value_span
            self.left = random.sample(range(base_min, base_max), self.n_items)
            self.right = random.sample(range(base_min, base_max), self.n_items)
            # Build bipartite adjacency
            adj = {i: [] for i in range(self.n_items)}
            for i, a in enumerate(self.left):
                for j, b in enumerate(self.right):
                    if self._predicate(a, b):
                        adj[i].append(j)
            if self._has_perfect_matching(adj):
                return self.left[:], self.right[:]
            if tries > 200:
                # Relax by shrinking predicate strictness if stuck
                if self.predicate_level > 1:
                    self.predicate_level -= 1
                tries = 0

    def _has_perfect_matching(self, adj: Dict[int, List[int]]) -> bool:
        # Standard bipartite matching via DFS (Kuhn's algorithm)
        n = self.n_items
        match_r = [-1] * n

        def dfs(u: int, seen: Set[int]) -> bool:
            for v in adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                if match_r[v] == -1 or dfs(match_r[v], seen):
                    match_r[v] = u
                    return True
            return False

        result = 0
        for u in range(n):
            if dfs(u, set()):
                result += 1
            else:
                return False
        return result == n

    def _build_adjacency(self) -> Dict[int, List[int]]:
        adj = {i: [] for i in range(self.n_items)}
        for i, a in enumerate(self.left):
            for j, b in enumerate(self.right):
                if self._predicate(a, b):
                    adj[i].append(j)
        return adj

    def _get_instructions(self) -> str:
        return (
            "You are solving a bipartite number matching puzzle.\n"
            "Goal: Pair every Left number to exactly one Right number forming a perfect matching.\n"
            "Admissibility rule (predicate): "
            f"{self._predicate_label()}  (see below).\n"
            "Instance:\n"
            f"- Left side (L): {self.left}\n"
            f"- Right side (R): {self.right}\n"
            "Rules:\n"
            "- Propose one pair per action using \\boxed{pair L=<left_value> R=<right_value>}.\n"
            "- Each Left and Right value can be used at most once.\n"
            "- A proposed pair must satisfy the predicate; otherwise it is rejected and counts as an error.\n"
            "- You may request a hint with \\boxed{hint L=<left_value>} to see admissible Right partners for an unmatched Left (limited uses).\n"
            f"- Error budget: {self.error_budget}; exceeding it ends the episode.\n"
            f"- Hint budget: {self.hint_budget}.\n"
            "Predicates:\n"
            "- sum_is_prime: a+b is a prime number.\n"
            "- gcd_is_one: gcd(a, b) = 1.\n"
            "- sum_is_square: a+b is a perfect square.\n"
            "- abs_diff_is_prime: |a-b| is a prime number.\n"
            f"Format examples:\n"
            r"- \boxed{pair L=12 R=19}" "\n"
            r"- \boxed{hint L=12}" "\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"State: unmatched Left={sorted([self.left[i] for i in self.unmatched_left])}, "
            f"unmatched Right={sorted([self.right[j] for j in self.unmatched_right])}, "
            f"pairs={[(self.left[i], self.right[j]) for i,j in self.pairs]}, "
            f"errors={self.errors_made}/{self.error_budget}, hints_used={self.hints_used}/{self.hint_budget}. "
            "Enter action as \\boxed{pair L=<int> R=<int>} or \\boxed{hint L=<int>}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
            self._instance_seed = seed
        self._apply_complexity_params()
        self.turn_count = 0
        self.errors_made = 0
        self.hints_used = 0
        self.predicate_name = self._predicate_label()
        self.left, self.right = self._generate_instance()
        self.unmatched_left = set(range(self.n_items))
        self.unmatched_right = set(range(self.n_items))
        self.pairs = []
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{pair L=<int> R=<int>} or \\boxed{hint L=<int>}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "")
        terminated = False
        truncated = False
        reward = 0.0

        if act == "hint":
            lval = parsed.get("L", None)
            if lval is None:
                obs = "PROTOCOL VIOLATION: hint requires L=<int>."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # Validate left value exists and is unmatched
            if lval not in self.left:
                obs = f"FAILED: Left value {lval} not in instance."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            li = self.left.index(lval)
            if li not in self.unmatched_left:
                obs = f"FAILED: Left value {lval} is already matched."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.hints_used >= self.hint_budget:
                obs = "FAILED: No hints remaining."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            # Provide admissible partners among unmatched right
            partners = []
            for rj in sorted(self.unmatched_right):
                if self._predicate(self.left[li], self.right[rj]):
                    partners.append(self.right[rj])
            self.hints_used += 1
            obs = f"HINT: Admissible partners for L={lval} under {self.predicate_name}: {partners if partners else 'None'}."
            if self.turn_count >= self.max_turns:
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if act == "pair":
            lval = parsed.get("L", None)
            rval = parsed.get("R", None)
            if lval is None or rval is None:
                obs = "PROTOCOL VIOLATION: pair requires L=<int> and R=<int>."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if lval not in self.left or rval not in self.right:
                obs = "FAILED: Values not found in instance."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            li = self.left.index(lval)
            rj = self.right.index(rval)

            if li not in self.unmatched_left:
                obs = f"FAILED: Left value {lval} already matched."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if rj not in self.unmatched_right:
                obs = f"FAILED: Right value {rval} already matched."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if not self._predicate(lval, rval):
                self.errors_made += 1
                over = self.errors_made > self.error_budget
                obs = (
                    f"REJECTED: Pair (L={lval}, R={rval}) violates predicate {self.predicate_name}. "
                    f"Errors: {self.errors_made}/{self.error_budget}."
                )
                if over:
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                if self.turn_count >= self.max_turns:
                    return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

            # Accept the pair
            self.unmatched_left.remove(li)
            self.unmatched_right.remove(rj)
            self.pairs.append((li, rj))
            if len(self.pairs) == self.n_items:
                obs = f"Success! Perfect matching completed: {[(self.left[i], self.right[j]) for i,j in self.pairs]}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns})"
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            obs = f"ACCEPTED: Paired L={lval} with R={rval}. Remaining unmatched L={len(self.unmatched_left)}, R={len(self.unmatched_right)}."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        obs = "UNSUPPORTED ACTION: Use 'pair' or 'hint'."
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
        tokens["action"] = parts[0].lower()
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k in ("L", "R"):
                    try:
                        tokens[k] = int(v)
                    except:
                        tokens[k] = None
                else:
                    tokens[k] = v
        return tokens

    def sample_random_action(self) -> str:
        # With some probability request a hint for a random unmatched left
        if self.unmatched_left and random.random() < 0.3:
            li = random.choice(list(self.unmatched_left))
            return rf"\boxed{{hint L={self.left[li]}}}"
        # Otherwise attempt a random admissible pair or a random pair
        if self.unmatched_left and self.unmatched_right:
            li = random.choice(list(self.unmatched_left))
            candidates = []
            for rj in self.unmatched_right:
                if self._predicate(self.left[li], self.right[rj]):
                    candidates.append(self.right[rj])
            if candidates:
                r = random.choice(candidates)
            else:
                r = random.choice([self.right[rj] for rj in self.unmatched_right])
            return rf"\boxed{{pair L={self.left[li]} R={r}}}"
        return r"\boxed{hint L=0}"


class PrimeWeaveMatchingEnvWithFeedback(PrimeWeaveMatchingEnv):
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
            error_detail["issue"] = "missing_or_malformed_boxed"
            hint = "Wrap your command in \\boxed{...} and use 'pair' or 'hint'."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["allowed"] = ["pair", "hint"]
            hint = "Use \\boxed{pair L=<int> R=<int>} or \\boxed{hint L=<int>}."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "hint requires" in text:
                error_detail["violation"] = "hint_missing_L"
                hint = "Supply a specific Left value: \\boxed{hint L=<left_value>}."
            elif "pair requires" in text:
                error_detail["violation"] = "pair_missing_L_or_R"
                hint = "Provide both endpoints: \\boxed{pair L=<int> R=<int>}."
            else:
                error_detail["violation"] = "general_protocol"
                hint = "Follow action formats exactly."
        elif text.startswith("failed:"):
            error_type = "WrongDecision"
            if "not in instance" in text:
                error_detail["issue"] = "value_not_in_instance"
                hint = "Choose L and R from the listed instance arrays."
            elif "already matched" in text:
                error_detail["issue"] = "double_use_value"
                hint = "Pick unmatched items only."
            elif "no hints remaining" in text:
                error_detail["issue"] = "hint_exhausted"
                hint = "Proceed by proposing admissible pairs without hints."
            else:
                error_detail["issue"] = "general_failure"
                hint = "Check unmatched lists and predicate constraints."
        elif "rejected" in text:
            error_type = "WrongDecision"
            error_detail["issue"] = "predicate_violation"
            error_detail["predicate"] = getattr(self, "predicate_name", None)
            hint = "Use hints to see admissible partners or verify the predicate before pairing."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act faster: use hints sparingly and focus on forcing moves."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "accepted" in text or "hint:" in text:
            error_type = "OK"
            error_detail["outcome"] = "progress"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "unmatched_left": sorted([self.left[i] for i in self.unmatched_left]),
                "unmatched_right": sorted([self.right[j] for j in self.unmatched_right]),
                "pairs": [(self.left[i], self.right[j]) for i, j in self.pairs],
                "errors_made": self.errors_made,
                "error_budget": self.error_budget,
                "hints_used": self.hints_used,
                "hint_budget": self.hint_budget,
                "predicate": getattr(self, "predicate_name", None),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start", "predicate": getattr(self, "predicate_name", None)},
            "hint": "Start by requesting a hint for a Left value with few options, or propose an obvious admissible pair.",
            "turn": 0,
            "state": {
                "unmatched_left": sorted(self.left),
                "unmatched_right": sorted(self.right),
                "pairs": [],
                "errors_made": 0,
                "error_budget": self.error_budget,
                "hints_used": 0,
                "hint_budget": self.hint_budget,
            },
        }
        return obs, info