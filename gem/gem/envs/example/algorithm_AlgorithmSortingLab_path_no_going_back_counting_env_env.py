from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmSortingLabEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = None,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self._base_max_turns = max_turns  # if None, computed each reset based on budgets

        # Evolvable parameters
        self.complexity_params = {
            # Problem size: larger arrays increase state space and planning complexity
            'array_length': (5, 30),
            # Comparison budget factor x10: smaller factor => fewer comparisons => harder (REVERSED via range)
            # compare_budget = int((compare_factor_x10/10) * n * log2(n+1))
            'compare_factor_x10': (70, 30),
            # Swap budget factor x10: smaller factor => fewer swaps => harder (REVERSED via range)
            # swap_budget = int((swap_factor_x10/10) * n)
            'swap_factor_x10': (40, 20),
            # Max swap distance ratio x100: smaller distance => more local moves needed => harder (REVERSED)
            'max_swap_distance_ratio_x100': (100, 50),
            # Disorder ratio x100: higher disorder => harder sorting
            'disorder_ratio_x100': (20, 100),
            # Duplicates ratio x100: higher duplicates introduce equality cases => slightly harder
            'duplicates_ratio_x100': (0, 50),
        }

        # Variance settings
        self.param_variance = {
            'array_length': 1,                   # ±1 within [5,30]
            'compare_factor_x10': 3,             # ±3 around center (x10 scale)
            'swap_factor_x10': 2,                # ±2 around center (x10 scale)
            'max_swap_distance_ratio_x100': 5,   # ±5% points
            'disorder_ratio_x100': 5,            # ±5% points
            'duplicates_ratio_x100': 5,          # ±5% points
        }

        # Placeholder attributes
        self.array_length: int = 0
        self.compare_factor_x10: int = 0
        self.swap_factor_x10: int = 0
        self.max_swap_distance_ratio_x100: int = 0
        self.disorder_ratio_x100: int = 0
        self.duplicates_ratio_x100: int = 0

        # State
        self.arr = []
        self.turn_count: int = 0
        self.compare_budget: int = 0
        self.swap_budget: int = 0
        self.used_compares: int = 0
        self.used_swaps: int = 0
        self.max_swap_distance: int = 0

        # Effective max turns used per episode (set in reset)
        self.max_turns: int = 0

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
            "Algorithm Sorting Lab:\n"
            "- A hidden array of length n must be sorted in non-decreasing order using only three actions:\n"
            "  1) compare i j  -> reveals whether A[i] < A[j], A[i] = A[j], or A[i] > A[j]\n"
            "  2) swap i j     -> swaps the elements at positions i and j (1-based indices)\n"
            "  3) submit       -> finishes the episode; success if the hidden array is sorted\n"
            "- You cannot see the array values directly. Plan using comparisons and apply swaps to sort.\n"
            "- Constraints:\n"
            "  • You have limited budgets for comparisons and swaps.\n"
            "  • Swaps must satisfy |i - j| <= max_swap_distance.\n"
            "  • Indices are 1-based and must be within [1, n].\n"
            "- The episode ends on 'submit', invalid format, running out of allowed operations, or reaching max turns.\n"
            "- Rewards:\n"
            "  • Success (array sorted) yields 1.0.\n"
            "  • Invalid format incurs a small penalty and ends the episode.\n"
            "  • Other steps yield 0.0.\n"
            "Format your action as \\boxed{your_command}. Examples:\n"
            f"  {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"State:\n"
            f"- n = {self.array_length}\n"
            f"- compare_budget_left = {self.compare_budget - self.used_compares}\n"
            f"- swap_budget_left = {self.swap_budget - self.used_swaps}\n"
            f"- max_swap_distance = {self.max_swap_distance}\n"
            f"- turn = {self.turn_count}\n"
            "Valid commands inside \\boxed{...}: 'compare i j', 'swap i j', or 'submit'. "
            "Indices are 1-based. Enter your action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        n = self.array_length
        dup_ratio = self.duplicates_ratio_x100 / 100.0
        dis_ratio = self.disorder_ratio_x100 / 100.0

        # Build a non-decreasing multiset with a target number of distinct values
        num_distinct = max(2, n - int(round(dup_ratio * n)))
        base_vals = list(range(1, num_distinct + 1))
        # Distribute counts across distinct values to fill n
        counts = [1] * num_distinct
        remaining = n - num_distinct
        while remaining > 0:
            idx = random.randint(0, num_distinct - 1)
            counts[idx] += 1
            remaining -= 1
        multiset = []
        for v, c in zip(base_vals, counts):
            multiset.extend([v] * c)
        multiset.sort()

        # Apply disorder: for high disorder random shuffle, else do local adjacent swaps
        arr = multiset[:]
        if dis_ratio >= 0.8:
            random.shuffle(arr)
        else:
            k = max(1, int(round(dis_ratio * n)))
            for _ in range(k):
                if n <= 1:
                    break
                p = random.randint(0, n - 2)
                arr[p], arr[p + 1] = arr[p + 1], arr[p]

        self.arr = arr
        self.turn_count = 0
        self.used_compares = 0
        self.used_swaps = 0

        # Budgets and constraints
        import math
        compare_factor = self.compare_factor_x10 / 10.0
        swap_factor = self.swap_factor_x10 / 10.0
        self.compare_budget = max(1, int(round(compare_factor * n * max(1.0, math.log2(n + 1)))))
        self.swap_budget = max(1, int(round(swap_factor * n)))
        self.max_swap_distance = max(1, int(round((self.max_swap_distance_ratio_x100 / 100.0) * max(1, n - 1))))

        # Effective max turns
        if self._base_max_turns is None:
            self.max_turns = self.compare_budget + self.swap_budget + 15
        else:
            self.max_turns = int(self._base_max_turns)

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed.get("cmd")
        info_suffix = {"suffix": self.get_task_suffix()}

        # Helper for indices validation
        def valid_idx(i: int) -> bool:
            return 1 <= i <= self.array_length

        # Forced termination if no operations left before executing action (except submit)
        if cmd != "submit" and (self.used_compares >= self.compare_budget and self.used_swaps >= self.swap_budget):
            obs = "No operations remaining; episode ends."
            return obs, 0.0, True, False, info_suffix

        if cmd == "compare":
            i, j = parsed.get("i"), parsed.get("j")
            if i is None or j is None or not valid_idx(i) or not valid_idx(j):
                obs = f"Protocol violation: indices out of range. Use 1..{self.array_length}."
                return obs, 0.0, False, False, info_suffix
            if self.used_compares >= self.compare_budget:
                obs = "Protocol violation: compare budget exhausted."
                return obs, 0.0, False, False, info_suffix
            ai = self.arr[i - 1]
            aj = self.arr[j - 1]
            self.used_compares += 1
            if ai < aj:
                rel = "<"
            elif ai > aj:
                rel = ">"
            else:
                rel = "="
            obs = (
                f"Compare result: A[{i}] {rel} A[{j}]. "
                f"Budgets left - compare: {self.compare_budget - self.used_compares}, "
                f"swap: {self.swap_budget - self.used_swaps}."
            )
            # Check turns timeout after a valid move
            if self.turn_count >= self.max_turns:
                return f"{obs} Reached max turns.", 0.0, True, True, info_suffix
            return obs, 0.0, False, False, info_suffix

        elif cmd == "swap":
            i, j = parsed.get("i"), parsed.get("j")
            if i is None or j is None or not valid_idx(i) or not valid_idx(j):
                obs = f"Protocol violation: indices out of range. Use 1..{self.array_length}."
                return obs, 0.0, False, False, info_suffix
            if self.used_swaps >= self.swap_budget:
                obs = "Protocol violation: swap budget exhausted."
                return obs, 0.0, False, False, info_suffix
            if abs(i - j) > self.max_swap_distance:
                obs = (
                    f"Protocol violation: swap distance exceeds max ({self.max_swap_distance}). "
                    "Use a sequence of shorter swaps."
                )
                return obs, 0.0, False, False, info_suffix
            # Perform swap
            self.arr[i - 1], self.arr[j - 1] = self.arr[j - 1], self.arr[i - 1]
            self.used_swaps += 1
            obs = (
                f"Swapped positions {i} and {j}. "
                f"Budgets left - compare: {self.compare_budget - self.used_compares}, "
                f"swap: {self.swap_budget - self.used_swaps}."
            )
            if self.turn_count >= self.max_turns:
                return f"{obs} Reached max turns.", 0.0, True, True, info_suffix
            return obs, 0.0, False, False, info_suffix

        elif cmd == "submit":
            # Check sortedness
            is_sorted = all(self.arr[k] <= self.arr[k + 1] for k in range(len(self.arr) - 1))
            if is_sorted:
                obs = (
                    "Success! The hidden array is sorted in non-decreasing order. "
                    f"Used compares: {self.used_compares}/{self.compare_budget}, "
                    f"swaps: {self.used_swaps}/{self.swap_budget}."
                )
                return obs, 1.0, True, False, info_suffix
            else:
                obs = (
                    "Failed! The array is not sorted. "
                    f"Used compares: {self.used_compares}/{self.compare_budget}, "
                    f"swaps: {self.used_swaps}/{self.swap_budget}."
                )
                return obs, 0.0, True, False, info_suffix

        else:
            obs = "Unsupported action. Valid commands: compare i j, swap i j, submit."
            if self.turn_count >= self.max_turns:
                return f"{obs} Reached max turns.", 0.0, True, True, info_suffix
            return obs, 0.0, False, False, info_suffix

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        # Normalize spaces
        tokens = re.split(r'\s+', content.lower())
        if not tokens:
            return {"cmd": "unknown", "raw": content}
        cmd = tokens[0]
        if cmd == "submit":
            return {"cmd": "submit"}
        if cmd in ("compare", "swap"):
            if len(tokens) != 3:
                return {"cmd": "unknown", "raw": content}
            try:
                i = int(tokens[1])
                j = int(tokens[2])
            except ValueError:
                return {"cmd": "unknown", "raw": content}
            return {"cmd": cmd, "i": i, "j": j}
        return {"cmd": "unknown", "raw": content}

    def sample_random_action(self) -> str:
        if self.array_length <= 1:
            return "\\boxed{submit}"
        op = random.choice(["compare", "swap", "compare", "compare"])
        i = random.randint(1, max(1, self.array_length))
        j = random.randint(1, max(1, self.array_length))
        if op == "swap":
            # Adjust to likely respect distance
            if abs(i - j) > max(1, self.max_swap_distance):
                j = max(1, min(self.array_length, i + random.randint(-self.max_swap_distance, self.max_swap_distance)))
                if j == i:
                    j = max(1, min(self.array_length, i + (1 if i < self.array_length else -1)))
        return f"\\boxed{{{op} {i} {j}}}"


class AlgorithmSortingLabEnvWithFeedback(AlgorithmSortingLabEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        # Classify errors/outcomes
        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command in \\boxed{...}, e.g., \\boxed{compare 1 2}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["expected"] = ["compare i j", "swap i j", "submit"]
            hint = "Use 'compare i j', 'swap i j', or 'submit'."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "indices out of range" in text:
                error_detail["violation"] = "index_out_of_range"
                error_detail["valid_range"] = [1, getattr(self, "array_length", None)]
                hint = f"Use 1-based indices within [1, {self.array_length}]."
            elif "compare budget exhausted" in text:
                error_detail["violation"] = "compare_budget_exhausted"
                hint = "Reduce comparisons; rely on previous outcomes or proceed to 'submit' when confident."
            elif "swap budget exhausted" in text:
                error_detail["violation"] = "swap_budget_exhausted"
                hint = "Plan fewer swaps; first compare to ensure swaps improve order."
            elif "swap distance exceeds max" in text:
                error_detail["violation"] = "swap_distance_limit"
                error_detail["max_swap_distance"] = getattr(self, "max_swap_distance", None)
                hint = "Use a sequence of shorter swaps to move elements gradually."
            else:
                error_detail["violation"] = "unknown_protocol_violation"
                hint = "Check budgets and index constraints before acting."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["reason"] = "max_turns"
            hint = "Act more decisively; prioritize comparisons that reduce uncertainty and swap when clear."
        elif "no operations remaining" in text:
            error_type = "Timeout"
            error_detail["reason"] = "no_operations_left"
            hint = "Budget carefully; avoid redundant comparisons and prefer swaps that fix multiple inversions."
        elif "failed!" in text:
            error_type = "WrongDecision"
            error_detail["outcome"] = "submit_on_unsorted"
            hint = "Perform additional comparisons near suspected inversions and swap locally within the distance limit."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        else:
            # Normal step (compare/swap result)
            error_type = "OK"
            if "compare result" in text:
                error_detail["action"] = "compare"
            elif "swapped positions" in text:
                error_detail["action"] = "swap"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["budgets"] = {
                "compare_left": getattr(self, "compare_budget", 0) - getattr(self, "used_compares", 0),
                "swap_left": getattr(self, "swap_budget", 0) - getattr(self, "used_swaps", 0),
            }
            diagnostic["max_swap_distance"] = getattr(self, "max_swap_distance", None)
            diagnostic["n"] = getattr(self, "array_length", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by comparing distant indices to locate large inversions, then use swaps within the distance limit.",
            "turn": 0,
            "n": getattr(self, "array_length", None),
            "budgets": {
                "compare_left": getattr(self, "compare_budget", 0),
                "swap_left": getattr(self, "swap_budget", 0),
            },
            "max_swap_distance": getattr(self, "max_swap_distance", None),
        }
        return obs, info