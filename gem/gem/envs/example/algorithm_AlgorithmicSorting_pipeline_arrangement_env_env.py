from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmicSortingEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            "list_length": (5, 25),          # Number of elements: larger lists increase state size and planning complexity
            "value_range": (10, 100),        # Range of values: wider ranges increase variability and inversions potential
            "slack_budget": (10, 0),         # REVERSED: extra swaps allowed above optimal; less slack makes budget tighter and harder
            "min_inv_percent": (10, 70),     # Target min inversion ratio (% of max possible inversions), more disorder is harder
        }
        self.param_variance = {
            "list_length": 2,        # ±2 within 5-25
            "value_range": 10,       # ±10 within 10-100
            "slack_budget": 2,       # ±2 within 0-10
            "min_inv_percent": 6,    # ±6 within 10-70
        }

        # Placeholders for evolvable params (set in _apply_complexity_params)
        self.list_length: int = 0
        self.value_range: int = 0
        self.slack_budget: int = 0
        self.min_inv_percent: int = 0

        # Domain state
        self.turn_count: int = 0
        self.sequence: list = []
        self.swap_count: int = 0
        self.max_swaps_allowed: int = 0
        self.optimal_swaps: int = 0
        self.last_parsed_action: Optional[Dict[str, Any]] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                var = self.param_variance.get(param_name, 0)
                if var > 0:
                    actual_value = center_value + random.uniform(-var, var)
            # Clamp to range, supporting reversed params
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _compute_inversions(self, seq_pairs: list) -> int:
        # Count inversions for stable ascending sort: count pairs (i<j, val[i] > val[j])
        vals = [v for v, _idx in seq_pairs]
        n = len(vals)
        inv = 0
        for i in range(n):
            vi = vals[i]
            for j in range(i + 1, n):
                if vi > vals[j]:
                    inv += 1
        return inv

    def _is_stably_sorted(self, seq_pairs: list) -> bool:
        for i in range(len(seq_pairs) - 1):
            v1, idx1 = seq_pairs[i]
            v2, idx2 = seq_pairs[i + 1]
            if v1 > v2:
                return False
            if v1 == v2 and idx1 > idx2:
                return False
        return True

    def _generate_instance(self):
        # Generate sequence with at least a target inversion ratio
        n = self.list_length
        target_ratio = self.min_inv_percent / 100.0
        max_possible = n * (n - 1) // 2
        attempts = 0
        while True:
            attempts += 1
            seq = [random.randint(0, self.value_range - 1) for _ in range(n)]
            seq_pairs = [(seq[i], i) for i in range(n)]
            inv = self._compute_inversions(seq_pairs)
            if max_possible == 0 or inv / max_possible >= target_ratio or attempts > 100:
                return seq_pairs, inv

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Algorithmic Sorting Game\n"
            "Goal: Stably sort the given list in ascending order using adjacent swaps and achieve the minimum number of swaps.\n"
            "Optimal swaps equal the number of inversions (pairs out of order by value). You have a swap budget: do not exceed it.\n"
            "Rules:\n"
            "- Allowed action: swap i j where j = i+1 and 0 <= i < len(list)-1.\n"
            "- Use zero-based indices.\n"
            "- 'submit' ends the episode and evaluates your result.\n"
            "- You must not exceed the swap budget.\n"
            f"Format: use \\boxed{{swap i j}} or \\boxed{{submit}}.\n"
            f"Example: {example}\n"
        )

    def get_task_suffix(self) -> str:
        vals = [str(v) for v, _idx in self.sequence]
        remaining = max(0, self.max_swaps_allowed - self.swap_count)
        return (
            f"Current list: {' '.join(vals)}\n"
            f"List length: {len(self.sequence)} | Swaps used: {self.swap_count} | Budget: {self.max_swaps_allowed} | Remaining: {remaining}\n"
            f"Turns: {self.turn_count}/{self.max_turns}\n"
            "Enter your action as \\boxed{swap i i+1} or \\boxed{submit}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.sequence, self.optimal_swaps = self._generate_instance()
        self.swap_count = 0
        self.max_swaps_allowed = self.optimal_swaps + self.slack_budget
        self.turn_count = 0
        self.last_parsed_action = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        self.last_parsed_action = parsed

        if parsed is None:
            obs = "Invalid action format. Use \\boxed{swap i i+1} or \\boxed{submit}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        if parsed["type"] == "unsupported":
            obs = f"Unsupported action: '{parsed.get('raw', '')}'. Allowed: swap i i+1 or submit."
            return (
                obs,
                -0.25,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        if parsed["type"] == "swap":
            i = parsed["i"]
            j = parsed["j"]
            n = len(self.sequence)

            if self.swap_count >= self.max_swaps_allowed:
                obs = "Protocol violation: swap budget exceeded. Episode terminated."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

            if i < 0 or j < 0 or i >= n or j >= n:
                obs = "Protocol violation: indices out of range. Episode terminated."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

            if j != i + 1:
                obs = "Protocol violation: non-adjacent swap requested. Episode terminated."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

            # Perform swap of adjacent elements (both value and original index)
            self.sequence[i], self.sequence[j] = self.sequence[j], self.sequence[i]
            self.swap_count += 1

            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns})."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

            vals = [str(v) for v, _idx in self.sequence]
            remaining = max(0, self.max_swaps_allowed - self.swap_count)
            obs = (
                f"Swap performed at positions {i} and {j}. "
                f"List now: {' '.join(vals)}. Swaps used: {self.swap_count}/{self.max_swaps_allowed}. "
                "Continue or submit."
            )
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        if parsed["type"] == "submit":
            is_sorted = self._is_stably_sorted(self.sequence)
            if not is_sorted:
                obs = "Failed: submitted list is not stably sorted ascending."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            if self.swap_count == self.optimal_swaps:
                obs = (
                    f"Success! Stably sorted with minimal swaps ({self.swap_count}). "
                    f"Optimal swaps = {self.optimal_swaps}."
                )
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

            obs = (
                f"Valid but suboptimal: sorted stably using {self.swap_count} swaps; "
                f"optimal is {self.optimal_swaps}."
            )
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        obs = "Unknown internal state."
        return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()

        if re.fullmatch(r'submit', extracted, flags=re.IGNORECASE):
            return {"type": "submit"}

        m = re.fullmatch(r'swap\s+(\d+)\s+(\d+)', extracted, flags=re.IGNORECASE)
        if m:
            i = int(m.group(1))
            j = int(m.group(2))
            return {"type": "swap", "i": i, "j": j}

        return {"type": "unsupported", "raw": extracted}

    def sample_random_action(self) -> str:
        if not self.sequence or len(self.sequence) < 2:
            return "\\boxed{submit}"
        i = random.randint(0, len(self.sequence) - 2)
        return f"\\boxed{{swap {i} {i+1}}}"


class AlgorithmicSortingEnvWithFeedback(AlgorithmicSortingEnv):
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
            error_detail["issue"] = "missing_boxed_or_bad_format"
            hint = "Use \\boxed{swap i i+1} or \\boxed{submit} with zero-based indices."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["raw_action"] = getattr(self, "last_parsed_action", {}).get("raw", "")
            hint = "Allowed actions are exactly: swap i i+1 and submit."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "non-adjacent" in text:
                error_detail["violation"] = "non_adjacent_swap"
                hint = "Choose adjacent positions only: j must equal i+1."
            elif "out of range" in text:
                error_detail["violation"] = "index_out_of_range"
                hint = "Use indices within 0..len(list)-1 and adjacent j=i+1."
            elif "budget exceeded" in text:
                error_detail["violation"] = "swap_budget_exceeded"
                hint = "Submit your result when near the budget; avoid extra swaps."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Act faster: perform adjacent swaps and submit before reaching the turn limit."

        elif "failed" in text:
            error_type = "WrongDecision"
            error_detail["reason"] = "not_sorted"
            hint = "Continue swapping adjacent out-of-order pairs until the list is ascending (stable ties)."

        elif "valid but suboptimal" in text:
            error_type = "WrongDecision"
            error_detail["reason"] = "suboptimal_swaps"
            hint = "Use insertion-like strategy: move each element left by swapping until it reaches its correct position; this matches the inversion count."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        # Add state info
        if self.feedback_level >= 1:
            state_vals = [v for v, _idx in getattr(self, "sequence", [])]
            error_detail["list_length"] = len(state_vals)
            error_detail["used_swaps"] = getattr(self, "swap_count", None)
            error_detail["optimal_swaps"] = getattr(self, "optimal_swaps", None)
            remaining = max(0, getattr(self, "max_swaps_allowed", 0) - getattr(self, "swap_count", 0))
            error_detail["remaining_budget"] = remaining
            error_detail["sequence"] = state_vals

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        hint = "Start by swapping adjacent out-of-order pairs; plan to match the inversion count."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": hint if self.feedback_level >= 2 else None,
            "turn": 0,
        }
        return obs, info