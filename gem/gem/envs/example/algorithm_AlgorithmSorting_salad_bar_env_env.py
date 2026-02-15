from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class AlgorithmSortingEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 200,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 200

        # Evolvable parameters
        self.complexity_params = {
            # Array length: longer arrays have more inversions and require more steps → harder
            'array_len': (5, 24),
            # REVERSED: budget multiplier on minimal swap cost; lower multiplier means tighter budget → harder
            'budget_multiplier': (3, 1),
            # Swap cost: each swap consumes more budget at higher levels → harder
            'swap_cost': (1, 3),
            # Compare cost: comparisons become costly at higher levels → harder
            'compare_cost': (0, 1),
            # REVERSED: extra slack beyond minimal cost; less slack makes it harder
            'swap_buffer': (12, 0),
            # Value range: broader range increases uniqueness but does not reduce solvability; included for variety
            'value_range': (9, 99),
        }

        # Variance settings for parameters
        self.param_variance = {
            'array_len': 1,
            'budget_multiplier': 0,
            'swap_cost': 0,
            'compare_cost': 0,
            'swap_buffer': 2,
            'value_range': 8,
        }

        # Placeholder/effective parameters
        self.array_len: int = 0
        self.budget_multiplier: int = 0
        self.swap_cost: int = 0
        self.compare_cost: int = 0
        self.swap_buffer: int = 0
        self.value_range: int = 0

        # State
        self.turn_count: int = 0
        self.array: list = []
        self.budget_left: int = 0
        self.inversion_target: int = 0
        self.adjacent_only: bool = True
        self.operation_history: list = []

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
                    low = min(min_val, max_val)
                    high = max(min_val, max_val)
                    actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _compute_inversions(self, arr: list) -> int:
        inv = 0
        n = len(arr)
        for i in range(n):
            ai = arr[i]
            for j in range(i + 1, n):
                if ai > arr[j]:
                    inv += 1
        return inv

    def _is_sorted(self, arr: list) -> bool:
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:
                return False
        return True

    def _get_instructions(self) -> str:
        return (
            "Algorithm Sorting Planner.\n"
            "Goal: Sort the array in nondecreasing order using adjacent operations while staying within budget.\n"
            "You may perform:\n"
            "- compare i j: reveal relation between A[i] and A[j] (adjacent indices only: j=i+1)\n"
            "- swap i j: swap A[i] and A[j] (adjacent indices only: j=i+1)\n"
            "- done: declare completion when you believe the array is sorted\n"
            "Costs: swap consumes swap_cost budget; compare consumes compare_cost budget.\n"
            "Invalid formats terminate the episode with a penalty.\n"
            "Use \\boxed{...} to submit actions.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        inv_remaining = self._compute_inversions(self.array)
        return (
            f"Instance: array={self.array} (length={len(self.array)}), "
            f"budget_left={self.budget_left} (swap_cost={self.swap_cost}, compare_cost={self.compare_cost}). "
            f"Inversions remaining: {inv_remaining}. "
            "Allowed actions: compare i j (adjacent only), swap i j (adjacent only), done. "
            "Format: use \\boxed{compare i j}, \\boxed{swap i j}, or \\boxed{done}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        self.turn_count = 0
        self.operation_history = []
        self.array = []
        # Ensure not pre-sorted to keep challenge meaningful
        while True:
            self.array = [random.randint(1, self.value_range) for _ in range(self.array_len)]
            if not self._is_sorted(self.array):
                break
        self.inversion_target = self._compute_inversions(self.array)
        base_swap_cost = self.swap_cost * self.inversion_target
        self.budget_left = self.budget_multiplier * base_swap_cost + self.swap_buffer

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if isinstance(parsed, dict) and parsed.get('type') == 'unknown':
            obs = f"Unsupported action '{parsed.get('raw', '')}'. Allowed: compare, swap, done."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if isinstance(parsed, dict) and parsed.get('type') == 'done':
            if self._is_sorted(self.array):
                obs = "Success! Array is sorted and you declared done."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Failed! Array is not sorted at done."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if isinstance(parsed, dict) and parsed.get('type') in ('compare', 'swap'):
            i = parsed.get('i')
            j = parsed.get('j')
            if not isinstance(i, int) or not isinstance(j, int):
                obs = "Protocol violation: indices must be integers."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            n = len(self.array)
            if i < 0 or j < 0 or i >= n or j >= n:
                obs = "Protocol violation: indices out of range."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.adjacent_only and j != i + 1:
                obs = "Protocol violation: only adjacent indices allowed (j must equal i+1)."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if parsed['type'] == 'compare':
                self.budget_left -= self.compare_cost
                a_i, a_j = self.array[i], self.array[j]
                relation = "<" if a_i < a_j else ">" if a_i > a_j else "="
                self.operation_history.append(('compare', i, j, relation))
                if self.budget_left < 0:
                    obs = f"Budget exceeded after compare {i} {j}."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                inv_remain = self._compute_inversions(self.array)
                obs = (
                    f"Compared A[{i}]={a_i} {relation} A[{j}]={a_j}. "
                    f"Budget_left={self.budget_left}. Inversions remaining: {inv_remain}."
                )
                reward = 0.0
                # Not terminal unless timeout
                if self.turn_count >= self.max_turns:
                    obs = f"Reached max turns ({self.max_turns})."
                    return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

            if parsed['type'] == 'swap':
                self.budget_left -= self.swap_cost
                a_i, a_j = self.array[i], self.array[j]
                self.array[i], self.array[j] = self.array[j], self.array[i]
                self.operation_history.append(('swap', i, j))
                if self.budget_left < 0:
                    obs = f"Budget exceeded after swap {i} {j}."
                    return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
                inv_remain = self._compute_inversions(self.array)
                if self._is_sorted(self.array):
                    obs = (
                        f"Performed swap {i} {j}. Array now sorted. "
                        f"Budget_left={self.budget_left}. Submit \\boxed{{done}} to finish."
                    )
                else:
                    obs = (
                        f"Performed swap {i} {j}. Array={self.array}. "
                        f"Budget_left={self.budget_left}. Inversions remaining: {inv_remain}."
                    )
                reward = 0.0
                if self.turn_count >= self.max_turns:
                    obs = f"Reached max turns ({self.max_turns})."
                    return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        obs = "Unexpected state."
        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip()
        if extracted.lower() == 'done':
            return {'type': 'done'}
        m = re.match(r'^(compare|swap)\s+(-?\d+)\s+(-?\d+)\s*$', extracted, flags=re.IGNORECASE)
        if m:
            op = m.group(1).lower()
            i = int(m.group(2))
            j = int(m.group(3))
            return {'type': op, 'i': i, 'j': j}
        return {'type': 'unknown', 'raw': extracted}

    def sample_random_action(self) -> str:
        if len(self.array) >= 2:
            i = random.randint(0, len(self.array) - 2)
            if random.random() < 0.5:
                return f"\\boxed{{compare {i} {i+1}}}"
            else:
                return f"\\boxed{{swap {i} {i+1}}}"
        else:
            return "\\boxed{done}"


class AlgorithmSortingEnvWithFeedback(AlgorithmSortingEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        inv_remaining = self._compute_inversions(self.array)
        state_snapshot = {
            "length": len(self.array),
            "budget_left": self.budget_left,
            "swap_cost": self.swap_cost,
            "compare_cost": self.compare_cost,
            "inversions_remaining": inv_remaining,
            "inversion_target": self.inversion_target,
            "turn": self.turn_count,
        }

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Submit actions as \\boxed{compare i i+1}, \\boxed{swap i i+1}, or \\boxed{done}."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "adjacent indices" in text:
                error_detail["violation"] = "non_adjacent_indices"
                hint = "Use adjacent indices: choose i in [0..n-2] and j=i+1."
            elif "out of range" in text:
                error_detail["violation"] = "index_out_of_range"
                hint = "Indices must be within [0..n-1]; j must equal i+1."
            else:
                error_detail["violation"] = "invalid_indices"
                hint = "Provide integer indices; adjacent-only operations are allowed."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["raw"] = obs
            hint = "Only compare, swap, and done are supported."

        elif "budget exceeded" in text:
            error_type = "WrongDecision"
            error_detail["cause"] = "budget_exceeded"
            hint = "Reduce unnecessary compares and focus swaps on adjacent inversions (left > right)."

        elif "failed! array is not sorted" in text:
            error_type = "WrongDecision"
            error_detail["cause"] = "premature_done"
            hint = "Continue fixing adjacent inversions until no pair A[i] > A[i+1] remains, then use done."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan moves efficiently; target adjacent inversions and avoid extra compares."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = {**error_detail, **state_snapshot}
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        diagnostic = {
            "error_type": "OK",
            "error_detail": {
                "outcome": "episode_start",
                "length": len(self.array),
                "budget_left": self.budget_left,
                "swap_cost": self.swap_cost,
                "compare_cost": self.compare_cost,
                "inversions_remaining": self._compute_inversions(self.array),
                "inversion_target": self.inversion_target,
                "turn": 0,
            },
            "hint": "Start by scanning for adjacent pairs where A[i] > A[i+1] and swap them.",
        }
        info["diagnostic"] = diagnostic
        return obs, info