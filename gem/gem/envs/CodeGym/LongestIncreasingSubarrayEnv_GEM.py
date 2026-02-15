from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LongestIncreasingSubarrayEnvGEM(Env):
    """
    GEM-compatible environment for finding the length of the longest strictly increasing contiguous subarray.
    Actions are language-style commands wrapped in \\boxed{...} following DungeonScout conventions.

    Available actions:
    - \\boxed{observe}
    - \\boxed{len}
    - \\boxed{get i}
    - \\boxed{cmp i j}
    - \\boxed{answer N}
    """

    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **kwargs,  # 忽略其他参数，但允许可选的 arr 注入
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围
        self.complexity_params = {
            "array_length": (5, 50),  # 数组长度
            "value_range": (10, 10000),  # 数值范围（绝对值上界）
        }

        # 参数方差（用于训练时微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 50,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 可选：外部注入数组（评测或调试）
        self.initial_arr = kwargs.get("arr", None)

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.arr = []

        self.reset()

    def _apply_complexity_params(self):
        """根据 complexity 等级计算参数值"""
        normalized = min(1.0, (self.complexity - 1) / 9.0)  # [0, 1]

        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value

            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    actual_value = max(min_val, min(max_val, actual_value))

            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Task: Find the length of the longest strictly increasing contiguous subarray.\n"
            "Array is hidden. You can probe it using actions below and then submit the answer.\n"
            "Available actions (wrap commands in \\boxed{...}):\n"
            "- Observe prompt: \\boxed{observe}\n"
            "- Get array length: \\boxed{len}\n"
            "- Get element at index i (0-based): \\boxed{get i}\n"
            "- Compare elements at indices i and j: \\boxed{cmp i j}\n"
            "  Returns 1 if arr[i] > arr[j], -1 if arr[i] < arr[j], 0 if equal.\n"
            "- Submit final answer N: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Turn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        if self.initial_arr is not None:
            self.arr = list(self.initial_arr)
        else:
            # 生成在 [-value_range, value_range] 之间的整数
            lo = -self.value_range
            hi = self.value_range
            self.arr = [random.randint(lo, hi) for _ in range(self.array_length)]
        return {"data": self.arr, "size": len(self.arr)}

    def _parse_action(self, action: str) -> Optional[Dict]:
        """
        解析 \\boxed{...} 格式，返回 {"content": str} 或 None
        """
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        return {"content": content}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        content = parsed["content"].strip()
        tokens = content.split()
        cmd = tokens[0].lower() if tokens else ""

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()
                # Non-terminal probe
            elif cmd == "len":
                obs = self.GetArrayLength()
            elif cmd == "get":
                if len(tokens) != 2:
                    obs = "Invalid parameters for get. Usage: \\boxed{get i}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i = int(tokens[1])
                res = self.GetElement(i)
                if res.startswith("Error:"):
                    obs = res
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = res
            elif cmd == "cmp":
                if len(tokens) != 3:
                    obs = "Invalid parameters for cmp. Usage: \\boxed{cmp i j}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i = int(tokens[1])
                j = int(tokens[2])
                res = self.CompareElements(i, j)
                if res.startswith("Error:"):
                    obs = res
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = res
            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Invalid parameters for answer. Usage: \\boxed{answer N}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans = int(tokens[1])
                ref_answer = self.get_ref_answer()
                correct = (ans == ref_answer)
                obs = f"Your answer: {ans}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
                reward = 1.0 if correct else -1.0
                terminated = True
            else:
                obs = f"Invalid action: {cmd}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
        except Exception as e:
            obs = f"Format error: {e}"
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    # -------------------------
    # Helper methods (preserved and adapted from original)
    # -------------------------
    def GetArrayLength(self) -> str:
        """
        Get the length of the array.
        Returns:
            str: The length of the array.
        Example Output: "5"
        """
        return str(len(self.arr))

    def CompareElements(self, index1: int, index2: int) -> str:
        """
        Compare the sizes of elements at two indices in the array.
        Args:
            index1 (int): The index of the first element.
            index2 (int): The index of the second element.
        Returns:
            str: The comparison result,
                 1 means arr[index1] > arr[index2],
                -1 means arr[index1] < arr[index2],
                 0 means equal.
        Example Output: "1"
        """
        if index1 < 0 or index1 >= len(self.arr) or index2 < 0 or index2 >= len(self.arr):
            return "Error: index out of range"

        if self.arr[index1] > self.arr[index2]:
            return "1"
        elif self.arr[index1] < self.arr[index2]:
            return "-1"
        else:
            return "0"

    def GetElement(self, index: int) -> str:
        """
        Get the value of the element at the specified index in the array.
        Args:
            index (int): index of the element.
        Returns:
            str: The value or error message.
        Example Output: "5"
        """
        if index < 0 or index >= len(self.arr):
            return "Error: index out of range"
        return str(self.arr[index])

    def Observe(self) -> str:
        """
        Return the observation information of the current state.
        """
        return "Please analyze the array to find the length of the longest strictly increasing contiguous subarray."

    def get_ref_answer(self) -> int:
        """
        Compute the length of the longest strictly increasing contiguous subarray.
        """
        if not self.arr:
            return 0

        max_length = 1
        current_length = 1

        for i in range(1, len(self.arr)):
            if self.arr[i] > self.arr[i - 1]:
                current_length += 1
                if current_length > max_length:
                    max_length = current_length
            else:
                current_length = 1

        return max_length

    # Optional utility: provide a random legal action
    def sample_random_action(self) -> str:
        if not self.arr:
            return "\\boxed{answer 0}"
        # Randomly choose an action
        choices = ["observe", "len", "get", "cmp"]
        choice = random.choice(choices)
        if choice == "observe":
            return "\\boxed{observe}"
        elif choice == "len":
            return "\\boxed{len}"
        elif choice == "get":
            i = random.randint(0, max(0, len(self.arr) - 1)) if self.arr else 0
            return f"\\boxed{{get {i}}}"
        else:
            if len(self.arr) >= 2:
                i = random.randint(0, len(self.arr) - 1)
                j = random.randint(0, len(self.arr) - 1)
                return f"\\boxed{{cmp {i} {j}}}"
            else:
                return "\\boxed{len}"

    # Non-essential but preserved for compatibility with original environment style
    def solve(self) -> str:
        """
        A simple internal solver not using the step language actions.
        Returns a human-readable string consistent with 'answer' outcome.
        """
        if not self.arr:
            ans = 0
            ref = self.get_ref_answer()
            result = "Correct" if ans == ref else "Incorrect"
            return f"Your answer: {ans}, Reference answer: {ref}, Result: {result}"

        max_length = 1
        current_length = 1
        n = len(self.arr)
        for i in range(n - 1):
            if self.arr[i + 1] > self.arr[i]:
                current_length += 1
                if current_length > max_length:
                    max_length = current_length
            else:
                current_length = 1

        ref = self.get_ref_answer()
        result = "Correct" if max_length == ref else "Incorrect"
        return f"Your answer: {max_length}, Reference answer: {ref}, Result: {result}"