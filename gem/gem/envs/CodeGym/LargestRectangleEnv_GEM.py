from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LargestRectangleEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围（根据原环境分析）
        # - array_length: 直方图数组长度
        # - value_range: 柱高取值范围（1..value_range）
        # - max_turns: 允许的最大步数（根据难度自动设置）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (5, 100),
            "max_turns": (20, 200),
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 10,
            "max_turns": 10,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0

        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.heights: list[int] = []

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

            ivalue = int(round(actual_value))
            if param_name == "array_length":
                self.array_length = ivalue
            elif param_name == "value_range":
                self.value_range = ivalue
            elif param_name == "max_turns":
                # 用难度控制步数上限
                self.max_turns = ivalue

    def _get_instructions(self) -> str:
        return (
            "Largest Rectangle in Histogram:\n"
            "You are given a histogram with non-negative integer heights.\n"
            "Find the largest rectangle area that can be formed.\n"
            "Available actions (use \\boxed{...}):\n"
            "- Observe prompt: \\boxed{observe}\n"
            "- Get bar count (including sentinel 0): \\boxed{count}\n"
            "- Get bar height at index i: \\boxed{height i}\n"
            "- Calculate area h*w: \\boxed{area h w}\n"
            "- Update max area: \\boxed{update current max}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "Notes:\n"
            "- Index i in [0..N-1] refers to actual bars; i==N (sentinel) returns height 0.\n"
            "- The sentinel 0 simplifies stack-based algorithms.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array length: {len(self.heights)} | Value range: 1..{self.value_range} | "
            f"Turn: {self.turn_count}/{self.max_turns}\nEnter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.heights = self.problem["heights"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 生成非负整数的柱高（至少 1，允许 0 以增加复杂度）
        heights = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        # 可选：在部分情况下插入 0 以改变地形（提高复杂度）
        if self.enable_param_randomization and self.array_length >= 5:
            zero_slots = random.randint(0, max(1, self.array_length // 10))
            for _ in range(zero_slots):
                idx = random.randint(0, self.array_length - 1)
                heights[idx] = 0
        return {"heights": heights, "size": self.array_length}

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

        content = parsed["content"]
        tokens = content.split()
        if len(tokens) == 0:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()

            elif cmd == "count":
                obs = self.GetBuildingCount()

            elif cmd == "height":
                if len(tokens) != 2:
                    obs = f"Invalid parameters for 'height'."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                idx = int(tokens[1])
                obs = self.GetBuildingHeight(idx)

            elif cmd == "area":
                if len(tokens) != 3:
                    obs = f"Invalid parameters for 'area'."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                h = int(tokens[1])
                w = int(tokens[2])
                obs = self.CalculateRectangleArea(h, w)

            elif cmd == "update":
                if len(tokens) != 3:
                    obs = f"Invalid parameters for 'update'."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                current_area = int(tokens[1])
                max_area = int(tokens[2])
                obs = self.UpdateMaxArea(current_area, max_area)

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = f"Invalid parameters for 'answer'."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                answer = int(tokens[1])
                # 验证答案并决定奖励
                done_msg = self.Done(answer)
                correct = self._is_correct_answer(answer)
                obs = done_msg
                reward = 1.0 if correct else -1.0
                terminated = True
                truncated = False

            else:
                obs = f"Invalid action: {tokens[0]}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except Exception as e:
            obs = f"Format/Execution error: {str(e)}"
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

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        return {"content": content}

    def sample_random_action(self) -> str:
        # 提供一个示例动作
        if random.random() < 0.5:
            return "\\boxed{observe}"
        else:
            return "\\boxed{count}"

    # 辅助方法（保留并转换自原环境）

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        max_area = 0
        stack = []
        temp_heights = self.heights.copy()
        temp_heights.append(0)

        for i, h in enumerate(temp_heights):
            while stack and temp_heights[stack[-1]] >= h:
                height = temp_heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)

        return max_area

    def _is_correct_answer(self, answer: int) -> bool:
        return answer == self.get_ref_answer()

    def GetBuildingHeight(self, index: int) -> str:
        r"""
        Gets the height of the building at the specified index.

        Args:
            index (int): The index of the building.

        Returns:
            str: The height of the building at the specified index.

        Example Output:
            "5"
        """
        # index in [0..len(heights)] — len(heights) returns sentinel 0
        if 0 <= index < len(self.heights):
            return str(int(self.heights[index]))
        elif index == len(self.heights):
            return "0"  # sentinel 0 height
        return "0"

    def GetBuildingCount(self) -> str:
        r"""
        Gets the total number of buildings (including the 0-height sentinel).

        Returns:
            str: The total number of buildings.

        Example Output:
            "6"
        """
        return str(len(self.heights) + 1)  # +1 for sentinel 0

    def CalculateRectangleArea(self, height: int, width: int) -> str:
        r"""
        Calculates the area of a rectangle.

        Args:
            height (int): The height of the rectangle.
            width (int): The width of the rectangle.

        Returns:
            str: The area of the rectangle.

        Example Output:
            "10"
        """
        return str(int(height) * int(width))

    def UpdateMaxArea(self, current_area: int, max_area: int) -> str:
        r"""
        Updates the maximum area.

        Args:
            current_area (int): The currently calculated area.
            max_area (int): The currently known maximum area.

        Returns:
            str: The updated maximum area.

        Example Output:
            "10"
        """
        return str(max(int(current_area), int(max_area)))

    def Observe(self) -> str:
        r"""
        Returns the observation information of the current state.

        Returns:
            str: A prompt message describing the current state.

        Example Output:
            "Please analyze the building heights and calculate the maximum rectangle area"
        """
        return "Please analyze the building heights and calculate the maximum rectangle area"

    def Done(self, answer: int) -> str:
        r"""
        Verifies whether the final answer is correct and returns the result information.

        Args:
            answer (int): The answer submitted by the user.

        Returns:
            str: Result information, including whether it is correct.

        Example Output:
            "Your answer: 10, Reference answer: 10, Result: Correct"
        """
        ref_answer = self.get_ref_answer()
        correct = int(answer) == ref_answer
        msg = f"Your answer: {int(answer)}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        Automatically call actions to compute the answer and submit for verification.

        Returns:
            str: The result information of the final answer verification.
        """
        # Get count (including sentinel)
        obs, _, term, _, _ = self.step("\\boxed{count}")
        if term:
            return obs
        try:
            n = int(obs)
        except Exception:
            # Fallback to direct computation if parsing fails
            n = len(self.heights) + 1

        max_area = 0  # Initialize max area to 0

        for i in range(n):
            obs, _, term, _, _ = self.step(f"\\boxed{ { 'height ' + str(i) } }")
            # Safe formatting fallback
            if not obs or "Format" in obs or "Invalid" in obs:
                obs, _, term, _, _ = self.step(f"\\boxed{{height {i}}}")
            if term:
                return obs
            try:
                current_height = int(obs)
            except Exception:
                current_height = 0

            left = i
            while left > 0:
                obs, _, term, _, _ = self.step(f"\\boxed{{height {left - 1}}}")
                if term:
                    return obs
                try:
                    left_height = int(obs)
                except Exception:
                    left_height = 0
                if left_height >= current_height:
                    left -= 1
                else:
                    break

            right = i
            while right < n - 1:
                obs, _, term, _, _ = self.step(f"\\boxed{{height {right + 1}}}")
                if term:
                    return obs
                try:
                    right_height = int(obs)
                except Exception:
                    right_height = 0
                if right_height >= current_height:
                    right += 1
                else:
                    break

            width = right - left + 1

            obs, _, term, _, _ = self.step(f"\\boxed{{area {current_height} {width}}}")
            if term:
                return obs
            try:
                current_area = int(obs)
            except Exception:
                current_area = 0

            obs, _, term, _, _ = self.step(f"\\boxed{{update {current_area} {max_area}}}")
            if term:
                return obs
            try:
                max_area = int(obs)
            except Exception:
                max_area = max(max_area, current_area)

        # Submit final answer
        obs, _, _, _, _ = self.step(f"\\boxed{{answer {max_area}}}")
        return obs