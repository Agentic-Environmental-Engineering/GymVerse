from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxWaterContainerEnvGEM(Env):
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

        # 定义难度参数范围
        self.complexity_params = {
            "array_length": (5, 50),      # 高度数组长度
            "value_range": (10, 10000),   # 高度值范围
            "turn_limit": (20, 200),      # 建议的回合上限（用于状态展示）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 2,   # ±2 的方差
            "value_range": 500,  # ±500 的方差
            "turn_limit": 10,    # ±10 的方差
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.turn_limit: int = 0  # 仅用于展示，不影响 self.max_turns

        # 状态变量
        self.turn_count: int = 0

        # 兼容原环境的状态变量
        self._reward: float = 0.0
        self._done: bool = False

        self.problem: Dict[str, Any] = {"heights": []}

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
            "Max Water Container (GEM): Use two-pointer method to compute maximum water capacity.\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- Initialize pointers: \\boxed{init}\n"
            "- Calculate current water: \\boxed{calc L R}\n"
            "- Update max water: \\boxed{update CURRENT MAX}\n"
            "- Move left pointer: \\boxed{move_left L}\n"
            "- Move right pointer: \\boxed{move_right R}\n"
            "- Check pointers: \\boxed{check L R}\n"
            "- Get line height: \\boxed{height INDEX}\n"
            "- Observe hint: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        length = len(self.problem.get("heights", []))
        return f"Array length: {length} | Suggested turn limit: {self.turn_limit} | Turn: {self.turn_count}/{self.max_turns} | Enter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        heights = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        return {"heights": heights}

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
        tokens = content.strip().split()
        if not tokens:
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
            if cmd == "init":
                obs = self.InitializePointers()
            elif cmd == "calc" and len(tokens) == 3:
                left = int(tokens[1])
                right = int(tokens[2])
                obs = self.CalculateCurrentWater(left, right)
            elif cmd == "update" and len(tokens) == 3:
                current_water = int(tokens[1])
                current_max = int(tokens[2])
                obs = self.UpdateMaxWater(current_water, current_max)
            elif cmd == "move_left" and len(tokens) == 2:
                left = int(tokens[1])
                obs = self.MoveLeftPointer(left)
            elif cmd == "move_right" and len(tokens) == 2:
                right = int(tokens[1])
                obs = self.MoveRightPointer(right)
            elif cmd == "check" and len(tokens) == 3:
                left = int(tokens[1])
                right = int(tokens[2])
                obs = self.CheckPointers(left, right)
            elif cmd == "height" and len(tokens) == 2:
                index = int(tokens[1])
                obs = self.GetLineHeight(index)
            elif cmd == "observe":
                obs = self.Observe()
            elif cmd == "answer" and len(tokens) == 2:
                answer = int(tokens[1])
                obs = self.Done(answer)
                # 根据 Done 的结果设置奖励
                reward = 1.0 if "Result: Correct" in obs else -1.0
                terminated = True
                truncated = False
            else:
                obs = f"Invalid action: {content}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
        except Exception as e:
            obs = f"Error: {str(e)}"
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
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
        return "\\boxed{observe}"

    # ----------------------------
    # 保留并转换原环境的辅助方法
    # ----------------------------
    def InitializePointers(self) -> str:
        """
        Initialize the two pointers, with the left pointer at 0 and the right pointer at the index of the last element.

        Returns:
            str: A string containing the positions of the initialized left and right pointers.

        Example Output:
            "{'left': 0, 'right': 8}"
        """
        left = 0
        right = len(self.problem["heights"]) - 1
        return f"{{'left': {left}, 'right': {right}}}"

    def CalculateCurrentWater(self, left: int, right: int) -> str:
        """
        Calculate the amount of water that the container formed by the lines pointed to by the current left and right pointers can hold.

        Args:
            left (int): Index of the left pointer
            right (int): Index of the right pointer

        Returns:
            str: The amount of water in the current container

        Example Output:
            "49"
        """
        heights = self.problem["heights"]
        if not (0 <= left < len(heights)) or not (0 <= right < len(heights)):
            raise IndexError("Left or right index out of range.")
        if right < left:
            raise ValueError("Right index must be >= left index.")

        width = right - left
        height_left = heights[left]
        height_right = heights[right]
        current_height = min(height_left, height_right)
        current_water = width * current_height
        return str(current_water)

    def UpdateMaxWater(self, current_water: int, current_max: int) -> str:
        """
        Compare the current water amount with the recorded maximum water amount and return the updated maximum water amount.

        Args:
            current_water (int): The currently calculated water amount
            current_max (int): The currently recorded maximum water amount

        Returns:
            str: The updated maximum water amount
        """
        return str(max(current_water, current_max))

    def MoveLeftPointer(self, left: int) -> str:
        """
        Move the left pointer one position to the right.

        Args:
            left (int): Current position of the left pointer

        Returns:
            str: Position of the left pointer after moving
        """
        return str(left + 1)

    def MoveRightPointer(self, right: int) -> str:
        """
        Move the right pointer one position to the left.

        Args:
            right (int): Current position of the right pointer

        Returns:
            str: Position of the right pointer after moving
        """
        return str(right - 1)

    def CheckPointers(self, left: int, right: int) -> str:
        """
        Check if the left pointer is less than the right pointer.

        Args:
            left (int): Index of the left pointer
            right (int): Index of the right pointer

        Returns:
            str: Returns "True" if the left pointer is less than the right pointer, otherwise returns "False"
        """
        return str(left < right)

    def GetLineHeight(self, index: int) -> str:
        """
        Get the height of the line at the specified index.

        Args:
            index (int): Index of the line whose height is to be obtained

        Returns:
            str: Height of the line at the specified index
        """
        heights = self.problem["heights"]
        if not (0 <= index < len(heights)):
            raise IndexError("Index out of range.")
        return str(heights[index])

    def Observe(self) -> str:
        """
        Return observation information of the current state.

        Returns:
            str: Prompt information describing the current state
        """
        return "Please use the two-pointer method to calculate the maximum water capacity."

    def Done(self, answer: int) -> str:
        """
        Verify whether the final answer is correct and return result information.

        Args:
            answer (int): The user-submitted answer for the maximum water storage capacity

        Returns:
            str: Result information, including correctness and reward information
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        heights = self.problem["heights"]
        left = 0
        right = len(heights) - 1
        max_water = 0

        while left < right:
            width = right - left
            current_height = min(heights[left], heights[right])
            current_water = width * current_height
            max_water = max(max_water, current_water)

            if heights[left] < heights[right]:
                left += 1
            else:
                right -= 1

        return max_water

    def solve(self) -> str:
        """
        Automatically call helper actions to compute and return the final verification message (not through step).
        This is a utility method mirroring the original environment's solve behavior.
        """
        # Directly use helper methods to simulate the optimal process
        init_result = self.InitializePointers()
        # Parse "{'left': 0, 'right': N}" string
        try:
            # Simple parsing to extract left and right
            parts = init_result.strip("{}").replace("'", "").split(",")
            left = int(parts[0].split(":")[1].strip())
            right = int(parts[1].split(":")[1].strip())
        except Exception:
            left, right = 0, len(self.problem["heights"]) - 1

        current_max = 0

        while str(left < right) == "True":
            current_water_str = self.CalculateCurrentWater(left, right)
            current_water = int(current_water_str)

            current_max_str = self.UpdateMaxWater(current_water, current_max)
            current_max = int(current_max_str)

            left_height = int(self.GetLineHeight(left))
            right_height = int(self.GetLineHeight(right))

            if left_height < right_height:
                left = int(self.MoveLeftPointer(left))
            else:
                right = int(self.MoveRightPointer(right))

        return self.Done(current_max)