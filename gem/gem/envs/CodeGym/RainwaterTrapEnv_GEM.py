from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class RainwaterTrapEnvGEM(Env):
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

        # 难度参数范围（根据原环境：数组长度、数值范围）
        self.complexity_params = {
            "array_length": (5, 50),  # 高度数组长度
            "value_range": (3, 100),  # 高度取值上限（包含 0）
        }

        # 参数方差（启用随机化时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 10,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题与计算状态
        self.problem: Dict[str, Any] = {}
        self.height = []
        self.left: Optional[int] = None
        self.right: Optional[int] = None
        self.left_max: Optional[int] = None
        self.right_max: Optional[int] = None
        self.water_trapped: int = 0

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
            "Rainwater Trap: Compute how much water can be trapped between bars.\n"
            "You can inspect the height list, initialize two pointers, move one pointer at a time,\n"
            "observe the state, and finally submit the answer.\n"
            "Available actions:\n"
            "- Get height list: \\boxed{get}\n"
            "- Initialize pointers: \\boxed{init}\n"
            "- Move pointer: \\boxed{move left} or \\boxed{move right}\n"
            "- Observe state: \\boxed{observe}\n"
            "- Submit answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        n = len(self.height) if self.height is not None else 0
        lp = "None" if self.left is None else str(self.left)
        rp = "None" if self.right is None else str(self.right)
        return (
            f"Array length: {n}\n"
            f"Pointers: left={lp}, right={rp}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            f"Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.height = self.problem["height"]

        # 重置状态
        self.left = None
        self.right = None
        self.left_max = None
        self.right_max = None
        self.water_trapped = 0
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 生成高度列表，长度为 array_length，数值范围 [0, value_range]
        heights = [random.randint(0, self.value_range) for _ in range(self.array_length)]
        return {"height": heights, "size": self.array_length}

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

        content = parsed["content"].strip().lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # 命令解析与执行
        if content in {"get", "get heights", "heights", "list", "get_height_list"}:
            obs = self.GetHeightList()
            # non-terminal

        elif content in {"init", "initialize", "initialize pointers", "initialize_pointers"}:
            obs = self.InitializePointers()
            # non-terminal

        elif content.startswith("move"):
            # 支持 move left/right 或 move l/r
            tokens = content.split()
            if len(tokens) < 2:
                obs = "Error: invalid move command. Use 'move left' or 'move right'."
            else:
                direction = tokens[1]
                if direction in {"l", "left"}:
                    direction = "left"
                elif direction in {"r", "right"}:
                    direction = "right"
                else:
                    obs = "Error: invalid direction, use 'left' or 'right'."
                    # Do not mark as invalid-action-terminal; keep non-terminal with neutral reward.
                if obs == "":
                    obs = self.MovePointer(direction)
            # non-terminal

        elif content in {"observe", "status", "state"}:
            obs = self.Observe()
            # non-terminal

        elif content.startswith("answer") or content.startswith("submit"):
            # extract integer
            m = re.match(r"(answer|submit)\s+(-?\d+)", content)
            if not m:
                obs = "Format error: use 'answer N' where N is an integer."
                return (
                    obs,
                    LanguageGameReward.format_error_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            answer = int(m.group(2))
            msg = self.Done(answer)
            # 判定奖励
            correct = (answer == self.get_ref_answer())
            reward = 1.0 if correct else -1.0
            terminated = True
            truncated = False
            obs = f"{msg}\nReward: {reward:.1f}"

        else:
            obs = f"Invalid action: '{content}'."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（放在 step 结尾）
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
        # 简单随机策略：观察或移动
        if self.left is None or self.right is None:
            return "\\boxed{init}"
        choices = ["\\boxed{observe}", "\\boxed{move left}", "\\boxed{move right}"]
        return random.choice(choices)

    # -------------------------
    # 辅助方法（从原环境转换保留）
    # -------------------------

    def GetHeightList(self) -> str:
        r"""
        Get the list of wall heights.

        Returns:
            str: JSON string of the wall height list (e.g., "[0,1,0,2,...]").
        """
        return json.dumps(self.height)

    def InitializePointers(self) -> str:
        r"""
        Initialize the left and right pointers and their corresponding maximum heights.

        Returns:
            str: Information containing the initial pointer positions and maximum heights.
        """
        if not self.height:
            self.left, self.right = 0, 0
            self.left_max, self.right_max = 0, 0
            self.water_trapped = 0
            return "height list is empty, left=0, right=0, left_max=0, right_max=0, water_trapped=0"

        self.left, self.right = 0, len(self.height) - 1
        self.left_max, self.right_max = self.height[self.left], self.height[self.right]
        self.water_trapped = 0
        return (
            f"left={self.left}, right={self.right}, "
            f"left_max={self.left_max}, right_max={self.right_max}, "
            f"water_trapped={self.water_trapped}"
        )

    def MovePointer(self, direction: str) -> str:
        r"""
        Move the pointer according to the specified direction and calculate newly added rainwater.

        Args:
            direction (str): "left" -> move left pointer; "right" -> move right pointer.

        Returns:
            str: Information about the pointer position after moving, maximum height, and accumulated rainwater volume.
        """
        if self.left is None or self.right is None:
            return "Error: pointers not initialized, call InitializePointers first"

        if self.left >= self.right:
            return (
                f"Pointers have met or crossed: left={self.left}, right={self.right}. "
                f"Total water_trapped={self.water_trapped}"
            )

        if direction == "left":
            self.left += 1
            self.left_max = max(self.left_max, self.height[self.left])
            self.water_trapped += max(0, self.left_max - self.height[self.left])
        elif direction == "right":
            self.right -= 1
            self.right_max = max(self.right_max, self.height[self.right])
            self.water_trapped += max(0, self.right_max - self.height[self.right])
        else:
            return "Error: invalid direction, use 'left' or 'right'"

        return (
            f"left={self.left}, right={self.right}, "
            f"left_max={self.left_max}, right_max={self.right_max}, "
            f"water_trapped={self.water_trapped}"
        )

    def Observe(self) -> str:
        r"""
        Return observation information of the current state.

        Returns:
            str: Prompt information describing the current state.
        """
        if self.left is None or self.right is None:
            return (
                "Current state: pointers not initialized. "
                "Please call InitializePointers first to initialize the pointers."
            )
        return (
            f"Current state: left={self.left}, right={self.right}, "
            f"left_max={self.left_max}, right_max={self.right_max}, "
            f"water_trapped={self.water_trapped}. "
            f"Call MovePointer to continue calculation or call answer N to submit the answer."
        )

    def Done(self, answer: int) -> str:
        r"""
        Verify whether the final answer is correct and return result information.

        Args:
            answer (int): The user-submitted answer of the maximum rainwater volume.

        Returns:
            str: Result information, including whether it is correct.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg

    def get_ref_answer(self) -> int:
        r"""
        Use the information in the environment to get the reference answer.
        """
        if not self.height:
            return 0

        left, right = 0, len(self.height) - 1
        left_max, right_max = self.height[left], self.height[right]
        water_trapped = 0

        while left < right:
            if left_max < right_max:
                left += 1
                left_max = max(left_max, self.height[left])
                water_trapped += max(0, left_max - self.height[left])
            else:
                right -= 1
                right_max = max(right_max, self.height[right])
                water_trapped += max(0, right_max - self.height[right])

        return water_trapped

    def solve(self) -> str:
        r"""
        Automatically interacts with the environment to compute the trapped water and submits the answer.

        Returns:
            str: The final verification message (observation string of the answer step).
        """
        # Get heights
        _ = self.step("\\boxed{get}")

        # Initialize pointers
        _ = self.step("\\boxed{init}")

        # Move pointers until they meet
        while self.left is not None and self.right is not None and self.left < self.right:
            # Decide direction based on current maxima
            if self.left_max is None or self.right_max is None:
                _ = self.step("\\boxed{observe}")
                break

            if self.left_max <= self.right_max:
                _ = self.step("\\boxed{move left}")
            else:
                _ = self.step("\\boxed{move right}")

            # Safety: avoid infinite loop due to turn limit interactions
            if self.turn_count >= self.max_turns:
                break

        # Final observation (optional)
        _ = self.step("\\boxed{observe}")

        # Submit answer
        answer = self.water_trapped
        obs, reward, terminated, truncated, _info = self.step(f"\\boxed{{answer {answer}}}")
        return obs