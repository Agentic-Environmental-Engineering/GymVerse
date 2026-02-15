from typing import Any, Dict, Optional, Tuple, List
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class RainWaterTrapEnvGEM(Env):
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

        # 难度参数范围：数组长度与高度最大值控制
        self.complexity_params = {
            "array_length": (5, 50),   # 高度数组长度
            "height_max": (2, 100),    # 高度最大值（均匀采样 [0, height_max]）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 3,
            "height_max": 5,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.height_max: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例与中间缓存
        self.problem: Dict[str, Any] = {}
        self.height: List[int] = []
        self.left_max_list: Optional[List[int]] = None
        self.right_max_list: Optional[List[int]] = None
        self.water_amounts: Optional[List[Optional[int]]] = None
        self.submitted: bool = False

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
            "Rain Water Trapping (GEM): Given a list of non-negative integers representing elevation map where the width of each bar is 1, compute how much water it can trap after raining.\n"
            "You can interact step-by-step using actions in boxed format.\n"
            "Available actions:\n"
            "- Observe heights: \\boxed{observe}\n"
            "- Compute left max array: \\boxed{left_max}\n"
            "- Compute right max array: \\boxed{right_max}\n"
            "- Compute water amount at index i (0-based): \\boxed{water i}\n"
            "- Sum all water amounts (auto-completes missing entries if needed): \\boxed{sum}\n"
            "- Submit final answer N: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        n = len(self.height) if self.height is not None else 0
        lm = "Y" if self.left_max_list is not None else "N"
        rm = "Y" if self.right_max_list is not None else "N"
        computed = 0
        if self.water_amounts is not None:
            computed = sum(1 for x in self.water_amounts if isinstance(x, int))
        return (
            f"Array length: {n} | LeftMax: {lm} | RightMax: {rm} | Water computed: {computed}/{n}\n"
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

        # 初始化状态
        self.turn_count = 0
        self.left_max_list = None
        self.right_max_list = None
        self.water_amounts = [None for _ in range(len(self.problem["heights"]))]
        self.submitted = False

        # 与原环境兼容的属性
        self.height = self.problem["heights"]

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        max_h = self.height_max
        heights = [random.randint(0, max_h) for _ in range(n)]
        return {"heights": heights, "size": n}

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
        content_lower = content.strip().lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            # Observe
            if content_lower == "observe":
                obs = self.Observe()

            # Compute left max
            elif content_lower == "left_max" or content_lower == "calc_left_max":
                left_max_json = self.CalculateLeftMax(self.height)
                self.left_max_list = json.loads(left_max_json)
                obs = left_max_json

            # Compute right max
            elif content_lower == "right_max" or content_lower == "calc_right_max":
                right_max_json = self.CalculateRightMax(self.height)
                self.right_max_list = json.loads(right_max_json)
                obs = right_max_json

            # Compute water at index i
            elif content_lower.startswith("water"):
                m = re.match(r"water\s+(\d+)$", content_lower)
                if not m:
                    obs = f"Invalid 'water' command format. Use: \\boxed{{water i}}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                index = int(m.group(1))
                if index < 0 or index >= len(self.height):
                    obs = f"Index out of range: {index}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                if self.left_max_list is None or self.right_max_list is None:
                    obs = "Please compute left_max and right_max first."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                water_str = self.CalculateWaterAtPosition(
                    index, self.left_max_list, self.right_max_list, self.height
                )
                water_val = int(water_str)
                if self.water_amounts is None:
                    self.water_amounts = [None for _ in range(len(self.height))]
                self.water_amounts[index] = water_val
                obs = water_str

            # Sum water amounts
            elif content_lower == "sum":
                if self.left_max_list is None or self.right_max_list is None:
                    obs = "Please compute left_max and right_max first."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                # 自动补齐未计算的 water_i
                if self.water_amounts is None or any(v is None for v in self.water_amounts):
                    n = len(self.height)
                    completed = []
                    for i in range(n):
                        w_str = self.CalculateWaterAtPosition(
                            i, self.left_max_list, self.right_max_list, self.height
                        )
                        completed.append(int(w_str))
                    self.water_amounts = completed
                total_str = self.SumWaterAmount([int(v) for v in self.water_amounts])
                obs = total_str

            # Submit final answer
            elif content_lower.startswith("answer"):
                m = re.match(r"answer\s+(-?\d+)$", content_lower)
                if not m:
                    obs = "Invalid 'answer' command format. Use: \\boxed{answer N}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                answer_val = int(m.group(1))
                ref_answer = self.get_ref_answer()
                correct = (answer_val == ref_answer)
                msg = self.Done(answer_val)
                obs = msg
                reward = 1.0 if correct else -1.0
                terminated = True

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
        # 简单的随机示例：优先 observe
        if self.left_max_list is None or self.right_max_list is None:
            return "\\boxed{observe}"
        n = len(self.height)
        if self.water_amounts is not None and any(v is None for v in self.water_amounts):
            idx = random.randint(0, max(0, n - 1))
            return f"\\boxed{{water {idx}}}"
        return "\\boxed{sum}"

    # ------------------------------
    # 以下为原环境的辅助方法（保持并转换）
    # ------------------------------
    @property
    def finished(self) -> bool:
        # GEM 接口不使用该属性控制，但保留兼容
        return self.submitted

    @property
    def reward(self):
        # GEM 接口中奖励在 step 返回，此处仅保留兼容
        return 0.0

    @staticmethod
    def from_env_str(env_str: str):
        # 保留接口以兼容原构造方式（将忽略字符串参数，直接返回默认实例）
        prefix = "RainWaterTrapEnv@"
        if not env_str.startswith(prefix):
            return None
        return RainWaterTrapEnvGEM()

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        if not self.height:
            return 0

        n = len(self.height)
        left_max = [0] * n
        right_max = [0] * n
        water_trapped = 0

        # Fill left_max
        left_max[0] = self.height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i - 1], self.height[i])

        # Fill right_max
        right_max[n - 1] = self.height[n - 1]
        for i in range(n - 2, -1, -1):
            right_max[i] = max(right_max[i + 1], self.height[i])

        # Calculate trapped water
        for i in range(n):
            water_trapped += max(0, min(left_max[i], right_max[i]) - self.height[i])

        return water_trapped

    def CalculateLeftMax(self, height_list: list):
        r"""
        Calculate the maximum height to the left of each position in the height list.

        Args:
            height_list (list[int]): List of block heights.

        Returns:
            str: A list of the maximum heights to the left of each position, in JSON format.

        Example Output:
            "[0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]"
        """
        if not height_list:
            return json.dumps([])

        n = len(height_list)
        left_max = [0] * n
        left_max[0] = height_list[0]

        for i in range(1, n):
            left_max[i] = max(left_max[i - 1], height_list[i])

        return json.dumps(left_max)

    def CalculateRightMax(self, height_list: list):
        r"""
        Calculate the maximum height to the right of each position in the height list.

        Args:
            height_list (list[int]): List of block heights.

        Returns:
            str: A list of the maximum heights to the right of each position, in JSON format.

        Example Output:
            "[3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1]"
        """
        if not height_list:
            return json.dumps([])

        n = len(height_list)
        right_max = [0] * n
        right_max[n - 1] = height_list[n - 1]

        for i in range(n - 2, -1, -1):
            right_max[i] = max(right_max[i + 1], height_list[i])

        return json.dumps(right_max)

    def CalculateWaterAtPosition(self, index: int, left_max: list, right_max: list, height_list: list):
        r"""
        Calculate the amount of rainwater that can be trapped at a specified position.

        Args:
            index (int): The index of the position to calculate.
            left_max (list[int]): List of the maximum heights to the left of each position.
            right_max (list[int]): List of the maximum heights to the right of each position.
            height_list (list[int]): List of block heights.

        Returns:
            str: The amount of rainwater that can be trapped at the specified position.

        Example Output:
            "1"
        """
        if not height_list or index < 0 or index >= len(height_list):
            return "0"

        water_amount = min(left_max[index], right_max[index]) - height_list[index]
        return str(max(0, water_amount))

    def SumWaterAmount(self, water_amounts: list):
        r"""
        Calculate the total amount of rainwater trapped at all positions.

        Args:
            water_amounts (list[int]): List of the amounts of rainwater trapped at each position.

        Returns:
            str: The total amount of rainwater.

        Example Output:
            "6"
        """
        total = sum(water_amounts)
        return str(total)

    def Observe(self):
        r"""
        Obtain information about the block heights in the current environment.

        Returns:
            str: Information describing the current block heights.

        Example Output:
            "Current block heights: [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]"
        """
        return f"Current block heights: {self.height}"

    def Done(self, answer):
        r"""
        Verify whether the final answer is correct and return result information.

        Args:
            answer (int): The user-submitted answer for the total amount of rainwater.

        Returns:
            str: Result information, including whether it is correct and reward information.

        Example Output:
            "Your answer: 6, Reference answer: 6, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        # 在 GEM 中，奖励由 step 决定，这里仅返回文本
        return msg + f", reward={'1' if correct else '0'}"

    def solve(self) -> str:
        """
        Automatically call actions to complete the process and submit the answer for verification.
        Returns:
            str: The result information of the final answer verification.
        """
        # Observe
        obs, _, _, _, _ = self.step("\\boxed{observe}")

        # Compute left and right max arrays
        lm_obs, _, _, _, _ = self.step("\\boxed{left_max}")
        rm_obs, _, _, _, _ = self.step("\\boxed{right_max}")

        # Compute water for all positions
        n = len(self.height)
        for i in range(n):
            self.step(f"\\boxed{{water {i}}}")

        # Sum all
        total_obs, _, _, _, _ = self.step("\\boxed{sum}")
        try:
            total = int(total_obs.strip())
        except Exception:
            total = self.get_ref_answer()

        # Submit answer
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {total}}}")
        return final_obs