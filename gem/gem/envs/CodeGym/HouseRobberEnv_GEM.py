from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class HouseRobberEnvGEM(Env):
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

        # 难度参数范围
        # - num_houses: 房屋数量（数组长度）
        # - value_low/value_high: 每个房屋金额的随机区间
        self.complexity_params = {
            "num_houses": (5, 50),
            "value_low": (1, 10),
            "value_high": (20, 200),
        }

        # 参数方差（启用 enable_param_randomization 时用于微调）
        self.param_variance = {
            "num_houses": 2,
            "value_low": 2,
            "value_high": 20,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.num_houses: int = 0
        self.value_low: int = 0
        self.value_high: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.houses: list[int] = []
        self.dp: list[int] = []

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

        # 确保 value_low <= value_high
        if self.value_low > self.value_high:
            self.value_low, self.value_high = self.value_high, self.value_low

    def _get_instructions(self) -> str:
        return (
            "House Robber (GEM): Compute the maximum amount that can be robbed without robbing adjacent houses.\n"
            "You can build a DP array where DP[i] stores the best up to house i.\n"
            "Available actions (wrap in \\boxed{ ... }):\n"
            "- Observe current state: \\boxed{observe}\n"
            "- Initialize DP array: \\boxed{init N}\n"
            "- Set DP value: \\boxed{set INDEX VALUE}\n"
            "- Calculate DP[i] with values: \\boxed{calc INDEX PREV1 PREV2 HOUSE_VALUE}\n"
            "- Submit final answer: \\boxed{answer VALUE}\n"
            "Goal: Submit the correct maximum robbable amount.\n"
        )

    def get_task_suffix(self) -> str:
        n = len(self.houses) if hasattr(self, "houses") and self.houses is not None else 0
        return f"Houses: {n}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.houses = self.problem["houses"]
        self.dp = []
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        houses = [random.randint(self.value_low, self.value_high) for _ in range(self.num_houses)]
        return {"houses": houses}

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
        if len(tokens) == 0:
            obs = "Empty action."
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
                if len(tokens) != 1:
                    obs = "Invalid 'observe' usage. Use: \\boxed{observe}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.Observe()

            elif cmd == "init":
                if len(tokens) != 2:
                    obs = "Invalid 'init' usage. Use: \\boxed{init N}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                length = int(tokens[1])
                if length < 0:
                    obs = "Error: DP length must be non-negative."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.InitializeDp(length)

            elif cmd == "set":
                if len(tokens) != 3:
                    obs = "Invalid 'set' usage. Use: \\boxed{set INDEX VALUE}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                index = int(tokens[1])
                value = int(tokens[2])
                obs = self.SetDpValue(index, value)
                # 若越界错误，判为无效动作
                if obs.startswith("Error:"):
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

            elif cmd == "calc":
                if len(tokens) != 5:
                    obs = "Invalid 'calc' usage. Use: \\boxed{calc INDEX PREV1 PREV2 HOUSE_VALUE}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                index = int(tokens[1])
                prev1 = int(tokens[2])
                prev2 = int(tokens[3])
                house_value = int(tokens[4])
                obs = self.CalculateMaxAmount(index, prev1, prev2, house_value)
                if obs.startswith("Error:"):
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Invalid 'answer' usage. Use: \\boxed{answer VALUE}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                answer_val = int(tokens[1])
                msg = self.Done(answer_val)
                # 根据正确与否设置奖励
                ref_answer = self.get_ref_answer()
                correct = (answer_val == ref_answer)
                reward = 1.0 if correct else -1.0
                obs = msg
                terminated = True
                truncated = False

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
            obs = f"Execution error: {str(e)}"
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
        return "\\boxed{observe}"

    # =========================
    # 原环境辅助方法（转换后保留）
    # =========================
    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        if not self.houses:
            return 0
        if len(self.houses) == 1:
            return self.houses[0]

        dp = [0] * len(self.houses)
        dp[0] = self.houses[0]
        dp[1] = max(self.houses[0], self.houses[1])

        for i in range(2, len(self.houses)):
            dp[i] = max(dp[i - 1], dp[i - 2] + self.houses[i])

        return dp[-1]

    def InitializeDp(self, length: int):
        r"""
        Initialize the DP array, set a DP array of the specified length with an initial value of 0.

        Args:
            length (int): The length of the DP array.

        Returns:
            str: Initialization result information, including the length of the DP array.

        Example Output:
            "DP array initialized, length is 5"
        """
        self.dp = [0] * length
        return f"DP array initialized, length is {length}"

    def SetDpValue(self, index: int, value: int):
        r"""
        Set the value at the specified index position in the DP array.

        Args:
            index (int): The index position to be set.
            value (int): The value to be set.

        Returns:
            str: Setting result information, including the index and the set value.

        Example Output:
            "DP[1] set to 5"
        """
        if 0 <= index < len(self.dp):
            self.dp[index] = value
            return f"DP[{index}] set to {value}"
        else:
            return f"Error: Index {index} is out of DP array bounds"

    def CalculateMaxAmount(self, index: int, prev1: int, prev2: int, house_value: int):
        r"""
        Calculate the maximum amount that can be stolen at the current house position and update the DP array.

        Args:
            index (int): The index of the current house.
            prev1 (int): The maximum amount at the previous house position (DP[i-1]).
            prev2 (int): The maximum amount at the house position two before (DP[i-2]).
            house_value (int): The amount of money in the current house.

        Returns:
            str: Calculation result information, including the current index and the calculated maximum amount.

        Example Output:
            "DP[3] calculated, value is 8"
        """
        if 0 <= index < len(self.dp):
            max_amount = max(prev1, prev2 + house_value)
            self.dp[index] = max_amount
            return f"DP[{index}] calculated, value is {max_amount}"
        else:
            return f"Error: Index {index} is out of DP array bounds"

    def Observe(self):
        r"""
        Obtain the current environment state, including the list of house amounts and the state of the DP array.

        Returns:
            str: Current environment state information.

        Example Output:
            "House amounts: [3, 2, 7, 10], DP array: [3, 3, 10, 0]"
        """
        return f"House amounts: {self.houses}, DP array: {self.dp}"

    def Done(self, answer: int):
        r"""
        Verify whether the final answer is correct and return the result information.

        Args:
            answer (int): The maximum stealable amount submitted by the user.

        Returns:
            str: Result information, including whether it is correct.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        Automatically call actions to complete the process, and submit the answer for verification.
        Returns:
            str: The result information of the final answer verification.
        """
        # Observe initial state (optional)
        self.step("\\boxed{observe}")

        n = len(self.houses)
        if n == 0:
            obs, reward, terminated, truncated, info = self.step("\\boxed{answer 0}")
            return obs

        # Initialize DP
        self.step(f"\\boxed{{init {n}}}")

        # Base cases
        self.step(f"\\boxed{{set 0 {self.houses[0]}}}")
        if n == 1:
            obs, reward, terminated, truncated, info = self.step(f"\\boxed{{answer {self.houses[0]}}}")
            return obs

        dp1_value = max(self.houses[0], self.houses[1])
        self.step(f"\\boxed{{set 1 {dp1_value}}}")
        if n == 2:
            obs, reward, terminated, truncated, info = self.step(f"\\boxed{{answer {dp1_value}}}")
            return obs

        # Transition
        for i in range(2, n):
            prev1 = self.dp[i - 1]
            prev2 = self.dp[i - 2]
            hv = self.houses[i]
            self.step(f"\\boxed{{calc {i} {prev1} {prev2} {hv}}}")

        max_amount = self.dp[-1]
        obs, reward, terminated, truncated, info = self.step(f"\\boxed{{answer {max_amount}}}")
        return obs