from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LongestConsecutiveSubsequenceEnvGEM(Env):
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
        # - array_length: 数组长度（越大越难）
        # - value_range: 数值范围（越大越难）
        # - duplication_rate: 重复率（百分比，越低越接近全唯一，越难）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 10000),
            "duplication_rate": (10, 50),  # 百分比（10%-50%）
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "duplication_rate": 5,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.duplication_rate: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例数据
        self.problem: Dict[str, Any] = {}
        self.nums: list[int] = []
        self.nums_set: set[int] = set()

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
            "Longest Consecutive Subsequence Game:\n"
            "Given a list of integers, find the length of the longest consecutive sequence.\n"
            "Available actions:\n"
            "- Observe the list: \\boxed{observe}\n"
            "- Convert to set: \\boxed{to_set}\n"
            "- Check if (N-1) exists: \\boxed{check_prev N}\n"
            "- Count consecutive starting from N: \\boxed{count_from N}\n"
            "- Update longest streak: \\boxed{update_longest C L}\n"
            "- Submit answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array length: {self.array_length} | Value range: [0, {self.value_range})\n"
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
        self.nums = self.problem["nums"]
        self.nums_set = set(self.nums)

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

        # 使用难度参数生成问题实例（受 seed 影响）

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # unique_count 根据重复率计算
        unique_count = max(
            1, min(self.array_length, int(round(self.array_length * (100 - self.duplication_rate) / 100.0)))
        )

        # 生成唯一值集合
        if self.value_range <= 0:
            # 容错：保证范围至少为1
            max_val = 1
        else:
            max_val = self.value_range

        if unique_count <= max_val:
            unique_values = random.sample(range(max_val), unique_count)
        else:
            # 如果唯一值数量超过 value_range，则允许有重复采样
            unique_values = [random.randint(0, max_val - 1) for _ in range(unique_count)]

        arr = unique_values.copy()
        while len(arr) < self.array_length:
            arr.append(random.choice(unique_values))
        random.shuffle(arr)
        return {"nums": arr}

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
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()

            elif cmd == "to_set":
                obs = self.ConvertToSet(self.nums)

            elif cmd == "check_prev":
                if len(tokens) != 2:
                    obs = "Invalid action: check_prev requires 1 parameter N."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                num = int(tokens[1])
                obs = self.CheckPreviousNumber(set(self.nums), num)

            elif cmd == "count_from":
                if len(tokens) != 2:
                    obs = "Invalid action: count_from requires 1 parameter N."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                start_num = int(tokens[1])
                obs = self.CountConsecutiveNumbers(set(self.nums), start_num)

            elif cmd == "update_longest":
                if len(tokens) != 3:
                    obs = "Invalid action: update_longest requires 2 parameters C and L."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                current_streak = int(tokens[1])
                longest_streak = int(tokens[2])
                obs = self.UpdateLongestStreak(current_streak, longest_streak)

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Invalid action: answer requires 1 parameter N."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                answer = int(tokens[1])
                msg, correct = self.Done(answer, return_reward_info=False)
                obs = msg
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
        choices = [
            "\\boxed{observe}",
            "\\boxed{to_set}",
            "\\boxed{check_prev 1}",
            "\\boxed{count_from 1}",
            "\\boxed{update_longest 1 2}",
            "\\boxed{answer 1}",
        ]
        return random.choice(choices)

    # 辅助方法（保留并转换）
    def ConvertToSet(self, numbers: list):
        """
        Convert a list of numbers into a set.
        Returns: JSON string of unique numbers.
        """
        num_set = set(numbers)
        return json.dumps(list(num_set))

    def CheckPreviousNumber(self, num_set: set, num: int):
        """
        Check if num-1 is in the set.
        Returns: "True" or "False".
        """
        return str(num - 1 in num_set)

    def CountConsecutiveNumbers(self, num_set: set, start_num: int):
        """
        Calculate the length of the consecutive number sequence starting from start_num.
        Returns: The length as string.
        """
        current_num = start_num
        current_streak = 1

        while current_num + 1 in num_set:
            current_num += 1
            current_streak += 1

        return str(current_streak)

    def UpdateLongestStreak(self, current_streak: int, longest_streak: int):
        """
        Update the length of the longest consecutive sequence.
        Returns: The updated length as string.
        """
        return str(max(current_streak, longest_streak))

    def Observe(self):
        """
        Obtain the list of numbers in the current environment.
        Returns: JSON string of numbers.
        """
        return json.dumps(self.nums)

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        nums_set = set(self.nums)
        longest_streak = 0

        for num in nums_set:
            if num - 1 not in nums_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in nums_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak

    def Done(self, answer: int, return_reward_info: bool = True):
        """
        Verify whether the final answer is correct and return the result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        if return_reward_info:
            msg += f", reward={'1.0' if correct else '-1.0'}"
        return msg, correct

    def solve(self) -> str:
        """
        Automatically call all actions to complete the process, and submit the answer for verification.
        Returns: The result information of the final answer verification.
        """
        # Observe numbers
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        numbers = json.loads(obs)

        # Convert to set
        set_obs, _, _, _, _ = self.step("\\boxed{to_set}")
        num_set = set(json.loads(set_obs))

        longest_streak = 0

        for num in num_set:
            check_obs, _, term, _, _ = self.step(f"\\boxed{{check_prev {num}}}")
            if term:
                return check_obs  # terminated due to error
            if check_obs == "False":
                count_obs, _, term, _, _ = self.step(f"\\boxed{{count_from {num}}}")
                if term:
                    return count_obs
                current_streak = int(count_obs)
                update_obs, _, term, _, _ = self.step(f"\\boxed{{update_longest {current_streak} {longest_streak}}}")
                if term:
                    return update_obs
                longest_streak = int(update_obs)

        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {longest_streak}}}")
        return final_obs