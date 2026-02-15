from typing import Any, Dict, Optional, Tuple
import random
import re
import json
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinSubsetSumDiffEnvGEM(Env):
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
        self.complexity_params = {
            "array_length": (5, 50),       # 数组长度
            "value_range": (10, 10000),    # 数值范围上限（值在 [1, value_range]）
            "max_turns_cfg": (20, 200),    # 建议的最大步数（仅作为信息展示，不覆盖 self.max_turns）
            "num_constraints": (1, 5),     # 约束条件数量（当前未使用，仅保留拓展位）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "max_turns_cfg": 10,
            "num_constraints": 1,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.max_turns_cfg: int = 0
        self.num_constraints: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 原环境状态变量
        self.n: int = 0
        self.arr: list[int] = []
        self.dp: Optional[list[bool]] = None

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
            "Min Subset Sum Difference (GEM): Given an integer array, find the minimum possible difference "
            "between sums of two subsets.\n"
            "Available actions (use the latest \\boxed{...} content):\n"
            "- Observe array: \\boxed{observe}\n"
            "- Calculate total sum: \\boxed{calc_sum}\n"
            "- Initialize DP array with target sum T: \\boxed{init_dp T}\n"
            "- Update DP with number N up to target T: \\boxed{update_dp N T}\n"
            "- Find max reachable subset sum up to target T: \\boxed{find_max T}\n"
            "- Calculate min difference from TOTAL and MAX: \\boxed{calc_diff TOTAL MAX}\n"
            "- Submit final answer: \\boxed{answer A}\n"
            "Notes:\n"
            "- DP target T is typically total_sum // 2\n"
            "- Example sequence: calc_sum -> init_dp T -> observe -> update_dp N T (for each N) -> find_max T -> calc_diff TOTAL MAX -> answer A\n"
        )

    def get_task_suffix(self) -> str:
        arr_len = len(self.arr) if self.arr else 0
        dp_info = f"DP={'init' if self.dp is not None else 'none'}"
        return f"Array length: {arr_len} | {dp_info} | Turn: {self.turn_count}/{self.max_turns} | Suggested max_turns: {self.max_turns_cfg}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 将问题注入原环境状态
        self.arr = self.problem["arr"]
        self.n = len(self.arr)
        self.dp = None

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        data = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        return {"arr": data, "size": self.array_length}

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
        cmd = tokens[0].lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()
                reward = 0.0
                terminated = False

            elif cmd == "calc_sum":
                obs_val = self.CalculateTotalSum()
                obs = f"Total sum = {obs_val}"
                reward = 0.0
                terminated = False

            elif cmd == "init_dp":
                if len(tokens) != 2 or not tokens[1].isdigit():
                    obs = "Invalid init_dp syntax. Use: \\boxed{init_dp T}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                target_sum = int(tokens[1])
                obs_array = self.InitializeDPArray(target_sum)
                obs = f"DP initialized up to {target_sum}: {obs_array}"
                reward = 0.0
                terminated = False

            elif cmd == "update_dp":
                if len(tokens) != 3 or (not tokens[1].isdigit()) or (not tokens[2].isdigit()):
                    obs = "Invalid update_dp syntax. Use: \\boxed{update_dp N T}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                number = int(tokens[1])
                target_sum = int(tokens[2])
                obs_array = self.UpdateDPArray(number, target_sum)
                if obs_array.startswith("Error:"):
                    return obs_array, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = f"DP updated with {number} (target {target_sum}): {obs_array}"
                reward = 0.0
                terminated = False

            elif cmd == "find_max":
                if len(tokens) != 2 or not tokens[1].isdigit():
                    obs = "Invalid find_max syntax. Use: \\boxed{find_max T}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                target_sum = int(tokens[1])
                obs_val = self.FindMaxSubsetSum(target_sum)
                if obs_val.startswith("Error:"):
                    return obs_val, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = f"Max reachable subset sum = {obs_val}"
                reward = 0.0
                terminated = False

            elif cmd == "calc_diff":
                if len(tokens) != 3 or (not tokens[1].isdigit()) or (not tokens[2].isdigit()):
                    obs = "Invalid calc_diff syntax. Use: \\boxed{calc_diff TOTAL MAX}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                total_sum = int(tokens[1])
                max_subset_sum = int(tokens[2])
                obs_val = self.CalculateMinDiff(total_sum, max_subset_sum)
                obs = f"Min difference = {obs_val}"
                reward = 0.0
                terminated = False

            elif cmd == "answer":
                if len(tokens) != 2 or not tokens[1].isdigit():
                    obs = "Invalid answer syntax. Use: \\boxed{answer A}"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                answer_val = int(tokens[1])
                ref_answer = self.get_ref_answer()
                correct = answer_val == ref_answer
                obs = f"Your answer: {answer_val}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
                reward = 1.0 if correct else -1.0
                terminated = True

            else:
                obs = f"Invalid action '{cmd}'."
                return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

        except Exception as e:
            obs = f"Error: {str(e)}"
            return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

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

    # ------------------------
    # 原环境辅助方法（转换保留）
    # ------------------------
    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        total_sum = sum(self.arr)
        dp = [False] * (total_sum // 2 + 1)
        dp[0] = True

        for num in self.arr:
            for j in range(total_sum // 2, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]

        for j in range(total_sum // 2, -1, -1):
            if dp[j]:
                return total_sum - 2 * j
        return total_sum

    def CalculateTotalSum(self):
        """
        Calculate the sum of all elements in the array.
        Returns: str
        """
        total_sum = sum(self.arr)
        return str(total_sum)

    def InitializeDPArray(self, target_sum: int):
        """
        Initialize the dynamic programming array to record reachable subset sums.
        Returns: str (JSON array of booleans)
        """
        self.dp = [False] * (target_sum + 1)
        self.dp[0] = True
        return json.dumps([bool(x) for x in self.dp])

    def UpdateDPArray(self, number: int, target_sum: int):
        """
        Update the dynamic programming array based on the current number.
        Returns: str (JSON array) or error message
        """
        if self.dp is None:
            return "Error: DP array not initialized. Call init_dp first."
        for j in range(target_sum, number - 1, -1):
            self.dp[j] = self.dp[j] or self.dp[j - number]
        return json.dumps([bool(x) for x in self.dp])

    def FindMaxSubsetSum(self, target_sum: int):
        """
        Find the maximum reachable subset sum up to target_sum.
        Returns: str (int) or error message
        """
        if self.dp is None:
            return "Error: DP array not initialized. Call init_dp first."
        for j in range(target_sum, -1, -1):
            if self.dp[j]:
                return str(j)
        return "0"

    def CalculateMinDiff(self, total_sum: int, max_subset_sum: int):
        """
        Calculate the minimum absolute difference between the sums of the two subsets.
        Returns: str
        """
        return str(total_sum - 2 * max_subset_sum)

    def Observe(self):
        """
        Obtain the array information in the current environment.
        Returns: str
        """
        return f"Current array: {self.arr}"

    def Done(self, answer: int):
        """
        Verify whether the final answer is correct and return the result information.
        Returns: str
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={'1' if correct else '0'}"

    def solve(self) -> str:
        """
        Automatically call all actions to complete the full process, and submit the answer for verification.
        Returns: str (final observation message)
        """
        # calc_sum
        obs, _, _, _, _ = self.step("\\boxed{calc_sum}")
        # parse total_sum from obs
        m = re.search(r"Total sum\s*=\s*(\d+)", obs)
        total_sum = int(m.group(1)) if m else sum(self.arr)
        target_sum = total_sum // 2

        # init_dp
        self.step(f"\\boxed{{init_dp {target_sum}}}")

        # observe
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        # parse array
        try:
            array_str = obs.split(": ", 1)[1]
            numbers = ast.literal_eval(array_str)
        except Exception:
            numbers = self.arr

        # update_dp for each number
        for num in numbers:
            self.step(f"\\boxed{{update_dp {num} {target_sum}}}")

        # find_max
        obs, _, _, _, _ = self.step(f"\\boxed{{find_max {target_sum}}}")
        m2 = re.search(r"Max reachable subset sum\s*=\s*(\d+)", obs)
        max_subset_sum = int(m2.group(1)) if m2 else 0

        # calc_diff
        obs, _, _, _, _ = self.step(f"\\boxed{{calc_diff {total_sum} {max_subset_sum}}}")
        m3 = re.search(r"Min difference\s*=\s*(\d+)", obs)
        min_diff = int(m3.group(1)) if m3 else self.get_ref_answer()

        # answer
        obs, _, _, _, _ = self.step(f"\\boxed{{answer {min_diff}}}")
        return obs