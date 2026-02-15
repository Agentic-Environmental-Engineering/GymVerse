from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class TeamScoreBalancingEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围（根据原环境分析）
        # 由于 DP 复杂度 ~ O(n * total_sum/2)，避免过大值导致资源消耗过高
        self.complexity_params = {
            "array_length": (5, 50),   # 玩家数量
            "value_range": (1, 200),   # 分数范围（最大值）
        }

        # 参数方差（用于增强训练时的多样性）
        self.param_variance = {
            "array_length": 2,  # ±2
            "value_range": 20,  # ±20
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例与中间变量
        self.n: int = 0
        self.scores: list[int] = []
        self.total_sum: Optional[int] = None
        self.dp_array: Optional[list[int]] = None
        self.max_subset_sum: Optional[int] = None
        self.min_diff: Optional[int] = None

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
            "Team Score Balancing: Split players into two teams to minimize the score difference.\n"
            "You can compute using a 0/1 knapsack DP up to total_sum//2.\n"
            "Available actions (use \\boxed{...}):\n"
            "- Observe players and scores: \\boxed{observe}\n"
            "- Calculate total sum: \\boxed{sum}\n"
            "- Initialize DP array: \\boxed{init T} or \\boxed{init target_sum=T} (default T=total_sum//2 if available)\n"
            "- Update DP with a score: \\boxed{update S} or \\boxed{update score=S}\n"
            "- Find max subset sum: \\boxed{find_max}\n"
            "- Calculate min difference: \\boxed{calc_diff} or \\boxed{calc_diff total_sum=X max_subset_sum=Y}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Players: {self.n} | Turn: {self.turn_count}/{self.max_turns} | "
            f"DP inited: {'yes' if self.dp_array is not None else 'no'} | "
            f"total_sum: {self.total_sum if self.total_sum is not None else 'N/A'} | "
            f"max_subset_sum: {self.max_subset_sum if self.max_subset_sum is not None else 'N/A'}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 重置中间变量
        self.total_sum = None
        self.dp_array = None
        self.max_subset_sum = None
        self.min_diff = None

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        self.n = self.array_length
        self.scores = [random.randint(1, self.value_range) for _ in range(self.n)]
        return {"n": self.n, "scores": self.scores}

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
        lc = content.lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if lc.startswith("observe"):
                obs = self.Observe()

            elif lc in ("sum", "calculate_total_sum", "total"):
                obs = self.CalculateTotalSum()
                # 保存内部状态
                try:
                    self.total_sum = int(obs)
                except Exception:
                    pass

            elif lc.startswith("init"):
                # 支持两种形式：init T 或 init target_sum=T
                target_sum = None
                m = re.search(r"init\s+(\d+)", lc)
                if not m:
                    m2 = re.search(r"target_sum\s*=\s*(\d+)", lc)
                    if m2:
                        target_sum = int(m2.group(1))
                else:
                    target_sum = int(m.group(1))

                if target_sum is None:
                    if self.total_sum is None:
                        obs = "Error: total_sum is not available; run \\boxed{sum} or specify target_sum."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        target_sum = self.total_sum // 2

                if not terminated:
                    obs = self.InitializeDPArray(target_sum)
                    # 保存内部状态
                    try:
                        self.dp_array = json.loads(obs)
                    except Exception:
                        self.dp_array = [0] * (target_sum + 1)

            elif lc.startswith("update"):
                # 支持 update S 或 update score=S
                if self.dp_array is None:
                    obs = "Error: DP array not initialized. Use \\boxed{init target_sum=T} first."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    score = None
                    m = re.search(r"update\s+(\d+)", lc)
                    if not m:
                        m2 = re.search(r"score\s*=\s*(\d+)", lc)
                        if m2:
                            score = int(m2.group(1))
                    else:
                        score = int(m.group(1))

                    if score is None:
                        obs = "Error: Missing score. Use \\boxed{update S}."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.UpdateDPArray(score, self.dp_array)
                        try:
                            self.dp_array = json.loads(obs)
                        except Exception:
                            pass

            elif lc in ("find_max", "find_max_subset_sum"):
                if self.dp_array is None:
                    obs = "Error: DP array not initialized. Use \\boxed{init target_sum=T} first."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.FindMaxSubsetSum(self.dp_array)
                    try:
                        self.max_subset_sum = int(obs)
                    except Exception:
                        pass

            elif lc.startswith("calc_diff") or lc.startswith("calculate_min_difference"):
                # 支持 calc_diff 或 calc_diff total_sum=X max_subset_sum=Y
                ts, ms = self.total_sum, self.max_subset_sum
                m_ts = re.search(r"total_sum\s*=\s*(\d+)", lc)
                m_ms = re.search(r"max_subset_sum\s*=\s*(\d+)", lc)
                if m_ts:
                    ts = int(m_ts.group(1))
                if m_ms:
                    ms = int(m_ms.group(1))

                if ts is None or ms is None:
                    obs = "Error: Missing total_sum or max_subset_sum. Use \\boxed{sum}, \\boxed{init}, \\boxed{update}, \\boxed{find_max} first."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.CalculateMinDifference(ts, ms)
                    try:
                        self.min_diff = int(obs)
                    except Exception:
                        pass

            elif lc.startswith("answer"):
                # 提交答案：answer N
                m = re.search(r"answer\s+(-?\d+)", lc)
                if not m:
                    obs = "Format error: use \\boxed{answer N}."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    ans = int(m.group(1))
                    obs = self.Done(ans)
                    # 解析正误
                    correct = "Result: Correct" in obs
                    reward = 1.0 if correct else -1.0
                    terminated = True

            else:
                obs = f"Invalid action: {content}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

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
        # 示例动作
        return "\\boxed{observe}"

    # -----------------------------
    # 保留并转换原环境的辅助方法
    # -----------------------------
    def CalculateTotalSum(self):
        r"""
        Calculate the sum of all players' scores.

        Args:
            None

        Returns:
            str: The total sum of players' scores.

        Example Output:
            "10"
        """
        total_sum = sum(self.scores)
        return str(total_sum)

    def InitializeDPArray(self, target_sum: int):
        r"""
        Initialize the dynamic programming array with a length of target_sum + 1 and all initial values set to 0.

        Args:
            target_sum (int): The target total sum, which determines the length of the array.

        Returns:
            str: The initialized dynamic programming array, converted to a string using json.dumps.

        Example Output:
            "[0, 0, 0, 0, 0]"
        """
        dp_array = [0] * (target_sum + 1)
        return json.dumps(dp_array)

    def UpdateDPArray(self, score: int, dp_array: list):
        r"""
        Update the dynamic programming array based on the current score, updating from the back to the front to avoid reusing the same score.

        Args:
            score (int): The current player's score to be processed.
            dp_array (list[int]): The current dynamic programming array.

        Returns:
            str: The updated dynamic programming array, converted to a string using json.dumps.

        Example Output:
            "[0, 1, 1, 2, 3]"
        """
        target_sum = len(dp_array) - 1
        for j in range(target_sum, score - 1, -1):
            dp_array[j] = max(dp_array[j], dp_array[j - score] + score)
        return json.dumps(dp_array)

    def FindMaxSubsetSum(self, dp_array: list):
        r"""
        Find the maximum value in the dynamic programming array, which is the maximum subset sum not exceeding the target total sum.

        Args:
            dp_array (list[int]): The dynamic programming array.

        Returns:
            str: The maximum value in the dynamic programming array.

        Example Output:
            "5"
        """
        max_subset_sum = max(dp_array)
        return str(max_subset_sum)

    def CalculateMinDifference(self, total_sum: int, max_subset_sum: int):
        r"""
        Calculate the minimum difference between the total scores of the two teams.

        Args:
            total_sum (int): The sum of all players' scores.
            max_subset_sum (int): The maximum subset sum not exceeding half of the total sum.

        Returns:
            str: The minimum difference between the total scores of the two teams.

        Example Output:
            "0"
        """
        min_difference = abs(total_sum - 2 * max_subset_sum)
        return str(min_difference)

    def Observe(self):
        r"""
        Obtain the number of players and the list of scores in the current environment.

        Args:
            None

        Returns:
            str: The number of players and the list of scores in the current environment.

        Example Output:
            "Number of players: 4, score list: [1, 2, 3, 4]"
        """
        return f"Number of players: {self.n}, score list: {self.scores}"

    def get_ref_answer(self):
        """
        获取参考答案：使用 0/1 背包在 total_sum//2 上得到最大子集和，再计算最小差值
        """
        total_sum = sum(self.scores)
        dp = [0] * (total_sum // 2 + 1)

        for score in self.scores:
            for j in range(total_sum // 2, score - 1, -1):
                dp[j] = max(dp[j], dp[j - score] + score)

        return abs(total_sum - 2 * dp[total_sum // 2])

    def Done(self, answer):
        r"""
        Verify whether the final answer is correct and return the result information.

        Args:
            answer (int): The minimum difference answer submitted by the user.

        Returns:
            str: Result information, including whether it is correct and reward information.

        Example Output:
            "Your answer: 0, Reference answer: 0, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={'1' if correct else '0'}"