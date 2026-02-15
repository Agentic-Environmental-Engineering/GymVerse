from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxNonBlockingTowersEnvGEM(Env):
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
            "array_length": (5, 50),     # 塔数量（数组长度）
            "value_range": (10, 1000),   # 属性范围（攻击范围与高度）
            "num_constraints": (2, 2),   # 约束条件数量（必须为2，严格递增两个维度）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 5,
            "value_range": 100,
            "num_constraints": 0,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.num_constraints: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}

        # 内部求解状态
        self.towers: list = []
        self.sorted_towers: list = []
        self.dp_array: list = []

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
            "Max Non-Blocking Towers: Given towers with (range, height), find the maximum number of towers "
            "such that both range and height strictly increase.\n"
            "Available actions (use the last \\boxed{...} in your message):\n"
            "- Sort towers: \\boxed{sort}\n"
            "- Initialize DP array: \\boxed{init}\n"
            "- Update DP value: \\boxed{update i j}  (0-based indices)\n"
            "- Find max DP value: \\boxed{max}\n"
            "- Observe state: \\boxed{observe}\n"
            "- Submit answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        sorted_status = "Yes" if self.sorted_towers else "No"
        dp_length = len(self.dp_array) if self.dp_array else 0
        return (
            f"Towers: {len(self.towers)} | Sorted: {sorted_status} | DP length: {dp_length}\n"
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
        self.towers = self.problem.get("towers", [])
        self.sorted_towers = []
        self.dp_array = []
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        towers = []
        max_val = max(1, self.value_range)
        for _ in range(self.array_length):
            r = random.randint(1, max_val)
            h = random.randint(1, max_val)
            towers.append([r, h])
        return {"towers": towers}

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
        if not tokens:
            obs = f"Invalid action at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
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
            if cmd == "sort":
                obs = self.SortTowers()

            elif cmd == "init":
                obs = self.InitializeDpArray()

            elif cmd == "update":
                if len(tokens) < 3:
                    obs = "Error: 'update' requires two indices i and j."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        i = int(tokens[1])
                        j = int(tokens[2])
                    except ValueError:
                        obs = "Error: indices must be integers."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.UpdateDpValue(i, j)
                        if obs.startswith("Error"):
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True

            elif cmd == "max":
                obs = self.FindMaxDpValue()

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "answer":
                if len(tokens) < 2:
                    obs = "Error: 'answer' requires an integer value."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        user_answer = int(tokens[1])
                    except ValueError:
                        obs = "Error: answer must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        # Done() 返回消息并内部设置状态；这里根据正确性赋奖励
                        ref = self.get_ref_answer()
                        correct = user_answer == ref
                        done_msg = self.Done(user_answer)
                        obs = done_msg
                        reward = 1.0 if correct else -1.0
                        terminated = True

            else:
                obs = f"Invalid action: {cmd}"
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
        return "\\boxed{observe}"

    # -----------------------
    # 保留原环境的辅助方法并转换
    # -----------------------

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        if not self.towers:
            return 0

        towers_copy = self.towers.copy()
        towers_copy.sort(key=lambda x: (x[0], x[1]))

        n = len(towers_copy)
        dp = [1] * n

        for i in range(n):
            for j in range(i):
                if towers_copy[j][0] < towers_copy[i][0] and towers_copy[j][1] < towers_copy[i][1]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp) if dp else 0

    def SortTowers(self):
        """
        Sort the towers by attack range and height.
        Returns: str (JSON array of towers)
        """
        self.sorted_towers = sorted(self.towers, key=lambda x: (x[0], x[1]))
        return json.dumps(self.sorted_towers)

    def InitializeDpArray(self):
        """
        Initialize the dp array with all elements set to 1.
        Returns: str (JSON array of dp)
        """
        if not self.sorted_towers:
            self.dp_array = []
            return json.dumps([])
        self.dp_array = [1] * len(self.sorted_towers)
        return json.dumps(self.dp_array)

    def UpdateDpValue(self, i: int, j: int):
        """
        Update dp[i] = max(dp[i], dp[j] + 1) if both range and height strictly increase.
        Returns: str (updated value of dp[i] or error message)
        """
        if not self.sorted_towers or not self.dp_array:
            return "0"

        if i < 0 or i >= len(self.dp_array) or j < 0 or j >= len(self.dp_array):
            return "Error: Index out of range"

        if self.sorted_towers[j][0] < self.sorted_towers[i][0] and self.sorted_towers[j][1] < self.sorted_towers[i][1]:
            self.dp_array[i] = max(self.dp_array[i], self.dp_array[j] + 1)

        return str(self.dp_array[i])

    def FindMaxDpValue(self):
        """
        Find the maximum value in the dp array.
        Returns: str (maximum dp value)
        """
        if not self.dp_array:
            return "0"
        return str(max(self.dp_array))

    def Observe(self):
        """
        Return observation information of the current state.
        Returns: str
        """
        sorted_status = "Yes" if self.sorted_towers else "No"
        dp_length = len(self.dp_array) if self.dp_array else 0
        return f"Number of towers: {len(self.towers)}, Sorted or not: {sorted_status}, Length of dp array: {dp_length}"

    def Done(self, answer):
        """
        Verify whether the final answer is correct and return result information.
        Returns: str (result info)
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={'1.0' if correct else '-1.0'}"

    def solve(self) -> str:
        """
        Automatically call all actions to complete the process and submit the answer for verification.
        Returns: str (final result info)
        """
        # sort towers
        obs, _, _, _, _ = self.step("\\boxed{sort}")
        try:
            sorted_towers = json.loads(obs)
        except Exception:
            sorted_towers = []
        n = len(sorted_towers)

        # init dp array
        self.step("\\boxed{init}")

        # update transitions
        for i in range(1, n):
            for j in range(i):
                if sorted_towers[j][0] < sorted_towers[i][0] and sorted_towers[j][1] < sorted_towers[i][1]:
                    self.step(f"\\boxed{{update {i} {j}}}")

        # get max dp
        obs, _, _, _, _ = self.step("\\boxed{max}")
        try:
            max_dp = int(obs)
        except Exception:
            max_dp = 0

        # submit answer
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {max_dp}}}")
        return final_obs