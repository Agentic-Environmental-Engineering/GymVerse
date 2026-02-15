from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxCutTreesEnvGEM(Env):
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
        # - array_length: 数组长度（树的数量）
        # - height_min: 树高度的最小值
        # - height_max: 树高度的最大值
        # - num_constraints: 约束（此任务为非相邻选择，固定为1，但保留用于扩展）
        self.complexity_params = {
            "array_length": (5, 50),
            "height_min": (1, 10),
            "height_max": (10, 1000),
            "num_constraints": (1, 1),
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "height_min": 1,
            "height_max": 25,
            "num_constraints": 0,
        }

        # 占位属性
        self.array_length: int = 0
        self.height_min: int = 0
        self.height_max: int = 0
        self.num_constraints: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.trees: list[int] = []

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

        # 确保 height_min <= height_max
        if self.height_min > self.height_max:
            self.height_min, self.height_max = self.height_max, self.height_min

    def _get_instructions(self) -> str:
        return (
            "Max-Cut Trees (House Robber variant): Given a list of tree heights, "
            "compute the maximum total height by selecting non-adjacent trees.\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- Observe the task: \\boxed{observe}\n"
            "- Get tree count: \\boxed{count}\n"
            "- Get height at index i (0-based): \\boxed{height i}\n"
            "- Calculate DP value max(prev1, prev2 + h): \\boxed{calc prev1 prev2 h}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Trees: {len(self.trees)}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.trees = self.problem["trees"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        trees = [random.randint(self.height_min, self.height_max) for _ in range(n)]
        return {"trees": trees, "size": n}

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
        if len(tokens) == 0:
            obs = f"Empty action at turn {self.turn_count}."
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
                reward = 0.0
                terminated = False

            elif cmd == "count":
                obs = self.GetTreeCount()
                reward = 0.0
                terminated = False

            elif cmd == "height":
                if len(tokens) != 2:
                    obs = "Format error: use height i (0-based)."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                idx_str = tokens[1]
                if not idx_str.isdigit():
                    obs = "Format error: index must be a non-negative integer."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                idx = int(idx_str)
                obs = self.GetTreeHeight(idx)
                if obs.startswith("Error:"):
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                reward = 0.0
                terminated = False

            elif cmd == "calc":
                if len(tokens) != 4:
                    obs = "Format error: use calc prev1 prev2 h."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    prev1 = int(tokens[1])
                    prev2 = int(tokens[2])
                    h = int(tokens[3])
                except ValueError:
                    obs = "Format error: prev1, prev2, h must be integers."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.CalculateDpValue(prev1, prev2, h)
                reward = 0.0
                terminated = False

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Format error: use answer N."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    ans = int(tokens[1])
                except ValueError:
                    obs = "Format error: N must be an integer."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                # 验证答案
                ref_answer = self.get_ref_answer()
                correct = ans == ref_answer
                obs = self.Done(ans)
                reward = 1.0 if correct else -1.0
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
            obs = f"Runtime error: {str(e)}"
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
        if not self.trees:
            return "\\boxed{observe}"
        idx = random.randint(0, max(0, len(self.trees) - 1))
        # 随机选择一种动作示例
        choices = [
            "\\boxed{observe}",
            "\\boxed{count}",
            f"\\boxed{{height {idx}}}",
            "\\boxed{calc 10 5 7}",
        ]
        return random.choice(choices)

    # =========================
    # 辅助方法（从原环境保留并转换）
    # =========================
    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        n = len(self.trees)
        if n == 0:
            return 0
        if n == 1:
            return self.trees[0]

        dp = [0] * n
        dp[0] = self.trees[0]
        dp[1] = max(self.trees[0], self.trees[1])

        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + self.trees[i])

        return dp[-1]

    def GetTreeCount(self) -> str:
        """
        Get the number of trees.
        Returns:
            str: The number of trees.
        """
        return str(len(self.trees))

    def GetTreeHeight(self, index: int) -> str:
        """
        Get the height of the tree at the specified index.
        Args:
            index (int): The index of the tree (starting from 0).
        Returns:
            str: The height of the tree at the specified index or error.
        """
        if 0 <= index < len(self.trees):
            return str(self.trees[index])
        return "Error: Invalid index"

    def CalculateDpValue(self, prev1: int, prev2: int, height: int) -> str:
        """
        Calculate the current value in dynamic programming, which is equal to the
        maximum of the previous value and the sum of the value two places before and the current height.
        Args:
            prev1 (int): The previous value in dynamic programming.
            prev2 (int): The value two places before in dynamic programming.
            height (int): The height of the current tree.
        Returns:
            str: The calculated current dynamic programming value.
        """
        return str(max(prev1, prev2 + height))

    def Observe(self) -> str:
        """
        Return observation information of the current state.
        Returns:
            str: Prompt information describing the current state.
        """
        return (
            "Use \\boxed{count} to get number of trees, \\boxed{height i} to get height at index i, "
            "\\boxed{calc prev1 prev2 h} for DP helper, and \\boxed{answer N} to submit final result."
        )

    def Done(self, answer: int) -> str:
        """
        Verify whether the final answer is correct and return result information.
        Args:
            answer (int): The answer submitted by the user, i.e., the maximum total height.
        Returns:
            str: Result information, including whether it is correct.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        Automatically call actions to compute and submit the answer for verification.
        Returns:
            str: The result information of the final answer verification.
        """
        # Get count
        obs, _, term, _, _ = self.step("\\boxed{count}")
        if term:
            return obs
        try:
            tree_count = int(obs)
        except Exception:
            return "Format error while reading tree count."

        if tree_count == 0:
            obs, reward, term, _, _ = self.step("\\boxed{answer 0}")
            return obs

        if tree_count == 1:
            obs, _, term, _, _ = self.step("\\boxed{height 0}")
            if term:
                return obs
            height0 = int(obs)
            obs, reward, term, _, _ = self.step(f"\\boxed{{answer {height0}}}")
            return obs

        # DP initialize
        obs, _, term, _, _ = self.step("\\boxed{height 0}")
        if term:
            return obs
        prev2 = int(obs)

        obs, _, term, _, _ = self.step("\\boxed{height 1}")
        if term:
            return obs
        h1 = int(obs)
        prev1 = max(prev2, h1)

        for i in range(2, tree_count):
            obs, _, term, _, _ = self.step(f"\\boxed{{height {i}}}")
            if term:
                return obs
            h = int(obs)
            obs, _, term, _, _ = self.step(f"\\boxed{{calc {prev1} {prev2} {h}}}")
            if term:
                return obs
            current_dp = int(obs)
            prev2, prev1 = prev1, current_dp

        obs, reward, term, _, _ = self.step(f"\\boxed{{answer {prev1}}}")
        return obs