from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class NearbyAlmostDuplicateEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 9,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围（原环境：数组长度、取值范围、k窗口/阈值）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 10000),
            "k_value": (1, 10),
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "k_value": 1,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.k_value: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.arr: list[int] = []
        self.k: int = 0

        # 状态变量
        self.turn_count: int = 0

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
            "Nearby Almost Duplicate:\n"
            "You are given an array arr and an integer k. Determine whether there exist distinct indices i and j such that:\n"
            "- |i - j| <= k, and\n"
            "- |arr[i] - arr[j]| <= k.\n"
            "Available actions:\n"
            "- Observe array and k: \\boxed{observe}\n"
            "- Check a specific pair: \\boxed{check_pair i j}\n"
            "- Check range around index i: \\boxed{check_range i}\n"
            "- Submit final answer: \\boxed{answer true} or \\boxed{answer false}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Array length: {len(self.arr)} | Turns: {self.turn_count}/{self.max_turns} | Enter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 同步便捷属性
        self.arr = self.problem["arr"]
        self.k = self.problem["k"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        arr = [random.randint(0, self.value_range) for _ in range(self.array_length)]
        k = self.k_value
        return {"arr": arr, "k": k}

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
            obs = f"Invalid action at turn {self.turn_count}: empty command."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        observation = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                observation = self.Observe()
                # 中间动作不终止
            elif cmd == "check_pair":
                if len(tokens) != 3:
                    observation = "Invalid parameters for check_pair. Use: \\boxed{check_pair i j}"
                    return (
                        observation,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i = int(tokens[1])
                j = int(tokens[2])
                observation = self.CheckPair(i, j)
            elif cmd == "check_range":
                if len(tokens) != 2:
                    observation = "Invalid parameters for check_range. Use: \\boxed{check_range i}"
                    return (
                        observation,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i = int(tokens[1])
                observation = self.CheckRange(i)
            elif cmd == "answer":
                if len(tokens) != 2:
                    observation = "Invalid parameters for answer. Use: \\boxed{answer true} or \\boxed{answer false}"
                    return (
                        observation,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans_token = tokens[1].lower()
                if ans_token not in ("true", "false"):
                    observation = "Invalid answer value. Use 'true' or 'false'."
                    return (
                        observation,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                answer = True if ans_token == "true" else False
                ref_answer = self.get_ref_answer()
                correct = answer == ref_answer
                observation = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
                reward = 1.0 if correct else -1.0
                terminated = True
            else:
                observation = f"Unknown action '{cmd}'."
                return (
                    observation,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
        except Exception as e:
            observation = f"Error: {str(e)}"
            return (
                observation,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            observation = f"{observation}\nReached max turns ({self.max_turns})."
            return observation, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return observation, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

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
        if len(self.arr) == 0:
            return "\\boxed{observe}"
        # Randomly choose among actions for demonstration
        choice = random.choice(["observe", "check_range", "check_pair", "answer"])
        if choice == "observe":
            return "\\boxed{observe}"
        elif choice == "check_range":
            i = random.randint(0, max(0, len(self.arr) - 1))
            return f"\\boxed{{check_range {i}}}"
        elif choice == "check_pair":
            if len(self.arr) < 2:
                return "\\boxed{observe}"
            i = random.randint(0, len(self.arr) - 1)
            j_candidates = [j for j in range(len(self.arr)) if j != i]
            j = random.choice(j_candidates)
            return f"\\boxed{{check_pair {i} {j}}}"
        else:
            return "\\boxed{answer false}"

    # ===== 保留并转换原环境的辅助方法 =====

    def get_ref_answer(self) -> bool:
        """
        Use the information in the environment to get the reference answer.
        Returns True if there exist indices i<j with j-i<=k and |arr[i]-arr[j]|<=k.
        """
        if self.k <= 0:
            return False

        n = len(self.arr)
        for i in range(n):
            for j in range(i + 1, min(i + self.k + 1, n)):
                if abs(self.arr[i] - self.arr[j]) <= self.k:
                    return True
        return False

    def CheckPair(self, i: int, j: int) -> str:
        """
        Check whether the elements at indices i and j in the array satisfy the condition:
        - i != j
        - |i - j| <= k
        - |arr[i] - arr[j]| <= k
        Returns "true" or "false".
        """
        if i < 0 or i >= len(self.arr) or j < 0 or j >= len(self.arr):
            return "false"
        if i == j:
            return "false"
        if abs(i - j) > self.k:
            return "false"
        return "true" if abs(self.arr[i] - self.arr[j]) <= self.k else "false"

    def CheckRange(self, i: int) -> str:
        """
        Check whether there exists a pair of elements that satisfy the condition
        within the range of k positions around index i in the array.
        Returns "true" or "false".
        """
        if i < 0 or i >= len(self.arr):
            return "false"
        start = i + 1
        end = min(i + self.k + 1, len(self.arr))
        for j in range(start, end):
            if abs(self.arr[i] - self.arr[j]) <= self.k:
                return "true"
        return "false"

    def Observe(self) -> str:
        """
        Return the array and threshold information in the current environment.
        Example: "Array: [1, 3, 5, 7], threshold k: 2"
        """
        return f"Array: {self.arr}, threshold k: {self.k}"

    def Done(self, answer: bool) -> str:
        """
        Verify whether the final answer is correct and return result information.
        This method is kept for compatibility, but does not alter GEM episode state.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        Automatically perform a simple heuristic to solve:
        - Observe
        - For each index i, check_range i; if any 'true', answer true; else answer false.
        Returns final observation message from the answer step.
        """
        # Observe
        obs, _, term, _, _ = self.step("\\boxed{observe}")
        if term:
            return obs

        # Try ranges
        n = len(self.arr)
        for i in range(n):
            obs, _, term, _, _ = self.step(f"\\boxed{{check_range {i}}}")
            if term:
                return obs
            if obs.strip().lower() == "true":
                final_obs, reward, term, trunc, _ = self.step("\\boxed{answer true}")
                return final_obs

        final_obs, reward, term, trunc, _ = self.step("\\boxed{answer false}")
        return final_obs