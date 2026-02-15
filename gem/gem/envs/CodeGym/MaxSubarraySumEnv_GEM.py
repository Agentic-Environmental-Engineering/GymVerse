from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxSubarraySumEnvGEM(Env):
    """
    GEM-compatible environment for the Max Subarray Sum of size k (sliding window) task.

    Actions (use the last \\boxed{...} block in your input):
    - \\boxed{observe}
        Returns the current array and k.
    - \\boxed{calc END_INDEX}
        Calculate sum of arr[:END_INDEX] (END_INDEX is exclusive).
    - \\boxed{slide CURRENT_SUM LEFT_INDEX K}
        Slide the window by removing arr[LEFT_INDEX] and adding arr[LEFT_INDEX + K] to CURRENT_SUM, return new sum.
    - \\boxed{compare SUM1 SUM2}
        Return the larger of the two sums.
    - \\boxed{answer N}
        Submit your final answer N. Episode ends with success (1.0) if correct, otherwise failure (-1.0).
    - \\boxed{help}
        Show instructions again.
    """

    def __init__(
        self,
        complexity: int = 4,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围（根据原环境分析）
        self.complexity_params = {
            # 数组长度：更高难度更长
            "array_length": (5, 50),
            # 数值范围（绝对值上限）：更高难度数值更大
            "value_abs_max": (10, 10000),
            # 允许负数比例（百分比）
            "neg_ratio_pct": (0, 60),
            # k 的下界和上界，受数组长度约束
            "k_min": (1, 5),
            "k_max": (2, 15),
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_abs_max": 200,
            "neg_ratio_pct": 10,
            "k_min": 1,
            "k_max": 2,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_abs_max: int = 0
        self.neg_ratio_pct: int = 0
        self.k_min: int = 0
        self.k_max: int = 0

        # 状态
        self.turn_count: int = 0
        self.problem: Dict[str, Any] = {}

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

        # 保证 k_min <= k_max 且至少为 1
        if self.k_min < 1:
            self.k_min = 1
        if self.k_max < self.k_min:
            self.k_max = self.k_min

        # 至少长度为2以形成滑窗变化（允许 k=1 也可）
        if self.array_length < 1:
            self.array_length = 1

    def _get_instructions(self) -> str:
        return (
            "Max Subarray Sum (Fixed window size k): Find the maximum sum of any contiguous subarray of size k.\n"
            "Available actions (use the last \\boxed{...} in your input):\n"
            "- Observe current instance: \\boxed{observe}\n"
            "- Calculate initial sum (exclusive end index): \\boxed{calc END_INDEX}\n"
            "- Slide window by one step: \\boxed{slide CURRENT_SUM LEFT_INDEX K}\n"
            "- Compare two sums: \\boxed{compare SUM1 SUM2}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "- Show instructions: \\boxed{help}\n"
        )

    def get_task_suffix(self) -> str:
        n = len(self.problem.get("arr", []))
        return f"Array length: {n}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        arr = []
        for _ in range(n):
            # 按 neg_ratio_pct 决定符号
            is_neg = random.uniform(0, 100) < float(self.neg_ratio_pct)
            magnitude = random.randint(0, self.value_abs_max)
            val = -magnitude if is_neg else magnitude
            arr.append(val)

        # 确定 k
        k_low = max(1, min(self.k_min, n))
        k_high = max(k_low, min(self.k_max, n))
        k = random.randint(k_low, k_high) if k_low <= k_high else k_low

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

        cmd = parsed.get("cmd")
        args = parsed.get("args", [])

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        # 命令处理
        try:
            if cmd == "help":
                obs = self._get_instructions()

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd in ("calc", "calculate", "calculateinitialsum", "calculate_initial_sum"):
                if len(args) != 1:
                    obs = "Invalid arguments for calc. Usage: \\boxed{calc END_INDEX}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    end_index = self._parse_int(args[0])
                    if end_index is None:
                        obs = "END_INDEX must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.CalculateInitialSum(end_index)

            elif cmd == "slide":
                if len(args) != 3:
                    obs = "Invalid arguments for slide. Usage: \\boxed{slide CURRENT_SUM LEFT_INDEX K}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    current_sum = self._parse_int(args[0])
                    left_index = self._parse_int(args[1])
                    k_val = self._parse_int(args[2])
                    if None in (current_sum, left_index, k_val):
                        obs = "CURRENT_SUM, LEFT_INDEX, and K must be integers."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        # 额外边界检查
                        if left_index < 0 or k_val <= 0 or left_index + k_val >= len(self.problem["arr"]):
                            obs = "Index out of range for sliding window."
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        else:
                            obs = self.SlideWindow(current_sum, left_index, k_val)

            elif cmd == "compare":
                if len(args) != 2:
                    obs = "Invalid arguments for compare. Usage: \\boxed{compare SUM1 SUM2}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    s1 = self._parse_int(args[0])
                    s2 = self._parse_int(args[1])
                    if None in (s1, s2):
                        obs = "SUM1 and SUM2 must be integers."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.CompareSums(s1, s2)

            elif cmd == "answer":
                if len(args) != 1:
                    obs = "Invalid arguments for answer. Usage: \\boxed{answer N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    ans = self._parse_int(args[0])
                    if ans is None:
                        obs = "Answer must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        msg, correct = self.Done(ans, return_correct=True)
                        obs = msg
                        reward = 1.0 if correct else -1.0
                        terminated = True

            else:
                obs = f"Unknown action: {cmd}"
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
        if not content:
            return None

        tokens = content.strip().split()
        if not tokens:
            return None

        cmd = tokens[0].lower()
        args = tokens[1:]
        return {"cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        # Provide a random but valid non-terminal action most of the time
        choices = ["observe", "help"]
        if random.random() < 0.2:
            # sometimes submit a random answer
            return f"\\boxed{{answer {random.randint(-100, 100)} }}"
        return f"\\boxed{{{random.choice(choices)}}}"

    # --------------------------
    # Helper methods (adapted from original environment)
    # --------------------------

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        arr = self.problem.get("arr", [])
        k = self.problem.get("k", 1)
        if not arr or k <= 0 or k > len(arr):
            return 0

        window_sum = sum(arr[:k])
        max_sum = window_sum

        for i in range(len(arr) - k):
            window_sum = window_sum - arr[i] + arr[i + k]
            max_sum = max(max_sum, window_sum)

        return max_sum

    def CalculateInitialSum(self, end_index: int) -> str:
        """
        Calculate the sum of the elements in the array from the start up to (but not including) end_index.
        """
        arr = self.problem.get("arr", [])
        if end_index <= 0 or end_index > len(arr):
            return "0"
        return str(sum(arr[:end_index]))

    def SlideWindow(self, current_sum: int, left_index: int, k: int) -> str:
        """
        Slide the window by removing the left element and adding a new right element to calculate the new window sum.
        """
        arr = self.problem.get("arr", [])
        new_sum = current_sum - arr[left_index] + arr[left_index + k]
        return str(new_sum)

    def CompareSums(self, sum1: int, sum2: int) -> str:
        """
        Compare the sizes of two sums and return the larger one.
        """
        return str(max(sum1, sum2))

    def Observe(self) -> str:
        """
        Return the array and the value of k in the current environment.
        """
        arr = self.problem.get("arr", [])
        k = self.problem.get("k", 1)
        return f"Array: {arr}, k: {k}"

    def Done(self, answer: int, return_correct: bool = False) -> str | Tuple[str, bool]:
        """
        Verify whether the final answer is correct and return result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        if return_correct:
            return msg, correct
        return msg

    # --------------------------
    # Utility
    # --------------------------
    @staticmethod
    def _parse_int(s: str) -> Optional[int]:
        try:
            return int(s)
        except Exception:
            return None