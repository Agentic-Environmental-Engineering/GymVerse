from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinimumPossibleSumEnvGEM(Env):
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
        # 外部 max_turns 作为上限；复杂度根据参数设置具体预算（之后会覆盖为复杂度预算或其上限）
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围
        # - array_length: 数组长度
        # - value_range: 元素值范围上限（下限固定为 1）
        # - turn_budget: 最大步数限制（根据复杂度调整）
        # - threshold_tightness: 阈值紧致度（越大越接近或超过数组和）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 10000),
            "turn_budget": (20, 200),
            "threshold_tightness": (1, 4),
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 500,
            "turn_budget": 10,
            "threshold_tightness": 1,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0
        self.turn_budget: int = 0
        self.threshold_tightness: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题状态
        self.n: int = 0
        self.threshold: int = 0
        self.arr: list[int] = []
        self.current_sum: int = 0

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

        # 用复杂度预算覆盖 max_turns（外部传入值作为上限）
        self.max_turns = min(self.max_turns, self.turn_budget)

    def _get_instructions(self) -> str:
        return (
            "Minimum Possible Sum: Given an array and a threshold, decide the minimum possible sum.\n"
            "The answer equals max(sum(array), threshold).\n"
            "Available actions:\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Sort array descending: \\boxed{sort}\n"
            "- Calculate current sum: \\boxed{sum}\n"
            "- Check threshold exceeded (use last computed sum or provide one): \\boxed{check} or \\boxed{check S}\n"
            "- Calculate required difference to reach threshold: \\boxed{diff} or \\boxed{diff S}\n"
            "- Update sum to threshold using sum and diff: \\boxed{update S D}\n"
            "- Submit answer: \\boxed{answer N}\n"
            "Notes:\n"
            "- If S/D/N are omitted where allowed, the environment uses the latest computed sum.\n"
            "- Turns are limited; submit your final answer via \\boxed{answer N}.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array size: {self.n}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            f"Enter action."
        )

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
        # 生成数组
        arr = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        n = len(arr)
        total = sum(arr)

        # 根据 threshold_tightness 生成阈值：
        # 越紧致（值越大），阈值越接近或超过总和；越松散，越有可能低于总和。
        # 采用两个档位以保持简单直观：
        if self.threshold_tightness <= 2:
            low = max(1, int(total * 0.70))
            high = max(low, int(total * 1.10))
        else:
            low = max(1, int(total * 0.95))
            high = max(low, int(total * 1.50))
        threshold = random.randint(low, high)

        # 写入状态
        self.arr = arr
        self.n = n
        self.threshold = threshold
        self.current_sum = 0

        # 返回问题实例数据
        return {"arr": arr, "threshold": threshold, "n": n}

    # 保留并转换原环境的辅助方法
    def get_ref_answer(self) -> int:
        arr_copy = self.arr.copy()
        arr_copy.sort(reverse=True)
        current_sum = sum(arr_copy)
        if current_sum >= self.threshold:
            return current_sum
        return self.threshold

    def SortArray(self) -> str:
        r"""
        Sort the array in descending order.

        Returns:
            str: The sorted array as a JSON-like string.
        """
        self.arr.sort(reverse=True)
        return str(self.arr)

    def CalculateCurrentSum(self) -> str:
        r"""
        Calculate the sum of the current array.

        Returns:
            str: The sum of the array.
        """
        self.current_sum = sum(self.arr)
        return str(self.current_sum)

    def CheckThresholdExceeded(self, sum_value: Optional[int] = None) -> str:
        r"""
        Check if the given or current sum exceeds the threshold.

        Args:
            sum_value (Optional[int]): The sum value to check. Uses self.current_sum if None.

        Returns:
            str: "True" or "False".
        """
        if sum_value is None:
            sum_value = self.current_sum
        return str(sum_value >= self.threshold)

    def CalculateRequiredDiff(self, sum_value: Optional[int] = None) -> str:
        r"""
        Calculate the difference required to reach the threshold.

        Args:
            sum_value (Optional[int]): The current sum value. Uses self.current_sum if None.

        Returns:
            str: The difference required to reach the threshold (can be negative or zero).
        """
        if sum_value is None:
            sum_value = self.current_sum
        return str(self.threshold - sum_value)

    def UpdateSumToThreshold(self, sum_value: Optional[int] = None, diff: Optional[int] = None) -> str:
        r"""
        Update the sum to the threshold by adding the diff.

        Args:
            sum_value (Optional[int]): Current sum value. Uses self.current_sum if None.
            diff (Optional[int]): Difference required to reach the threshold. If None, computed from current_sum.

        Returns:
            str: The updated sum (sum_value + diff).
        """
        if sum_value is None:
            sum_value = self.current_sum
        if diff is None:
            diff = self.threshold - sum_value
        updated_sum = sum_value + diff
        return str(updated_sum)

    def Observe(self) -> str:
        r"""
        Get the current environment state information.

        Returns:
            str: State information including array size, threshold, and current array.
        """
        return f"Array size: {self.n}, Threshold: {self.threshold}, Current array: {self.arr}"

    def Done(self, answer: int) -> str:
        r"""
        Submit the final answer and verify if it is correct.

        Args:
            answer (int): The submitted minimum possible sum.

        Returns:
            str: Information including the answer verification result.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            reward = LanguageGameReward.format_error_reward
            terminated = True
            truncated = False
            info = {"suffix": self.get_task_suffix()}
            # 超时检查（统一放在 step 结尾）
            if not terminated and self.turn_count >= self.max_turns:
                obs = f"{obs}\nReached max turns ({self.max_turns})."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, reward, terminated, truncated, info

        content = parsed["content"]
        tokens = content.split()
        action_name = tokens[0].lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        def parse_int(token: str) -> Optional[int]:
            try:
                return int(token)
            except Exception:
                return None

        if action_name == "observe":
            obs = self.Observe()

        elif action_name == "sort":
            obs = self.SortArray()

        elif action_name == "sum":
            obs = self.CalculateCurrentSum()

        elif action_name == "check":
            # \boxed{check} or \boxed{check S}
            if len(tokens) >= 2:
                s_val = parse_int(tokens[1])
                if s_val is None:
                    obs = "Invalid 'check' parameter. Expect integer S."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.CheckThresholdExceeded(s_val)
            else:
                obs = self.CheckThresholdExceeded()

        elif action_name == "diff":
            # \boxed{diff} or \boxed{diff S}
            if len(tokens) >= 2:
                s_val = parse_int(tokens[1])
                if s_val is None:
                    obs = "Invalid 'diff' parameter. Expect integer S."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.CalculateRequiredDiff(s_val)
            else:
                obs = self.CalculateRequiredDiff()

        elif action_name == "update":
            # \boxed{update S D}
            if len(tokens) < 3:
                obs = "Invalid 'update' parameters. Expect two integers: S D."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                s_val = parse_int(tokens[1])
                d_val = parse_int(tokens[2])
                if s_val is None or d_val is None:
                    obs = "Invalid 'update' parameters. Expect integer S and integer D."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.UpdateSumToThreshold(s_val, d_val)

        elif action_name == "answer":
            # \boxed{answer N}
            if len(tokens) < 2:
                obs = "Invalid 'answer' parameter. Expect integer N."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                n_val = parse_int(tokens[1])
                if n_val is None:
                    obs = "Invalid 'answer' parameter. Expect integer N."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs_msg = self.Done(n_val)
                    # 判定成功与否
                    ref_answer = self.get_ref_answer()
                    if n_val == ref_answer:
                        reward = 1.0
                    else:
                        reward = -1.0
                    terminated = True
                    obs = obs_msg

        else:
            obs = f"Invalid action '{action_name}'."
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
        # 随机示例动作
        candidates = [
            "\\boxed{observe}",
            "\\boxed{sort}",
            "\\boxed{sum}",
            "\\boxed{check}",
            "\\boxed{diff}",
        ]
        return random.choice(candidates)

    # 提供一个自动求解器，保留原环境思路但采用 GEM 风格动作
    def solve(self) -> str:
        # 观测
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        # 计算和
        obs, _, _, _, _ = self.step("\\boxed{sum}")
        try:
            current_sum = int(self.current_sum)
        except Exception:
            current_sum = sum(self.arr)
        # 检查阈值
        obs, _, term, _, _ = self.step(f"\\boxed{{check {current_sum}}}")
        if term:
            return obs
        threshold_exceeded = obs.strip() == "True"

        if threshold_exceeded:
            # 直接提交当前和
            final_obs, reward, terminated, truncated, _ = self.step(f"\\boxed{{answer {current_sum}}}")
            return final_obs
        else:
            # 计算差值并更新到阈值
            obs, _, term, _, _ = self.step(f"\\boxed{{diff {current_sum}}}")
            if term:
                return obs
            try:
                required_diff = int(obs)
            except Exception:
                required_diff = self.threshold - current_sum
            obs, _, term, _, _ = self.step(f"\\boxed{{update {current_sum} {required_diff}}}")
            if term:
                return obs
            updated_sum = obs.strip()
            final_obs, reward, terminated, truncated, _ = self.step(f"\\boxed{{answer {updated_sum}}}")
            return final_obs