from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinimizeMaxSubarraySumEnvGEM(Env):
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
        # - array_length: 数组长度
        # - value_max: 元素最大值（元素在 [1, value_max] 随机）
        # - m_subarrays: 可允许的最大子数组数（最终 m 在 [1, min(m_subarrays, array_length)] 随机）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_max": (10, 10000),
            "m_subarrays": (2, 10),
        }

        # 参数方差（仅在 enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 2,
            "value_max": 500,
            "m_subarrays": 1,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_max: int = 0
        self.m_subarrays: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.problem: Dict[str, Any] = {}
        self.has_submitted: bool = False

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

        # 保障边界
        if self.array_length < 1:
            self.array_length = 1
        if self.value_max < 1:
            self.value_max = 1
        if self.m_subarrays < 1:
            self.m_subarrays = 1

    def _get_instructions(self) -> str:
        return (
            "Minimize Max Subarray Sum: Given an array of positive integers and an integer m,\n"
            "split the array into m or fewer non-empty contiguous subarrays to minimize the largest subarray sum.\n"
            "Available actions:\n"
            "- Observe the array and m: \\boxed{observe}\n"
            "- Query maximum element: \\boxed{max}\n"
            "- Query total sum: \\boxed{sum}\n"
            "- Check feasibility for a candidate max_sum S: \\boxed{check S}\n"
            "- Submit final answer (the minimized largest sum): \\boxed{answer S}\n"
        )

    def get_task_suffix(self) -> str:
        length = len(self.problem.get("nums", []))
        m = self.problem.get("m", 0)
        return f"Array length: {length}, m: {m}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self.has_submitted = False
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        length = self.array_length
        nums = [random.randint(1, self.value_max) for _ in range(length)]
        # m 至少 1，且不超过数组长度和 m_subarrays
        max_m = max(1, min(self.m_subarrays, length))
        m = random.randint(1, max_m)
        return {"nums": nums, "m": m}

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

        if self.has_submitted:
            obs = "Episode already finished. Please reset the environment."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        action_type = parsed.get("type", "")
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if action_type == "observe":
                obs = self.Observe()

            elif action_type == "max":
                obs = self.GetMaxValue()

            elif action_type == "sum":
                obs = self.GetSum()

            elif action_type == "check":
                max_sum = int(parsed.get("value", 0))
                obs = self.CanSplit(max_sum)

            elif action_type == "answer":
                answer = int(parsed.get("value", 0))
                msg, correct = self.Done(answer)
                obs = msg
                reward = 1.0 if correct else -1.0
                terminated = True
                self.has_submitted = True

            else:
                obs = f"Invalid action: {action_type}"
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

        # Map content to action types
        if re.fullmatch(r"observe", content, flags=re.IGNORECASE):
            return {"type": "observe"}
        if re.fullmatch(r"max", content, flags=re.IGNORECASE):
            return {"type": "max"}
        if re.fullmatch(r"sum", content, flags=re.IGNORECASE):
            return {"type": "sum"}

        m_check = re.fullmatch(r"check\s+(-?\d+)", content, flags=re.IGNORECASE)
        if m_check:
            return {"type": "check", "value": int(m_check.group(1))}

        m_answer = re.fullmatch(r"answer\s+(-?\d+)", content, flags=re.IGNORECASE)
        if m_answer:
            return {"type": "answer", "value": int(m_answer.group(1))}

        return None

    def sample_random_action(self) -> str:
        # 随机示例动作
        actions = [
            "\\boxed{observe}",
            "\\boxed{max}",
            "\\boxed{sum}",
            f"\\boxed{{check {random.randint(1, max(1, sum(self.problem.get('nums', [1]))))}}}",
        ]
        return random.choice(actions)

    # -----------------------------
    # 保留原环境的所有辅助方法（转换为内部使用）
    # -----------------------------
    def GetMaxValue(self) -> str:
        """Get the maximum value in the array."""
        nums = self.problem.get("nums", [])
        if not nums:
            return "Max value: 0"
        return f"Max value: {max(nums)}"

    def GetSum(self) -> str:
        """Get the sum of all elements in the array."""
        nums = self.problem.get("nums", [])
        return f"Total sum: {sum(nums)}"

    def CanSplit(self, max_sum: int) -> str:
        """
        Determine whether the array can be split into no more than m subarrays
        such that the sum of each subarray does not exceed max_sum.
        Returns: "Can split: True/False"
        """
        nums = self.problem.get("nums", [])
        m = self.problem.get("m", 1)

        current_sum = 0
        count = 1
        for num in nums:
            if current_sum + num > max_sum:
                current_sum = num
                count += 1
                if count > m:
                    return "Can split: False"
            else:
                current_sum += num
        return "Can split: True"

    def Observe(self) -> str:
        """Return the array and the number of subarrays."""
        nums = self.problem.get("nums", [])
        m = self.problem.get("m", 1)
        return f"Array: {nums}, needs to be split into {m} subarrays"

    def get_ref_answer(self) -> int:
        """Compute the minimized maximum subarray sum using binary search."""
        nums = self.problem.get("nums", [])
        m = self.problem.get("m", 1)

        def can_split(max_sum: int) -> bool:
            current_sum = 0
            count = 1
            for num in nums:
                if current_sum + num > max_sum:
                    current_sum = num
                    count += 1
                    if count > m:
                        return False
                else:
                    current_sum += num
            return True

        left = max(nums) if nums else 0
        right = sum(nums) if nums else 0
        while left < right:
            mid = (left + right) // 2
            if can_split(mid):
                right = mid
            else:
                left = mid + 1
        return left

    def Done(self, answer: int) -> Tuple[str, bool]:
        """Verify whether the final answer is correct and return result information."""
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, correct