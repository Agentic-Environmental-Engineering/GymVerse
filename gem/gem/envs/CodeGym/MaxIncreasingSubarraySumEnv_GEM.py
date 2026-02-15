from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxIncreasingSubarraySumEnvGEM(Env):
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
        # - 数组长度
        # - 数值范围
        # - 约束条件数量（示意，不影响核心逻辑）
        # - 搜索空间大小（示意，不影响核心逻辑）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 10000),
            "num_constraints": (1, 5),
            "search_space": (10, 1000),
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "num_constraints": 1,
            "search_space": 50,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.num_constraints: int = 0
        self.search_space: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.nums: list[int] = []

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
            "Task: Compute the maximum sum of strictly increasing contiguous subarrays.\n"
            "You can interact using the following actions inside a box:\n"
            "- \\boxed{observe}\n"
            "- \\boxed{length}\n"
            "- \\boxed{get INDEX}\n"
            "- \\boxed{compare I J}  # returns True if nums[I] > nums[J]\n"
            "- \\boxed{add VALUE CURRENT_SUM}\n"
            "- \\boxed{update MAX_SUM CURRENT_SUM}\n"
            "- \\boxed{set VALUE}  # set current sum to VALUE\n"
            "- \\boxed{answer N}  # submit your final answer\n"
        )

    def get_task_suffix(self) -> str:
        return f"Array size: {self.problem.get('size', 0)} | Turn: {self.turn_count}/{self.max_turns} | Enter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.nums = self.problem["nums"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

        # 返回说明与初始状态后，环境准备好接受动作

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 生成随机数组
        nums = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        return {
            "nums": nums,
            "size": self.array_length,
            "constraints": self.num_constraints,
            "search_space": self.search_space,
        }

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
            obs = "Invalid action: empty content."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        def _to_int(x: str) -> Optional[int]:
            try:
                return int(x)
            except Exception:
                return None

        if cmd == "observe":
            obs = self.Observe()

        elif cmd in ("length", "len"):
            obs = self.GetArrayLength()

        elif cmd in ("get", "element"):
            if len(tokens) != 2:
                obs = "Invalid parameters for 'get'. Expected: get INDEX"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                idx = _to_int(tokens[1])
                if idx is None:
                    obs = "Invalid index for 'get'."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.GetElement(idx)

        elif cmd in ("compare", "cmp"):
            if len(tokens) != 3:
                obs = "Invalid parameters for 'compare'. Expected: compare I J"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                i = _to_int(tokens[1])
                j = _to_int(tokens[2])
                if i is None or j is None:
                    obs = "Invalid indices for 'compare'."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.CompareElements(i, j)

        elif cmd == "add":
            if len(tokens) != 3:
                obs = "Invalid parameters for 'add'. Expected: add VALUE CURRENT_SUM"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                v = _to_int(tokens[1])
                s = _to_int(tokens[2])
                if v is None or s is None:
                    obs = "Invalid numbers for 'add'."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.AddToCurrentSum(v, s)

        elif cmd in ("update", "update_max"):
            if len(tokens) != 3:
                obs = "Invalid parameters for 'update'. Expected: update MAX_SUM CURRENT_SUM"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                m = _to_int(tokens[1])
                s = _to_int(tokens[2])
                if m is None or s is None:
                    obs = "Invalid numbers for 'update'."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.UpdateMaxSum(m, s)

        elif cmd in ("set", "set_current_sum"):
            if len(tokens) != 2:
                obs = "Invalid parameters for 'set'. Expected: set VALUE"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                v = _to_int(tokens[1])
                if v is None:
                    obs = "Invalid number for 'set'."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.SetCurrentSum(v)

        elif cmd in ("answer", "done"):
            if len(tokens) != 2:
                obs = "Invalid parameters for 'answer'. Expected: answer N"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                ans = _to_int(tokens[1])
                if ans is None:
                    obs = "Invalid number for 'answer'."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    verification_msg, is_correct = self.Done(ans)
                    obs = verification_msg
                    reward = 1.0 if is_correct else -1.0
                    terminated = True

        else:
            obs = f"Invalid action: {cmd}"
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
        # 示例动作：请求数组长度
        return "\\boxed{length}"

    # ===== 保留并转换原环境辅助方法 =====
    def Observe(self) -> str:
        """
        Returns basic information about the current environment.
        """
        return "Need to find the maximum sum of strictly increasing subarrays in the array"

    def GetArrayLength(self) -> str:
        """
        Gets the length of the array.
        """
        return str(len(self.nums))

    def GetElement(self, index: int) -> str:
        """
        Gets the value of the element at the specified index in the array.
        """
        if 0 <= index < len(self.nums):
            return str(self.nums[index])
        return "Error: Index out of range"

    def CompareElements(self, i: int, j: int) -> str:
        """
        Compares the sizes of the elements at indices i and j in the array.
        Returns "True" if nums[i] > nums[j], else "False".
        """
        if 0 <= i < len(self.nums) and 0 <= j < len(self.nums):
            return str(self.nums[i] > self.nums[j])
        return "Error: Index out of range"

    def AddToCurrentSum(self, value: int, current_sum: int) -> str:
        """
        Adds a value to the current sum.
        """
        return str(current_sum + value)

    def UpdateMaxSum(self, max_sum: int, current_sum: int) -> str:
        """
        Updates the maximum sum by taking max(max_sum, current_sum).
        """
        return str(max(max_sum, current_sum))

    def SetCurrentSum(self, value: int) -> str:
        """
        Sets the sum of the current subarray to the specified value.
        """
        return str(value)

    def Done(self, answer: int) -> Tuple[str, bool]:
        """
        Verifies whether the final answer is correct and returns result information.
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, correct

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        if not self.nums:
            return 0
        n = len(self.nums)
        if n == 1:
            return self.nums[0]

        max_sum = current_sum = self.nums[0]
        for i in range(1, n):
            if self.nums[i] > self.nums[i - 1]:
                current_sum += self.nums[i]
            else:
                max_sum = max(max_sum, current_sum)
                current_sum = self.nums[i]
        max_sum = max(max_sum, current_sum)
        return max_sum

    # 可选：提供一个内部 solve 方法（不用于交互，仅便于调试）
    def solve(self) -> str:
        """
        Non-interactive solve using direct access to nums for debugging convenience.
        """
        return f"Reference answer: {self.get_ref_answer()}"