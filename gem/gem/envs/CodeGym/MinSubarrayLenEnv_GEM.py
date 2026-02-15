from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinSubarrayLenEnvGEM(Env):
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
        self.complexity_params = {
            "array_length": (5, 50),          # 数组长度范围
            "value_max": (10, 10000),         # 数值最大值范围（元素为 1..value_max）
            "max_sum_queries": (10, 200),     # 允许的 sum 查询次数上限
            "max_turns_param": (20, 200),     # 难度建议的最大步数（不覆盖 self.max_turns，仅作状态展示）
        }

        # 参数方差（启用随机化时生效）
        self.param_variance = {
            "array_length": 2,
            "value_max": 100,
            "max_sum_queries": 5,
            "max_turns_param": 10,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_max: int = 0
        self.max_sum_queries: int = 0
        self.max_turns_param: int = 0

        # 问题实例
        self.nums: list[int] = []
        self.target: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.sum_query_count: int = 0

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
            "MinSubarrayLen: Find the minimal length of a contiguous subarray whose sum >= target.\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- Observe array and target: \\boxed{observe}\n"
            "- Get array length: \\boxed{len}\n"
            "- Get element by index i: \\boxed{get i}\n"
            "- Calculate sum of subarray [l, r]: \\boxed{sum l r}\n"
            "- Update minimum length: \\boxed{update current candidate}\n"
            "- Submit final answer N: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array length: {len(self.nums)} | "
            f"Sum queries: {self.sum_query_count}/{self.max_sum_queries} | "
            f"Turn: {self.turn_count}/{self.max_turns} (suggested {self.max_turns_param})\n"
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
        self.sum_query_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        self.nums = [random.randint(1, self.value_max) for _ in range(self.array_length)]
        total_sum = sum(self.nums)
        lower = max(1, int(self.value_max * 0.5))
        upper = max(lower, total_sum)
        self.target = random.randint(lower, upper)
        return {"nums": self.nums, "target": self.target}

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
            if lc == "observe":
                obs = self.Observe()
            elif lc in ("len", "length"):
                obs = self.GetArrayLength()
            elif lc.startswith("get"):
                # pattern: get i
                tokens = content.split()
                if len(tokens) != 2:
                    obs = "Invalid 'get' usage. Expected: get i"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                idx = int(tokens[1])
                result = self.GetElementByIndex(idx)
                if result.startswith("Error"):
                    obs = result
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = result
            elif lc.startswith("sum"):
                # pattern: sum l r
                tokens = content.split()
                if len(tokens) != 3:
                    obs = "Invalid 'sum' usage. Expected: sum l r"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                l = int(tokens[1])
                r = int(tokens[2])
                self.sum_query_count += 1
                result = self.CalculateSubarraySum(l, r)
                if result.startswith("Error"):
                    obs = result
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = result
            elif lc.startswith("update"):
                # pattern: update current candidate
                tokens = content.split()
                if len(tokens) != 3:
                    obs = "Invalid 'update' usage. Expected: update current candidate"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                cur_token = tokens[1].lower()
                if cur_token in ("inf", "infinity"):
                    current = float("inf")
                else:
                    current = float(tokens[1])
                candidate = int(tokens[2])
                obs = self.UpdateMinLength(current, candidate)
            elif lc.startswith("answer"):
                # pattern: answer N
                tokens = content.split()
                if len(tokens) != 2:
                    obs = "Invalid 'answer' usage. Expected: answer N"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans = int(tokens[1])
                ref_answer = self.get_ref_answer()
                correct = ans == ref_answer
                obs = f"Your answer: {ans}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
                reward = 1.0 if correct else -1.0
                terminated = True
            else:
                obs = "Invalid action."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

            # 查询上限检查（在执行动作后进行）
            if not terminated and self.sum_query_count > self.max_sum_queries:
                obs = f"{obs}\nExceeded sum query limit ({self.max_sum_queries})."
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
        return "\\boxed{observe}"

    # -----------------------
    # 辅助方法（从原环境保留并转换）
    # -----------------------
    def get_ref_answer(self) -> int:
        """
        使用环境中的信息获取参考答案。
        目标：最短子数组长度，使得子数组和 >= target；若不存在，返回 0。
        """
        n = len(self.nums)
        min_length = float('inf')
        left = 0
        current_sum = 0

        for right in range(n):
            current_sum += self.nums[right]
            while current_sum >= self.target:
                min_length = min(min_length, right - left + 1)
                current_sum -= self.nums[left]
                left += 1

        return 0 if min_length == float('inf') else int(min_length)

    def Observe(self) -> str:
        """
        获取当前环境中的数组与目标信息。
        返回示例：
        "Array: [2, 3, 1, 2, 4, 3], Target value: 7"
        """
        return f"Array: {self.nums}, Target value: {self.target}"

    def GetArrayLength(self) -> str:
        """
        获取数组长度。
        返回示例："6"
        """
        return str(len(self.nums))

    def GetElementByIndex(self, index: int) -> str:
        """
        获取数组指定索引的元素值。
        索引从 0 开始。
        返回示例："4"
        """
        if 0 <= index < len(self.nums):
            return str(self.nums[index])
        else:
            return "Error: Index out of range"

    def CalculateSubarraySum(self, left: int, right: int) -> str:
        """
        计算子数组 [left, right] 的元素和（包含左右端点）。
        返回示例："7"
        """
        if left < 0 or right >= len(self.nums) or left > right:
            return "Error: Invalid indices"
        subarray_sum = sum(self.nums[left:right + 1])
        return str(subarray_sum)

    def UpdateMinLength(self, current_length: float, candidate_length: int) -> str:
        """
        比较当前最小长度与候选长度，返回更小者。
        返回示例："2"
        """
        if candidate_length < current_length:
            return str(candidate_length)
        else:
            return str(current_length)

    def solve(self) -> str:
        """
        自动调用动作完成流程，并提交答案进行验证。
        返回最终验证的 observation 字符串。
        注意：此方法使用 step 接口，遵循查询上限与步数限制。
        """
        # 观察环境
        obs, _, term, trunc, _ = self.step("\\boxed{observe}")
        if term:
            return obs

        # 获取数组长度
        obs_len, _, term, trunc, _ = self.step("\\boxed{len}")
        if term:
            return obs_len
        try:
            n = int(obs_len)
        except Exception:
            n = len(self.nums)

        target = self.target
        min_len = float('inf')

        # 朴素双层循环 + sum 查询（遇到满足条件即更新并 break）
        for left in range(n):
            if term or trunc:
                break
            for right in range(left, n):
                if term or trunc:
                    break
                action_sum = f"\\boxed{{sum {left} {right}}}"
                sum_obs, _, term, trunc, _ = self.step(action_sum)
                if term or trunc:
                    return sum_obs
                try:
                    current_sum = int(sum_obs)
                except Exception:
                    # 非法结果，中止
                    return sum_obs
                if current_sum >= target:
                    candidate_len = right - left + 1
                    action_update = f"\\boxed{{update {min_len if min_len != float('inf') else 'inf'} {candidate_len}}}"
                    upd_obs, _, term, trunc, _ = self.step(action_update)
                    if term or trunc:
                        return upd_obs
                    try:
                        min_len = float(upd_obs)
                    except Exception:
                        # 解析失败，直接终止
                        return upd_obs
                    break  # 某 left 的最短已找到，移动到下一个 left

        final_answer = int(min_len) if min_len != float('inf') else 0
        ans_obs, _, _, _, _ = self.step(f"\\boxed{{answer {final_answer}}}")
        return ans_obs