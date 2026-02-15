from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class UniqueSubstringsWithOddEnvGEM(Env):
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
        self.complexity_params = {
            "array_length": (5, 50),     # 数组长度
            "value_range": (10, 10000),  # 数值范围（0 到 value_range）
            "k_value": (1, 10),          # 子数组长度上限（实际会被 clamp 到 array_length）
            "num_constraints": (1, 1),   # 约束条件数量（此任务固定为1：至少一个奇数）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "k_value": 2,
            "num_constraints": 0,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.k_value: int = 0
        self.num_constraints: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.arr: list = []
        self.k: int = 1
        self.unique_substrings: set = set()

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
            "Unique Substrings with Odd:\n"
            "Given a hidden array arr and integer k, count unique length-k subarrays that contain at least one odd number.\n"
            "Available actions (use the latest \\boxed{...} only):\n"
            "- Observe array and k: \\boxed{observe}\n"
            "- Extract substring: \\boxed{extract START_INDEX} or \\boxed{extract START_INDEX K}\n"
            "- Check contains odd: \\boxed{check JSON_SUBARRAY}  e.g., \\boxed{check [1,2,3]}\n"
            "- Add to set: \\boxed{add JSON_SUBARRAY}\n"
            "- Get current set size: \\boxed{size}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array length: {len(self.arr)} | k: {self.k} | Unique set size: {len(self.unique_substrings)}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 将问题实例载入环境状态
        self.arr = self.problem["arr"]
        self.k = self.problem["k"]

        self.unique_substrings = set()

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        vmax = self.value_range
        arr = [random.randint(0, vmax) for _ in range(n)]

        # 计算 k，确保 1 <= k <= n
        k = max(1, min(self.k_value, n if n > 0 else 1))

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
        tokens = content.split()
        head = tokens[0].lower() if tokens else ""

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if head == "observe":
                obs = self.Observe()

            elif head == "extract":
                # extract START_INDEX [K]
                if len(tokens) < 2:
                    obs = "Error: extract requires START_INDEX."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                start_index_str = tokens[1]
                try:
                    start_index = int(start_index_str)
                except Exception:
                    obs = "Error: START_INDEX must be an integer."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

                if len(tokens) >= 3:
                    try:
                        k_val = int(tokens[2])
                    except Exception:
                        obs = "Error: K must be an integer when provided."
                        return (
                            obs,
                            LanguageGameReward.format_error_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                else:
                    k_val = self.k

                # 边界检查
                if start_index < 0 or k_val <= 0 or start_index + k_val > len(self.arr):
                    obs = "Error: extract out of range."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

                obs = self.ExtractSubstring(start_index, k_val)

            elif head == "check":
                # check JSON_SUBARRAY
                if len(tokens) < 2:
                    obs = "Error: check requires JSON_SUBARRAY."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                payload = content[len("check"):].strip()
                try:
                    substring = json.loads(payload)
                except Exception:
                    obs = "Error: JSON_SUBARRAY parse failed."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.CheckContainsOdd(substring)

            elif head == "add":
                # add JSON_SUBARRAY
                if len(tokens) < 2:
                    obs = "Error: add requires JSON_SUBARRAY."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                payload = content[len("add"):].strip()
                try:
                    substring = json.loads(payload)
                except Exception:
                    obs = "Error: JSON_SUBARRAY parse failed."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.AddToSet(substring)

            elif head == "size":
                obs = self.GetSetSize()

            elif head == "answer":
                # answer N
                if len(tokens) < 2:
                    obs = "Error: answer requires an integer N."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    ans = int(tokens[1])
                except Exception:
                    obs = "Error: N must be an integer."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

                # 验证答案并给出奖励
                result_msg, correct = self._verify_answer(ans)
                obs = result_msg
                reward = 1.0 if correct else -1.0
                terminated = True

            else:
                obs = f"Invalid action: {head}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except Exception as e:
            obs = f"Error: {str(e)}"
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
        # 简单示例动作
        return "\\boxed{observe}"

    # -------------------------
    # 保留并转换原环境的辅助方法
    # -------------------------

    def get_ref_answer(self) -> int:
        unique_substrings = set()
        for i in range(len(self.arr) - self.k + 1):
            substring = tuple(self.arr[i:i + self.k])
            if any(x % 2 != 0 for x in substring):
                unique_substrings.add(substring)
        return len(unique_substrings)

    def ExtractSubstring(self, start_index: int, k: int) -> str:
        """
        Extract a subarray of length k starting from start_index from the array.

        Returns:
            str: The extracted subarray returned as a JSON string
        """
        end_index = start_index + k
        substring = self.arr[start_index:end_index]
        return json.dumps(substring)

    def CheckContainsOdd(self, substring: list) -> str:
        """
        Check if the subarray contains at least one odd number.

        Returns:
            str: "True" or "False"
        """
        contains_odd = any(x % 2 != 0 for x in substring)
        return str(contains_odd)

    def AddToSet(self, substring: list) -> str:
        """
        Add the subarray to the set to track unique subarrays.

        Returns:
            str: The size of the current set
        """
        tuple_sub = tuple(substring)
        self.unique_substrings.add(tuple_sub)
        return str(len(self.unique_substrings))

    def GetSetSize(self) -> str:
        """
        Get the size of the current set.

        Returns:
            str: The size of the set
        """
        return str(len(self.unique_substrings))

    def Observe(self) -> str:
        """
        Obtain the observation information of the current environment.

        Returns:
            str: Observation information containing the array and the k value
        """
        return f"Array: {self.arr}, k value: {self.k}"

    def _verify_answer(self, answer: int) -> Tuple[str, bool]:
        """
        Internal helper to verify final answer.
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg, correct

    def Done(self, answer: int) -> str:
        """
        Kept for compatibility with original helper style.
        Returns validation message only (reward is handled in step).
        """
        msg, _ = self._verify_answer(answer)
        return msg

    def solve(self) -> str:
        """
        Automatically simulate actions to compute and submit the answer.
        This helper uses the boxed action interface and returns final message.
        """
        # Observe to get arr and k
        obs, _, terminated, _, _ = self.step("\\boxed{observe}")
        if terminated:
            return obs

        # Parse array and k from the observation
        try:
            array_str = obs.split('Array: ')[1].split(', k value: ')[0]
            k_str = obs.split('k value: ')[1]
            array = json.loads(array_str)
            k = int(k_str)
        except Exception:
            # If parsing fails, terminate
            return "Error: Failed to parse observation."

        n = len(array)
        max_start = n - k
        if max_start < 0:
            final_obs, _, _, _, _ = self.step("\\boxed{answer 0}")
            return final_obs

        # Iterate and process substrings
        for start in range(max_start + 1):
            act_extract = f"\\boxed{{extract {start} {k}}}"
            sub_json, _, termd, trunc, _ = self.step(act_extract)
            if termd:
                return sub_json
            try:
                substring = json.loads(sub_json)
            except Exception:
                # invalid extract result, try to continue but likely to fail soon
                continue
            act_check = f"\\boxed{{check {json.dumps(substring)}}}"
            has_odd_str, _, termd, trunc, _ = self.step(act_check)
            if termd:
                return has_odd_str
            if has_odd_str == "True":
                act_add = f"\\boxed{{add {json.dumps(substring)}}}"
                add_res, _, termd, trunc, _ = self.step(act_add)
                if termd:
                    return add_res

        set_size_str, _, termd, trunc, _ = self.step("\\boxed{size}")
        if termd:
            return set_size_str
        try:
            set_size = int(set_size_str)
        except Exception:
            set_size = 0

        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {set_size}}}")
        return final_obs