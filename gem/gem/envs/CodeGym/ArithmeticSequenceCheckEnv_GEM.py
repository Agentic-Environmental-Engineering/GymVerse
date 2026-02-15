from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class ArithmeticSequenceCheckEnvGEM(Env):
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

        # 难度参数范围设计
        self.complexity_params = {
            "array_length": (5, 50),  # 数组长度
            "value_range": (10, 10000),  # 数值范围 [0, value_range]
            "min_seq_len": (3, 8),  # 要求的最短等差序列长度
            "duplicates_ratio_pct": (0, 40),  # 重复元素比例（百分比）
            "max_turns_by_complexity": (20, 200),  # 步数限制随难度变化
        }

        # 参数方差（启用随机化时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "min_seq_len": 1,
            "duplicates_ratio_pct": 5,
            "max_turns_by_complexity": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0
        self.min_seq_len: int = 0
        self.duplicates_ratio_pct: int = 0
        self.max_turns_by_complexity: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.nums: list[int] = []

        # 问题实例
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

        # 将复杂度驱动的最大步数应用到环境
        self.max_turns = int(round(self.max_turns_by_complexity))

    def _get_instructions(self) -> str:
        return (
            "Arithmetic Sequence Check: Determine if the given array contains a consecutive arithmetic subsequence "
            f"of length at least {self.min_seq_len} after sorting.\n"
            "Available actions (use the last \\boxed{...} in your message):\n"
            "- Observe current array: \\boxed{observe}\n"
            "- Sort an array: \\boxed{sort [5, 2, 7, 3, 9]} (if omitted, sorts the current array)\n"
            "- Calculate difference: \\boxed{diff [1, 3, 5, 8, 10] 0}\n"
            "- Check consecutive differences: \\boxed{check [1, 3, 5, 8, 10] 1 2}\n"
            "- Submit final answer: \\boxed{answer true} or \\boxed{answer false}\n"
            "Rewards: Correct answer=1.0, Incorrect=-1.0, Format/Invalid action penalties apply. Timeout=0.0.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array target length: {self.array_length} | "
            f"Min seq len: {self.min_seq_len} | "
            f"Value range: [0, {self.value_range}] | "
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

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        L = self.array_length
        vr = self.value_range
        dup_pct = self.duplicates_ratio_pct

        # 计算重复数量
        dup_count = int(round(L * dup_pct / 100.0))
        unique_count = max(0, L - dup_count)

        nums: list[int] = []
        # 生成唯一值
        if vr + 1 >= unique_count and unique_count > 0:
            unique_vals = random.sample(range(0, vr + 1), unique_count)
        else:
            # 范围过小或 unique_count==0 时退化为随机选择
            unique_vals = [random.randint(0, vr) for _ in range(unique_count)]
        nums.extend(unique_vals)

        # 添加重复值
        for _ in range(dup_count):
            if nums:
                nums.append(random.choice(nums))
            else:
                nums.append(random.randint(0, vr))

        random.shuffle(nums)
        self.nums = nums
        return {"nums": nums, "size": len(nums)}

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

        name = parsed.get("name")
        params = parsed.get("parameters", {})

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if name == "observe":
                obs = self.Observe()
                terminated = False

            elif name == "sort":
                array = params.get("array", None)
                if array is None:
                    array = self.nums
                if not isinstance(array, list):
                    raise ValueError("Invalid parameter: 'array' must be a list.")
                obs = self.SortArray(array)
                terminated = False

            elif name == "diff":
                array = params.get("array", None)
                index = params.get("index", None)
                if array is None or index is None:
                    raise ValueError("Missing parameters for 'diff': require 'array' and 'index'.")
                if not isinstance(array, list) or not isinstance(index, int):
                    raise ValueError("Invalid parameter types for 'diff'.")
                obs = self.CalculateDifference(array, index)
                terminated = False

            elif name == "check":
                array = params.get("array", None)
                index = params.get("index", None)
                diff = params.get("diff", None)
                if array is None or index is None or diff is None:
                    raise ValueError("Missing parameters for 'check': require 'array', 'index', and 'diff'.")
                if not isinstance(array, list) or not isinstance(index, int) or not isinstance(diff, int):
                    raise ValueError("Invalid parameter types for 'check'.")
                obs = self.CheckConsecutiveDifferences(array, index, diff)
                terminated = False

            elif name == "answer":
                answer = params.get("answer", None)
                if isinstance(answer, str):
                    answer_lower = answer.strip().lower()
                    if answer_lower in ["true", "t", "1"]:
                        answer = True
                    elif answer_lower in ["false", "f", "0"]:
                        answer = False
                    else:
                        raise ValueError("Invalid 'answer' string. Use true/false.")
                if not isinstance(answer, bool):
                    raise ValueError("Missing or invalid 'answer' parameter (must be boolean).")

                ref_answer = self.get_ref_answer()
                correct = (answer == ref_answer)
                obs = self.Done(answer)
                reward = 1.0 if correct else -1.0
                terminated = True
                truncated = False

            else:
                obs = f"Invalid action: {name}"
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

        # Normalize and decide action
        lower = content.lower()
        # observe
        if lower == "observe":
            return {"name": "observe", "parameters": {}}

        # answer true/false
        ans_match = re.fullmatch(r"answer\s+(\S+)", lower)
        if ans_match:
            ans_token = ans_match.group(1)
            return {"name": "answer", "parameters": {"answer": ans_token}}

        # sort [array]
        if lower.startswith("sort"):
            # try to extract JSON array
            arr = self._extract_json_array(content)
            params: Dict[str, Any] = {}
            if arr is not None:
                params["array"] = arr
            return {"name": "sort", "parameters": params}

        # diff [array] index
        if lower.startswith("diff"):
            arr = self._extract_json_array(content)
            idx = self._extract_trailing_int(content)
            if arr is None or idx is None:
                return None
            return {"name": "diff", "parameters": {"array": arr, "index": idx}}

        # check [array] index diff
        if lower.startswith("check"):
            arr = self._extract_json_array(content)
            # Extract two integers at the end
            ints = self._extract_trailing_ints(content, count=2)
            if arr is None or ints is None:
                return None
            index, diff = ints
            return {"name": "check", "parameters": {"array": arr, "index": index, "diff": diff}}

        return None

    def _extract_json_array(self, text: str) -> Optional[list]:
        try:
            # Attempt to find the last [...] block
            bracket_pattern = re.compile(r"\[.*?\]", re.DOTALL)
            matches = list(bracket_pattern.finditer(text))
            if not matches:
                return None
            arr_str = matches[-1].group(0)
            arr = json.loads(arr_str)
            if isinstance(arr, list):
                return arr
            return None
        except Exception:
            return None

    def _extract_trailing_int(self, text: str) -> Optional[int]:
        # find last integer (possibly negative) in the string
        m = re.findall(r"(-?\d+)", text)
        if not m:
            return None
        try:
            return int(m[-1])
        except Exception:
            return None

    def _extract_trailing_ints(self, text: str, count: int = 2) -> Optional[Tuple[int, ...]]:
        m = re.findall(r"(-?\d+)", text)
        if not m or len(m) < count:
            return None
        try:
            vals = tuple(int(x) for x in m[-count:])
            return vals
        except Exception:
            return None

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{observe}",
            f"\\boxed{{sort {json.dumps(random.sample(range(0, max(1, self.value_range)), k=min(5, self.array_length)))}}}",
            "\\boxed{answer false}",
        ]
        return random.choice(choices)

    # ===== 原环境辅助方法（转换保留） =====

    def SortArray(self, array: list):
        """
        Sorts the input array.
        Args:
            array (list[int]): The array to be sorted.
        Returns:
            str: The sorted array returned as a JSON string. e.g. "[1, 3, 5, 7, 9]"
        """
        sorted_array = sorted(array)
        return json.dumps(sorted_array)

    def CalculateDifference(self, array: list, index: int):
        """
        Calculates the difference between the element at the specified index and the next element in the array.
        Args:
            array (list[int]): The input array.
            index (int): The starting index for calculating the difference.
        Returns:
            str: The calculated difference. e.g. "2"
        """
        if index < 0 or index >= len(array) - 1:
            return "Error: index out of range"
        diff = array[index + 1] - array[index]
        return str(diff)

    def CheckConsecutiveDifferences(self, array: list, index: int, diff: int):
        """
        Checks whether the array elements starting from the specified index maintain a constant difference.
        Args:
            array (list[int]): The input array.
            index (int): The starting index for the check.
            diff (int): The constant difference that needs to be maintained.
        Returns:
            str: Returns "true" if the constant difference is maintained, otherwise returns "false".
        """
        for j in range(index, len(array) - 1):
            if array[j + 1] - array[j] != diff:
                return "false"
        return "true"

    def Observe(self):
        """
        Obtains the array information in the current environment.
        Returns:
            str: The array in the current environment returned as a JSON string. e.g. "[5, 2, 7, 3, 9]"
        """
        return json.dumps(self.nums)

    def Done(self, answer: bool):
        """
        Submits the final answer and verifies its correctness.
        Args:
            answer (bool): The submitted answer; true indicates existence of an arithmetic sequence (length>=min_seq_len).
        Returns:
            str: Result information, including correctness and reward details.
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        # Reward is handled in step(), but we keep trace text here for consistency
        return msg + f", reward={'1.0' if correct else '-1.0'}"

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        Checks whether the sorted array contains a consecutive arithmetic subsequence
        of length at least self.min_seq_len.
        """
        sorted_nums = sorted(self.nums)
        n = len(sorted_nums)
        if n < self.min_seq_len:
            return False

        # Find any run of at least min_seq_len with constant difference
        for i in range(n - 1):
            diff = sorted_nums[i + 1] - sorted_nums[i]
            run_len = 2  # at least two consecutive elements
            j = i + 1
            while j < n - 1 and sorted_nums[j + 1] - sorted_nums[j] == diff:
                run_len += 1
                j += 1
            if run_len >= self.min_seq_len:
                return True
        return False