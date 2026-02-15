from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxNonAdjacentSumEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 3,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围
        self.complexity_params = {
            "array_length": (5, 50),  # 数组长度
            "value_abs_max": (10, 1000),  # 值的绝对最大值
            "negative_ratio_percent": (0, 50),  # 负数比例（百分比）
            "turn_allowance": (20, 200),  # 建议的最大步数（可用于覆盖 max_turns）
        }

        # 参数方差（用于启用随机化时的微调）
        self.param_variance = {
            "array_length": 2,
            "value_abs_max": 50,
            "negative_ratio_percent": 5,
            "turn_allowance": 5,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_abs_max: int = 0
        self.negative_ratio_percent: int = 0
        self.turn_allowance: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.arr: list[int] = []
        self._reward: float = 0.0
        self._done: bool = False

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

        # 可选：让难度控制步数上限（优先使用传入的 max_turns，如果传入则不覆盖）
        # 如果需要让难度完全控制，可取消以下注释以覆盖外部 max_turns：
        # self.max_turns = self.turn_allowance

    def _get_instructions(self) -> str:
        return (
            "Max Non-Adjacent Sum: Compute the maximum sum of non-adjacent elements in the array.\n"
            "Available actions (use \\boxed{...}):\n"
            "- Observe array: \\boxed{observe}\n"
            "- Initialize include/exclude at index i: \\boxed{init i}\n"
            "- Update at index i with current include/exclude: \\boxed{update i include X exclude Y}\n"
            "- Get max from include/exclude: \\boxed{getmax include X exclude Y}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "Goal: Submit the correct maximum non-adjacent sum. Success yields reward=1.0; wrong answer yields -1.0.\n"
        )

    def get_task_suffix(self) -> str:
        arr_len = len(self.arr)
        return f"Array length: {arr_len}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        max_abs = max(1, self.value_abs_max)
        neg_ratio = max(0, min(100, self.negative_ratio_percent)) / 100.0

        arr = []
        for _ in range(n):
            val = random.randint(0, max_abs)
            is_negative = random.random() < neg_ratio
            if is_negative:
                val = -val
            arr.append(val)

        # 调整边界情形：如果全为负数，保持原样（允许），如果数组长度为0（理论上不会），则生成空数组
        self.arr = arr
        return {"arr": arr, "size": n}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}. Expect \\boxed{{...}}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        if parsed.get("type") == "invalid":
            obs = f"Invalid action at turn {self.turn_count}: {parsed.get('raw', '')}"
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        t = parsed["type"]

        try:
            if t == "observe":
                obs = self.Observe()

            elif t == "init":
                index = parsed["index"]
                obs = self.InitializeIncludeExclude(index)

            elif t == "update":
                index = parsed["index"]
                include = parsed["include"]
                exclude = parsed["exclude"]
                obs = self.UpdateIncludeExclude(index, include, exclude)

            elif t == "getmax":
                include = parsed["include"]
                exclude = parsed["exclude"]
                obs = self.GetMaxSum(include, exclude)

            elif t == "answer":
                answer = parsed["answer"]
                ref_answer = self.get_ref_answer()
                correct = answer == ref_answer
                self._reward = 1.0 if correct else -1.0
                self._done = True
                obs = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
                reward = self._reward
                terminated = True
                truncated = False
        except Exception as e:
            obs = f"Error processing action: {str(e)}"
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
        raw = content
        content_lc = content.lower()

        # observe
        if re.fullmatch(r"observe", content_lc):
            return {"type": "observe"}

        # init i
        m = re.fullmatch(r"init\s+(\d+)", content_lc)
        if m:
            idx = int(m.group(1))
            return {"type": "init", "index": idx}

        # update i include X exclude Y
        m = re.fullmatch(r"update\s+(\d+)\s+include\s+(-?\d+)\s+exclude\s+(-?\d+)", content_lc)
        if m:
            idx = int(m.group(1))
            inc = int(m.group(2))
            exc = int(m.group(3))
            return {"type": "update", "index": idx, "include": inc, "exclude": exc}

        # getmax include X exclude Y
        m = re.fullmatch(r"getmax\s+include\s+(-?\d+)\s+exclude\s+(-?\d+)", content_lc)
        if m:
            inc = int(m.group(1))
            exc = int(m.group(2))
            return {"type": "getmax", "include": inc, "exclude": exc}

        # answer N
        m = re.fullmatch(r"answer\s+(-?\d+)", content_lc)
        if m:
            ans = int(m.group(1))
            return {"type": "answer", "answer": ans}

        return {"type": "invalid", "raw": raw}

    def sample_random_action(self) -> str:
        if not self.arr:
            return "\\boxed{observe}"
        choices = []
        choices.append("\\boxed{observe}")
        if len(self.arr) > 0:
            idx = random.randint(0, len(self.arr) - 1)
            choices.append(f"\\boxed{{init {idx}}}")
            inc = random.randint(-self.value_abs_max, self.value_abs_max)
            exc = random.randint(-self.value_abs_max, self.value_abs_max)
            choices.append(f"\\boxed{{update {idx} include {inc} exclude {exc}}}")
            choices.append(f"\\boxed{{getmax include {inc} exclude {exc}}}")
        # Random guess for answer
        guess = random.randint(-self.value_abs_max * len(self.arr), self.value_abs_max * len(self.arr) if self.arr else 0)
        choices.append(f"\\boxed{{answer {guess}}}")
        return random.choice(choices)

    # 辅助方法（保留并转换自原环境）
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    def get_ref_answer(self) -> int:
        """使用环境信息得到参考答案"""
        if not self.arr:
            return 0
        if len(self.arr) == 1:
            return self.arr[0]
        include = self.arr[0]
        exclude = 0
        for i in range(1, len(self.arr)):
            new_exclude = max(include, exclude)
            include = exclude + self.arr[i]
            exclude = new_exclude
        return max(include, exclude)

    def InitializeIncludeExclude(self, index: int):
        """
        初始化 include 和 exclude 变量
        返回 JSON 字符串：{"include": X, "exclude": 0}
        """
        if index < 0 or index >= len(self.arr):
            return json.dumps({"error": "Invalid index"})
        include = self.arr[index]
        exclude = 0
        return json.dumps({"include": include, "exclude": exclude})

    def UpdateIncludeExclude(self, index: int, include: int, exclude: int):
        """
        更新 include 和 exclude 变量
        返回 JSON 字符串：{"new_include": X, "new_exclude": Y}
        """
        if index < 0 or index >= len(self.arr):
            return json.dumps({"error": "Invalid index"})
        new_exclude = max(include, exclude)
        new_include = exclude + self.arr[index]
        return json.dumps({"new_include": new_include, "new_exclude": new_exclude})

    def GetMaxSum(self, include: int, exclude: int):
        """
        返回 include 和 exclude 的最大值（字符串）
        """
        max_sum = max(include, exclude)
        return str(max_sum)

    def Observe(self):
        """
        返回当前数组信息（字符串）
        """
        return f"Current array: {self.arr}"

    def Done(self, answer):
        """
        校验最终答案并返回结果信息（字符串，不改变 GEM 的终止与奖励逻辑）
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        自动调用所有动作完成求解，并提交答案用于验证。
        返回最终的验证信息字符串。
        """
        # 观察数组
        obs, _, terminated, _, _ = self.step("\\boxed{observe}")
        if terminated:
            return obs

        # 解析数组
        try:
            array_str = obs.split(": ")[1]
            nums = json.loads(array_str.replace("'", "\""))
        except Exception:
            nums = self.arr[:]

        n = len(nums)
        if n == 0:
            max_sum = 0
        elif n == 1:
            max_sum = nums[0]
        else:
            # init at index 0
            init_action = "\\boxed{init 0}"
            init_result, _, terminated, _, _ = self.step(init_action)
            if terminated:
                return init_result
            init_data = json.loads(init_result)
            current_include = init_data.get("include", 0)
            current_exclude = init_data.get("exclude", 0)

            for i in range(1, n):
                update_action = f"\\boxed{{update {i} include {current_include} exclude {current_exclude}}}"
                update_result, _, terminated, _, _ = self.step(update_action)
                if terminated:
                    return update_result
                update_data = json.loads(update_result)
                current_include = update_data.get("new_include", current_include)
                current_exclude = update_data.get("new_exclude", current_exclude)

            max_action = f"\\boxed{{getmax include {current_include} exclude {current_exclude}}}"
            max_sum_str, _, terminated, _, _ = self.step(max_action)
            if terminated:
                return max_sum_str
            try:
                max_sum = int(max_sum_str)
            except ValueError:
                # 解析失败则退回参考答案
                max_sum = self.get_ref_answer()

        final_action = f"\\boxed{{answer {max_sum}}}"
        final_result, _, _, _, _ = self.step(final_action)
        return final_result