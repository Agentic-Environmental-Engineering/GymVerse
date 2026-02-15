from typing import Any, Dict, Optional, Tuple
import random
import re
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class SumArrayConstructionEnvGEM(Env):
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
        # - value_range: 元素数值范围（1 到 value_range）
        # - num_constraints: 约束条件数量（可用于未来扩展）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 10000),
            "num_constraints": (1, 5),
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "num_constraints": 1,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0
        self.num_constraints: int = 0

        # 环境状态变量
        self.turn_count: int = 0
        self._reward: float = 0.0
        self._done: bool = False

        # 问题实例
        self.n: int = 0
        self.arr: list[int] = []
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

    def _get_instructions(self) -> str:
        return (
            "Sum Array Construction: Build a new array where each element equals the total sum of the original array minus the element at the same index.\n"
            "Available actions (use the latest \\boxed{...} block in your message):\n"
            "- Observe the array and its length: \\boxed{observe}\n"
            "- Calculate total sum: \\boxed{total}\n"
            "- Calculate element value at index i given total_sum: \\boxed{elem i total_sum}\n"
            "  Example: \\boxed{elem 2 15}  # computes total_sum - arr[2]\n"
            "- Submit final answer (list of integers): \\boxed{answer [a0, a1, ..., a{n-1}]}\n"
            "Notes:\n"
            "- Indices are 0-based.\n"
            "- The final answer must be a list of length n.\n"
        )

    def get_task_suffix(self) -> str:
        size_info = f"Array size: {self.n if self.n else 'unknown'}"
        return f"{size_info}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        if seed is not None:
            random.seed(seed)
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        size = self.array_length
        values = [random.randint(1, self.value_range) for _ in range(size)]
        self.n = size
        self.arr = values
        return {"data": values, "size": size}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}. Expect a \\boxed{{...}} action."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        act_type = parsed.get("type")

        # 处理动作
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        if act_type == "observe":
            obs = self.Observe()
        elif act_type == "total":
            obs = self.CalculateTotalSum()
        elif act_type == "elem":
            idx = parsed.get("index")
            t_sum = parsed.get("total_sum")
            if idx is None or t_sum is None:
                obs = "Error: 'elem' requires index and total_sum, e.g., \\boxed{elem 2 15}."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            # 参数有效性检查
            if not isinstance(idx, int) or idx < 0 or idx >= self.n:
                obs = f"Error: index out of range. Valid range: 0..{self.n - 1}."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            if not isinstance(t_sum, int):
                obs = "Error: total_sum must be an integer."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            obs = self.CalculateElementSum(idx, t_sum)
        elif act_type == "answer":
            ans = parsed.get("answer")
            if not isinstance(ans, list):
                obs = "Error: answer must be a list of integers, e.g., \\boxed{answer [1,2,3]}."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            # 验证答案长度
            if len(ans) != self.n:
                obs = f"Error: answer length mismatch. Expected {self.n}, got {len(ans)}."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            # 提交答案
            obs = self.Done(ans)
            terminated = True
            reward = 1.0 if self._reward == 1 else -1.0
        else:
            obs = f"Invalid action: {parsed.get('raw', '')}"
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

        # 简单解析
        lc = content.lower()
        tokens = lc.split()

        # observe
        if lc == "observe":
            return {"type": "observe", "raw": raw}

        # total
        if lc == "total" or lc == "sum" or lc == "calculate_total":
            return {"type": "total", "raw": raw}

        # elem i total_sum
        if tokens and tokens[0] in {"elem", "element"}:
            if len(tokens) < 3:
                return None
            # 从原始 content 中提取两个参数（避免 lower 后的数字出错）
            # 尝试解析第二、第三个 token 为整数
            try:
                parts = content.split()
                idx = int(parts[1])
                t_sum = int(parts[2])
                return {"type": "elem", "index": idx, "total_sum": t_sum, "raw": raw}
            except Exception:
                return None

        # answer [list]
        if tokens and tokens[0] == "answer":
            # 提取 answer 后面的列表
            try:
                # 找到第一个 '[' 到最后一个 ']'
                start = content.find("[")
                end = content.rfind("]")
                if start == -1 or end == -1 or end < start:
                    return None
                list_str = content[start : end + 1]
                # 安全解析为 Python 列表
                ans = ast.literal_eval(list_str)
                # 强制为 int 列表
                if not isinstance(ans, list) or any(not isinstance(x, int) for x in ans):
                    return None
                return {"type": "answer", "answer": ans, "raw": raw}
            except Exception:
                return None

        return None

    def sample_random_action(self) -> str:
        # 示例：首先观察
        return "\\boxed{observe}"

    # 保留并转换原环境的辅助方法

    def get_ref_answer(self):
        """
        使用环境中的数组生成参考答案
        """
        total_sum = sum(self.arr)
        return [total_sum - self.arr[i] for i in range(self.n)]

    def CalculateTotalSum(self):
        """
        计算原始数组所有元素之和
        返回: 字符串形式的总和
        """
        total_sum = sum(self.arr)
        return str(total_sum)

    def CalculateElementSum(self, index: int, total_sum: int):
        """
        计算新数组指定索引的元素值：total_sum - arr[index]
        返回: 字符串形式的元素值
        """
        element_sum = total_sum - self.arr[index]
        return str(element_sum)

    def Observe(self):
        """
        获取当前环境的数组信息
        返回: 字符串，包含数组长度与数组元素
        """
        return f"Array length: {self.n}, Array elements: {self.arr}"

    def Done(self, answer):
        """
        校验最终答案，并返回结果信息
        返回: 字符串，包含答案、参考答案、是否正确、奖励信息
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self) -> str:
        """
        自动使用动作完成整个流程并提交答案进行验证
        返回: 最终答案验证的结果信息（字符串）
        """
        # 观察
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        # 从观察中解析长度
        try:
            len_str = obs.split("Array length: ")[1].split(",")[0]
            array_length = int(len_str)
        except Exception:
            array_length = self.n

        # 计算总和
        total_str, _, _, _, _ = self.step("\\boxed{total}")
        try:
            total_sum = int(total_str)
        except Exception:
            total_sum = sum(self.arr)

        # 构造新数组
        new_array = []
        for index in range(array_length):
            elem_str, _, term, _, _ = self.step(f"\\boxed{{elem {index} {total_sum}}}")
            # 如果意外终止，提前退出
            if term:
                break
            try:
                element = int(elem_str)
            except Exception:
                element = total_sum - self.arr[index]
            new_array.append(element)

        # 提交答案
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {new_array}}}")
        return final_obs