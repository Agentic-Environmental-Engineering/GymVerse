from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class RotatedArrayMinEnvGEM(Env):
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

        # 难度参数范围（根据原任务特性设计）
        # - 数组长度：控制问题规模
        # - 数值范围：控制元素值的范围（影响分布与可能的比较难度）
        # - 步数限制：最大可用步数（越高越宽松）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 10000),
            "max_turns": (20, 200),
        }

        # 参数方差（启用随机化时用于微调）
        self.param_variance = {
            "array_length": 2,
            "value_range": 100,
            "max_turns": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.array = []

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

            value = int(round(actual_value))
            if param_name == "array_length":
                self.array_length = value
            elif param_name == "value_range":
                self.value_range = value
            elif param_name == "max_turns":
                # 根据难度控制步数上限
                self.max_turns = value

    def _get_instructions(self) -> str:
        return (
            "Rotated Array Minimum: Find the smallest element in a rotated sorted array.\n"
            "Available actions (use the boxed command format):\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Get array length: \\boxed{len}\n"
            "- Get element at index i: \\boxed{get i}\n"
            "- Compare elements at indices i and j: \\boxed{cmp i j}\n"
            "- Submit final answer (integer or 'none'): \\boxed{answer X}\n"
            "Goal: Use binary search style querying to find the minimum element.\n"
        )

    def get_task_suffix(self) -> str:
        size = len(self.array)
        return f"Array size: {size}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()
        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.array = self.problem["array"]
        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = max(0, self.array_length)
        # 生成唯一递增数组
        # 为确保唯一性和范围足够，使用 random.sample 当 value_range >= n，否则用累加偏移
        if self.value_range >= n and self.value_range > 0:
            base_values = sorted(random.sample(range(self.value_range), n))
        else:
            # 退化情况：范围太小，生成递增序列
            start = random.randint(0, 10)
            base_values = [start + i for i in range(n)]

        # 随机旋转
        pivot = random.randint(0, n - 1) if n > 0 else 0
        rotated = base_values[pivot:] + base_values[:pivot]

        return {"array": rotated, "pivot": pivot, "sorted_base": base_values}

    # 解析动作
    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        return {"content": content}

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
        tokens = content.split()
        if len(tokens) == 0:
            obs = "Empty action."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()
            elif cmd in ("len", "length"):
                obs = self.GetArrayLength()
            elif cmd == "get":
                if len(tokens) < 2:
                    obs = "Error: index required for get."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                idx = int(tokens[1])
                # 边界检查
                if idx < 0 or idx >= len(self.array):
                    obs = "Error: Index out of bounds"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.GetElement(idx)
            elif cmd == "cmp":
                if len(tokens) < 3:
                    obs = "Error: two indices required for cmp."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                i = int(tokens[1])
                j = int(tokens[2])
                if i < 0 or i >= len(self.array) or j < 0 or j >= len(self.array):
                    obs = "Error: Index out of bounds"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.CompareElements(i, j)
            elif cmd == "answer":
                if len(tokens) < 2:
                    obs = "Error: answer value required."
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                ans_token = tokens[1].lower()
                answer_val: Optional[int]
                if ans_token in ("none", "null"):
                    answer_val = None
                else:
                    # 允许负数
                    try:
                        answer_val = int(tokens[1])
                    except Exception:
                        obs = "Error: invalid answer value."
                        return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                obs = self.Done(answer_val)
                # 根据 Done 的正确性分配奖励
                # Done() 消息中包含 "Result: Correct/Incorrect"
                terminated = True
                if "Result: Correct" in obs:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                obs = f"Invalid action: {cmd}"
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

    def sample_random_action(self) -> str:
        # 简单随机动作示例
        if len(self.array) == 0:
            return "\\boxed{observe}"
        r = random.random()
        if r < 0.2:
            return "\\boxed{observe}"
        elif r < 0.4:
            return "\\boxed{len}"
        elif r < 0.7 and len(self.array) > 0:
            i = random.randint(0, len(self.array) - 1)
            return f"\\boxed{{get {i}}}"
        else:
            if len(self.array) > 1:
                i = random.randint(0, len(self.array) - 1)
                j = random.randint(0, len(self.array) - 1)
                return f"\\boxed{{cmp {i} {j}}}"
            return "\\boxed{len}"

    # 保留原环境的辅助方法并转换

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "RotatedArrayMinEnvGEM@"
        if not env_str.startswith(prefix):
            return None
        # 解析 options（支持 JSON 风格或 Python dict 字符串）
        payload = env_str[len(prefix):]
        payload = payload.strip()
        # 尝试解析为字典
        try:
            import ast
            options = ast.literal_eval(payload)
        except Exception:
            try:
                import json
                options = json.loads(payload)
            except Exception:
                options = {}

        env = RotatedArrayMinEnvGEM(
            complexity=options.get("complexity", 5),
            enable_param_randomization=options.get("enable_param_randomization", False),
            max_turns=options.get("max_turns", None),
        )
        # 如果提供了数组，直接加载该数组
        if "array" in options and isinstance(options["array"], list):
            env.load_array(options["array"])
        return env

    def load_array(self, array: list):
        """外部注入数组，用于评测或固定实例。"""
        self.array = list(array)
        self.array_length = len(self.array)
        self.problem = {"array": self.array, "pivot": None, "sorted_base": None}
        self.turn_count = 0

    # 参考答案（原环境逻辑）
    def get_ref_answer(self):
        """Use the information in the environment to get the reference answer."""
        if not self.array:
            return None

        left, right = 0, len(self.array) - 1
        while left < right:
            mid = (left + right) // 2
            if self.array[mid] > self.array[right]:
                left = mid + 1
            else:
                right = mid
        return self.array[left]

    # 原动作的封装（返回 string，与 GEM step 中使用一致）
    def Observe(self):
        return "A rotated sorted array has been loaded. Please use the relevant operations to find the smallest element in it."

    def GetArrayLength(self):
        return str(len(self.array))

    def GetElement(self, index: int):
        if 0 <= index < len(self.array):
            return str(self.array[index])
        else:
            return "Error: Index out of bounds"

    def CompareElements(self, index1: int, index2: int):
        if 0 <= index1 < len(self.array) and 0 <= index2 < len(self.array):
            return "1" if self.array[index1] > self.array[index2] else "0"
        else:
            return "Error: Index out of bounds"

    def Done(self, answer):
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={'1' if correct else '0'}"

    # 自动求解流程（调用 GEM 风格的 step）
    def solve(self) -> str:
        # 获取长度
        obs, _, term, _, _ = self.step("\\boxed{len}")
        if term and obs.startswith("Error"):
            return obs

        try:
            n = int(obs)
        except Exception:
            # 如果长度输出非纯数字，则尝试提取数字
            import re as _re
            m = _re.search(r"(\d+)", obs)
            n = int(m.group(1)) if m else 0

        if n == 0:
            final_obs, reward, terminated, truncated, _ = self.step("\\boxed{answer none}")
            return final_obs

        left = 0
        right = n - 1

        while left < right and self.turn_count < self.max_turns:
            mid = (left + right) // 2
            cmp_cmd = f"\\boxed{{cmp {mid} {right}}}"
            cmp_obs, _, cmp_term, _, _ = self.step(cmp_cmd)
            if cmp_term and ("Error" in cmp_obs or "Invalid" in cmp_obs):
                return cmp_obs
            if cmp_obs.strip() == "1":
                left = mid + 1
            else:
                right = mid

        get_cmd = f"\\boxed{{get {left}}}"
        min_obs, _, get_term, _, _ = self.step(get_cmd)
        if get_term and ("Error" in min_obs or "Invalid" in min_obs):
            return min_obs

        try:
            min_element = int(min_obs)
        except Exception:
            # 无法解析时返回错误
            return "Error: failed to obtain minimum element."

        final_cmd = f"\\boxed{{answer {min_element}}}"
        final_obs, reward, terminated, truncated, _ = self.step(final_cmd)
        return final_obs