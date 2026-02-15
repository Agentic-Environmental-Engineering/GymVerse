from typing import Any, Dict, Optional, Tuple
import random
import re
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class DailyTemperaturesEnvGEM(Env):
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
            # 温度数组长度
            "array_length": (5, 50),
            # 温度取值下界（保持常数）
            "value_min": (30, 30),
            # 温度取值上界
            "value_max": (80, 110),
        }

        # 参数随机方差（仅在 enable_param_randomization=True 时使用）
        self.param_variance = {
            "array_length": 2,
            "value_min": 0,
            "value_max": 5,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_min: int = 0
        self.value_max: int = 0

        # 运行时状态
        self.turn_count: int = 0

        # 问题数据
        self.problem: Dict[str, Any] = {}
        self.temperatures = []
        self.answer = []
        self.stack = []

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
            "Daily Temperatures (GEM): Compute for each day how many days until a warmer temperature.\n"
            "You can interact with the environment using actions inside a box (DungeonScout style):\n"
            "Available actions:\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Get temperatures length: \\boxed{len}\n"
            "- Get temperature at index i: \\boxed{temp i}\n"
            "- Initialize answer array of length n: \\boxed{ans init n}\n"
            "- Initialize stack: \\boxed{stack init}\n"
            "- Check if stack empty: \\boxed{stack empty}\n"
            "- Get stack top index: \\boxed{stack top}\n"
            "- Pop stack top index: \\boxed{stack pop}\n"
            "- Push index i into stack: \\boxed{stack push i}\n"
            "- Update answer[i] = v: \\boxed{ans set i v}\n"
            "- Get current answer array: \\boxed{ans get}\n"
            "- Submit final answer (e.g., [1,1,4,2,1,1,0,0]): \\boxed{submit [..]}\n"
            "Rewards:\n"
            "- Correct submit: +1.0 (terminated)\n"
            "- Wrong submit: -1.0 (terminated)\n"
            "- Format error: LanguageGameReward.format_error_reward (terminated)\n"
            "- Invalid action: LanguageGameReward.invalid_action_reward\n"
            "- Timeout: 0.0 (terminated, truncated)\n"
        )

    def get_task_suffix(self) -> str:
        tlen = len(self.temperatures)
        return (
            f"Temperatures length: {tlen}\n"
            f"Stack size: {len(self.stack)}; Answer length: {len(self.answer)}\n"
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

        # 初始化状态
        self.temperatures = list(self.problem["temperatures"])
        self.answer = []
        self.stack = []
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        temps = [random.randint(self.value_min, self.value_max) for _ in range(self.array_length)]
        return {"temperatures": temps}

    # --------------------------
    # 原环境的辅助方法（转换保留）
    # --------------------------
    def get_ref_answer(self):
        n = len(self.temperatures)
        answer = [0] * n
        stack = []
        for i in range(n):
            while stack and self.temperatures[i] > self.temperatures[stack[-1]]:
                previous_day = stack.pop()
                answer[previous_day] = i - previous_day
            stack.append(i)
        return answer

    # 内部动作实现（基于原环境）
    def _observe(self) -> str:
        return "Daily temperature analysis environment; use actions to calculate the waiting days array"

    def _get_temperature_length(self) -> str:
        return str(len(self.temperatures))

    def _get_temperature_at_index(self, index: int) -> str:
        if 0 <= index < len(self.temperatures):
            return str(self.temperatures[index])
        return "Error: Index out of range"

    def _initialize_answer_array(self, length: int) -> str:
        self.answer = [0] * length
        return f"Answer array initialized as a zero-filled array of length {length}"

    def _initialize_stack(self) -> str:
        self.stack = []
        return "Stack initialized as empty"

    def _is_stack_empty(self) -> str:
        return str(len(self.stack) == 0)

    def _get_stack_top(self) -> str:
        if self.stack:
            return str(self.stack[-1])
        return "Error: Stack is empty"

    def _pop_stack(self) -> str:
        if self.stack:
            return str(self.stack.pop())
        return "Error: Stack is empty"

    def _push_stack(self, element: int) -> str:
        self.stack.append(element)
        return f"Element {element} pushed onto stack; top of stack is {self.stack[-1]}"

    def _update_answer_at_index(self, index: int, value: int) -> str:
        if 0 <= index < len(self.answer):
            self.answer[index] = value
            return f"Value at index {index} in answer array updated to {value}"
        return "Error: Index out of range"

    def _get_answer_array(self) -> str:
        return str(self.answer)

    def _submit(self, answer_list) -> Tuple[str, float, bool, bool]:
        ref_answer = self.get_ref_answer()
        correct = list(answer_list) == ref_answer
        obs = f"Your answer: {list(answer_list)}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        reward = 1.0 if correct else -1.0
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated

    # --------------------------
    # 交互逻辑
    # --------------------------
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        # 格式错误（未找到 \boxed{} 或内容为空）
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
        content_raw = content
        content = content.strip()
        content_lc = content.lower()

        reward = 0.0
        terminated = False
        truncated = False

        obs = ""
        invalid_action = False

        try:
            # 解析指令
            if content_lc == "observe":
                obs = self._observe()

            elif content_lc == "len":
                obs = self._get_temperature_length()

            elif content_lc.startswith("temp"):
                m = re.match(r"temp\s+(\d+)$", content_lc)
                if not m:
                    invalid_action = True
                else:
                    idx = int(m.group(1))
                    obs = self._get_temperature_at_index(idx)

            elif content_lc.startswith("ans init"):
                m = re.match(r"ans\s+init\s+(\d+)$", content_lc)
                if not m:
                    invalid_action = True
                else:
                    n = int(m.group(1))
                    obs = self._initialize_answer_array(n)

            elif content_lc == "stack init":
                obs = self._initialize_stack()

            elif content_lc in ("stack empty", "stack empty?"):
                obs = self._is_stack_empty()

            elif content_lc == "stack top":
                obs = self._get_stack_top()

            elif content_lc == "stack pop":
                obs = self._pop_stack()

            elif content_lc.startswith("stack push"):
                m = re.match(r"stack\s+push\s+(\d+)$", content_lc)
                if not m:
                    invalid_action = True
                else:
                    e = int(m.group(1))
                    obs = self._push_stack(e)

            elif content_lc.startswith("ans set"):
                m = re.match(r"ans\s+set\s+(\d+)\s+(\d+)$", content_lc)
                if not m:
                    invalid_action = True
                else:
                    i = int(m.group(1))
                    v = int(m.group(2))
                    obs = self._update_answer_at_index(i, v)

            elif content_lc == "ans get":
                obs = self._get_answer_array()

            elif content_lc.startswith("submit"):
                # 提交答案，支持多种格式：submit [1,2,3] 或 submit 1,2,3 或 submit 1 2 3
                payload = content.strip()[len("submit"):].strip()
                ans_list = self._parse_answer_list(payload)
                if ans_list is None:
                    invalid_action = True
                    obs = "Invalid submit format. Use: \\boxed{submit [1,2,3,...]}"
                else:
                    obs, reward, terminated, truncated = self._submit(ans_list)

            else:
                invalid_action = True

        except Exception as e:
            # 任意运行时异常都视为无效动作
            obs = f"Error: {str(e)}"
            invalid_action = True

        if invalid_action and not terminated:
            # 无效动作：不给终止，但给予无效动作惩罚
            reward = LanguageGameReward.invalid_action_reward
            if not obs:
                obs = f"Invalid action: {content_raw}"

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
        return {"content": content}

    def _parse_answer_list(self, payload: str) -> Optional[list]:
        # 优先解析标准 python 列表
        if payload.startswith("["):
            try:
                obj = ast.literal_eval(payload)
                if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
                    return obj
                return None
            except Exception:
                return None
        # 兼容 "1,2,3" 或 "1 2 3"
        if payload:
            # 去除两端括号（如果错误地只写了括号）
            p = payload.strip()
            p = p.strip("()")
            parts = re.split(r"[,\s]+", p)
            ints = []
            for part in parts:
                if part == "":
                    continue
                if not re.match(r"^-?\d+$", part):
                    return None
                ints.append(int(part))
            if ints:
                return ints
        return None

    def sample_random_action(self) -> str:
        # 随机选择一个合法动作
        actions = [
            "\\boxed{observe}",
            "\\boxed{len}",
            "\\boxed{stack init}",
            "\\boxed{stack empty}",
            "\\boxed{stack top}",
            "\\boxed{stack pop}",
            "\\boxed{ans get}",
        ]
        # 随机参数化的动作
        if self.temperatures:
            i = random.randint(0, max(0, len(self.temperatures) - 1))
            actions.append(f"\\boxed{{temp {i}}}")
            actions.append(f"\\boxed{{stack push {i}}}")
        if self.answer:
            i = random.randint(0, max(0, len(self.answer) - 1))
            v = random.randint(0, len(self.temperatures))
            actions.append(f"\\boxed{{ans set {i} {v}}}")
        if not self.answer:
            actions.append(f"\\boxed{{ans init {len(self.temperatures)}}}")
        # 随机提交（大概率错误，仅作演示）
        if self.temperatures:
            dummy = [0] * len(self.temperatures)
            actions.append(f"\\boxed{{submit {dummy}}}")
        return random.choice(actions)

    # 可选：自动求解器（使用当前动作语法调用）
    def auto_solve(self) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        # 初始化
        obs, info = "", {}
        # 初始化答案数组与栈
        self.step("\\boxed{ans init " + str(len(self.temperatures)) + "}")
        self.step("\\boxed{stack init}")
        # 单调栈解法
        for i in range(len(self.temperatures)):
            temp_i_obs, *_ = self.step(f"\\boxed{{temp {i}}}")
            # 尝试弹栈，直到栈空或当前温度不更高
            while True:
                is_empty_obs, *_ = self.step("\\boxed{stack empty}")
                if is_empty_obs == "True":
                    break
                top_idx_str, *_ = self.step("\\boxed{stack top}")
                if not re.match(r"^-?\d+$", top_idx_str.strip()):
                    break
                top_idx = int(top_idx_str.strip())
                top_temp_str, *_ = self.step(f"\\boxed{{temp {top_idx}}}")
                # 若数据错误，停止
                if not re.match(r"^-?\d+$", top_temp_str.strip()):
                    break
                if int(temp_i_obs) > int(top_temp_str):
                    days = i - top_idx
                    self.step(f"\\boxed{{ans set {top_idx} {days}}}")
                    self.step("\\boxed{stack pop}")
                else:
                    break
            self.step(f"\\boxed{{stack push {i}}}")
        # 获取答案并提交
        ans_str, *_ = self.step("\\boxed{ans get}")
        try:
            ans_list = ast.literal_eval(ans_str)
        except Exception:
            ans_list = []
        return self.step(f"\\boxed{{submit {ans_list}}}")