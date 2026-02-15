from typing import Any, Dict, Optional, Tuple
import random
import re
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class DigitalRootEnvGEM(Env):
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

        # 难度参数范围设计：控制初始数字的位数和允许的提交次数预算（用于生成更大数字）
        self.complexity_params = {
            "initial_num_digits": (2, 10),   # 初始数字的位数范围（复杂度越高位数越多）
            "turn_budget_hint": (10, 60),    # 提示性的步数预算（不覆盖 max_turns，仅用于描述）
        }

        # 参数方差（训练时用于微调随机性）
        self.param_variance = {
            "initial_num_digits": 1,
            "turn_budget_hint": 2,
        }

        # 占位属性
        self.initial_num_digits: int = 0
        self.turn_budget_hint: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 原环境状态兼容字段
        self._reward: float = 0.0
        self._done: bool = False
        self.step_count: int = 0  # 保留原命名兼容

        # 问题数据
        self.problem: Dict[str, Any] = {}
        self.num: int = 0  # 与原环境一致，用于 Observe 与 get_ref_answer

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
            "Digital Root: Reduce a number to its digital root (repeated digit sums until single-digit).\n"
            "Available actions (use LaTeX boxed syntax):\n"
            "- Observe initial number: \\boxed{observe}\n"
            "- Get digits of a number: \\boxed{get_digits N}\n"
            "- Sum a list of digits: \\boxed{sum_digits [d1, d2, ...]} or \\boxed{sum_digits d1,d2,...}\n"
            "- Check single-digit: \\boxed{is_single_digit N}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Digits: {self.initial_num_digits} | "
            f"Turn: {self.turn_count}/{self.max_turns} (hint budget: {self.turn_budget_hint})\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.turn_count = 0
        self.step_count = 0
        self._reward = 0.0
        self._done = False

        # 与原环境保持一致
        self.num = self.problem["initial_number"]

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 生成指定位数的随机正整数（首位非零）
        first_digit = random.randint(1, 9)
        other_digits = [random.randint(0, 9) for _ in range(max(0, self.initial_num_digits - 1))]
        number_str = str(first_digit) + "".join(str(d) for d in other_digits)
        initial_number = int(number_str)
        return {"initial_number": initial_number}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        self.step_count += 1

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
        lower = content.lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if lower == "observe":
                obs = self.Observe()
            elif lower.startswith("get_digits"):
                m = re.match(r"get_digits\s+(-?\d+)\s*$", lower)
                if not m:
                    obs = "Invalid get_digits format. Use: \\boxed{get_digits N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    number = int(m.group(1))
                    obs = self.GetDigits(number)
            elif lower.startswith("sum_digits"):
                # Accept either Python list or comma/space separated numbers
                payload = content[len("sum_digits"):].strip()
                if not payload:
                    obs = "Invalid sum_digits format. Provide digits as [1,2,3] or 1,2,3"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    digits: Optional[list] = None
                    try:
                        # Try literal list parse
                        if "[" in payload or "(" in payload:
                            digits = list(ast.literal_eval(payload))
                        else:
                            # Comma-separated or space-separated
                            if "," in payload:
                                parts = [p.strip() for p in payload.split(",") if p.strip()]
                            else:
                                parts = [p.strip() for p in payload.split() if p.strip()]
                            digits = [int(p) for p in parts]
                    except Exception:
                        digits = None

                    if digits is None or not isinstance(digits, list):
                        obs = "Invalid digits for sum_digits."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.SumDigits(digits)
            elif lower.startswith("is_single_digit"):
                m = re.match(r"is_single_digit\s+(-?\d+)\s*$", lower)
                if not m:
                    obs = "Invalid is_single_digit format. Use: \\boxed{is_single_digit N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    number = int(m.group(1))
                    obs = self.IsSingleDigit(number)
            elif lower.startswith("answer"):
                m = re.match(r"answer\s+(-?\d+)\s*$", lower)
                if not m:
                    obs = "Invalid answer format. Use: \\boxed{answer N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    answer = int(m.group(1))
                    # 调用原环境 Done 方法进行验证与信息生成
                    msg = self.Done(answer)
                    obs = msg
                    # 根据正确性设定奖励（与 Done 中 _reward 一致）
                    ref_answer = self.get_ref_answer()
                    correct = answer == ref_answer
                    reward = 1.0 if correct else -1.0
                    terminated = True
            else:
                obs = "Invalid action."
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
        return {"content": content}

    def sample_random_action(self) -> str:
        return "\\boxed{observe}"

    # -----------------------------
    # 保留原环境的辅助方法并转换
    # -----------------------------

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        current = self.num
        while current >= 10:
            current = sum(int(digit) for digit in str(current))
        return current

    def GetDigits(self, number: int):
        """
        Convert an integer into a list composed of its individual digits.
        Returns: string representation like "[3, 8]"
        """
        digits = [int(digit) for digit in str(number)]
        return str(digits)

    def SumDigits(self, digits: list):
        """
        Calculate the sum of all elements in the list of digits.
        Returns: the sum as string
        """
        return str(sum(int(d) for d in digits))

    def IsSingleDigit(self, number: int):
        """
        Determine if a number is a single-digit number (less than 10).
        Returns: "True" or "False"
        """
        return str(number < 10)

    def Observe(self):
        """
        Obtain the initial number in the current environment.
        Returns: "The number to be processed is: X"
        """
        return f"The number to be processed is: {self.num}"

    def Done(self, answer):
        """
        Verify whether the final answer is correct and return result information.
        Returns: "Your answer: a, Reference answer: r, Result: Correct/Incorrect, reward=1/0"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self):
        """
        Automatically compute the digital root and submit the answer for verification.
        Returns: final verification message.
        """
        current_number = self.num
        while current_number >= 10:
            digits = [int(d) for d in str(current_number)]
            current_number = sum(digits)
        return self.Done(current_number)