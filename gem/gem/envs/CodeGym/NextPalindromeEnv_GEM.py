from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class NextPalindromeEnvGEM(Env):
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

        # 难度参数范围设计：
        # - value_range_max: 初始数字 N 的最大值
        # - gap_upper: 控制 N 到下一个回文的距离上限（越大越难）
        self.complexity_params = {
            "value_range_max": (100, 100000),  # 初始 N 的最大值（复杂度越高越大）
            "gap_upper": (5, 1000),            # 允许的回文间隙上限（复杂度越高越大）
        }

        # 参数随机化方差（训练增强多样性）
        self.param_variance = {
            "value_range_max": 500,
            "gap_upper": 10,
        }

        # 占位属性
        self.value_range_max: int = 0
        self.gap_upper: int = 0

        # 状态
        self.turn_count: int = 0
        self.N: int = 0
        self._last_reward: float = 0.0

        # 问题实例
        self.problem: Dict[str, Any] = {}

        self.reset()

    def _apply_complexity_params(self):
        """根据 complexity 等级计算参数值"""
        normalized = min(1.0, (self.complexity - 1) / 9.0)  # 归一化到 [0, 1]
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
            "Next Palindrome: Given an initial number N, find the smallest palindrome strictly greater than N.\n"
            "Available actions:\n"
            "- Observe initial number: \\boxed{observe}\n"
            "- Check palindrome: \\boxed{check X}\n"
            "- Increment a number: \\boxed{inc X}\n"
            "- Submit final answer: \\boxed{answer X}\n"
            "Notes:\n"
            "- X must be an integer.\n"
            "- The 'answer' must be the next palindrome greater than N."
        )

    def get_task_suffix(self) -> str:
        return (
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            f"N range <= {self.value_range_max}, gap <= {self.gap_upper}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.N = self.problem["N"]

        self.turn_count = 0
        self._last_reward = 0.0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 目标：生成 N，使得 N 到下一个回文的距离 <= gap_upper
        # 若在尝试次数内无法满足，则退化为任意 N
        attempts = 2000
        min_N = 1
        max_N = max(min_N + 1, self.value_range_max - 1)

        def is_palindrome(num: int) -> bool:
            s = str(num)
            return s == s[::-1]

        def next_palindrome(num: int) -> int:
            candidate = num + 1
            while not is_palindrome(candidate):
                candidate += 1
            return candidate

        for _ in range(attempts):
            n = random.randint(min_N, max_N)
            gap = next_palindrome(n) - n
            if 1 <= gap <= self.gap_upper:
                return {"N": n, "gap": gap}

        # 退化：选择一个随机 N
        n = random.randint(min_N, max_N)
        gap = next_palindrome(n) - n
        return {"N": n, "gap": gap}

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
        content_lower = content.strip().lower()

        # 解析指令
        # 支持的形式：
        # - "observe"
        # - "check X"
        # - "inc X"
        # - "answer X"
        terminated = False
        truncated = False
        reward = 0.0
        obs_msg = ""

        # observe
        if re.fullmatch(r"observe", content_lower):
            obs_msg = self.Observe()
            terminated = False
            reward = 0.0

        # check X
        elif re.fullmatch(r"check\s+[-+]?\d+", content_lower):
            try:
                number = int(content.split()[1])
            except Exception:
                obs_msg = "Format error: 'check' expects an integer."
                return (
                    obs_msg,
                    LanguageGameReward.format_error_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            obs_msg = self.CheckPalindrome(number)
            terminated = False
            reward = 0.0

        # inc X
        elif re.fullmatch(r"inc\s+[-+]?\d+", content_lower):
            try:
                number = int(content.split()[1])
            except Exception:
                obs_msg = "Format error: 'inc' expects an integer."
                return (
                    obs_msg,
                    LanguageGameReward.format_error_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            obs_msg = self.IncrementNumber(number)
            terminated = False
            reward = 0.0

        # answer X
        elif re.fullmatch(r"answer\s+[-+]?\d+", content_lower):
            try:
                answer = int(content.split()[1])
            except Exception:
                obs_msg = "Format error: 'answer' expects an integer."
                return (
                    obs_msg,
                    LanguageGameReward.format_error_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            msg, correct = self.Done(answer, include_reward=False)
            reward = 1.0 if correct else -1.0
            obs_msg = f"{msg}, reward={reward}"
            terminated = True

        else:
            obs_msg = f"Invalid action: {content}"
            return (
                obs_msg,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs_msg = f"{obs_msg}\nReached max turns ({self.max_turns})."
            return obs_msg, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs_msg, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        return {"content": content}

    def sample_random_action(self) -> str:
        # 随机示例动作
        choices = [
            "\\boxed{observe}",
            f"\\boxed{{check {self.N}}}",
            f"\\boxed{{inc {self.N}}}",
            f"\\boxed{{answer {self.get_ref_answer()}}}",
        ]
        return random.choice(choices)

    # 原环境辅助方法保留并适配
    def CheckPalindrome(self, number: int) -> str:
        """
        Check if a number is a palindrome.
        Returns "True" or "False".
        """
        return str(str(number) == str(number)[::-1])

    def IncrementNumber(self, number: int) -> str:
        """
        Add 1 to the number and return the result as string.
        """
        return str(number + 1)

    def Observe(self) -> str:
        """
        Obtain the initial number N in the current environment.
        """
        return f"Initial number N is: {self.N}"

    def Done(self, answer: int, include_reward: bool = True) -> Tuple[str, bool]:
        """
        Verify whether the final answer is correct and return the result information.
        If include_reward=True, append reward info; otherwise only correctness message.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        if include_reward:
            reward = 1 if correct else 0
            msg = msg + f", reward={reward}"
        return msg, correct

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer:
        the smallest palindrome strictly greater than N.
        """
        def is_palindrome(num: int) -> bool:
            s = str(num)
            return s == s[::-1]

        candidate = self.N + 1
        while not is_palindrome(candidate):
            candidate += 1
        return candidate

    def solve(self) -> str:
        """
        Automatically call all actions to complete the process, and submit the answer for verification.
        Returns the final verification result string.
        """
        # Observe N
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        try:
            n = int(obs.split(": ")[1])
        except Exception:
            # Fallback to stored N
            n = self.N

        current = n
        while True:
            inc_obs, _, terminated, truncated, _ = self.step(f"\\boxed{{inc {current}}}")
            if terminated or truncated:
                return inc_obs
            try:
                current = int(inc_obs)
            except Exception:
                # If format error occurs, stop
                return inc_obs

            check_obs, _, terminated, truncated, _ = self.step(f"\\boxed{{check {current}}}")
            if terminated or truncated:
                return check_obs

            if check_obs == "True":
                final_obs, reward, terminated, truncated, _ = self.step(f"\\boxed{{answer {current}}}")
                return final_obs