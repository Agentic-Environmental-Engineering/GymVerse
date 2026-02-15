from typing import Any, Dict, Optional, Tuple
import random
import re
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class PrimeFilteringEnvGEM(Env):
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
        self.complexity_params = {
            # 数组长度（隐藏的数字列表长度）
            "array_length": (5, 50),
            # 数值范围（数字取样范围上界）
            "value_range": (10, 10000),
        }

        # 参数方差（启用参数随机化时生效）
        self.param_variance = {
            "array_length": 2,  # ±2
            "value_range": 200,  # ±200
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 任务数据
        self.numbers = []
        self.collected_primes = []
        self._length_revealed = False

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
            "Prime Filtering: Identify all prime numbers in a hidden list.\n"
            "You can interact via boxed actions. Available actions:\n"
            "- Get list length: \\boxed{len}\n"
            "- Get element by index (0-based): \\boxed{get I}\n"
            "- Check if a number is prime: \\boxed{check N}\n"
            "- Collect a prime number: \\boxed{collect N}\n"
            "- Observe collected primes: \\boxed{observe}\n"
            "- Submit final answer (Python-style list): \\boxed{answer [p1, p2, ...]}\n"
            "Goal: Submit the exact list of primes from the hidden numbers in ascending index order as they appear."
        )

    def get_task_suffix(self) -> str:
        length_info = str(len(self.numbers)) if self._length_revealed else "?"
        return (
            f"Length: {length_info} | "
            f"Collected: {len(self.collected_primes)} | "
            f"Turn: {self.turn_count}/{self.max_turns}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.turn_count = 0
        self.collected_primes = []
        self._length_revealed = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        self.numbers = [random.randint(1, self.value_range) for _ in range(self.array_length)]
        return {"numbers": self.numbers, "size": self.array_length, "value_range": self.value_range}

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
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            tokens = content.strip().split(maxsplit=1)
            cmd = tokens[0].lower()

            if cmd == "len":
                obs = self.GetListLength()
                self._length_revealed = True

            elif cmd == "get":
                if len(tokens) < 2:
                    obs = "Error: missing index for 'get' action."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    index = int(tokens[1])
                except Exception:
                    obs = "Error: index must be an integer."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.GetListElement(index)

            elif cmd == "check":
                if len(tokens) < 2:
                    obs = "Error: missing number for 'check' action."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    number = int(tokens[1])
                except Exception:
                    obs = "Error: number must be an integer."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.CheckPrime(number)

            elif cmd == "collect":
                if len(tokens) < 2:
                    obs = "Error: missing number for 'collect' action."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    number = int(tokens[1])
                except Exception:
                    obs = "Error: number must be an integer."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.CollectPrime(number)

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "answer":
                if len(tokens) < 2:
                    obs = "Error: missing list for 'answer'. Example: answer [2, 3, 5]"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    answer_obj = ast.literal_eval(tokens[1])
                    if not isinstance(answer_obj, list) or not all(isinstance(x, int) for x in answer_obj):
                        raise ValueError("Answer must be a list of integers.")
                except Exception as e:
                    obs = f"Format error in answer payload: {e}"
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

                done_msg, success = self._do_done(answer_obj)
                obs = done_msg
                reward = 1.0 if success else -1.0
                terminated = True

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
        # 随机选择一个动作示例
        choices = [
            "\\boxed{len}",
            "\\boxed{observe}",
        ]
        # 若已知长度，随机尝试 get
        if self.numbers:
            idx = random.randint(0, len(self.numbers) - 1)
            choices.append(f"\\boxed{{get {idx}}}")
            # 也随机选一个值检查
            val = random.randint(1, self.value_range)
            choices.append(f"\\boxed{{check {val}}}")
        return random.choice(choices)

    # ========== 保留并转换原环境辅助方法 ==========

    def get_ref_answer(self):
        """
        使用环境中的信息获取参考答案。
        """
        def is_prime(n):
            if n <= 1:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            max_divisor = int(n ** 0.5) + 1
            for d in range(3, max_divisor, 2):
                if n % d == 0:
                    return False
            return True

        return [x for x in self.numbers if is_prime(x)]

    def CheckPrime(self, number: int):
        """
        判断一个数是否为素数。
        返回字符串 "True"/"False"
        """
        if number <= 1:
            return "False"
        if number == 2:
            return "True"
        if number % 2 == 0:
            return "False"
        max_divisor = int(number ** 0.5) + 1
        for d in range(3, max_divisor, 2):
            if number % d == 0:
                return "False"
        return "True"

    def GetListElement(self, index: int):
        """
        获取列表中指定下标的元素。
        返回字符串化的元素或错误信息。
        """
        if 0 <= index < len(self.numbers):
            return str(self.numbers[index])
        return "Error: Index out of range"

    def GetListLength(self):
        """
        获取列表长度（字符串）。
        """
        return str(len(self.numbers))

    def CollectPrime(self, number: int):
        """
        收集一个素数到当前集合（不强制校验素性，最终以提交答案为准）。
        """
        self.collected_primes.append(number)
        return f"Collected prime: {number}, total collected primes so far: {len(self.collected_primes)}"

    def Observe(self):
        """
        观察当前收集的素数列表。
        """
        return f"Collected primes: {self.collected_primes}"

    def Done(self, answer):
        """
        验证最终答案是否正确并返回信息（字符串，不改变 GEM 的 step 流程）。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def _do_done(self, answer):
        """
        GEM 风格的 Done 执行（返回消息字符串和布尔是否成功）。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = self.Done(answer)
        return msg, correct

    # 可选求解器（演示用）
    def solve(self, seed: Optional[int] = None) -> str:
        """
        自动完成流程（使用内部方法，不调用 step）。
        """
        if seed is not None:
            random.seed(seed)
            self.reset(seed=seed)

        length = int(self.GetListLength())
        primes = []
        for index in range(length):
            val_str = self.GetListElement(index)
            if val_str.startswith("Error"):
                continue
            number = int(val_str)
            if self.CheckPrime(number) == "True":
                self.CollectPrime(number)
                primes.append(number)
        _ = self.Observe()
        return self.Done(primes)