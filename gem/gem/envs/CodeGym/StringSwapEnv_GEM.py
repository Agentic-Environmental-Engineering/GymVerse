from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class StringSwapEnvGEM(Env):
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
        # 通过 complexity 控制字符串长度和字符集规模；max_turns_param 仅用于信息展示
        self.complexity_params = {
            "min_length": (2, 20),        # 字符串最小长度
            "max_length": (8, 200),       # 字符串最大长度
            "charset_size": (2, 26),      # 字符集大小（从 a-z 中取前 N 个）
            "max_turns_param": (5, 30),   # 建议步数预算（不覆盖 self.max_turns，仅展示）
        }

        # 参数方差（启用参数随机化时生效）
        self.param_variance = {
            "min_length": 1,
            "max_length": 5,
            "charset_size": 2,
            "max_turns_param": 2,
        }

        # 占位属性
        self.min_length: int = 0
        self.max_length: int = 0
        self.charset_size: int = 0
        self.max_turns_param: int = 0

        # 状态
        self.turn_count: int = 0

        # 任务相关缓存
        self.problem: Dict[str, Any] = {}
        self.input_string: str = ""

        # 交互中间结果
        self.last_length: Optional[int] = None
        self.last_mid: Optional[int] = None
        self.last_swapped: Optional[str] = None

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

        # 保护性修正：确保 min_length 不大于 max_length
        if self.min_length > self.max_length:
            self.min_length, self.max_length = self.max_length, self.min_length

        # 下界至少为 0
        self.min_length = max(0, self.min_length)

        # charset_size 下界至少 1
        self.charset_size = max(1, self.charset_size)

    def _get_instructions(self) -> str:
        return (
            "String Swap: Given a hidden string S, split it at midpoint and swap halves.\n"
            "Midpoint rule: if len(S) is even, mid = len(S)//2; if odd, mid = len(S)//2 + 1.\n"
            "Your goal is to submit the swapped string T = S[mid:] + S[:mid].\n"
            "Available actions (use the last \\boxed{...} in your message as the command):\n"
            "- Observe the string: \\boxed{observe}\n"
            "- Get the string length: \\boxed{length}\n"
            "- Compute midpoint from a length L: \\boxed{mid L} (e.g., mid 8) or \\boxed{mid} (auto)\n"
            "- Swap using midpoint M: \\boxed{swap M} (e.g., swap 4) or \\boxed{swap auto}\n"
            "- Submit your answer: \\boxed{answer YOUR_STRING}\n"
            "- Show this help: \\boxed{help}\n"
            "Rewards: correct answer=1.0, incorrect answer=-1.0, invalid/format error penalties apply.\n"
        )

    def get_task_suffix(self) -> str:
        # 展示当前回合、总步数限制、难度相关参数
        hints = []
        if self.last_length is not None:
            hints.append(f"len={self.last_length}")
        if self.last_mid is not None:
            hints.append(f"mid={self.last_mid}")
        hints_str = (" | " + ", ".join(hints)) if hints else ""
        return (
            f"Turn: {self.turn_count}/{self.max_turns} | "
            f"LengthRange=[{self.min_length},{self.max_length}] | "
            f"CharsetSize={self.charset_size} | "
            f"BudgetHint~{self.max_turns_param}"
            f"{hints_str}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响随机实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.input_string = self.problem["input_string"]

        # 清空交互状态
        self.turn_count = 0
        self.last_length = None
        self.last_mid = None
        self.last_swapped = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        length = random.randint(self.min_length, self.max_length)
        alphabet = "abcdefghijklmnopqrstuvwxyz"[: max(1, self.charset_size)]
        s = "".join(random.choice(alphabet) for _ in range(length))
        return {"input_string": s, "length": length}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None
        parts = content.split()
        cmd = parts[0].lower()
        args = parts[1:]
        return {"cmd": cmd, "args": args, "raw": content}

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

        cmd = parsed["cmd"]
        args = parsed["args"]

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "help":
                obs = self._get_instructions()

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "length":
                obs = self.GetStringLength()
                # 缓存长度（数值）
                try:
                    self.last_length = int(obs)
                except Exception:
                    self.last_length = None

            elif cmd == "mid":
                # 允许无参（自动根据真实长度），或提供一个长度参数
                if len(args) == 0 or (len(args) == 1 and args[0].lower() == "auto"):
                    L = len(self.input_string)
                else:
                    # 解析用户提供的长度
                    try:
                        L = int(args[0])
                        if L < 0:
                            raise ValueError
                    except Exception:
                        obs = "Invalid argument for 'mid'. Expect a non-negative integer length or 'auto'."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                obs = self.CalculateMidPoint(L)
                # 缓存 mid（数值）
                try:
                    self.last_mid = int(obs)
                except Exception:
                    self.last_mid = None

            elif cmd == "swap":
                # 需要 mid 数值，或 auto（使用 last_mid 或自动计算）
                if len(args) == 0:
                    obs = "Missing argument for 'swap'. Provide an integer midpoint or 'auto'."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                use_auto = args[0].lower() == "auto"
                if use_auto:
                    if self.last_mid is not None:
                        mid_val = self.last_mid
                    else:
                        # 自动计算基于实际长度
                        mid_val = int(self.CalculateMidPoint(len(self.input_string)))
                else:
                    try:
                        mid_val = int(args[0])
                    except Exception:
                        obs = "Invalid midpoint for 'swap'. Expect an integer or 'auto'."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )

                # 校验范围
                n = len(self.input_string)
                if mid_val < 0 or mid_val > n:
                    obs = f"Midpoint out of range: {mid_val}. Valid range: [0,{n}]."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

                obs = self.SplitAndSwapString(mid_val)
                self.last_swapped = obs

            elif cmd == "answer":
                if len(args) == 0:
                    obs = "Missing argument for 'answer'. Provide the final swapped string."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                # 原始字符串不包含空格，此处直接拼接
                candidate = " ".join(args)
                ref_answer = self.get_ref_answer()
                correct = (candidate == ref_answer)
                result_msg = self.Done(candidate)
                obs = result_msg
                reward = 1.0 if correct else -1.0
                terminated = True
                truncated = False

            else:
                obs = f"Invalid action: {parsed['raw']}"
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

        # 超时检查（统一在结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def sample_random_action(self) -> str:
        # 简单示例：优先观察
        return "\\boxed{observe}"

    # -----------------------
    # 原环境的辅助方法（保留并转换）
    # -----------------------

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        n = len(self.input_string)
        if n % 2 == 0:
            mid = n // 2
        else:
            mid = (n // 2) + 1
        return self.input_string[mid:] + self.input_string[:mid]

    def Observe(self) -> str:
        """
        Get the current string to be processed.

        Returns:
            str: The current string to be processed.
        """
        return self.input_string

    def GetStringLength(self) -> str:
        """
        Get the length of the current string.

        Returns:
            str: The length of the current string.
        """
        return str(len(self.input_string))

    def CalculateMidPoint(self, length: int) -> str:
        """
        Calculate the split point based on the string length.
        For even length, return half the length; for odd length, return (length//2)+1.

        Args:
            length (int): The length of the string.

        Returns:
            str: The calculated split point.
        """
        if length % 2 == 0:
            mid = length // 2
        else:
            mid = (length // 2) + 1
        return str(mid)

    def SplitAndSwapString(self, mid: int) -> str:
        """
        Split the string into two parts based on the split point and swap them.

        Args:
            mid (int): The position of the split point.

        Returns:
            str: The swapped string.
        """
        return self.input_string[mid:] + self.input_string[:mid]

    def Done(self, answer: str) -> str:
        """
        Verify if the final answer is correct and return the result information.

        Args:
            answer (str): User-submitted answer string.

        Returns:
            str: Result information, including correctness.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={'1' if correct else '0'}"

    def solve(self) -> str:
        """
        Automatically call actions to finish the task and submit the answer for verification.

        Returns:
            str: The result information of the final answer verification.
        """
        # Observe (optional)
        _ = self.step("\\boxed{observe}")
        # Get length
        obs, _, _, _, _ = self.step("\\boxed{length}")
        try:
            length = int(obs)
        except Exception:
            length = len(self.input_string)
        # Compute mid
        mid_obs, _, _, _, _ = self.step(f"\\boxed{{mid {length}}}")
        try:
            mid = int(mid_obs)
        except Exception:
            mid = int(self.CalculateMidPoint(len(self.input_string)))
        # Swap
        swapped_obs, _, _, _, _ = self.step(f"\\boxed{{swap {mid}}}")
        swapped = swapped_obs
        # Answer
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {swapped}}}")
        return final_obs