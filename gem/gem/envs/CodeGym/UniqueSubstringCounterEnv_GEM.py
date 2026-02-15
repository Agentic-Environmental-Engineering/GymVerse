from typing import Any, Dict, Optional, Tuple
import random
import re
from collections import defaultdict
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class UniqueSubstringCounterEnvGEM(Env):
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

        # 定义难度参数范围（根据原环境分析）
        # 影响任务复杂度的要素：字符串长度、子串长度、字母表大小
        self.complexity_params = {
            "string_length": (5, 120),     # 字符串长度
            "substr_length": (1, 12),      # 子串长度 L
            "alphabet_size": (2, 26),      # 随机生成时使用的字母表大小
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "string_length": 3,
            "substr_length": 1,
            "alphabet_size": 1,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.string_length: int = 0
        self.substr_length: int = 0
        self.alphabet_size: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 任务变量（生成后填充）
        self.s: str = ""
        self.L: int = 1
        self.substring_counts: Dict[str, int] = defaultdict(int)

        # 兼容属性（非必须）
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

        # 约束：L 至少为 1，且不超过字符串长度
        self.substr_length = max(1, min(self.substr_length, self.string_length))

    def _get_instructions(self) -> str:
        return (
            "Unique Substring Counter: Count substrings of length L that appear exactly once in a hidden string.\n"
            "The string is hidden; you may inspect its length and L, extract substrings by index, and maintain your own counts.\n"
            "Available actions (use the last \\boxed{...} in your message):\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Extract substring: \\boxed{extract INDEX}  (INDEX is 0-based)\n"
            "- Increment count: \\boxed{count SUBSTRING}\n"
            "- Compute unique count from your maintained counts: \\boxed{unique}\n"
            "- Submit final answer: \\boxed{answer N}  (N is the number of substrings of length L appearing exactly once in the hidden string)\n"
        )

    def get_task_suffix(self) -> str:
        return f"Turn: {self.turn_count}/{self.max_turns}\nEnter an action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.s = self.problem["s"]
        self.L = self.problem["L"]
        self.substring_counts = defaultdict(int)
        self.turn_count = 0
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        alphabet = [chr(ord("a") + i) for i in range(self.alphabet_size)]
        s = "".join(random.choice(alphabet) for _ in range(self.string_length))
        L = self.substr_length
        return {"s": s, "L": L}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}. Expected \\boxed{{...}}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        content = parsed["content"]
        tokens = content.strip().split()
        if len(tokens) == 0:
            obs = f"Invalid action at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        verb = tokens[0].lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if verb == "observe":
                # Show environment info (without revealing the hidden string)
                obs = self.Observe()
                terminated = False

            elif verb == "extract":
                if len(tokens) < 2:
                    obs = "Usage: \\boxed{extract INDEX}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    index = int(tokens[1])
                except ValueError:
                    obs = "INDEX must be an integer. Usage: \\boxed{extract INDEX}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                substring = self.ExtractSubstring(index)
                obs = substring
                terminated = False

            elif verb == "count":
                if len(tokens) < 2:
                    obs = "Usage: \\boxed{count SUBSTRING}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                # Everything after 'count ' is the substring
                substring = content[len(tokens[0]):].strip()
                updated = self.CountSubstring(substring)
                obs = f"Count('{substring}') -> {updated}"
                terminated = False

            elif verb == "unique":
                # Compute unique count from maintained counts (NOT the ground truth)
                unique_count = self.GetUniqueCount()
                obs = unique_count
                terminated = False

            elif verb == "answer":
                if len(tokens) < 2:
                    obs = "Usage: \\boxed{answer N}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    answer_val = int(tokens[1])
                except ValueError:
                    obs = "N must be an integer. Usage: \\boxed{answer N}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                done_msg, success = self.Done(answer_val)
                obs = done_msg
                terminated = True
                reward = 1.0 if success else -1.0

            else:
                obs = f"Unknown action '{verb}'."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )

        except Exception as e:
            obs = f"Error while processing action: {str(e)}"
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
        # Randomly pick a valid action to demonstrate the format
        choices = [
            "\\boxed{observe}",
            "\\boxed{extract 0}",
            "\\boxed{count abc}",
            "\\boxed{unique}",
            f"\\boxed{{answer {random.randint(0, self.string_length)}}}",
        ]
        return random.choice(choices)

    # ---------------------------
    # 保留并转换原环境的辅助方法
    # ---------------------------

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        substring_count: Dict[str, int] = {}
        if self.L <= 0 or self.L > len(self.s):
            return 0
        for i in range(len(self.s) - self.L + 1):
            substring = self.s[i:i + self.L]
            substring_count[substring] = substring_count.get(substring, 0) + 1
        return sum(1 for count in substring_count.values() if count == 1)

    def ExtractSubstring(self, index: int) -> str:
        """
        Extract a substring of length L starting from the specified index in the string.

        Args:
            index (int): The starting index of the substring.

        Returns:
            str: The extracted substring, or an empty string if the index is out of range.

        Example Output:
            "aba"
        """
        if 0 <= index <= len(self.s) - self.L:
            return self.s[index:index + self.L]
        return ""

    def CountSubstring(self, substring: str) -> str:
        """
        Increment the count of the specified substring.

        Args:
            substring (str): The substring to be counted.

        Returns:
            str: The updated count of the substring.

        Example Output:
            "2"
        """
        self.substring_counts[substring] += 1
        return str(self.substring_counts[substring])

    def GetUniqueCount(self) -> str:
        """
        Count the number of substrings with a count exactly equal to 1.

        Args:
            None

        Returns:
            str: The number of substrings with a count exactly equal to 1.

        Example Output:
            "3"
        """
        return str(sum(1 for count in self.substring_counts.values() if count == 1))

    def Observe(self) -> str:
        """
        Return the observation information of the current environment, including the string length and the value of L.

        Args:
            None

        Returns:
            str: Information describing the current state of the environment.

        Example Output:
            "String length: 7, L value: 3"
        """
        return f"String length: {len(self.s)}, L value: {self.L}"

    def Done(self, answer: int) -> Tuple[str, bool]:
        """
        Verify whether the final answer is correct and return result information.

        Args:
            answer (int): The answer submitted by the user.

        Returns:
            Tuple[str, bool]: (Result information, whether it is correct)

        Example Output:
            ("Your answer: 3, Reference answer: 3, Result: Correct", True)
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, correct