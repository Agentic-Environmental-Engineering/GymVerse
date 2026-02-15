from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class StringSplitCheckerEnvGEM(Env):
    """
    GEM-converted environment for the String Split Checker task.
    The agent interacts via boxed actions and tries to determine if a string can be split
    into two non-empty parts such that the number of vowels on the left equals the number on the right.

    Available actions (use inside \\boxed{...}):
    - observe
    - length
    - count
    - check <position>            (0-based index)
    - answer <true|false|1|0|yes|no>

    Reward schema:
    - Success (correct answer): 1.0
    - Failure (incorrect answer): -1.0
    - Format error: LanguageGameReward.format_error_reward
    - Invalid action: LanguageGameReward.invalid_action_reward
    - Timeout (max turns reached): 0.0 (terminated=True, truncated=True)

    Difficulty (complexity 1-10) controls:
    - string_length: (4, 80)
    - alphabet_size: (5, 26)
    - vowel_percent: (10, 60)  -> probability (%) that a generated character is a vowel
    """

    VOWELS = set("aeiou")

    def __init__(
        self,
        complexity: int = 5,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围
        self.complexity_params = {
            "string_length": (4, 80),
            "alphabet_size": (5, 26),
            "vowel_percent": (10, 60),  # 表示 10% 到 60% 的概率生成元音
        }

        # 参数方差（启用随机化时生效）
        self.param_variance = {
            "string_length": 3,
            "alphabet_size": 2,
            "vowel_percent": 5,
        }

        # 占位属性
        self.string_length: int = 0
        self.alphabet_size: int = 0
        self.vowel_percent: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.s: str = ""
        self._done: bool = False
        self._reward: float = 0.0

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
            "String Split Checker (GEM): Determine if the string can be split into two non-empty parts\n"
            "so that the number of vowels on the left equals the number on the right.\n"
            "Vowels: a, e, i, o, u (lowercase)\n"
            "Available actions (use in a single boxed command):\n"
            "- Observe status: \\boxed{observe}\n"
            "- Get string length: \\boxed{length}\n"
            "- Count total vowels: \\boxed{count}\n"
            "- Check if a position is vowel (0-based): \\boxed{check 3}\n"
            "- Submit final answer: \\boxed{answer true} or \\boxed{answer false}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Turn: {self.turn_count}/{self.max_turns}. Enter an action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.s = self.problem["s"]

        self.turn_count = 0
        self._done = False
        self._reward = 0.0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        length = self.string_length
        # 构建字母表，确保包含元音
        vowels = list(self.VOWELS)
        # 选择一些辅音来满足 alphabet_size
        consonants_all = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in self.VOWELS]
        k = max(0, self.alphabet_size - len(vowels))
        random.shuffle(consonants_all)
        consonants = consonants_all[:k] if k > 0 else []

        # 生成字符串，按概率选择元音或辅音
        s_chars = []
        pv = max(0, min(100, self.vowel_percent)) / 100.0
        pool_consonants = consonants if consonants else ["b"]
        for _ in range(length):
            if random.random() < pv:
                s_chars.append(random.choice(vowels))
            else:
                s_chars.append(random.choice(pool_consonants))
        s = "".join(s_chars)
        return {"s": s, "length": length}

    # -------------------------
    # 原环境的辅助方法（转换保留）
    # -------------------------
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
        if len(self.s) < 2:
            return False

        vowels = self.VOWELS
        total_vowels = sum(1 for char in self.s if char in vowels)

        if total_vowels < 2 or total_vowels % 2 != 0:
            return False

        left_vowel_count = 0
        for i in range(len(self.s) - 1):
            if self.s[i] in vowels:
                left_vowel_count += 1
            if left_vowel_count == total_vowels // 2:
                return True
        return False

    def CountTotalVowels(self):
        """
        Calculate the total number of vowels in the string.
        Returns: str (e.g., "3")
        """
        vowels = self.VOWELS
        total_vowels = sum(1 for char in self.s if char in vowels)
        return str(total_vowels)

    def CheckVowelAtPosition(self, position: int):
        """
        Check if the character at the specified position in the string is a vowel.
        Args: position (int)
        Returns: "true" or "false"
        """
        if position < 0 or position >= len(self.s):
            return "false"
        vowels = self.VOWELS
        return "true" if self.s[position] in vowels else "false"

    def CheckStringLength(self):
        """
        Get the length of the string.
        Returns: str (e.g., "5")
        """
        return str(len(self.s))

    def Observe(self):
        """
        Return observation information of the current state.
        """
        return "The string to be checked is now ready"

    def Done(self, answer):
        """
        Verify whether the final answer is correct and return result information.
        Args: answer (bool)
        Returns: str result message
        """
        ref_answer = self.get_ref_answer()
        correct = (bool(answer) == ref_answer)
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg

    def solve(self) -> str:
        """
        Automatically interact using actions to submit the final answer.
        Returns: str final verification message
        """
        # Get length
        obs, _, _, _, _ = self.step("\\boxed{length}")
        try:
            length = int(obs.strip())
        except Exception:
            # Fallback if unexpected response
            length = len(self.s)

        if length < 2:
            obs, _, _, _, _ = self.step("\\boxed{answer false}")
            return obs

        # Count total vowels
        obs, _, _, _, _ = self.step("\\boxed{count}")
        try:
            total_vowels = int(obs.strip())
        except Exception:
            total_vowels = sum(1 for c in self.s if c in self.VOWELS)

        if total_vowels < 2 or total_vowels % 2 != 0:
            obs, _, _, _, _ = self.step("\\boxed{answer false}")
            return obs

        target = total_vowels // 2
        current_count = 0
        for position in range(length - 1):
            obs, _, terminated, _, _ = self.step(f"\\boxed{{check {position}}}")
            if terminated:
                # We already ended (shouldn't happen for check), return last obs
                return obs
            is_vowel = obs.strip().lower()
            if is_vowel == "true":
                current_count += 1
                if current_count == target:
                    obs, _, _, _, _ = self.step("\\boxed{answer true}")
                    return obs
            if current_count > target:
                break

        obs, _, _, _, _ = self.step("\\boxed{answer false}")
        return obs

    # -------------------------
    # GEM-required methods
    # -------------------------
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
            self._done = True
            self._reward = LanguageGameReward.format_error_reward
            return (
                obs,
                self._reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        content = parsed["content"].strip().lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # Parse and execute actions
        try:
            if content == "observe":
                obs = self.Observe()
            elif content == "length":
                obs = self.CheckStringLength()
            elif content == "count" or content == "count_vowels":
                obs = self.CountTotalVowels()
            elif content.startswith("check"):
                # Accept formats: "check 3", "check pos=3", "check position 3"
                position = None
                # Try simple "check N"
                tokens = content.split()
                if len(tokens) == 2 and tokens[1].isdigit():
                    position = int(tokens[1])
                else:
                    # Try pos=K
                    m = re.search(r"(pos|position)\s*=?\s*(-?\d+)", content)
                    if m:
                        position = int(m.group(2))
                if position is None:
                    obs = "Invalid check command. Use: \\boxed{check <position>}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.CheckVowelAtPosition(position)
            elif content.startswith("answer"):
                # Formats: "answer true/false/1/0/yes/no"
                ans_token = content.replace("answer", "", 1).strip()
                if ans_token == "":
                    obs = "Missing answer. Use: \\boxed{answer true} or \\boxed{answer false}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    if ans_token in ("true", "1", "yes"):
                        answer_bool = True
                    elif ans_token in ("false", "0", "no"):
                        answer_bool = False
                    else:
                        obs = "Invalid answer token. Use true/false/1/0/yes/no."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                        # set done flags below
                        self._done = True
                        self._reward = reward
                        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

                    ref = self.get_ref_answer()
                    correct = (answer_bool == ref)
                    obs = self.Done(answer_bool)
                    reward = 1.0 if correct else -1.0
                    terminated = True
            else:
                obs = f"Invalid action: {content}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
        except Exception as e:
            obs = f"Error executing action: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            self._done = True
            self._reward = 0.0
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        # 更新内部完成状态与奖励（仅在 terminated 时）
        if terminated:
            self._done = True
            self._reward = reward
        else:
            self._reward = 0.0  # ongoing step has no reward

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def sample_random_action(self) -> str:
        # Simple random sampler, biased toward non-terminal actions
        choices = [
            "\\boxed{observe}",
            "\\boxed{length}",
            "\\boxed{count}",
            f"\\boxed{{check {random.randint(0, max(0, self.string_length - 1))}}}",
        ]
        # Occasionally try answering randomly
        if random.random() < 0.1:
            choices.append("\\boxed{answer true}")
        if random.random() < 0.1:
            choices.append("\\boxed{answer false}")
        return random.choice(choices)