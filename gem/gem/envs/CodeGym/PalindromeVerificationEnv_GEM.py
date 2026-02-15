from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class PalindromeVerificationEnvGEM(Env):
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

        # 定义难度参数范围
        self.complexity_params = {
            "string_length": (5, 50),      # 字符串长度范围
            "alphabet_size": (2, 26),      # 字母表大小
            "noise_level": (0, 3),         # 扰动级别（插入字符数量的期望控制）
            "turn_limit": (20, 200),       # 难度驱动的最大步数上限
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "string_length": 2,
            "alphabet_size": 1,
            "noise_level": 1,
            "turn_limit": 10,
        }

        # 占位属性
        self.string_length: int = 0
        self.alphabet_size: int = 0
        self.noise_level: int = 0
        self.turn_limit: int = 0

        # 原环境的动作名称映射（内部使用）
        self.CHECK_PALINDROME_RANGE = 0
        self.GET_STRING_LENGTH = 1
        self.GET_CHARACTER_AT = 2
        self.OBSERVE = 3
        self.DONE = 4
        self.func_mapping = {
            "CheckPalindromeRange": self.CHECK_PALINDROME_RANGE,
            "GetStringLength": self.GET_STRING_LENGTH,
            "GetCharacterAt": self.GET_CHARACTER_AT,
            "Observe": self.OBSERVE,
            "Done": self.DONE,
        }

        # 状态变量
        self.turn_count: int = 0
        self._done: bool = False
        self._reward: float = 0.0
        self.s: str = ""

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

        # 将难度驱动的 turn_limit 与用户提供的 max_turns 结合
        self.max_turns = min(self.max_turns, self.turn_limit)

    def _get_instructions(self) -> str:
        return (
            "Palindrome Verification: Determine if a string can be a palindrome by deleting at most one character.\n"
            "Available actions:\n"
            "- Observe state: \\boxed{observe}\n"
            "- Get length: \\boxed{len}\n"
            "- Get character at index i: \\boxed{char i}\n"
            "- Check if substring [i..j] is palindrome: \\boxed{check i j}\n"
            "- Submit final answer (True/False): \\boxed{answer True} or \\boxed{answer False}\n"
            "Indexes are 0-based."
        )

    def get_task_suffix(self) -> str:
        return f"String length: {len(self.s)}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 状态重置
        self.turn_count = 0
        self._done = False
        self._reward = 0.0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 构造字母表
        alphabet = [chr(ord('a') + i) for i in range(self.alphabet_size)]
        if not alphabet:
            alphabet = ['a']

        # 生成基准回文串长度
        target_len = self.string_length
        base_len = max(1, target_len - 1)  # 基准回文的长度至少为1

        # 生成基准回文串
        half_len = base_len // 2
        left_half = [random.choice(alphabet) for _ in range(half_len)]
        if base_len % 2 == 1:
            middle = random.choice(alphabet)
            base_pal = left_half + [middle] + left_half[::-1]
        else:
            base_pal = left_half + left_half[::-1]

        # 根据 noise_level 决定插入的字符数量（0/1/2）来控制难度
        k = 0
        if self.noise_level <= 0:
            k = random.choice([0, 1])
        elif self.noise_level == 1:
            k = random.choice([0, 1, 1])
        elif self.noise_level == 2:
            k = random.choice([1, 2])
        else:
            k = 2

        s_list = base_pal[:]
        for _ in range(k):
            pos = random.randint(0, len(s_list))
            s_list.insert(pos, random.choice(alphabet))

        # 若长度不足或超出，截断/填充
        if len(s_list) < target_len:
            # 随机填充至目标长度（这可能破坏回文性质，增加难度）
            while len(s_list) < target_len:
                pos = random.randint(0, len(s_list))
                s_list.insert(pos, random.choice(alphabet))
        elif len(s_list) > target_len:
            s_list = s_list[:target_len]

        self.s = "".join(s_list)
        return {"string": self.s, "length": len(self.s), "alphabet_size": self.alphabet_size}

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
        tokens = content.strip().split()
        if not tokens:
            obs = f"Format error at turn {self.turn_count}."
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
            elif cmd == "len":
                obs = self.GetStringLength()
            elif cmd == "char":
                if len(tokens) < 2 or not tokens[1].isdigit():
                    obs = f"Format error: expected 'char i'."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                index = int(tokens[1])
                obs = self.GetCharacterAt(index)
            elif cmd == "check":
                if len(tokens) < 3 or (not tokens[1].isdigit()) or (not tokens[2].isdigit()):
                    obs = f"Format error: expected 'check i j'."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i = int(tokens[1])
                j = int(tokens[2])
                obs = self.CheckPalindromeRange(i, j)
            elif cmd == "answer":
                if len(tokens) < 2 or tokens[1].lower() not in ("true", "false"):
                    obs = f"Format error: expected 'answer True' or 'answer False'."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                answer = tokens[1].lower() == "true"
                obs_msg = self.Done(answer)
                obs = obs_msg
                # 根据结果设置奖励
                ref_answer = self.get_ref_answer()
                if answer == ref_answer:
                    reward = 1.0
                else:
                    reward = -1.0
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
            obs = f"Error: {str(e)}"
            return (
                obs,
                LanguageGameReward.format_error_reward,
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
        # 随机生成一个有效动作示例
        if not self.s:
            return "\\boxed{observe}"
        choice = random.choice(["observe", "len", "char", "check"])
        if choice == "observe":
            return "\\boxed{observe}"
        elif choice == "len":
            return "\\boxed{len}"
        elif choice == "char":
            idx = random.randint(0, max(0, len(self.s) - 1))
            return f"\\boxed{{char {idx}}}"
        else:
            if len(self.s) == 0:
                return "\\boxed{observe}"
            i = random.randint(0, len(self.s) - 1)
            j = random.randint(i, len(self.s) - 1)
            return f"\\boxed{{check {i} {j}}}"

    # 保留原环境的辅助方法并转换
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        """
        Create environment from string like:
        PalindromeVerificationEnv@{"s": "abcba"}
        or
        PalindromeVerificationEnvGEM@{"s": "abcba"}
        """
        prefixes = ["PalindromeVerificationEnv@", "PalindromeVerificationEnvGEM@"]
        if not any(env_str.startswith(p) for p in prefixes):
            return None
        try:
            payload = env_str.split("@", 1)[1]
            options = json.loads(payload)
        except Exception:
            return None
        s = options.get("s", "")
        env = PalindromeVerificationEnvGEM(enable_param_randomization=False)
        # Override generated problem with provided string
        env.s = s
        env.problem = {"string": s, "length": len(s)}
        env.turn_count = 0
        env._done = False
        env._reward = 0.0
        return env

    def reset_with_string(self, s: str, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset and override the internal string with provided s."""
        instr, info = self.reset(seed=seed)
        self.s = s or ""
        self.problem = {"string": self.s, "length": len(self.s)}
        return instr, info

    # Reference answer logic
    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        Determine if s can be a palindrome by deleting at most one character.
        """
        def is_palindrome_range(i, j):
            return all(self.s[k] == self.s[j - k + i] for k in range(i, j))

        left, right = 0, len(self.s) - 1
        while left < right:
            if self.s[left] != self.s[right]:
                return is_palindrome_range(left + 1, right) or is_palindrome_range(left, right - 1)
            left += 1
            right -= 1
        return True

    # All the actions of the environment (converted to internal helpers)
    def CheckPalindromeRange(self, i: int, j: int):
        """
        Check if the substring s[i:j+1] is a palindrome.
        Returns "True" or "False".
        """
        if i < 0 or j < 0 or i >= len(self.s) or j >= len(self.s) or i > j:
            return "False"
        is_pal = all(self.s[k] == self.s[j - k + i] for k in range(i, j))
        return str(is_pal)

    def GetStringLength(self):
        """
        Get the length of the current string.
        Returns the length as string.
        """
        return str(len(self.s))

    def GetCharacterAt(self, index: int):
        """
        Get the character at the specified index in the string.
        Returns the character or empty string if out of bounds.
        """
        if 0 <= index < len(self.s):
            return self.s[index]
        return ""

    def Observe(self):
        """
        Return observation information of the current state.
        """
        return "The string to be verified is ready"

    def Done(self, answer: bool):
        """
        Verify whether the final answer is correct and return the result info.
        Also sets internal reward and done state (for compatibility).
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    # A solve method using the GEM-style actions
    def solve(self) -> str:
        """
        Automatically call actions to determine if the given string can be a palindrome by deleting at most one character, and submit the answer.
        Returns the final verification result info.
        """
        # Get length
        n_str, _, _, _, _ = self.step("\\boxed{len}")
        try:
            n = int(n_str)
        except Exception:
            n = len(self.s)
        left = 0
        right = n - 1

        while left < right:
            obs_left, _, term, _, _ = self.step(f"\\boxed{{char {left}}}")
            if term:
                return obs_left
            obs_right, _, term, _, _ = self.step(f"\\boxed{{char {right}}}")
            if term:
                return obs_right

            c_left = obs_left
            c_right = obs_right

            if c_left == c_right:
                left += 1
                right -= 1
            else:
                check_left, _, term, _, _ = self.step(f"\\boxed{{check {left + 1} {right}}}")
                if term:
                    return check_left
                check_right, _, term, _, _ = self.step(f"\\boxed{{check {left} {right - 1}}}")
                if term:
                    return check_right
                answer = (check_left == "True") or (check_right == "True")
                final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {'True' if answer else 'False'}}}")
                return final_obs

        final_obs, _, _, _, _ = self.step("\\boxed{answer True}")
        return final_obs


# Example usage (optional)
if __name__ == "__main__":
    # Test with random instance
    env = PalindromeVerificationEnvGEM(complexity=5, enable_param_randomization=True, max_turns=100)
    instructions, info = env.reset(seed=42)
    print(instructions)
    print(info["suffix"])
    print(env.solve())

    # Test with provided strings via from_env_str
    print("\nTest Case 1:")
    test_str = "abcba"
    env1 = PalindromeVerificationEnvGEM.from_env_str(f"PalindromeVerificationEnvGEM@{{\"s\": \"{test_str}\"}}")
    instr1, info1 = env1.reset_with_string(test_str)
    print(env1.solve())
    print("turn count:", env1.turn_count)

    print("\nTest Case 2:")
    test_str = "abcca"
    env2 = PalindromeVerificationEnvGEM.from_env_str(f"PalindromeVerificationEnv@{{\"s\": \"{test_str}\"}}")
    instr2, info2 = env2.reset_with_string(test_str)
    print(env2.solve())
    print("turn count:", env2.turn_count)

    print("\nTest Case 3:")
    test_str = "abcde"
    env3 = PalindromeVerificationEnvGEM.from_env_str(f"PalindromeVerificationEnvGEM@{{\"s\": \"{test_str}\"}}")
    instr3, info3 = env3.reset_with_string(test_str)
    print(env3.solve())
    print("turn count:", env3.turn_count)