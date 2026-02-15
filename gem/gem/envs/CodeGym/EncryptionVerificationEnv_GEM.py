from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class EncryptionVerificationEnvGEM(Env):
    """
    GEM-style environment converted from the original CodeGym EncryptionVerificationEnv.

    Task:
      - You will be given a secret key (0-25), an original lowercase message, and a "reference encrypted message".
      - Your goal is to determine whether encrypting the original message with the key (Caesar shift) matches the reference message.
      - Submit the final answer as MATCH or NO MATCH.

    Actions (use the \\boxed{...} format, only the last \\boxed{...} in the input is parsed):
      - Observe:
          \\boxed{observe}
        Returns: "Key: K, Original message: MSG, Reference encrypted message: REF"

      - Encrypt a single character with a key (lowercase only):
          \\boxed{encrypt char=a key=3}
        Returns: the encrypted single character (e.g., "d")

      - Compare two strings (exact match):
          \\boxed{compare str1=abc str2=abd}
        Returns: "True" or "False"

      - Submit final answer:
          \\boxed{answer MATCH}
        or
          \\boxed{answer NO MATCH}
        Returns the verification result and ends the episode.

    Rewards:
      - Success (correct final answer): 1.0
      - Failure (incorrect final answer): -1.0
      - Format error: LanguageGameReward.format_error_reward
      - Invalid action / missing parameter: LanguageGameReward.invalid_action_reward
      - Timeout (reach max turns): 0.0 with terminated=True, truncated=True

    Difficulty control (complexity 1-10):
      - message_length: length of the message to encrypt
      - key_max: maximum possible key value (actual key sampled in [0, key_max])
      - max_turns_param: per-instance turn budget (combined with user-provided max_turns)
      - mismatch_prob: probability (%) to inject a mismatch into the reference message
    """

    def __init__(
        self,
        complexity: int = 6,  # 难度等级 1-10，默认中等
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
            "message_length": (5, 50),     # 原消息长度
            "key_max": (3, 25),            # 密钥上限（实际密钥范围 [0, key_max]）
            "max_turns_param": (20, 200),  # 每局最大步数预算（与构造参数组合）
            "mismatch_prob": (0, 35),      # 参考密文被扰动的概率（百分比）
        }

        # 参数方差（启用随机化时生效）
        self.param_variance = {
            "message_length": 3,
            "key_max": 2,
            "max_turns_param": 10,
            "mismatch_prob": 5,
        }

        # 占位属性（由 _apply_complexity_params 设置）
        self.message_length: int = 0
        self.key_max: int = 0
        self.max_turns_param: int = 0
        self.mismatch_prob: int = 0

        # 有效步数上限（由 reset 合成）
        self.effective_max_turns: int = self.max_turns

        # 状态变量
        self.turn_count: int = 0

        # 原环境的内部状态（保持兼容）
        self._reward: float = 0.0
        self._done: bool = False

        # 问题实例
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
            "Encryption Verification (Caesar Shift): Determine whether encrypt(key, message) matches the reference.\n"
            "Available actions:\n"
            "- Observe problem: \\boxed{observe}\n"
            "- Encrypt a character: \\boxed{encrypt char=a key=3}\n"
            "- Compare two strings: \\boxed{compare str1=abc str2=abd}\n"
            "- Submit final answer: \\boxed{answer MATCH} or \\boxed{answer NO MATCH}\n"
        )

    def get_task_suffix(self) -> str:
        msg_len = self.problem.get("message_len", 0)
        return f"Message length: {msg_len}\nTurn: {self.turn_count}/{self.effective_max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响实例生成，不影响难度参数

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 合成有效步数上限（遵循构造传入的 max_turns 与难度上限的下界）
        self.effective_max_turns = min(self.max_turns, self.max_turns_param)

        # 重置状态
        self.turn_count = 0
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        # 随机消息和密钥
        message = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(self.message_length))
        key = random.randint(0, max(0, self.key_max))

        # 计算参考密文（初步）
        encrypted = []
        for ch in message:
            shifted = chr(((ord(ch) - ord('a') + key) % 26) + ord('a'))
            encrypted.append(shifted)
        encrypted_message = "".join(encrypted)

        # 以一定概率扰动参考密文，制造 NO MATCH
        will_mismatch = random.random() < (self.mismatch_prob / 100.0)
        reference = encrypted_message
        if will_mismatch and len(reference) > 0:
            idx = random.randrange(len(reference))
            current = reference[idx]
            # 变成不同的字母
            choices = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != current]
            new_c = random.choice(choices)
            reference = reference[:idx] + new_c + reference[idx + 1 :]

        return {
            "key": key,
            "message": message,
            "reference": reference,
            "message_len": len(message),
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        # 默认返回
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = parsed.get("command", "").lower()
        params = parsed.get("params", {})

        try:
            if cmd == "observe":
                obs = self.Observe()

            elif cmd == "encrypt":
                # 需要 char 和 key
                if "char" not in params or "key" not in params:
                    obs = "Missing parameter: require char and key."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                char = params["char"]
                key_str = params["key"]
                if not isinstance(char, str) or len(char) != 1 or not ('a' <= char <= 'z'):
                    obs = "Invalid parameter: char must be a single lowercase letter."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    key = int(key_str)
                except Exception:
                    obs = "Invalid parameter: key must be an integer."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.EncryptCharacter(char, key)

            elif cmd == "compare":
                if "str1" not in params or "str2" not in params:
                    obs = "Missing parameter: require str1 and str2."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                str1 = params["str1"]
                str2 = params["str2"]
                obs = self.CompareStrings(str1, str2)

            elif cmd == "answer":
                if "arg" not in params:
                    obs = "Missing parameter: require final answer (MATCH or NO MATCH)."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans = params["arg"].upper()
                if ans not in {"MATCH", "NO", "NO MATCH", "NOMATCH"}:
                    obs = "Invalid answer. Use MATCH or NO MATCH."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                normalized_ans = "MATCH" if ans == "MATCH" else "NO MATCH"
                done_msg = self.Done(normalized_ans)
                ref_answer = self.get_ref_answer()
                correct = (normalized_ans == ref_answer)

                obs = done_msg
                reward = 1.0 if correct else -1.0
                terminated = True
                truncated = False

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

        # 超时检查（放在 step 结尾，以便本回合动作仍生效）
        if not terminated and self.turn_count >= self.effective_max_turns:
            obs = f"{obs}\nReached max turns ({self.effective_max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        """
        Parse the last \\boxed{...} content.
        Supported:
          - observe
          - encrypt char=a key=3
          - compare str1=abc str2=abd
          - answer MATCH
        """
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        if not content:
            return None

        # Split into command and args
        parts = content.split()
        if len(parts) == 0:
            return None

        command = parts[0].lower()
        params: Dict[str, str] = {}

        if command == "answer":
            # answer <MATCH|NO MATCH>
            if len(parts) >= 2:
                arg = " ".join(parts[1:]).strip()
                params["arg"] = arg
            return {"command": command, "params": params}

        if command in {"observe"}:
            return {"command": command, "params": params}

        # key=value pairs for other commands
        # values are expected to be contiguous (no spaces). Quotes are optionally stripped.
        def strip_quotes(s: str) -> str:
            if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
                return s[1:-1]
            return s

        for tok in parts[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                k = k.strip().lower()
                v = strip_quotes(v.strip())
                params[k] = v
        return {"command": command, "params": params}

    def sample_random_action(self) -> str:
        return "\\boxed{observe}"

    # ==============================
    # Helper/Compatibility Methods
    # ==============================

    def get_ref_answer(self) -> str:
        """Compute the reference answer for the current problem instance."""
        key = self.problem.get("key", 0)
        message = self.problem.get("message", "")
        reference = self.problem.get("reference", "")

        encrypted = []
        for char in message:
            shifted_char = chr(((ord(char) - ord('a') + key) % 26) + ord('a'))
            encrypted.append(shifted_char)
        encrypted_message = ''.join(encrypted)
        return "MATCH" if encrypted_message == reference else "NO MATCH"

    def EncryptCharacter(self, char: str, key: int) -> str:
        r"""
        Perform cyclic shift encryption on a single character using the given key.

        Args:
            char (str): The single lowercase letter character to be encrypted.
            key (int): The number of shift positions, typically 0 to 25.

        Returns:
            str: The encrypted character.

        Example Output:
            "d"
        """
        shifted_char = chr(((ord(char) - ord('a') + key) % 26) + ord('a'))
        return shifted_char

    def CompareStrings(self, str1: str, str2: str) -> str:
        r"""
        Compare whether two strings are equal.

        Args:
            str1 (str): The first string to be compared.
            str2 (str): The second string to be compared.

        Returns:
            str: "True" if the strings are equal, "False" otherwise.

        Example Output:
            "True"
        """
        return "True" if str1 == str2 else "False"

    def Observe(self) -> str:
        r"""
        Obtain the key, original message, and reference encrypted message in the current environment.

        Returns:
            str: Information containing the key, original message, and reference encrypted message.

        Example Output:
            "Key: 3, Original message: abcde, Reference encrypted message: defgh"
        """
        key = self.problem.get("key", 0)
        message = self.problem.get("message", "")
        reference = self.problem.get("reference", "")
        return f"Key: {key}, Original message: {message}, Reference encrypted message: {reference}"

    def Done(self, answer: str) -> str:
        r"""
        Submit the final answer and verify its correctness.

        Args:
            answer (str): The answer to be submitted, which should be "MATCH" or "NO MATCH".

        Returns:
            str: Result information, including whether it is correct and reward information.

        Example Output:
            "Your answer: MATCH, Reference answer: MATCH, Result: Correct"
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        Automatically compute the answer and return the verification message (non-interactive helper).
        """
        key = self.problem.get("key", 0)
        message = self.problem.get("message", "")
        reference = self.problem.get("reference", "")

        encrypted_chars = []
        for ch in message:
            encrypted_chars.append(self.EncryptCharacter(ch, key))
        encrypted_msg = "".join(encrypted_chars)

        compare_result = self.CompareStrings(encrypted_msg, reference)
        answer = "MATCH" if compare_result == "True" else "NO MATCH"
        return self.Done(answer)