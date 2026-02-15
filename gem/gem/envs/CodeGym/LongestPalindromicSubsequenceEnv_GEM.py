from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LongestPalindromicSubsequenceEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 4,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围（根据原环境：字符串长度、字母表大小、步数限制）
        self.complexity_params = {
            "string_length": (5, 50),      # 字符串长度
            "alphabet_size": (2, 26),      # 字母表大小（不同字符数）
            "turn_limit": (30, 200),       # 最大步数上限（受难度控制）
        }

        # 参数方差（启用随机化时生效）
        self.param_variance = {
            "string_length": 2,
            "alphabet_size": 2,
            "turn_limit": 10,
        }

        # 占位属性
        self.string_length: int = 0
        self.alphabet_size: int = 0
        self.turn_limit: int = 0

        # 有效最大步数（综合难度与外部 max_turns）
        self.effective_max_turns: int = self.max_turns

        # 状态变量
        self.turn_count: int = 0

        # 问题数据
        self.string: str = ""
        self.dp_table: Optional[list] = None
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

        # 计算有效最大步数（不受 seed 影响）
        self.effective_max_turns = min(self.max_turns, self.turn_limit)

    def _get_instructions(self) -> str:
        return (
            "Longest Palindromic Subsequence (LPS) Task:\n"
            "You interact with a hidden string to compute the length of its longest palindromic subsequence.\n"
            "Available actions (use the last \\boxed{...} as input):\n"
            "- Get length: \\boxed{len}\n"
            "- Get character at index i: \\boxed{char i} (0-based)\n"
            "- Initialize DP table (n x n): \\boxed{init n}\n"
            "- Set DP cell: \\boxed{set i j v}\n"
            "- Get DP cell: \\boxed{get i j}\n"
            "- Observe state: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer N} (N = length of LPS)\n"
        )

    def get_task_suffix(self) -> str:
        dp_status = "Initialized" if self.dp_table is not None else "Not initialized"
        return (
            f"Turn: {self.turn_count}/{self.effective_max_turns}\n"
            f"DP table: {dp_status}\n"
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
        self.string = self.problem["string"]
        self.dp_table = None
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 构造字母表
        alphabet = [chr(ord("a") + i) for i in range(26)]
        if self.alphabet_size < len(alphabet):
            alphabet = alphabet[: self.alphabet_size]

        # 生成随机字符串
        s = "".join(random.choice(alphabet) for _ in range(self.string_length))

        return {"string": s, "alphabet": alphabet, "length": self.string_length}

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
        tokens = content.split()
        verb = tokens[0].lower() if tokens else ""

        obs = ""
        reward: float = 0.0
        terminated = False
        truncated = False

        if verb == "len":
            obs = self.GetStringLength()

        elif verb == "char":
            if len(tokens) != 2:
                obs = "Error: char requires 1 parameter: i"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                try:
                    i = int(tokens[1])
                    obs = self.GetCharacterAt(i)
                except Exception:
                    obs = "Error: invalid index format"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

        elif verb == "init":
            if len(tokens) != 2:
                obs = "Error: init requires 1 parameter: n"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                try:
                    n = int(tokens[1])
                    obs = self.InitializeDPTable(n)
                except Exception:
                    obs = "Error: invalid size format"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

        elif verb == "set":
            if len(tokens) != 4:
                obs = "Error: set requires 3 parameters: i j v"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                try:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    v = int(tokens[3])
                    obs = self.SetDPTableCell(i, j, v)
                except Exception:
                    obs = "Error: invalid parameters for set"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

        elif verb == "get":
            if len(tokens) != 3:
                obs = "Error: get requires 2 parameters: i j"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                try:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    obs = self.GetDPTableCell(i, j)
                except Exception:
                    obs = "Error: invalid parameters for get"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

        elif verb == "observe":
            obs = self.Observe()

        elif verb == "answer":
            if len(tokens) != 2:
                obs = "Error: answer requires 1 parameter: N"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                try:
                    ans = int(tokens[1])
                    obs, reward = self.Done(ans)
                    terminated = True
                except Exception:
                    obs = "Error: invalid answer format"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

        else:
            obs = f"Invalid action: {verb}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.effective_max_turns:
            obs = f"{obs}\nReached max turns ({self.effective_max_turns})."
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

    def sample_random_action(self) -> str:
        # 简单随机动作示例
        actions = [
            "\\boxed{observe}",
            "\\boxed{len}",
            "\\boxed{char 0}",
        ]
        return random.choice(actions)

    # 辅助方法（保留并转换）

    def GetStringLength(self) -> str:
        """
        Get the length of the current string.
        Returns: str
        """
        return str(len(self.string))

    def GetCharacterAt(self, index: int) -> str:
        """
        Get the character at the specified index in the string.
        Args: index (int)
        Returns: str
        """
        if 0 <= index < len(self.string):
            return self.string[index]
        return "Error: Index out of range"

    def InitializeDPTable(self, size: int) -> str:
        """
        Initialize a size x size DP table with all elements initialized to 0.
        Args: size (int)
        Returns: str
        """
        self.dp_table = [[0] * size for _ in range(size)]
        return f"DP table initialized as a {size}x{size} matrix"

    def SetDPTableCell(self, i: int, j: int, value: int) -> str:
        """
        Set the value of the cell at position (i, j) in the DP table.
        Args: i (int), j (int), value (int)
        Returns: str
        """
        if self.dp_table is None:
            return "Error: DP table not initialized yet"

        if 0 <= i < len(self.dp_table) and 0 <= j < len(self.dp_table[0]):
            self.dp_table[i][j] = value
            return f"DP table [{i}][{j}] set to {value}"
        return "Error: Index out of DP table range"

    def GetDPTableCell(self, i: int, j: int) -> str:
        """
        Get the value of the cell at position (i, j) in the DP table.
        Args: i (int), j (int)
        Returns: str
        """
        if self.dp_table is None:
            return "Error: DP table not initialized yet"

        if 0 <= i < len(self.dp_table) and 0 <= j < len(self.dp_table[0]):
            return str(self.dp_table[i][j])
        return "Error: Index out of DP table range"

    def Observe(self) -> str:
        """
        Return observation information of the current state.
        Returns: str
        """
        dp_status = "Initialized" if self.dp_table is not None else "Not initialized"
        return f"Current string: {self.string}, DP table status: {dp_status}"

    def Done(self, answer: int) -> Tuple[str, float]:
        """
        Verify whether the final answer is correct and return result information.
        Args: answer (int)
        Returns: (str, float) -> message and reward
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        reward = 1.0 if correct else -1.0
        return msg, reward

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        Returns: int
        """
        s = self.string
        n = len(s)
        if n == 0:
            return 0
        dp = [[0] * n for _ in range(n)]

        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2 if j - i > 1 else 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

        return dp[0][n - 1]

    def solve(self) -> str:
        """
        Automatically call actions to complete the process and submit the answer for verification.
        Returns: str
        """
        # Initialize DP table of appropriate size
        n = len(self.string)
        self.InitializeDPTable(n)

        # Fill base cases
        if self.dp_table is None:
            self.InitializeDPTable(n)
        for i in range(n):
            self.SetDPTableCell(i, i, 1)

        # Fill DP table using LPS recurrence
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                char_i = self.GetCharacterAt(i)
                char_j = self.GetCharacterAt(j)

                if char_i == "Error: Index out of range" or char_j == "Error: Index out of range":
                    continue

                if char_i == char_j:
                    if length == 2:
                        self.SetDPTableCell(i, j, 2)
                    else:
                        sub_val_str = self.GetDPTableCell(i + 1, j - 1)
                        if sub_val_str.startswith("Error"):
                            continue
                        sub_val = int(sub_val_str)
                        self.SetDPTableCell(i, j, sub_val + 2)
                else:
                    val1_str = self.GetDPTableCell(i + 1, j)
                    val2_str = self.GetDPTableCell(i, j - 1)
                    if val1_str.startswith("Error") or val2_str.startswith("Error"):
                        continue
                    val1 = int(val1_str)
                    val2 = int(val2_str)
                    max_val = max(val1, val2)
                    self.SetDPTableCell(i, j, max_val)

        max_len_str = self.GetDPTableCell(0, n - 1) if n > 0 else "0"
        max_len = int(max_len_str) if not max_len_str.startswith("Error") else self.get_ref_answer()
        msg, _reward = self.Done(max_len)
        return msg