from typing import Any, Dict, Optional, Tuple
import random
import re
import json
import ast
from collections import defaultdict
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class SubstringIndicesEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 1,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围设计（影响字符串长度、字典规模、词长度、字母表规模、重复词数量）
        self.complexity_params = {
            "s_length": (20, 400),        # 字符串 S 的长度
            "dict_size": (1, 8),          # 字典中的词数
            "word_length": (3, 8),        # 每个词的长度（所有词等长）
            "alphabet_size": (2, 12),     # 字母表大小（用于生成词）
            "num_duplicates": (0, 3),     # 允许重复词的数量上限（用于增加词频约束）
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "s_length": 25,
            "dict_size": 2,
            "word_length": 1,
            "alphabet_size": 2,
            "num_duplicates": 1,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.s_length: int = 0
        self.dict_size: int = 0
        self.word_length: int = 0
        self.alphabet_size: int = 0
        self.num_duplicates: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.S: str = ""
        self.dictionary: list[str] = []

        self.reset()

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "SubstringIndicesEnvGEM@"
        if not env_str.startswith(prefix):
            return None
        try:
            options = ast.literal_eval(env_str.split("@", 1)[1])
        except Exception:
            return None
        env = SubstringIndicesEnvGEM()
        env.S = options.get("S", "")
        env.dictionary = options.get("dictionary", [])
        env.problem = {"S": env.S, "dictionary": env.dictionary}
        env.turn_count = 0
        return env

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
            "Substring Indices: Find starting indices where the substring is a concatenation of all dictionary words.\n"
            "All words have the same length and must be used exactly once, contiguously.\n"
            "Available actions (use \\boxed{...}):\n"
            "- Observe current problem: \\boxed{observe}\n"
            "- Get word length: \\boxed{getlen}\n"
            "- Get total concat length: \\boxed{totallen} or \\boxed{totallen L}\n"
            "- Create word frequency dict: \\boxed{dict}\n"
            "- Check substring: \\boxed{check I} or \\boxed{check I L}\n"
            "- Submit final answer (comma-separated indices): \\boxed{answer i1,i2,...}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"S length: {len(self.S)} | Dict size: {len(self.dictionary)} | "
            f"Turn: {self.turn_count}/{self.max_turns}\nEnter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.S = self.problem["S"]
        self.dictionary = self.problem["dictionary"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 构造字母表
        alphabet = [chr(ord('a') + i) for i in range(min(self.alphabet_size, 26))]
        if not alphabet:
            alphabet = ['a']

        # 生成唯一词集合
        unique_words = set()
        attempts = 0
        max_attempts = 1000
        while len(unique_words) < max(1, min(self.dict_size, 100)) and attempts < max_attempts:
            word = ''.join(random.choice(alphabet) for _ in range(self.word_length))
            unique_words.add(word)
            attempts += 1
        unique_words = list(unique_words)
        if not unique_words:
            unique_words = [''.join(random.choice(alphabet) for _ in range(self.word_length))]

        # 构建字典，可能包含重复词以增加词频约束
        dictionary: list[str] = []
        # 首先选择至少一个词
        base_choices = unique_words[:]
        for _ in range(self.dict_size):
            dictionary.append(random.choice(base_choices))

        # 注入重复（最多 num_duplicates 个）
        for _ in range(min(self.num_duplicates, self.dict_size // 2)):
            dictionary[random.randrange(self.dict_size)] = random.choice(base_choices)

        # 生成 S，尽量嵌入一个有效的拼接以保持可解性
        total_length = self.word_length * len(dictionary)
        s_length = max(self.s_length, total_length + random.randint(0, max(0, self.s_length - total_length)))
        S_chars = [random.choice(alphabet) for _ in range(s_length)]
        if s_length >= total_length:
            start_index = random.randint(0, s_length - total_length) if s_length > total_length else 0
            concat = ''.join(random.sample(dictionary, len(dictionary)))
            S_chars[start_index:start_index + total_length] = list(concat)
        S = ''.join(S_chars)

        return {"S": S, "dictionary": dictionary}

    # 保留原环境的辅助方法（转换为类内方法）
    def get_ref_answer(self):
        """
        使用环境中的信息计算参考答案。
        """
        if not self.S or not self.dictionary:
            return []

        word_length = len(self.dictionary[0])
        total_length = word_length * len(self.dictionary)
        word_count = len(self.dictionary)
        word_dict = defaultdict(int)

        for word in self.dictionary:
            word_dict[word] += 1

        result_indices = []

        for i in range(0, len(self.S) - total_length + 1):
            seen = defaultdict(int)
            j = 0
            while j < word_count:
                word = self.S[i + j * word_length: i + (j + 1) * word_length]
                if word in word_dict:
                    seen[word] += 1
                    if seen[word] > word_dict[word]:
                        break
                else:
                    break
                j += 1

            if j == word_count:
                result_indices.append(i)

        return result_indices

    def GetWordLength(self):
        """
        获取字典中词的长度。
        返回字符串形式长度；若字典为空返回 "0"。
        """
        if not self.dictionary:
            return "0"
        return str(len(self.dictionary[0]))

    def GetTotalLength(self, word_length: int):
        """
        计算所有词拼接后的总长度。
        """
        return str(word_length * len(self.dictionary))

    def CreateWordFreqDict(self):
        """
        创建字典中词的频率映射，返回 JSON 字符串。
        """
        word_dict = defaultdict(int)
        for word in self.dictionary:
            word_dict[word] += 1
        return json.dumps(dict(word_dict))

    def CheckSubstring(self, i: int, word_length: int, word_count: int, word_dict: dict):
        """
        检查从索引 i 开始的子串是否为所有词按任意顺序拼接的结果。
        返回 "True" 或 "False"。
        """
        seen = defaultdict(int)
        j = 0
        while j < word_count:
            start = i + j * word_length
            end = start + word_length
            if end > len(self.S):
                return "False"

            word = self.S[start:end]
            if word in word_dict:
                seen[word] += 1
                if seen[word] > word_dict[word]:
                    return "False"
            else:
                return "False"
            j += 1

        return "True" if j == word_count else "False"

    def Observe(self):
        """
        返回当前环境的观察信息。
        """
        return f"Current string: {self.S}, Dictionary: {self.dictionary}"

    def Done(self, answer):
        """
        校验最终答案，并返回结果信息。
        """
        ref_answer = self.get_ref_answer()
        correct = sorted(answer) == sorted(ref_answer)
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, 1.0 if correct else -1.0

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
        cmd = tokens[0].lower() if tokens else ""

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()
                reward = 0.0
                terminated = False

            elif cmd == "getlen":
                obs = self.GetWordLength()
                reward = 0.0
                terminated = False

            elif cmd == "totallen":
                # 支持可选参数 L
                if len(tokens) >= 2:
                    try:
                        L = int(tokens[1])
                    except Exception:
                        obs = "Error: invalid parameter L for totallen."
                        return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
                else:
                    if not self.dictionary:
                        L = 0
                    else:
                        L = len(self.dictionary[0])
                obs = self.GetTotalLength(L)
                reward = 0.0
                terminated = False

            elif cmd == "dict":
                obs = self.CreateWordFreqDict()
                reward = 0.0
                terminated = False

            elif cmd == "check":
                if len(tokens) < 2:
                    obs = "Error: missing index for check."
                    return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
                try:
                    i = int(tokens[1])
                except Exception:
                    obs = "Error: invalid index for check."
                    return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
                # 可选 L；否则从字典推断
                if len(tokens) >= 3:
                    try:
                        word_length = int(tokens[2])
                    except Exception:
                        obs = "Error: invalid word_length for check."
                        return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
                else:
                    if not self.dictionary:
                        word_length = 0
                    else:
                        word_length = len(self.dictionary[0])

                word_count = len(self.dictionary)
                try:
                    word_dict_str = self.CreateWordFreqDict()
                    word_dict = json.loads(word_dict_str)
                except Exception:
                    word_dict = {}
                obs = self.CheckSubstring(i, word_length, word_count, word_dict)
                reward = 0.0
                terminated = False

            elif cmd == "answer":
                if len(tokens) < 2:
                    obs = "Error: missing indices for answer."
                    return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
                indices_raw = ' '.join(tokens[1:])
                # 支持逗号或空格分隔
                parts = [p for p in re.split(r"[,\s]+", indices_raw.strip()) if p != ""]
                try:
                    answer_list = [int(p) for p in parts]
                except Exception:
                    obs = "Error: invalid indices format."
                    return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}
                msg, final_reward = self.Done(answer_list)
                obs = msg
                reward = final_reward
                terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

        except Exception as e:
            obs = f"Runtime error: {str(e)}"
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

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
        # 简单示例动作
        return "\\boxed{observe}"

    # 可选：提供自动求解流程（使用 GEM 风格动作）
    def solve(self) -> str:
        # 观察
        observe_info, _, _, _, _ = self.step("\\boxed{observe}")
        # 获取词长
        word_length_str, _, _, _, _ = self.step("\\boxed{getlen}")
        try:
            word_length = int(word_length_str)
        except Exception:
            word_length = 0
        if word_length == 0:
            obs, reward, _, _, _ = self.step("\\boxed{answer }")
            return f"{obs}, reward={reward}"

        # 总长度
        total_length_str, _, _, _, _ = self.step(f"\\boxed{{totallen {word_length}}}")
        try:
            total_length = int(total_length_str)
        except Exception:
            total_length = word_length * len(self.dictionary)
        word_count = total_length // word_length

        # 词频字典
        word_dict_str, _, _, _, _ = self.step("\\boxed{dict}")
        try:
            word_dict = json.loads(word_dict_str)
        except Exception:
            word_dict = {}

        # 解析 S
        s_part = observe_info.split('Current string: ')[1].split(', Dictionary: ')[0] if 'Current string: ' in observe_info else self.S
        len_S = len(s_part)
        max_i = len_S - total_length

        result = []
        if max_i >= 0:
            for i in range(0, max_i + 1):
                check_obs, _, _, _, _ = self.step(f"\\boxed{{check {i} {word_length}}}")
                if check_obs == "True":
                    result.append(i)

        final_obs, reward, _, _, _ = self.step(f"\\boxed{{answer {','.join(map(str, result))}}}")
        return f"{final_obs}, reward={reward}"