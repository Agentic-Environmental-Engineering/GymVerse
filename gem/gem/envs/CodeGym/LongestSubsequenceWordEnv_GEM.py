from typing import Any, Dict, Optional, Tuple
import random
import re
import string
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LongestSubsequenceWordEnvGEM(Env):
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

        # 难度参数范围（根据原环境设计）
        self.complexity_params = {
            "keyword_length": (5, 50),
            "num_words": (5, 50),
            "alphabet_size": (3, 26),
            "word_length_min": (1, 3),
            "word_length_max": (4, 12),
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "keyword_length": 2,
            "num_words": 2,
            "alphabet_size": 2,
            "word_length_min": 1,
            "word_length_max": 1,
        }

        # 占位属性
        self.keyword_length: int = 0
        self.num_words: int = 0
        self.alphabet_size: int = 0
        self.word_length_min: int = 0
        self.word_length_max: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题相关状态
        self.problem: Dict[str, Any] = {}
        self.reference_answer: str = ""
        self.last_sorted_words: Optional[list] = None
        self.last_candidate: Optional[str] = None

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

        # 保证最小长度不超过最大长度
        if self.word_length_min > self.word_length_max:
            self.word_length_min, self.word_length_max = self.word_length_max, self.word_length_min
        self.word_length_min = max(1, self.word_length_min)

    def _get_instructions(self) -> str:
        return (
            "Longest Subsequence Word: Find the longest word from the given list that is a subsequence of the keyword.\n"
            "Words are sorted by: length(desc), then lexicographic(asc). If no valid word exists, the answer is an empty string.\n"
            "Available actions:\n"
            "- Observe problem: \\boxed{observe}\n"
            "- Sort words: \\boxed{sort}\n"
            "- Check if a word can be formed: \\boxed{check WORD}\n"
            "- Find the first valid word from the sorted list: \\boxed{find}\n"
            "- Submit final answer: \\boxed{answer WORD}\n"
            "Note: To submit an empty answer, you may use '-' or '' (empty quotes).\n"
        )

    def get_task_suffix(self) -> str:
        kw = self.problem.get("keyword", "")
        words = self.problem.get("words", [])
        return (
            f"Keyword length: {len(kw) if kw else 0} | Words: {len(words)} | "
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

        # 预计算参考答案
        self.reference_answer = self.get_ref_answer()

        # 清理状态
        self.turn_count = 0
        self.last_sorted_words = None
        self.last_candidate = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 随机字母表
        alphabet = random.sample(list(string.ascii_lowercase), self.alphabet_size)
        # 生成 keyword
        keyword = "".join(random.choice(alphabet) for _ in range(self.keyword_length))
        # 生成 words 列表
        words = []
        for _ in range(self.num_words):
            wlen = random.randint(self.word_length_min, self.word_length_max)
            word = "".join(random.choice(alphabet) for _ in range(wlen))
            words.append(word)
        return {"keyword": keyword, "words": words, "n": len(words)}

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

        content = parsed["content"].strip()
        lower = content.lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if lower == "observe":
                obs = self.Observe()
            elif lower == "sort":
                sorted_json = self.SortWords(self.problem["words"])
                self.last_sorted_words = self._safe_json_loads(sorted_json)
                obs = f"Sorted words: {sorted_json}"
            elif lower.startswith("check"):
                m = re.match(r"^check\s+([A-Za-z]+)\s*$", content, re.IGNORECASE)
                if not m:
                    obs = f"Invalid 'check' action format. Use \\boxed{{check WORD}}."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    word = m.group(1)
                    res = self.CheckCanForm(word)
                    obs = f"CheckCanForm('{word}') -> {res}"
            elif lower == "find":
                # 如果未排序，自动排序
                if not self.last_sorted_words:
                    sorted_json = self.SortWords(self.problem["words"])
                    self.last_sorted_words = self._safe_json_loads(sorted_json)
                candidate = self.FindFirstValid(self.last_sorted_words)
                self.last_candidate = candidate
                obs = f"First valid: {candidate if candidate else ''}"
            elif lower.startswith("answer"):
                m = re.match(r"^answer(?:\s+(.*))?$", content, re.IGNORECASE | re.DOTALL)
                if not m:
                    obs = f"Invalid 'answer' action format. Use \\boxed{{answer WORD}}."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    raw_ans = m.group(1)
                    answer = "" if raw_ans is None else raw_ans.strip()
                    if answer in ["-", "''", '""']:
                        answer = ""
                    result_msg = self.Done(answer)
                    correct = (answer == self.reference_answer)
                    reward = 1.0 if correct else -1.0
                    terminated = True
                    obs = result_msg
            else:
                obs = f"Invalid action: {content}"
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
        # 随机采样一个合法动作示例
        choices = ["\\boxed{observe}", "\\boxed{sort}", "\\boxed{find}"]
        # 也可能尝试检查或回答
        if self.problem and self.problem.get("words"):
            w = random.choice(self.problem["words"])
            choices.append(f"\\boxed{{check {w}}}")
            # 随机尝试提交（可能正确也可能错误）
            if random.random() < 0.3:
                ans = random.choice([w, "-", "''"])
                choices.append(f"\\boxed{{answer {ans}}}")
        return random.choice(choices)

    # ---------- 原环境辅助方法（保留并适配内部状态） ----------

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        def can_form_by_deleting(keyword, word):
            it = iter(keyword)
            return all(char in it for char in word)

        words = self.problem.get("words", [])
        keyword = self.problem.get("keyword", "")

        sorted_words = sorted(words, key=lambda w: (-len(w), w))
        for word in sorted_words:
            if can_form_by_deleting(keyword, word):
                return word
        return ""

    def CheckCanForm(self, word: str):
        """
        Check if a word can be formed by deleting some characters from the keyword without changing the order of the characters.

        Args:
            word (str): The word to check.

        Returns:
            str: "True" if it can be formed, "False" otherwise.
        """
        keyword = self.problem.get("keyword", "")
        it = iter(keyword)
        result = all(char in it for char in word)
        return str(result)

    def SortWords(self, word_list: list):
        """
        Sort the list of words first by descending length, then by ascending lexicographical order.

        Args:
            word_list (list[str]): The list of words to sort.

        Returns:
            str: A JSON string of the sorted word list.
        """
        import json
        sorted_words = sorted(word_list, key=lambda w: (-len(w), w))
        return json.dumps(sorted_words)

    def FindFirstValid(self, sorted_words: list):
        """
        Find the first word in the sorted word list that can be formed from the keyword.

        Args:
            sorted_words (list[str]): The sorted list of words.

        Returns:
            str: The found word, or an empty string if none is found.
        """
        keyword = self.problem.get("keyword", "")
        for word in sorted_words:
            it = iter(keyword)
            if all(char in it for char in word):
                return word
        return ""

    def Observe(self):
        """
        Get the keyword and word list in the current environment.

        Returns:
            str: Information containing the keyword and word list.
        """
        import json
        keyword = self.problem.get("keyword", "")
        words = self.problem.get("words", [])
        return f"Keyword: {keyword}, word list: {json.dumps(words)}"

    def Done(self, answer: str):
        """
        Verify if the final answer is correct and return the result information.

        Args:
            answer (str): User-submitted answer string.

        Returns:
            str: Result information, including correctness and reward information.
        """
        ref_answer = self.reference_answer
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        # 奖励由 step 控制，这里不附加 reward 数值
        return msg

    # ---------- 工具方法 ----------

    @staticmethod
    def _safe_json_loads(s: str) -> Optional[list]:
        import json
        try:
            return json.loads(s)
        except Exception:
            return None