from typing import Any, Dict, Optional, Tuple
import random
import re
from collections import deque
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinWordsToFormTargetEnvGEM(Env):
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
            "dictionary_size": (5, 50),     # 词典规模
            "target_length_hint": (5, 50),  # 目标长度提示（用于生成近似长度）
            "alphabet_size": (2, 8),        # 使用字母数量
            "max_word_length": (3, 12),     # 单词最大长度
        }

        # 参数方差
        self.param_variance = {
            "dictionary_size": 2,
            "target_length_hint": 3,
            "alphabet_size": 1,
            "max_word_length": 2,
        }

        # 占位属性
        self.dictionary_size: int = 0
        self.target_length_hint: int = 0
        self.alphabet_size: int = 0
        self.max_word_length: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题数据
        self.words: list[str] = []
        self.target: str = ""
        self.queue: Optional[deque] = None
        self.visited: set[str] = set()

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
            "Min Words to Form Target (GEM): Compute minimal number of words to form the target by repeatedly removing a prefix word.\n"
            "You control a BFS-like process via actions.\n"
            "Available actions (use \\boxed{...}):\n"
            "- Observe problem: \\boxed{observe}\n"
            "- Initialize BFS: \\boxed{init}\n"
            "- Check queue empty: \\boxed{is_empty}\n"
            "- Pop queue: \\boxed{dequeue}\n"
            "- Check prefix: \\boxed{check WORD CURRENT_STRING}\n"
            "- Get remaining string: \\boxed{remain WORD CURRENT_STRING}\n"
            "- Add to queue: \\boxed{add_queue STRING STEPS}\n"
            "- Add to visited: \\boxed{add_visited STRING}\n"
            "- Check visited: \\boxed{is_visited STRING}\n"
            "- Submit answer: \\boxed{answer N}\n"
            "Goal: Provide the minimal number of words (or -1 if impossible).\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Words count: {len(self.words)}, Target length: {len(self.target)}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter an action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 从生成的实例中设置环境状态
        self.words = self.problem["words"]
        self.target = self.problem["target"]
        self.queue = None
        self.visited = set()

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 生成字母表
        base_alphabet = [chr(ord('a') + i) for i in range(26)]
        alphabet = random.sample(base_alphabet, k=min(self.alphabet_size, len(base_alphabet)))

        # 生成词典
        words_set = set()
        while len(words_set) < self.dictionary_size:
            length = random.randint(1, max(1, self.max_word_length))
            word = "".join(random.choice(alphabet) for _ in range(length))
            words_set.add(word)
        words = sorted(words_set)

        # 生成目标：70% 可解，30% 不可解
        solvable = random.random() < 0.7
        # 估计段数
        avg_len = max(1, sum(len(w) for w in random.sample(words, k=min(5, len(words)))) // min(5, len(words)))
        estimated_segments = max(1, int(round(self.target_length_hint / max(1, avg_len))))
        estimated_segments = max(1, min(estimated_segments, self.target_length_hint))  # 保守

        segments = []
        if len(words) == 0:
            # 极端情况，空词典
            target = ""
        else:
            for _ in range(estimated_segments):
                segments.append(random.choice(words))
            target = "".join(segments)

        if not solvable:
            # 添加一个不在词典字母表中的字符，确保不可解
            other_chars = [c for c in base_alphabet if c not in alphabet]
            if other_chars:
                target += random.choice(other_chars)
            else:
                # 没有可用新字母，则添加一个特殊字符
                target += "!"

        return {"words": words, "target": target, "solvable": solvable}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        # 格式错误
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
        tokens = parsed["tokens"]
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            cmd = tokens[0].lower()

            if cmd == "observe":
                obs = self.Observe()

            elif cmd == "init":
                # 初始化队列和已访问
                obs = self.InitQueue(self.target, 0)
                obs2 = self.AddToVisited(self.target)
                obs = f"{obs}\n{obs2}"

            elif cmd == "is_empty":
                obs = self.IsQueueEmpty()

            elif cmd == "dequeue":
                obs = self.Dequeue()

            elif cmd == "check":
                # check WORD CURRENT_STRING
                if len(tokens) < 3:
                    obs = "Error: 'check' requires WORD and CURRENT_STRING."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    word = self._strip_quotes(tokens[1])
                    current_string = self._strip_quotes(" ".join(tokens[2:]))
                    obs = self.CheckPrefix(word, current_string)

            elif cmd == "remain":
                # remain WORD CURRENT_STRING
                if len(tokens) < 3:
                    obs = "Error: 'remain' requires WORD and CURRENT_STRING."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    word = self._strip_quotes(tokens[1])
                    current_string = self._strip_quotes(" ".join(tokens[2:]))
                    obs = self.GetRemainingString(word, current_string)

            elif cmd == "add_queue":
                # add_queue STRING STEPS
                if len(tokens) < 3:
                    obs = "Error: 'add_queue' requires STRING and STEPS."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    remaining_string = self._strip_quotes(tokens[1])
                    try:
                        steps = int(self._strip_quotes(tokens[2]))
                    except ValueError:
                        obs = "Error: STEPS must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.AddToQueue(remaining_string, steps)

            elif cmd == "add_visited":
                # add_visited STRING
                if len(tokens) < 2:
                    obs = "Error: 'add_visited' requires STRING."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    remaining_string = self._strip_quotes(tokens[1])
                    obs = self.AddToVisited(remaining_string)

            elif cmd == "is_visited":
                # is_visited STRING
                if len(tokens) < 2:
                    obs = "Error: 'is_visited' requires STRING."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    remaining_string = self._strip_quotes(tokens[1])
                    obs = self.IsVisited(remaining_string)

            elif cmd == "answer":
                # answer N
                if len(tokens) < 2:
                    obs = "Error: 'answer' requires N."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        answer_val = int(self._strip_quotes(tokens[1]))
                    except ValueError:
                        obs = "Error: N must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        msg, success = self._evaluate_answer(answer_val)
                        obs = msg
                        reward = 1.0 if success else -1.0
                        terminated = True

            else:
                obs = f"Invalid action '{cmd}'."
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
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        tokens = content.split()
        if not tokens:
            return None
        return {"content": content, "tokens": tokens}

    def sample_random_action(self) -> str:
        # 简单随机动作示例
        options = [
            "\\boxed{observe}",
            "\\boxed{init}",
            "\\boxed{is_empty}",
            "\\boxed{answer -1}",
        ]
        return random.choice(options)

    # 评估答案
    def _evaluate_answer(self, answer: int) -> Tuple[str, bool]:
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, correct

    # 保留并转换原环境的辅助方法

    def get_ref_answer(self) -> int:
        """
        使用 BFS 获取参考答案（最少词数），不可形成时返回 -1
        """
        queue = deque([(self.target, 0)])
        visited = set([self.target])

        while queue:
            current, steps = queue.popleft()

            if current == "":
                return steps

            for word in self.words:
                if current.startswith(word):
                    new_target = current[len(word):]
                    if new_target not in visited:
                        visited.add(new_target)
                        queue.append((new_target, steps + 1))

        return -1

    def CheckPrefix(self, word: str, current_string: str) -> str:
        r"""
        检查 word 是否为 current_string 的前缀。
        返回 "True" 或 "False"。
        """
        return str(current_string.startswith(word))

    def GetRemainingString(self, word: str, current_string: str) -> str:
        r"""
        返回移除前缀 word 后的剩余字符串。
        如果 word 不是前缀，则返回原字符串。
        """
        if current_string.startswith(word):
            return current_string[len(word):]
        return current_string

    def InitQueue(self, initial_string: str, initial_steps: int) -> str:
        r"""
        初始化队列，包含初始字符串与步数。
        """
        self.queue = deque([(initial_string, initial_steps)])
        return f"Queue initialized with {list(self.queue)}"

    def Dequeue(self) -> str:
        r"""
        从队列中移除并返回首元素，格式为 "('str', steps)"；若空返回 "None"。
        """
        if self.queue and len(self.queue) > 0:
            item = self.queue.popleft()
            return f"('{item[0]}', {item[1]})"
        return "None"

    def IsQueueEmpty(self) -> str:
        r"""
        检查队列是否为空，返回 "True"/"False"。
        """
        return str(not self.queue or len(self.queue) == 0)

    def AddToQueue(self, remaining_string: str, steps: int) -> str:
        r"""
        将 (remaining_string, steps) 加入队列，并返回队列状态描述。
        """
        if not self.queue:
            self.queue = deque()
        self.queue.append((remaining_string, steps))
        return f"Added ('{remaining_string}', {steps}) to queue. Queue now: {list(self.queue)}"

    def AddToVisited(self, remaining_string: str) -> str:
        r"""
        将 remaining_string 加入 visited 集合，返回集合大小描述。
        """
        self.visited.add(remaining_string)
        return f"Added '{remaining_string}' to visited. Visited size: {len(self.visited)}"

    def IsVisited(self, remaining_string: str) -> str:
        r"""
        检查 remaining_string 是否已访问，返回 "True"/"False"。
        """
        return str(remaining_string in self.visited)

    def Observe(self) -> str:
        r"""
        返回观察信息：词典和目标字符串。
        """
        return f"Words: {self.words}, Target: '{self.target}'"

    # 工具：去除可选引号
    @staticmethod
    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
            return s[1:-1]
        return s