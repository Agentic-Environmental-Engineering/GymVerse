from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class PrefixPalindromeEnvGEM(Env):
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

        # 难度参数范围（根据原环境：字符串列表、搜索空间（最大单词长度）、约束数量、字符范围）
        self.complexity_params = {
            "array_length": (5, 50),      # 单词数量
            "search_space": (3, 40),      # 最大单词长度
            "num_constraints": (0, 10),   # 至少含有“长回文前缀”的单词数量
            "value_range": (4, 26),       # 字母表大小（从 'a' 开始的前 N 个字母）
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "search_space": 2,
            "num_constraints": 1,
            "value_range": 0,
        }

        # 占位属性
        self.array_length: int = 0
        self.search_space: int = 0
        self.num_constraints: int = 0
        self.value_range: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题数据
        self.problem: Dict[str, Any] = {"words": []}

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
            "Prefix Palindrome: 对给定的单词列表，计算每个单词的最长回文前缀长度。\n"
            "可用动作（使用 \\boxed{...} 包裹指令）：\n"
            "- 观察列表：\\boxed{observe}\n"
            "- 获取前缀（字符串形式）：\\boxed{get_prefix s=\"abcba\" length=3}\n"
            "- 获取前缀（索引形式）：\\boxed{get_prefix i=0 length=3}\n"
            "- 回文判定（字符串）：\\boxed{check s=\"abba\"}\n"
            "- 提交答案：\\boxed{answer [5, 2, 1]}  注意：答案需为 JSON 数组形式。\n"
            "成功时得分 1.0；错误答案 -1.0；格式错误使用专用惩罚；超时 0.0。\n"
        )

    def get_task_suffix(self) -> str:
        n_words = len(self.problem.get("words", []))
        return (
            f"Words: {n_words} | "
            f"Turn: {self.turn_count}/{self.max_turns} | "
            f"Complexity: {self.complexity}. Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 构建字母表
        alphabet_size = max(1, min(26, self.value_range))
        alphabet = [chr(ord('a') + i) for i in range(alphabet_size)]

        words = []
        # 先生成满足约束的若干回文前缀单词
        constraint_indices = set(random.sample(range(self.array_length), k=min(self.num_constraints, self.array_length)))
        for i in range(self.array_length):
            max_len = random.randint(1, self.search_space)
            if i in constraint_indices and max_len >= 2:
                # 生成一个指定长度的较长回文前缀
                pal_len = random.randint(2, max_len)  # 至少长度为2的回文前缀
                half = [random.choice(alphabet) for _ in range((pal_len + 1) // 2)]
                if pal_len % 2 == 0:
                    pal_prefix = "".join(half + half[::-1])
                else:
                    pal_prefix = "".join(half[:-1] + [half[-1]] + half[:-1][::-1])
                suffix_len = max_len - pal_len
                suffix = "".join(random.choice(alphabet) for _ in range(suffix_len))
                word = pal_prefix + suffix
            else:
                # 普通随机单词
                word = "".join(random.choice(alphabet) for _ in range(max_len))
            words.append(word)

        return {"words": words}

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

        cmd = parsed.get("cmd", "").lower()
        args = parsed.get("args", {})

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd in ["observe"]:
                obs = self.Observe()
                # 观察不终止
                terminated = False
                reward = 0.0

            elif cmd in ["get_prefix", "prefix"]:
                if "i" in args and "length" in args:
                    idx = int(args["i"])
                    length = int(args["length"])
                    words = self.problem.get("words", [])
                    if idx < 0 or idx >= len(words):
                        obs = "Invalid index."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        s = words[idx]
                        obs = self.GetPrefix(s, length)
                        terminated = False
                        reward = 0.0
                elif "s" in args and "length" in args:
                    s = str(args["s"])
                    length = int(args["length"])
                    obs = self.GetPrefix(s, length)
                    terminated = False
                    reward = 0.0
                else:
                    obs = "Error: missing 'i' or 's', and 'length' for get_prefix."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

            elif cmd in ["check_palindrome", "check"]:
                if "s" in args:
                    s = str(args["s"])
                    obs = self.CheckPalindrome(s)
                    terminated = False
                    reward = 0.0
                else:
                    obs = "Error: missing 's' for check_palindrome."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

            elif cmd in ["answer", "done", "submit"]:
                # 解析答案数组（JSON 格式）
                answer_str = args.get("answer_raw", "")
                # 如果 _parse_action 未捕获，尝试原始 content 中的剩余字符串
                if not answer_str and "raw" in parsed:
                    m = re.search(r"\[.*\]", parsed["raw"])
                    if m:
                        answer_str = m.group(0)

                try:
                    answer = json.loads(answer_str)
                except Exception:
                    obs = "Format error: answer must be a JSON array, e.g., [5, 2, 1]."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

                obs = self.Done(answer)
                ref = self.get_ref_answer()
                try:
                    correct = list(answer) == list(ref)
                except Exception:
                    correct = False
                reward = 1.0 if correct else -1.0
                terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Runtime error: {str(e)}"
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
        tokens = self._tokenize_content(content)
        if not tokens:
            return None

        cmd = tokens[0].lower()
        args: Dict[str, Any] = {}

        # 解析 key=value 或者位置参数（仅在需要时）
        # 支持 s="...", i=0, length=3, 以及 answer [..] 的原始载荷
        if cmd in ["get_prefix", "prefix", "check_palindrome", "check"]:
            for t in tokens[1:]:
                if "=" in t:
                    key, val = t.split("=", 1)
                    key = key.strip().lower()
                    val = val.strip()
                    # 去除可能的引号
                    if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                        val = val[1:-1]
                    args[key] = val
        elif cmd in ["answer", "done", "submit"]:
            # 捕获 JSON 数组
            m = re.search(r"\[.*\]", content, re.DOTALL)
            if m:
                args["answer_raw"] = m.group(0)
        elif cmd in ["observe"]:
            pass
        else:
            # 未知命令也返回原始内容以便上层给出 invalid_action
            pass

        return {"cmd": cmd, "args": args, "raw": content}

    def _tokenize_content(self, content: str) -> Optional[list]:
        # 简单分词：按空白分割，但保留引号中的空白
        # 使用正则匹配带引号的字符串或非空白序列
        pattern = re.compile(r"""("[^"]*"|'[^']*'|\S+)""")
        parts = pattern.findall(content)
        return parts if parts else None

    def sample_random_action(self) -> str:
        return "\\boxed{observe}"

    # ===============================
    # 保留原环境的辅助方法并转换
    # ===============================

    def get_ref_answer(self):
        """
        使用环境中的信息获取参考答案。
        返回每个单词的最长回文前缀长度列表。
        """
        words = self.problem.get("words", [])
        result = []
        for word in words:
            max_length = 0
            for i in range(len(word) + 1):
                if word[:i] == word[:i][::-1]:
                    max_length = i
            result.append(max_length)
        return result

    def CheckPalindrome(self, s: str):
        """
        检查字符串 s 是否为回文。
        返回 "True" 或 "False"。
        """
        return str(s == s[::-1])

    def GetPrefix(self, s: str, length: int):
        """
        获取字符串 s 的指定长度前缀。
        """
        return s[:max(0, int(length))]

    def Observe(self):
        """
        返回当前环境中的字符串列表（JSON 字符串）。
        """
        return json.dumps(self.problem.get("words", []))

    def Done(self, answer):
        """
        验证最终答案是否正确并返回结果信息。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        自动调用动作完成任务并提交答案。
        采用观察-枚举前缀-回文判定策略（直接使用内部方法计算，效率更高）。
        """
        words = self.problem.get("words", [])
        answer = []
        for s in words:
            max_len = 0
            # 从长到短尝试前缀
            for length in range(len(s), 0, -1):
                prefix = self.GetPrefix(s, length)
                is_palindrome = self.CheckPalindrome(prefix)
                if is_palindrome == "True":
                    max_len = length
                    break
            answer.append(max_len)
        # 提交答案（返回与 step 一致的验证信息字符串）
        return self.Done(answer)