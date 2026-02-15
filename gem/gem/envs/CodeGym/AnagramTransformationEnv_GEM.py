from typing import Any, Dict, Optional, Tuple
import random
import re
import json
import string
from collections import Counter
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class AnagramTransformationEnvGEM(Env):
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

        # 定义难度参数范围（根据原环境任务分析）
        self.complexity_params: Dict[str, Tuple[int, int]] = {
            # 字符串长度（等长情况下）
            "string_length": (5, 50),
            # 字母表大小（从小到大）
            "alphabet_size": (2, 26),
            # 目标差异比率（百分比，控制需要的更改数）
            "delta_ratio_percent": (0, 100),
            # 长度不匹配概率（百分比）
            "mismatch_chance_percent": (0, 40),
        }

        # 参数方差（用于训练时增加多样性）
        self.param_variance: Dict[str, int] = {
            "string_length": 3,
            "alphabet_size": 2,
            "delta_ratio_percent": 10,
            "mismatch_chance_percent": 10,
        }

        # 占位属性
        self.string_length: int = 0
        self.alphabet_size: int = 0
        self.delta_ratio_percent: int = 0
        self.mismatch_chance_percent: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.S: str = ""
        self.T: str = ""

        # 记忆最近一次计数结果，供 calculate_changes 无参数使用
        self._last_count_s: Optional[Dict[str, int]] = None
        self._last_count_t: Optional[Dict[str, int]] = None

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
            "Anagram Transformation: Compute the minimum number of changes needed to transform the character frequency of S to that of T.\n"
            "Rules:\n"
            "- If |S| != |T|, the correct answer is -1.\n"
            "- Otherwise, the answer is the sum over characters where count_T[c] > count_S[c] of (count_T[c] - count_S[c]).\n"
            "Available actions (use the latest \\boxed{...} command in your message):\n"
            "- Observe strings: \\boxed{observe}\n"
            "- Check lengths: \\boxed{check_lengths}\n"
            "- Count characters of S or T: \\boxed{count S} or \\boxed{count T}\n"
            "- Calculate changes (uses last counted S/T if available): \\boxed{calculate_changes}\n"
            "- Submit answer: \\boxed{answer N}\n"
            "Output format for observations is JSON strings where applicable."
        )

    def get_task_suffix(self) -> str:
        s_len = len(self.S) if self.S is not None else 0
        t_len = len(self.T) if self.T is not None else 0
        return (
            f"Turn: {self.turn_count}/{self.max_turns} | "
            f"S length: {s_len}, T length: {t_len} | "
            f"Complexity: {self.complexity}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.S = self.problem["S"]
        self.T = self.problem["T"]

        self._last_count_s = None
        self._last_count_t = None

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        # 构造用于 S 的字母表
        all_letters = list(string.ascii_lowercase)
        random.shuffle(all_letters)
        alphabet_for_s = all_letters[: max(1, self.alphabet_size)]

        # 生成 S
        s_len = max(1, self.string_length)
        S_chars = [random.choice(alphabet_for_s) for _ in range(s_len)]
        S = "".join(S_chars)

        # 决定是否长度不匹配
        mismatch = random.randint(1, 100) <= max(0, min(100, self.mismatch_chance_percent))

        # 生成 T
        if mismatch:
            # 通过偏移量改变长度
            max_offset = max(1, int(0.3 * s_len))
            offset = random.randint(1, max_offset)
            # 随机决定增加或减少长度（确保长度至少为 1）
            if random.random() < 0.5:
                t_len = max(1, s_len - offset)
            else:
                t_len = s_len + offset
            # 重新生成一个随机的 T
            T = "".join(random.choice(all_letters) for _ in range(t_len))
        else:
            # 等长，控制差异度（目标更改数）
            target_delta = int(round(s_len * max(0, min(100, self.delta_ratio_percent)) / 100.0))
            target_delta = max(0, min(s_len, target_delta))

            # 起始 T 为 S 的副本
            T_list = list(S_chars)

            # 优先选择不在 S 中的字母作为替换，提高更改数的确定性
            letters_not_in_s = [ch for ch in all_letters if ch not in set(alphabet_for_s)]
            for _ in range(target_delta):
                pos = random.randrange(s_len)
                if letters_not_in_s:
                    new_char = random.choice(letters_not_in_s)
                else:
                    # 如果 S 已覆盖所有字母，则选择 S 中的某个字母也能增加更改数
                    new_char = random.choice(alphabet_for_s)
                T_list[pos] = new_char
            T = "".join(T_list)

        return {"S": S, "T": T}

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
            elif lower == "check_lengths":
                obs = self.CheckLengths()
            elif lower.startswith("count"):
                parts = content.split()
                if len(parts) != 2 or parts[1] not in ("S", "T"):
                    obs = "Invalid count action. Use 'count S' or 'count T'."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                target = parts[1]
                obs = self.CountCharacters(target)
                try:
                    count_dict = json.loads(obs)
                    if target == "S":
                        self._last_count_s = count_dict
                    else:
                        self._last_count_t = count_dict
                except Exception:
                    # 如果输出不是有效 JSON（理论上不会发生），标记为格式错误
                    return (
                        "CountCharacters output JSON parse error.",
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
            elif lower.startswith("calculate_changes"):
                # 支持两种方式：
                # 1) 无参数，使用最近一次计数的 S/T
                # 2) 带两个 JSON dict：calculate_changes {..} {..}
                dict_matches = list(re.finditer(r"\{.*?\}", content, re.DOTALL))
                if len(dict_matches) >= 2:
                    try:
                        count_s_str = dict_matches[0].group(0)
                        count_t_str = dict_matches[1].group(0)
                        count_s = json.loads(count_s_str)
                        count_t = json.loads(count_t_str)
                        obs = self.CalculateChanges(count_s, count_t)
                    except Exception:
                        return (
                            "calculate_changes JSON parameters parse error.",
                            LanguageGameReward.format_error_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                else:
                    if self._last_count_s is None or self._last_count_t is None:
                        obs = "Error: missing counts. Run 'count S' and 'count T' first or provide JSON dicts."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    obs = self.CalculateChanges(self._last_count_s, self._last_count_t)
            elif lower.startswith("answer"):
                # 解析 answer N
                m = re.match(r"answer\s+(-?\d+)", lower)
                if not m:
                    obs = "Invalid answer format. Use \\boxed{answer N}."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                answer_val = int(m.group(1))
                # 验证答案
                msg = self.Done(answer_val)
                ref = self.get_ref_answer()
                correct = (answer_val == ref)
                reward = 1.0 if correct else -1.0
                obs = msg
                terminated = True
            else:
                obs = "Invalid action."
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
        # 随机选择一个动作示例
        choices = [
            "\\boxed{observe}",
            "\\boxed{check_lengths}",
            "\\boxed{count S}",
            "\\boxed{count T}",
            "\\boxed{calculate_changes}",
            "\\boxed{answer -1}",
        ]
        return random.choice(choices)

    # =========================
    # 原环境辅助方法（转换保留）
    # =========================
    def get_ref_answer(self) -> int:
        if len(self.S) != len(self.T):
            return -1

        count_s = Counter(self.S)
        count_t = Counter(self.T)

        if count_s == count_t:
            return 0

        changes_needed = 0
        for char in count_t:
            if char in count_s:
                if count_t[char] > count_s[char]:
                    changes_needed += count_t[char] - count_s[char]
            else:
                changes_needed += count_t[char]

        return changes_needed

    def CheckLengths(self) -> str:
        r"""
        Check if the lengths of strings S and T are the same.

        Returns:
            str: A JSON string indicating whether the lengths are the same, containing the "same_length" key with a boolean value.

        Example Output:
            "{\"same_length\": true}"
        """
        same_length = len(self.S) == len(self.T)
        return json.dumps({"same_length": same_length})

    def CountCharacters(self, string_name: str) -> str:
        r"""
        Count the frequency of each character in the specified string.

        Args:
            string_name (str): The name of the string to count, either "S" or "T".

        Returns:
            str: A JSON string containing the character frequencies.

        Example Output:
            "{\"a\": 2, \"b\": 2}"
        """
        if string_name == "S":
            count = Counter(self.S)
        elif string_name == "T":
            count = Counter(self.T)
        else:
            return json.dumps({"error": "Invalid string name. Use 'S' or 'T'."})
        return json.dumps(dict(count))

    def CalculateChanges(self, count_s: Dict[str, int], count_t: Dict[str, int]) -> str:
        r"""
        Calculate the minimum number of changes needed to convert the character frequency of S to that of T.

        Args:
            count_s (dict): The character frequency dictionary of string S.
            count_t (dict): The character frequency dictionary of string T.

        Returns:
            str: A JSON string containing the number of changes needed.

        Example Output:
            "{\"changes_needed\": 3}"
        """
        if count_s == count_t:
            return json.dumps({"changes_needed": 0})

        changes_needed = 0
        for char in count_t:
            count_t_char = count_t[char]
            count_s_char = count_s.get(char, 0)
            if count_t_char > count_s_char:
                changes_needed += count_t_char - count_s_char

        return json.dumps({"changes_needed": changes_needed})

    def Observe(self) -> str:
        r"""
        Return the current strings S and T in the environment.

        Returns:
            str: A JSON string containing S and T.

        Example Output:
            "{\"S\": \"abab\", \"T\": \"baba\"}"
        """
        return json.dumps({"S": self.S, "T": self.T})

    def Done(self, answer: int) -> str:
        r"""
        Verify if the final answer is correct and return the result information.

        Args:
            answer (int): User-submitted answer, i.e., the minimum number of operations or -1.

        Returns:
            str: Result information, including correctness.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        r"""
        Automatically call actions to complete the process and submit the answer for verification.

        Returns:
            str: The result information of the final answer verification.
        """
        # Check lengths
        obs, _, _, _, _ = self.step("\\boxed{check_lengths}")
        try:
            length_data = json.loads(obs)
        except Exception:
            # 如果解析失败，尝试 observe 再计算
            length_data = {"same_length": len(self.S) == len(self.T)}

        if not length_data.get("same_length", False):
            obs, reward, terminated, _, _ = self.step("\\boxed{answer -1}")
            return obs

        # Count S and T
        obs_s, _, _, _, _ = self.step("\\boxed{count S}")
        count_s = json.loads(obs_s)
        obs_t, _, _, _, _ = self.step("\\boxed{count T}")
        count_t = json.loads(obs_t)

        # Calculate changes
        calc_action = f"\\boxed{{calculate_changes {json.dumps(count_s)} {json.dumps(count_t)}}}"
        obs_calc, _, _, _, _ = self.step(calc_action)
        changes_data = json.loads(obs_calc)
        changes_needed = changes_data.get("changes_needed", 0)

        # Submit answer
        obs_final, _, _, _, _ = self.step(f"\\boxed{{answer {changes_needed}}}")
        return obs_final