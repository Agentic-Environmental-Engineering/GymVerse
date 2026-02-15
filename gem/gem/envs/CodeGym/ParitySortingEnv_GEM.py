from typing import Any, Dict, Optional, Tuple
import random
import re
import json
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class ParitySortingEnvGEM(Env):
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

        # 定义难度参数范围（根据原环境分析）
        self.complexity_params = {
            "array_length": (5, 50),  # 数组长度
            "value_max": (10, 10000),  # 数值范围上限（下限为0）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "value_max": 50,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_max: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题与中间变量
        self.problem: Dict[str, Any] = {}
        self.arr: list[int] = []
        self.last_observed: Optional[list] = None
        self.last_evens: Optional[list] = None
        self.last_odds: Optional[list] = None
        self.last_result: Optional[list] = None
        self._terminated: bool = False

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
            "Parity Sorting: Reorder an array so that all even numbers come before odd numbers, preserving relative order.\n"
            "Available actions:\n"
            "- Observe original array: \\boxed{observe}\n"
            "- Collect evens: \\boxed{even} or \\boxed{even [list]}\n"
            "- Collect odds: \\boxed{odd} or \\boxed{odd [list]}\n"
            "- Concatenate lists: \\boxed{concat} (uses last evens and odds) or \\boxed{concat [list1] [list2]}\n"
            "- Submit answer: \\boxed{answer [list]} or \\boxed{answer} (uses last result)\n"
        )

    def get_task_suffix(self) -> str:
        info_parts = [
            f"Array length: {len(self.arr)}",
            f"Turn: {self.turn_count}/{self.max_turns}",
        ]
        state_info = []
        if self.last_evens is not None:
            state_info.append(f"evens={len(self.last_evens)}")
        if self.last_odds is not None:
            state_info.append(f"odds={len(self.last_odds)}")
        if self.last_result is not None:
            state_info.append(f"result_len={len(self.last_result)}")
        if state_info:
            info_parts.append("State: " + ", ".join(state_info))
        return "\n".join(info_parts)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        self._terminated = False

        # 重置中间变量
        self.arr = self.problem["array"]
        self.last_observed = None
        self.last_evens = None
        self.last_odds = None
        self.last_result = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        length = max(1, self.array_length)
        data = [random.randint(0, self.value_max) for _ in range(length)]
        return {"array": data, "size": length, "value_max": self.value_max}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        # 若已经终止，拒绝进一步动作
        if self._terminated:
            obs = "Episode already terminated. Please reset."
            return obs, LanguageGameReward.invalid_action_reward, True, False, {
                "suffix": self.get_task_suffix()
            }

        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            self._terminated = True
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = parsed.get("cmd")
        args = parsed.get("args", [])

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd == "observe":
                obs = self.Observe()
                # 更新 last_observed
                self.last_observed = ast.literal_eval(obs)

            elif cmd == "even":
                # 支持：even 或 even [list]
                target_list = None
                if args:
                    target_list = self._parse_list(args[0])
                else:
                    target_list = self.last_observed if self.last_observed is not None else self.arr
                obs = self.CollectEvenNumbers(target_list)
                self.last_evens = ast.literal_eval(obs)

            elif cmd == "odd":
                target_list = None
                if args:
                    target_list = self._parse_list(args[0])
                else:
                    target_list = self.last_observed if self.last_observed is not None else self.arr
                obs = self.CollectOddNumbers(target_list)
                self.last_odds = ast.literal_eval(obs)

            elif cmd == "concat":
                # 支持：concat 或 concat [list1] [list2]
                if len(args) >= 2:
                    list1 = self._parse_list(args[0])
                    list2 = self._parse_list(args[1])
                else:
                    if self.last_evens is None or self.last_odds is None:
                        self._terminated = True
                        return (
                            "Error: concat requires two lists. You can use previous 'even' and 'odd' results or provide two lists.",
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    list1 = self.last_evens
                    list2 = self.last_odds
                obs = self.ConcatenateLists(list1, list2)
                self.last_result = ast.literal_eval(obs)

            elif cmd == "answer":
                # 支持：answer 或 answer [list]
                if args:
                    answer_list = self._parse_list(args[0])
                else:
                    if self.last_result is None:
                        self._terminated = True
                        return (
                            "Error: No result found. Use 'concat' or provide an explicit list in answer.",
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    answer_list = self.last_result

                done_msg = self.Done(answer_list)
                obs = done_msg
                ref_answer = self.get_ref_answer()
                correct = answer_list == ref_answer
                reward = 1.0 if correct else -1.0
                terminated = True
                self._terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
                self._terminated = True

        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True
            self._terminated = True

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True
            self._terminated = True

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()

        # 命令解析
        # 支持：observe | even [list] | odd [list] | concat [list1] [list2] | answer [list]
        # 列表用 Python 风格，如 [1, 2, 3]
        tokens = self._split_tokens_preserve_lists(content)
        if not tokens:
            return None

        cmd = tokens[0].lower()
        args = tokens[1:]
        return {"cmd": cmd, "args": args, "content": content}

    def _split_tokens_preserve_lists(self, content: str) -> list:
        """
        将内容拆分为令牌，保留方括号列表为一个整体令牌。
        例如：'concat [1,2] [3,4]' -> ['concat', '[1,2]', '[3,4]']
        """
        tokens = []
        i = 0
        n = len(content)
        while i < n:
            if content[i].isspace():
                i += 1
                continue
            if content[i] == '[':
                # 捕获列表
                bracket = 1
                j = i + 1
                while j < n and bracket > 0:
                    if content[j] == '[':
                        bracket += 1
                    elif content[j] == ']':
                        bracket -= 1
                    j += 1
                tokens.append(content[i:j])
                i = j
            else:
                # 常规单词
                j = i + 1
                while j < n and not content[j].isspace():
                    j += 1
                tokens.append(content[i:j])
                i = j
        return tokens

    def _parse_list(self, s: str) -> list:
        """
        从字符串解析 Python 风格列表。
        """
        try:
            obj = ast.literal_eval(s)
            if not isinstance(obj, list):
                raise ValueError("not a list")
            return obj
        except Exception:
            raise ValueError(f"Invalid list format: {s}")

    def sample_random_action(self) -> str:
        return "\\boxed{observe}"

    # 保留原环境的辅助方法并转换

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer. 
        """
        even_numbers = [num for num in self.arr if num % 2 == 0]
        odd_numbers = [num for num in self.arr if num % 2 != 0]
        return even_numbers + odd_numbers

    def CollectEvenNumbers(self, array: list):
        """
        Collect all even numbers from the given array while maintaining their relative order.

        Args:
            array (list[int]): The integer array to process.

        Returns:
            str: String representation of the list containing all even numbers.

        Example Output:
            "[2, 4, 6]"
        """
        even_numbers = [num for num in array if num % 2 == 0]
        return json.dumps(even_numbers)

    def CollectOddNumbers(self, array: list):
        """
        Collect all odd numbers from the given array while maintaining their relative order.

        Args:
            array (list[int]): The integer array to process.

        Returns:
            str: String representation of the list containing all odd numbers.

        Example Output:
            "[1, 3, 5]"
        """
        odd_numbers = [num for num in array if num % 2 != 0]
        return json.dumps(odd_numbers)

    def ConcatenateLists(self, list1: list, list2: list):
        """
        Concatenate two lists, with elements of the first list followed by elements of the second list.

        Args:
            list1 (list): The first list.
            list2 (list): The second list.

        Returns:
            str: String representation of the concatenated list.

        Example Output:
            "[2, 4, 1, 3]"
        """
        concatenated = list1 + list2
        return json.dumps(concatenated)

    def Observe(self):
        """
        Get the original array in the current environment.

        Returns:
            str: String representation of the original array in the current environment.

        Example Output:
            "[3, 1, 2, 4, 5]"
        """
        return json.dumps(self.arr)

    def Done(self, answer):
        """
        Verify if the final answer is correct and return the result information. 

        Args:
            answer (list[int]): The sorted array submitted by the user. 

        Returns:
            str: Result information, including correctness and reward information. 

        Example Output:
            "Your answer: [2, 4, 3, 1, 5], Reference answer: [2, 4, 3, 1, 5], Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        # 奖励由 step 统一控制，这里仅返回信息
        return msg

    def solve(self):
        """
        Automatically call actions to complete the process, and submit the answer for verification. 
    
        Returns:
            str: The result information of the final answer verification. 
        """
        # Observe
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        original_array = ast.literal_eval(obs)

        # Evens
        evens_str, _, _, _, _ = self.step("\\boxed{even}")
        evens = ast.literal_eval(evens_str)

        # Odds
        odds_str, _, _, _, _ = self.step("\\boxed{odd}")
        odds = ast.literal_eval(odds_str)

        # Concat
        result_str, _, _, _, _ = self.step("\\boxed{concat}")
        _ = ast.literal_eval(result_str)

        # Answer (use last result)
        final_obs, reward, terminated, truncated, _ = self.step("\\boxed{answer}")
        # 返回最终验证信息
        return final_obs