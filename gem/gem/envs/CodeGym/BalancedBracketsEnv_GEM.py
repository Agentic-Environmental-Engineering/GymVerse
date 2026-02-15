from typing import Any, Dict, Optional, Tuple
import random
import re
import json
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class BalancedBracketsEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 7,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100
        self._max_turns_user_overridden = max_turns is not None

        # 定义难度参数范围
        self.complexity_params = {
            "string_length": (6, 120),     # 字符串总长度（含噪声）
            "bracket_types": (1, 3),       # 使用的括号类型数：1-3
            "noise_ratio": (0, 30),        # 噪声百分比（非括号字符占比，0-30%）
            "imbalance_rate": (20, 50),    # 生成不平衡串的概率百分比（20%-50%）
            "max_turns": (20, 200),        # 最大步数限制（可由复杂度控制）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "string_length": 3,
            "bracket_types": 1,
            "noise_ratio": 5,
            "imbalance_rate": 5,
            "max_turns": 10,
        }

        # 占位属性
        self.string_length: int = 0
        self.bracket_types: int = 0
        self.noise_ratio: int = 0
        self.imbalance_rate: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.s: str = ""
        self.stack: list = []
        self.cursor: int = 0
        self._imbalance_detected: bool = False

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

            val = int(round(actual_value))
            if param_name == "max_turns":
                # 仅当用户未显式覆盖时由复杂度控制
                if not self._max_turns_user_overridden:
                    self.max_turns = val
            elif param_name == "string_length":
                self.string_length = val
            elif param_name == "bracket_types":
                self.bracket_types = max(1, min(3, val))
            elif param_name == "noise_ratio":
                self.noise_ratio = max(0, min(100, val))
            elif param_name == "imbalance_rate":
                self.imbalance_rate = max(0, min(100, val))

    def _get_instructions(self) -> str:
        return (
            "Balanced Brackets: Determine whether the given string has balanced brackets.\n"
            "Available actions:\n"
            "- Observe the string: \\boxed{observe}\n"
            "- Process next character: \\boxed{process next}\n"
            "- Process a specific character: \\boxed{process X} (X is a single character)\n"
            "- Check final stack: \\boxed{check}\n"
            "- Submit answer: \\boxed{answer true} or \\boxed{answer false}\n"
            "Notes:\n"
            "- You may process characters one by one to simulate stack behavior.\n"
            "- Non-bracket characters are ignored in processing.\n"
            "- Reward: success=1.0, failure=-1.0, format error/invalid action use LanguageGameReward constants.\n"
        )

    def get_task_suffix(self) -> str:
        types = "()" if self.bracket_types == 1 else ("(){}" if self.bracket_types == 2 else "(){}[]")
        return (
            f"Length: {len(self.s)} | Types: {types} | "
            f"Processed: {self.cursor}/{len(self.s)} | Stack size: {len(self.stack)} | "
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

        self.s = self.problem["s"]
        self.turn_count = 0
        self.stack = []
        self.cursor = 0
        self._imbalance_detected = False
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 可用括号集合
        all_types = [
            ("(", ")"),
            ("{", "}"),
            ("[", "]"),
        ]
        types = all_types[: self.bracket_types]
        opens = [t[0] for t in types]
        closes = [t[1] for t in types]

        total_len = self.string_length
        noise_pct = self.noise_ratio / 100.0
        noise_len = int(round(total_len * noise_pct))
        bracket_len = max(0, total_len - noise_len)

        # 生成平衡的括号序列
        bracket_seq = []
        stack = []
        # 随机生成 bracket_len 个字符，确保最终可闭合
        for _ in range(bracket_len):
            # 决定是开还是关：如果栈为空，一定开；否则随机
            do_open = (len(stack) == 0) or (random.random() < 0.6)
            if do_open:
                t = random.choice(types)
                bracket_seq.append(t[0])
                stack.append(t)
            else:
                # 关闭栈顶
                t = stack.pop()
                bracket_seq.append(t[1])

        # 如果仍有未关闭的，全部关闭
        while stack:
            t = stack.pop()
            bracket_seq.append(t[1])

        # 如果长度超出 bracket_len，截断（很少发生）
        bracket_seq = bracket_seq[:bracket_len]

        # 插入噪声字符
        noise_chars_pool = "abcdefgXYZ012345!@#$%^&*"
        noise_seq = [random.choice(noise_chars_pool) for _ in range(noise_len)]

        # 合并并打乱位置：我们将 bracket_seq 与 noise_seq 交织
        combined = bracket_seq + noise_seq
        random.shuffle(combined)

        # 生成不平衡字符串的概率
        make_imbalanced = random.randint(1, 100) <= self.imbalance_rate
        if make_imbalanced and bracket_seq:
            # 简单破坏策略：随机替换一个括号为错误类型或删除一个字符
            idxs = [i for i, ch in enumerate(combined) if ch in opens + closes]
            if idxs:
                i = random.choice(idxs)
                ch = combined[i]
                if ch in opens:
                    # 替换为另一个开括号或随机噪声
                    candidates = [o for o in opens if o != ch]
                    if candidates:
                        combined[i] = random.choice(candidates)
                    else:
                        combined[i] = random.choice(noise_chars_pool)
                else:
                    # 替换为错误的闭括号或删除
                    candidates = [c for c in closes if c != ch]
                    if candidates and random.random() < 0.7:
                        combined[i] = random.choice(candidates)
                    else:
                        del combined[i]
            # 若没有括号字符，则追加一个多余的开括号以破坏平衡
            else:
                combined.append(random.choice(opens))

        s = "".join(combined)
        return {"s": s}

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
        cmd = parsed.get("cmd")
        args = parsed.get("args", [])

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        if cmd == "observe":
            obs = self.Observe()
        elif cmd == "process":
            if not args:
                obs = "Invalid process usage. Use \\boxed{process next} or \\boxed{process X}."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            arg = args[0]
            if arg.lower() == "next":
                if self.cursor >= len(self.s):
                    obs = "No more characters to process."
                else:
                    ch = self.s[self.cursor]
                    result = self.ProcessCharacter(ch, self.stack)
                    if result == "false":
                        self._imbalance_detected = True
                        obs = "false"
                    else:
                        try:
                            self.stack = ast.literal_eval(result)
                            obs = result
                        except Exception:
                            # 如果解析失败，视为格式错误
                            obs = "Error: internal stack parse failure."
                            return (
                                obs,
                                LanguageGameReward.format_error_reward,
                                True,
                                False,
                                {"suffix": self.get_task_suffix()},
                            )
                    self.cursor += 1
            else:
                # 显式处理一个字符，不移动 cursor
                ch = arg
                if len(ch) != 1:
                    obs = "Invalid character for process. Use a single character."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                result = self.ProcessCharacter(ch, self.stack)
                if result == "false":
                    self._imbalance_detected = True
                    obs = "false"
                else:
                    try:
                        self.stack = ast.literal_eval(result)
                        obs = result
                    except Exception:
                        obs = "Error: internal stack parse failure."
                        return (
                            obs,
                            LanguageGameReward.format_error_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
        elif cmd == "check":
            obs = self.CheckFinalStack(self.stack)
        elif cmd == "answer":
            if not args:
                obs = "Answer requires a boolean: \\boxed{answer true} or \\boxed{answer false}."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            ans_token = args[0].lower()
            if ans_token not in ("true", "false"):
                obs = "Answer must be 'true' or 'false'."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            answer_bool = ans_token == "true"
            msg = self.Done(answer_bool)
            ref = self.get_ref_answer()
            obs = msg
            reward = 1.0 if answer_bool == ref else -1.0
            terminated = True
            truncated = False
        else:
            obs = f"Invalid action: {content}"
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
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].lower()
        args = tokens[1:]
        return {"content": content, "cmd": cmd, "args": args}

        # All the actions of the environment

    def ProcessCharacter(self, char: str, stack: list):
        r"""
        Processes a single character and updates the stack state according to the bracket type.

        Args:
            char (str): The character to be processed
            stack (list): The current stack state

        Returns:
            str: The processed stack state or "false" (if imbalance is detected)

        Example Output:
            "[ '(', '{' ]" or "false"
        """
        bracket_map = {
            ')': '(',
            '}': '{',
            ']': '['
        }

        if char in bracket_map.values():
            new_stack = stack.copy()
            new_stack.append(char)
            return json.dumps(new_stack)
        elif char in bracket_map:
            if not stack or bracket_map[char] != stack[-1]:
                return "false"
            new_stack = stack.copy()
            new_stack.pop()
            return json.dumps(new_stack)
        return json.dumps(stack)

    def CheckFinalStack(self, stack: list):
        r"""
        Checks if the final stack is empty; if empty, the brackets are balanced.

        Args:
            stack (list): The stack state after processing all characters

        Returns:
            str: "true" (if the stack is empty) or "false" (if the stack is not empty)

        Example Output:
            "true"
        """
        return "true" if not stack else "false"

    def Observe(self):
        r"""
        Obtains the string in the current environment.

        Args:
            None

        Returns:
            str: The string in the current environment

        Example Output:
            "([{}])"
        """
        return self.s

    def Done(self, answer):
        r"""
        Verifies whether the final answer is correct and returns result information.

        Args:
            answer (bool): The answer submitted by the user

        Returns:
            str: Result information, including correctness and reward details

        Example Output:
            "Your answer: true, Reference answer: true, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer.
        """
        stack = []
        bracket_map = {
            ')': '(',
            '}': '{',
            ']': '['
        }

        for char in self.s:
            if char in bracket_map.values():
                stack.append(char)
            elif char in bracket_map:
                if stack == [] or bracket_map[char] != stack.pop():
                    return False

        return stack == []

    def solve(self) -> str:
        r"""
        Automatically calls actions in the environment to complete bracket balance check and submit the answer for verification.

        Returns:
            str: The result information of the final answer verification.
        """
        # Observe s
        _ = self.step("\\boxed{observe}")[0]
        # Process all characters
        for _ in range(len(self.s)):
            obs, _, term, _, _ = self.step("\\boxed{process next}")
            if term:
                # If terminated early due to format/invalid error, return obs
                return obs
            if obs == "false":
                # Imbalance detected, answer false
                result = self.step("\\boxed{answer false}")[0]
                return result
        # Final check
        final_check = self.step("\\boxed{check}")[0]
        answer = (final_check == "true")
        result = self.step(f"\\boxed{{answer {'true' if answer else 'false'}}}")[0]
        return result

    def sample_random_action(self) -> str:
        # Simple random sampler for demonstration
        if self.cursor < len(self.s):
            return "\\boxed{process next}"
        else:
            return "\\boxed{check}"