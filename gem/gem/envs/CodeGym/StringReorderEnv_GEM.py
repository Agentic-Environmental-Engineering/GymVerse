from typing import Any, Dict, Optional, Tuple
import random
import re
import string
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class StringReorderEnvGEM(Env):
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

        # 难度参数范围
        self.complexity_params = {
            "array_length": (5, 50),       # 字符串长度
            "char_pool_size": (4, 26),     # 字符集大小（英文小写字母的前 N 个）
            "observe_allowance": (1, 5),   # 最大观察次数
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 2,
            "char_pool_size": 2,
            "observe_allowance": 1,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.char_pool_size: int = 0
        self.observe_allowance: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例与操作状态
        self.problem: Dict[str, Any] = {}
        self.s: str = ""
        self.indices: list[int] = []
        self.result_list: Optional[list[str]] = None
        self.final_string: Optional[str] = None
        self.observe_count: int = 0

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
            "String Reorder: Reconstruct the string by placing characters at target indices.\n"
            "Available actions (use boxed format):\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Create result list: \\boxed{create length=L}\n"
            "- Place a character: \\boxed{place idx=I char=C pos=P}\n"
            "- Join characters: \\boxed{join}\n"
            "- Submit answer: \\boxed{answer YOUR_STRING}\n"
        )

    def get_task_suffix(self) -> str:
        placed = (
            sum(1 for c in (self.result_list or []) if c != "")
            if self.result_list is not None
            else 0
        )
        rl_status = "yes" if self.result_list is not None else "no"
        joined_status = "yes" if self.final_string is not None else "no"
        return (
            f"String length: {len(self.s)} | Result list created: {rl_status} | Placed: {placed}/{len(self.s) if self.s else 0} | "
            f"Joined: {joined_status} | Observes: {self.observe_count}/{self.observe_allowance} | "
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

        # 初始化状态
        self.s = self.problem["s"]
        self.indices = self.problem["indices"]
        self.result_list = None
        self.final_string = None
        self.turn_count = 0
        self.observe_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        length = self.array_length
        pool = string.ascii_lowercase[: self.char_pool_size]
        s_chars = [random.choice(pool) for _ in range(length)]
        s = "".join(s_chars)
        indices = list(range(length))
        random.shuffle(indices)
        return {"s": s, "indices": indices}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()

        # Recognize commands
        if content.lower() == "observe":
            return {"type": "observe", "params": {}}
        if content.lower().startswith("create"):
            # format: create length=L
            m = re.search(r"length\s*=\s*(\d+)", content, re.IGNORECASE)
            if not m:
                return None
            return {"type": "create", "params": {"length": int(m.group(1))}}
        if content.lower().startswith("place"):
            # format: place idx=I char=C pos=P
            mi = re.search(r"idx\s*=\s*(\d+)", content, re.IGNORECASE)
            mc = re.search(r"char\s*=\s*([^\s]+)", content, re.IGNORECASE)
            mp = re.search(r"pos\s*=\s*(\d+)", content, re.IGNORECASE)
            if not (mi and mc and mp):
                return None
            return {
                "type": "place",
                "params": {
                    "index": int(mi.group(1)),
                    "character": mc.group(1),
                    "position": int(mp.group(1)),
                },
            }
        if content.lower() == "join":
            return {"type": "join", "params": {}}
        if content.lower().startswith("answer"):
            # format: answer YOUR_STRING (take rest after first space)
            parts = content.split(None, 1)
            if len(parts) < 2:
                return None
            answer_str = parts[1]
            return {"type": "answer", "params": {"answer": answer_str}}
        return None

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

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        cmd_type = parsed["type"]
        params = parsed["params"]

        if cmd_type == "observe":
            if self.observe_count >= self.observe_allowance:
                obs = (
                    f"Observe limit reached ({self.observe_allowance}). Invalid action."
                )
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            self.observe_count += 1
            obs = self.Observe()
        elif cmd_type == "create":
            length = params["length"]
            # Validate length must match target length
            if length != len(self.s):
                obs = (
                    f"Error: length mismatch (provided={length}, required={len(self.s)})."
                )
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            obs = self.CreateResultList(length)
        elif cmd_type == "place":
            index = params["index"]
            character = params["character"]
            position = params["position"]
            # Basic validations
            if index < 0 or index >= len(self.s):
                obs = f"Error: index {index} out of range."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            if position < 0 or position >= len(self.s):
                obs = f"Error: position {position} out of range."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            # Character must match the original string at index
            expected_char = self.s[index]
            if character != expected_char:
                obs = f"Error: character mismatch at idx={index} (provided='{character}', expected='{expected_char}')."
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            obs = self.PlaceCharacter(index, character, position)
        elif cmd_type == "join":
            obs = self.JoinCharacters()
            # Non-terminal
        elif cmd_type == "answer":
            answer = params["answer"]
            done_msg = self.Done(answer)
            obs = done_msg
            # Set rewards based on correctness
            ref_answer = self.get_ref_answer()
            reward = 1.0 if answer == ref_answer else -1.0
            terminated = True
        else:
            obs = f"Invalid action: {cmd_type}"
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

    def sample_random_action(self) -> str:
        # Provide a reasonable random action
        if self.result_list is None:
            return f"\\boxed{{create length={len(self.s)}}}"
        # Randomly choose to observe or place
        if random.random() < 0.3:
            return "\\boxed{observe}"
        # Attempt a random place (using correct character)
        i = random.randint(0, len(self.s) - 1)
        c = self.s[i]
        p = self.indices[i]
        return f"\\boxed{{place idx={i} char={c} pos={p}}}"

    # ------------- 原环境的辅助方法（已转换） -------------

    @property
    def finished(self) -> bool:
        # GEM中通过 step 返回 terminated 来结束，但保留该属性以兼容
        return self.final_string is not None and self.turn_count >= self.max_turns

    @property
    def reward(self):
        # 在 GEM 中奖励通过 step 返回，但保留该属性（用于兼容）
        # 这里不追踪累计奖励，返回 0.0 作为占位
        return 0.0

    def get_ref_answer(self):
        r"""
        使用环境中的信息获取参考答案。
        """
        reordered = [''] * len(self.s)
        for i, char in enumerate(self.s):
            reordered[self.indices[i]] = char
        return ''.join(reordered)

    def CreateResultList(self, length: int):
        r"""
        创建指定长度的空列表以存储重排后的字符。
        """
        self.result_list = [''] * length
        return f"An empty list of length {length} has been created"

    def PlaceCharacter(self, index: int, character: str, position: int):
        r"""
        将指定字符放置在结果列表的指定位置。
        """
        if self.result_list is None:
            return "Error: Please create the result list first"
        if position < 0 or position >= len(self.result_list):
            return f"Error: position {position} out of range"
        self.result_list[position] = character
        return f"The character '{character}' at index {index} has been placed at position {position}"

    def JoinCharacters(self):
        r"""
        将结果列表中的所有字符连接为一个字符串。
        """
        if self.result_list is None:
            return "Error: Please first create the result list and place characters"
        self.final_string = ''.join(self.result_list)
        return self.final_string

    def Observe(self):
        r"""
        返回当前环境中的字符串和索引数组信息。
        """
        return f"Current string: {self.s}, index array: {self.indices}"

    def Done(self, answer: str):
        r"""
        验证最终答案是否正确并返回结果信息。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={1 if correct else 0}"

    def solve(self) -> str:
        r"""
        自动调用所有动作完成流程，并提交答案进行验证。
        返回最终答案验证的结果信息。
        """
        # Observe
        obs, _, term, _, _ = self.step("\\boxed{observe}")
        if term:
            return obs
        # Parse observed info
        try:
            s_part = obs.split('Current string: ')[1].split(', index array: ')[0]
            indices_part = obs.split('index array: ')[1]
            s = s_part.strip()
            indices = eval(indices_part)
        except Exception:
            s = self.s
            indices = self.indices
        # Create result list
        obs, _, term, _, _ = self.step(f"\\boxed{{create length={len(s)}}}")
        if term:
            return obs
        # Place characters
        for i in range(len(s)):
            character = s[i]
            position = indices[i]
            obs, _, term, _, _ = self.step(f"\\boxed{{place idx={i} char={character} pos={position}}}")
            if term:
                return obs
        # Join
        obs, _, term, _, _ = self.step("\\boxed{join}")
        if term:
            return obs
        result = obs
        # Submit answer
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {result}}}")
        return final_obs