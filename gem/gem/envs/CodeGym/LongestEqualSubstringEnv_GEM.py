from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LongestEqualSubstringEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 8,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization

        # 步数限制（将由 _apply_complexity_params 控制）
        self.max_turns = max_turns if max_turns is not None else 100

        # 定义难度参数范围（根据原环境：字符串长度、字母表大小、步数限制）
        self.complexity_params = {
            "string_length": (5, 50),      # 字符串长度
            "alphabet_size": (2, 3),       # 字母表大小（a,b,c 的子集）
            "max_turns_param": (20, 200),  # 难度驱动的步数限制
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "string_length": 2,
            "alphabet_size": 0,
            "max_turns_param": 10,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.string_length: int = 0
        self.alphabet_size: int = 0
        self.max_turns_param: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题定义（受 reset 控制）
        self.s: str = ""
        self.count_map: Dict[Tuple[int, int], int] = {}
        self.max_len: int = 0

        # 动作映射（保留原环境的辅助方法）
        self.INITIALIZE_COUNTERS = 0
        self.UPDATE_COUNTER = 1
        self.CALCULATE_KEY = 2
        self.CHECK_KEY_IN_MAP = 3
        self.ADD_KEY_TO_MAP = 4
        self.CALCULATE_DISTANCE = 5
        self.UPDATE_MAX_LENGTH = 6
        self.OBSERVE = 7

        self.func_mapping = {
            "InitializeCounters": self.INITIALIZE_COUNTERS,
            "UpdateCounter": self.UPDATE_COUNTER,
            "CalculateKey": self.CALCULATE_KEY,
            "CheckKeyInMap": self.CHECK_KEY_IN_MAP,
            "AddKeyToMap": self.ADD_KEY_TO_MAP,
            "CalculateDistance": self.CALCULATE_DISTANCE,
            "UpdateMaxLength": self.UPDATE_MAX_LENGTH,
            "Observe": self.OBSERVE,
        }

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

        # 使用难度驱动的步数限制
        self.max_turns = int(self.max_turns_param)

    def _get_instructions(self) -> str:
        return (
            "Longest Equal Substring: Find the longest substring where counts(a)=counts(b)=counts(c).\n"
            "Available actions (use \\boxed{...} format):\n"
            "- Observe current state: \\boxed{observe}\n"
            "- Call helper functions: \\boxed{call FunctionName <json_params>}\n"
            "  FunctionName ∈ {InitializeCounters, UpdateCounter, CalculateKey, CheckKeyInMap, AddKeyToMap, CalculateDistance, UpdateMaxLength, Observe}\n"
            "  Example: \\boxed{call UpdateCounter {\"char\":\"a\",\"counts\":{\"a\":0,\"b\":0,\"c\":0}}}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        alphabet = "".join([chr(ord("a") + i) for i in range(self.alphabet_size)])
        return (
            f"String length: {len(self.s)} | Alphabet: {alphabet} | "
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.s = self.problem["s"]

        # 初始化状态
        self.count_map = {(0, 0): -1}
        self.max_len = 0
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        alphabet = [chr(ord("a") + i) for i in range(self.alphabet_size)]
        s = "".join(random.choice(alphabet) for _ in range(self.string_length))
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
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            # answer
            if content.lower().startswith("answer"):
                m = re.match(r"answer\s+(-?\d+)\s*$", content.strip(), re.IGNORECASE)
                if not m:
                    obs = "Answer format error. Use \\boxed{answer N}."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    ans = int(m.group(1))
                    ref = self.get_ref_answer()
                    correct = (ans == ref)
                    obs = f"Your answer: {ans}, Reference answer: {ref}, Result: {'Correct' if correct else 'Incorrect'}"
                    reward = 1.0 if correct else -1.0
                    terminated = True

            # observe
            elif content.lower().strip() == "observe":
                obs = self.Observe()

            # call function
            elif content.lower().startswith("call"):
                cm = re.match(r"call\s+([A-Za-z_][A-Za-z0-9_]*)\s*(.*)$", content.strip(), re.IGNORECASE | re.DOTALL)
                if not cm:
                    obs = "Call format error. Use \\boxed{call FunctionName <json_params?>}."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    func_name = cm.group(1)
                    params_str = cm.group(2).strip()
                    params = {}
                    if params_str:
                        try:
                            params = json.loads(params_str)
                        except Exception:
                            obs = "Parameters JSON parse error."
                            reward = LanguageGameReward.format_error_reward
                            terminated = True
                    if not terminated:
                        if func_name not in self.func_mapping:
                            obs = f"Invalid action: {func_name}"
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        else:
                            action_code = self.func_mapping[func_name]
                            msg = ""

                            if action_code == self.INITIALIZE_COUNTERS:
                                msg = self.InitializeCounters()

                            elif action_code == self.UPDATE_COUNTER:
                                if "char" in params and "counts" in params:
                                    char = params["char"]
                                    counts = params["counts"]
                                    msg = self.UpdateCounter(char, counts)
                                else:
                                    msg = "Error: 'char' or 'counts' parameter is missing for UPDATE_COUNTER action."

                            elif action_code == self.CALCULATE_KEY:
                                if "a" in params and "b" in params and "c" in params:
                                    a = params["a"]
                                    b = params["b"]
                                    c = params["c"]
                                    msg = self.CalculateKey(a, b, c)
                                else:
                                    msg = "Error: 'a', 'b' or 'c' parameter is missing for CALCULATE_KEY action."

                            elif action_code == self.CHECK_KEY_IN_MAP:
                                if "key" in params:
                                    # Convert list to tuple if needed
                                    key_val = params["key"]
                                    if isinstance(key_val, list):
                                        key_val = tuple(key_val)
                                    elif isinstance(key_val, tuple):
                                        pass
                                    else:
                                        # try to coerce
                                        try:
                                            key_val = tuple(key_val)
                                        except Exception:
                                            msg = "Error: 'key' must be a list or tuple of two integers."
                                            obs = msg
                                            reward = LanguageGameReward.format_error_reward
                                            terminated = True
                                            obs += ""
                                            msg = None
                                    if not terminated and msg is None:
                                        msg = self.CheckKeyInMap(key_val)
                                else:
                                    msg = "Error: 'key' parameter is missing for CHECK_KEY_IN_MAP action."

                            elif action_code == self.ADD_KEY_TO_MAP:
                                if "key" in params and "index" in params:
                                    key_val = params["key"]
                                    if isinstance(key_val, list):
                                        key_val = tuple(key_val)
                                    msg = self.AddKeyToMap(key_val, params["index"])
                                else:
                                    msg = "Error: 'key' or 'index' parameter is missing for ADD_KEY_TO_MAP action."

                            elif action_code == self.CALCULATE_DISTANCE:
                                if "current_index" in params and "stored_index" in params:
                                    current_index = params["current_index"]
                                    stored_index = params["stored_index"]
                                    msg = self.CalculateDistance(current_index, stored_index)
                                else:
                                    msg = "Error: 'current_index' or 'stored_index' parameter is missing for CALCULATE_DISTANCE action."

                            elif action_code == self.UPDATE_MAX_LENGTH:
                                if "current_max" in params and "new_length" in params:
                                    current_max = params["current_max"]
                                    new_length = params["new_length"]
                                    msg = self.UpdateMaxLength(current_max, new_length)
                                else:
                                    msg = "Error: 'current_max' or 'new_length' parameter is missing for UPDATE_MAX_LENGTH action."

                            elif action_code == self.OBSERVE:
                                msg = self.Observe()

                            if not terminated:
                                obs = msg

            else:
                obs = "Unknown command. Use observe/call/answer."
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
        return {"content": content}

    def sample_random_action(self) -> str:
        # 随机示例动作
        choices = [
            "\\boxed{observe}",
            "\\boxed{call InitializeCounters}",
            "\\boxed{answer 0}",
        ]
        return random.choice(choices)

    # --------------------------
    # 保留原环境的所有辅助方法
    # --------------------------

    def InitializeCounters(self):
        r"""
        Initialize the counters for a, b, and c to 0.

        Returns:
            str: A JSON string containing the initialized counter values.
        """
        return json.dumps({"a": 0, "b": 0, "c": 0})

    def UpdateCounter(self, char: str, counts: dict):
        r"""
        Update the counter for the specified character.

        Args:
            char (str): The character whose count needs to be updated, must be 'a', 'b', or 'c'.
            counts (dict): The current count dictionary, containing the three keys "a", "b", and "c".

        Returns:
            str: A JSON string of the updated count dictionary.
        """
        if char not in ['a', 'b', 'c']:
            return json.dumps(counts)

        new_counts = counts.copy()
        new_counts[char] = new_counts.get(char, 0) + 1
        return json.dumps(new_counts)

    def CalculateKey(self, a: int, b: int, c: int):
        r"""
        Calculate the key value based on the counts of a, b, and c. The key value is a tuple of (a-b, a-c).

        Returns:
            str: A JSON string of the calculated key value.
        """
        key = (a - b, a - c)
        return json.dumps(list(key))

    def CheckKeyInMap(self, key: tuple):
        r"""
        Check if the key value exists in the count map.

        Args:
            key (tuple): The key value to be checked, a tuple containing two integers.

        Returns:
            str: A JSON string containing the check result and the corresponding index. If the key does not exist, the index is -1.
        """
        if key in self.count_map:
            return json.dumps({"exists": True, "index": self.count_map[key]})
        return json.dumps({"exists": False, "index": -1})

    def AddKeyToMap(self, key: tuple, index: int):
        r"""
        Add the key value and its corresponding index to the count map.

        Args:
            key (tuple): The key value to be added, a tuple containing two integers.
            index (int): The string index corresponding to the key value.

        Returns:
            str: Result information of the addition operation.
        """
        self.count_map[key] = index
        return f"Key{str(key)} has been added to the map, index is {index}"

    def CalculateDistance(self, current_index: int, stored_index: int):
        r"""
        Calculate the distance between the current index and the stored index.

        Args:
            current_index (int): The current string index.
            stored_index (int): The stored index value.

        Returns:
            str: The calculated distance.
        """
        distance = current_index - stored_index
        return str(distance)

    def UpdateMaxLength(self, current_max: int, new_length: int):
        r"""
        Update the maximum length to the larger value between the current maximum and the new length.

        Args:
            current_max (int): The current maximum length.
            new_length (int): The newly calculated length.

        Returns:
            str: The updated maximum length.
        """
        return str(max(current_max, new_length))

    def Observe(self):
        r"""
        Obtain observation information of the current environment, including the target string and the current maximum length.

        Returns:
            str: Information describing the current state of the environment.
        """
        return f"Target string: {self.s}, current maximum length: {self.max_len}"

    def get_ref_answer(self):
        def to_key(a, b):
            return a - b

        n = len(self.s)
        if n == 0:
            return 0

        count_a, count_b, count_c = 0, 0, 0
        count_map = {(0, 0): -1}
        max_len = 0

        for i in range(n):
            if self.s[i] == 'a':
                count_a += 1
            elif self.s[i] == 'b':
                count_b += 1
            elif self.s[i] == 'c':
                count_c += 1

            key = (to_key(count_a, count_b), to_key(count_a, count_c))

            if key in count_map:
                max_len = max(max_len, i - count_map[key])
            else:
                count_map[key] = i

        return max_len

    def solve(self) -> str:
        """
        Automatically call all actions to complete the process, and submit the answer for verification.
        Returns a summary string.
        """
        # Observe
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        target_str = obs.split("Target string: ")[1].split(", ")[0]

        # Initialize counters
        obs, _, _, _, _ = self.step("\\boxed{call InitializeCounters}")
        counts = json.loads(obs)

        # Initial key
        key_json_action = f'\\boxed{{call CalculateKey {json.dumps({"a": counts["a"], "b": counts["b"], "c": counts["c"]})}}}'
        obs, _, _, _, _ = self.step(key_json_action)
        initial_key = json.loads(obs)

        add_initial_action = f'\\boxed{{call AddKeyToMap {json.dumps({"key": initial_key, "index": -1})}}}'
        self.step(add_initial_action)

        max_length = 0

        for current_index, char in enumerate(target_str):
            update_action = f'\\boxed{{call UpdateCounter {json.dumps({"char": char, "counts": counts})}}}'
            obs, _, _, _, _ = self.step(update_action)
            counts = json.loads(obs)

            calc_key_action = f'\\boxed{{call CalculateKey {json.dumps({"a": counts["a"], "b": counts["b"], "c": counts["c"]})}}}'
            obs, _, _, _, _ = self.step(calc_key_action)
            key = json.loads(obs)

            check_action = f'\\boxed{{call CheckKeyInMap {json.dumps({"key": key})}}}'
            obs, _, _, _, _ = self.step(check_action)
            check_result = json.loads(obs)

            if check_result["exists"]:
                dist_action = f'\\boxed{{call CalculateDistance {json.dumps({"current_index": current_index, "stored_index": check_result["index"]})}}}'
                obs, _, _, _, _ = self.step(dist_action)
                distance = int(obs)

                update_max_action = f'\\boxed{{call UpdateMaxLength {json.dumps({"current_max": max_length, "new_length": distance})}}}'
                obs, _, _, _, _ = self.step(update_max_action)
                max_length = int(obs)
            else:
                add_action = f'\\boxed{{call AddKeyToMap {json.dumps({"key": key, "index": current_index})}}}'
                self.step(add_action)

        final_obs, reward, terminated, truncated, _ = self.step(f"\\boxed{{answer {max_length}}}")
        status = "Success" if reward > 0 and terminated and not truncated else "Failure"
        return f"{final_obs} | {status}"