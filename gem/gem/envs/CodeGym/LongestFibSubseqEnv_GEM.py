from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LongestFibSubseqEnvGEM(Env):
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
            "array_length": (5, 50),      # 数组长度
            "value_range": (10, 10000),   # 数值范围（上界）
        }

        # 参数随机化方差
        self.param_variance = {
            "array_length": 2,
            "value_range": 200,
        }

        # 难度参数占位
        self.array_length: int = 0
        self.value_range: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 任务数据结构
        self.nums = []
        self.value_index_map: Dict[int, int] = {}
        self.longest_sequences: Dict[Tuple[int, int], int] = {}
        self.max_length: int = 0

        # 保持与原环境一致的内部变量（不直接用于奖励）
        self._reward: float = 0.0
        self._done: bool = False

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
            "Longest Fibonacci-like Subsequence: Compute the maximum length.\n"
            "The array is strictly increasing. Build a value->index map, check pairs, update lengths, and submit the answer.\n"
            "Available actions:\n"
            "- Create value->index map: \\boxed{create_map}\n"
            "- Check pair j,k: \\boxed{check j k}\n"
            "- Update sequence i,j,k: \\boxed{update i j k}\n"
            "- Get current max length: \\boxed{maxlen}\n"
            "- Observe array: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Array length: {self.array_length}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.nums = self.problem["nums"]
        self.value_index_map = {}
        self.longest_sequences = {}
        self.max_length = 0
        self._reward = 0.0
        self._done = False

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        upper_bound = max(self.value_range, n * 3)

        # 构造一个可嵌入的斐波那契样序列，确保至少长度为3
        # 选择较小的起始值，避免快速溢出范围
        base_a = random.randint(1, max(2, upper_bound // 20))
        base_b = base_a + random.randint(1, max(2, upper_bound // 20))

        target_len = random.randint(3, min(8, n))  # 斐波序列目标长度
        fib_seq = [base_a, base_b]
        while len(fib_seq) < target_len:
            nxt = fib_seq[-1] + fib_seq[-2]
            if nxt > upper_bound:
                break
            fib_seq.append(nxt)
        # 至少保证长度>=3
        if len(fib_seq) < 3:
            fib_seq = [1, 2, 3]

        # 填充其余元素，保持唯一性和升序
        pool_upper = upper_bound
        existing = set(fib_seq)
        remaining = n - len(fib_seq)
        # 构造剩余的唯一随机元素
        # 确保有足够的候选
        candidate_pool = list(set(range(1, pool_upper + 1)) - existing)
        if remaining > len(candidate_pool):
            # 如果候选不够，扩大范围
            extra_upper = pool_upper + remaining * 2
            candidate_pool = list(set(range(1, extra_upper + 1)) - existing)

        additional = random.sample(candidate_pool, remaining) if remaining > 0 else []
        nums = sorted(fib_seq + additional)
        return {"nums": nums, "size": n}

    # ---------------------------
    # 原环境的辅助方法（转换保留）
    # ---------------------------
    def CreateValueToIndexMap(self) -> str:
        """
        Create a mapping from array values to their indices.
        """
        self.value_index_map = {x: i for i, x in enumerate(self.nums)}
        return "Value to index map has been created"

    def CheckFibPair(self, j: int, k: int) -> str:
        """
        Check if there exists an index i such that nums[i] = nums[k] - nums[j] and i < j.
        Returns i as string or '-1'.
        """
        if j >= len(self.nums) or k >= len(self.nums) or j >= k or j < 0 or k < 0:
            return "-1"
        target = self.nums[k] - self.nums[j]
        i = self.value_index_map.get(target, -1)
        return str(i) if i != -1 and i < j else "-1"

    def UpdateLongestSequence(self, i: int, j: int, k: int) -> str:
        """
        Update the length of the longest Fibonacci-like sequence ending with (j, k).
        """
        seq_len = self.longest_sequences.get((i, j), 2) + 1
        self.longest_sequences[(j, k)] = seq_len
        if seq_len > self.max_length:
            self.max_length = seq_len
        return str(seq_len)

    def FindMaxLength(self) -> str:
        """
        Get the length of the longest Fibonacci-like sequence, return 0 if it is less than 3.
        """
        return str(self.max_length if self.max_length >= 3 else 0)

    def Observe(self) -> str:
        """
        Return the observation information of the current array.
        """
        return f"Current array: {self.nums}"

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        if len(self.nums) < 3:
            return 0

        index = {x: i for i, x in enumerate(self.nums)}
        longest = {}
        max_len = 0

        for k in range(len(self.nums)):
            for j in range(k):
                i = index.get(self.nums[k] - self.nums[j], None)
                if i is not None and i < j:
                    seq_len = longest.get((i, j), 2) + 1
                    longest[(j, k)] = seq_len
                    max_len = max(max_len, seq_len)

        return max_len if max_len >= 3 else 0

    def Done(self, answer: int) -> str:
        """
        Verify whether the final answer is correct and return the result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    # ---------------------------
    # GEM 交互方法
    # ---------------------------
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

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        a_type = parsed.get("type")
        params = parsed.get("params", {})

        try:
            if a_type == "create_map":
                obs = self.CreateValueToIndexMap()
                reward = 0.0
                terminated = False

            elif a_type == "check":
                j = params.get("j")
                k = params.get("k")
                if j is None or k is None:
                    obs = "Error: missing parameters j or k"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    result = self.CheckFibPair(j, k)
                    obs = result
                    reward = 0.0
                    terminated = False

            elif a_type == "update":
                i = params.get("i")
                j = params.get("j")
                k = params.get("k")
                if i is None or j is None or k is None:
                    obs = "Error: missing parameters i, j, or k"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    result = self.UpdateLongestSequence(i, j, k)
                    obs = result
                    reward = 0.0
                    terminated = False

            elif a_type == "maxlen":
                obs = self.FindMaxLength()
                reward = 0.0
                terminated = False

            elif a_type == "observe":
                obs = self.Observe()
                reward = 0.0
                terminated = False

            elif a_type == "answer":
                ans = params.get("answer")
                if ans is None:
                    obs = "Error: missing parameter answer"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    ref = self.get_ref_answer()
                    correct = (int(ans) == ref)
                    obs = f"Your answer: {ans}, Reference answer: {ref}, Result: {'Correct' if correct else 'Incorrect'}"
                    reward = 1.0 if correct else -1.0
                    terminated = True

            else:
                obs = f"Invalid action: {a_type}"
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

        cmd = tokens[0].lower()

        if cmd in ["create_map", "createmap", "createvaluetoindexmap", "map"]:
            return {"type": "create_map", "params": {}}

        if cmd in ["check", "checkfibpair", "pair"]:
            if len(tokens) < 3:
                return {"type": "check", "params": {}}
            try:
                j = int(tokens[1])
                k = int(tokens[2])
            except ValueError:
                return None
            return {"type": "check", "params": {"j": j, "k": k}}

        if cmd in ["update", "updatelongestsequence", "upd"]:
            if len(tokens) < 4:
                return {"type": "update", "params": {}}
            try:
                i = int(tokens[1])
                j = int(tokens[2])
                k = int(tokens[3])
            except ValueError:
                return None
            return {"type": "update", "params": {"i": i, "j": j, "k": k}}

        if cmd in ["maxlen", "findmaxlength", "max"]:
            return {"type": "maxlen", "params": {}}

        if cmd in ["observe", "obs"]:
            return {"type": "observe", "params": {}}

        if cmd in ["answer", "done", "submit"]:
            if len(tokens) < 2:
                return {"type": "answer", "params": {}}
            try:
                ans = int(tokens[1])
            except ValueError:
                return None
            return {"type": "answer", "params": {"answer": ans}}

        return None

    def sample_random_action(self) -> str:
        # 简单示例动作
        choices = [
            "\\boxed{observe}",
            "\\boxed{create_map}",
            "\\boxed{maxlen}",
            "\\boxed{check 0 1}" if self.array_length >= 2 else "\\boxed{observe}",
        ]
        return random.choice(choices)