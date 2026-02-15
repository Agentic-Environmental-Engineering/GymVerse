from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from collections import Counter
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LargestHarmonicSubsetEnvGEM(Env):
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
            "array_length": (5, 50),       # 数组长度
            "value_range": (10, 10000),    # 值域上限（含）
            "num_constraints": (1, 5),     # 锚点对（v, v+1）数量，用于保证可解性
        }

        # 参数方差（启用随机化时生效）
        self.param_variance = {
            "array_length": 3,
            "value_range": 100,
            "num_constraints": 1,
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.num_constraints: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题与便捷属性
        self.problem: Dict[str, Any] = {}
        self.nums: list[int] = []

        # 内部缓存
        self._last_count_dict: Optional[Dict[str, int]] = None
        self._done: bool = False
        self._reward: float = 0.0

        self.reset()

    def _apply_complexity_params(self):
        """根据 complexity 等级计算参数值"""
        normalized = min(1.0, (self.complexity - 1) / 9.0)  # 归一化到 [0, 1]

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
            "Largest Harmonic Subset: Given an integer array, find the size of the largest harmonic subset.\n"
            "A harmonic subset contains elements whose max and min differ by exactly 1.\n"
            "You can interact using the following actions:\n"
            "- Observe the array: \\boxed{observe}\n"
            "- Count frequencies: \\boxed{count}\n"
            "- Calculate subset size for a base value v (counts of v and v+1): \\boxed{calc v}\n"
            "- Take max from a list of sizes: \\boxed{max s1 s2 s3 ...}\n"
            "- Submit final answer N: \\boxed{answer N}\n"
            "Notes:\n"
            "- calc v uses the last counted frequencies if available, otherwise it counts first.\n"
            "- The goal is to output the size of the largest harmonic subset."
        )

    def get_task_suffix(self) -> str:
        return f"Array length: {len(self.nums)} | Turn: {self.turn_count}/{self.max_turns} | Enter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.nums = self.problem["nums"]

        self.turn_count = 0
        self._last_count_dict = None
        self._done = False
        self._reward = 0.0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        V = max(2, self.value_range)  # 至少为 2，确保存在 v+1
        k = max(1, min(self.num_constraints, V - 1))

        # 选择 k 个锚点（保证 v+1 存在）
        anchors = sorted(random.sample(range(0, V - 1), k))

        nums = []
        # 预留一部分位置由锚点对 (v, v+1) 生成，确保存在非平凡的和声子集
        anchored_fraction = 0.6
        anchored_slots = int(round(n * anchored_fraction))
        noise_slots = n - anchored_slots

        # 生成锚点对样本
        for _ in range(anchored_slots):
            a = random.choice(anchors)
            val = a if random.random() < 0.5 else a + 1
            nums.append(val)

        # 加入噪声，增加难度
        for _ in range(noise_slots):
            nums.append(random.randint(0, V - 1))

        # 打乱
        random.shuffle(nums)

        return {"nums": nums, "anchors": anchors, "value_range": V, "array_length": n}

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
        content_stripped = content.strip()
        cmd = content_stripped.lower().split()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if not cmd:
                obs = "Empty command."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

            elif cmd[0] == "observe":
                obs = self.Observe()

            elif cmd[0] == "count":
                obs = self.CountOccurrences()
                # 缓存 count dict
                try:
                    self._last_count_dict = json.loads(obs)
                except Exception:
                    self._last_count_dict = None

            elif cmd[0] in ("calc", "calculate"):
                if len(cmd) < 2:
                    obs = "Error: missing base value v for calc."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        num = int(cmd[1])
                    except ValueError:
                        obs = "Error: v must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        # 如果没有缓存，先 count
                        count_dict = self._last_count_dict
                        if count_dict is None:
                            count_json = self.CountOccurrences()
                            count_dict = json.loads(count_json)
                            self._last_count_dict = count_dict
                        obs = self.CalculateSubsetSize(count_dict, num)

            elif cmd[0] == "max":
                if len(cmd) < 2:
                    obs = "Error: provide a list of sizes for max."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    # 允许空格/逗号混合的数字序列
                    tail = content_stripped[len(cmd[0]):].strip()
                    nums_str = re.findall(r"-?\d+", tail)
                    if not nums_str:
                        obs = "Error: no valid integers provided to max."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        sizes = [int(x) for x in nums_str]
                        obs = self.FindMaxSize(sizes)

            elif cmd[0] == "answer":
                if len(cmd) < 2:
                    obs = "Error: missing answer value."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        ans = int(cmd[1])
                    except ValueError:
                        obs = "Error: answer must be an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        # 使用 Done 进行验证并返回信息
                        msg = self.Done(ans)
                        obs = msg
                        reward = self._reward if isinstance(self._reward, (int, float)) else 0.0
                        terminated = True

            elif cmd[0] == "help":
                obs = self._get_instructions()

            else:
                obs = f"Invalid action: {cmd[0]}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Runtime error: {e}"
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
        # 随机采样一个动作（示例）
        choices = [
            "\\boxed{observe}",
            "\\boxed{count}",
            "\\boxed{calc 1}",
            "\\boxed{max 1 2 3}",
        ]
        # 偶尔尝试提交答案（随机猜测）
        if random.random() < 0.1:
            guess = random.randint(0, len(self.nums))
            return f"\\boxed{{answer {guess}}}"
        return random.choice(choices)

    # -------------------------
    # 以下为保留并转换的辅助方法
    # -------------------------

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        count = Counter(self.nums)
        max_size = 0
        for num in count:
            current_size = count[num] + count.get(num + 1, 0)
            max_size = max(max_size, current_size)
        return max_size

    def CountOccurrences(self):
        r"""
        Count the occurrence frequency of each number in the array.

        Returns:
            str: A dictionary containing the occurrence frequency of each number, in json format.

        Example Output:
            "{\"1\": 2, \"2\": 3, \"3\": 1}"
        """
        count = Counter(self.nums)
        # 将键转换为字符串以兼容原格式
        str_key_dict = {str(k): v for k, v in count.items()}
        return json.dumps(str_key_dict)

    def CalculateSubsetSize(self, count_dict: dict, num: int):
        r"""
        Calculate the size of the harmonic subset containing num and num+1.

        Args:
            count_dict (dict): A dictionary of the occurrence frequency of each number.
            num (int): The number to be calculated.

        Returns:
            str: The size of the harmonic subset.

        Example Output:
            "4"
        """
        # count_dict 的键为字符串
        current_size = count_dict.get(str(num), 0) + count_dict.get(str(num + 1), 0)
        return str(current_size)

    def FindMaxSize(self, sizes: list):
        r"""
        Find the maximum value from multiple subset sizes.

        Args:
            sizes (list[int]): A list containing multiple subset sizes.

        Returns:
            str: The maximum subset size.

        Example Output:
            "5"
        """
        max_size = max(sizes) if sizes else 0
        return str(max_size)

    def Observe(self):
        r"""
        Return the observation information of the current array.

        Returns:
            str: Information describing the current array.

        Example Output:
            "Current array: [1, 2, 2, 3, 4]"
        """
        return f"Current array: {self.nums}"

    def Done(self, answer):
        r"""
        Verify whether the final answer is correct and return result information.

        Args:
            answer (int): The answer submitted by the user.

        Returns:
            str: Result information, including correctness and reward information.

        Example Output:
            "Your answer: 3, Reference answer: 3, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self):
        r"""
        Automatically call all actions to complete the process, and submit the answer for verification.

        Returns:
            str: The result information of the final answer verification.
        """
        # 直接调用内部辅助函数进行求解（与原逻辑等价）
        count_json = self.CountOccurrences()
        count_dict = json.loads(count_json)
        nums = list(map(int, count_dict.keys()))
        if not nums:
            return self.Done(0)
        sizes = []
        for num in nums:
            size_str = self.CalculateSubsetSize(count_dict, num)
            sizes.append(int(size_str))
        max_size_str = self.FindMaxSize(sizes)
        max_size = int(max_size_str)
        return self.Done(max_size)