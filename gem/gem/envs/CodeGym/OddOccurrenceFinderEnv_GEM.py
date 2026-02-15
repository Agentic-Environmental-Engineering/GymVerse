from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class OddOccurrenceFinderEnvGEM(Env):
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
            "array_length": (5, 50),         # 数组长度
            "num_constraints": (1, 5),       # 干扰元素的种类数量（均为偶数次出现）
            "value_range": (10, 10000),      # 数值范围上限（1..value_range）
            "max_turns_param": (20, 200),    # 难度建议的最大步数
        }

        # 参数方差（仅在 enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 2,
            "num_constraints": 1,
            "value_range": 200,
            "max_turns_param": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.num_constraints: int = 0
        self.value_range: int = 0
        self.max_turns_param: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题数据
        self.problem: Dict[str, Any] = {
            "array": [],
            "odd_num": None,
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

        # 将难度建议的最大步数与用户传入的上限结合（取更小者）
        if isinstance(self.max_turns_param, int) and self.max_turns_param > 0:
            self.max_turns = min(self.max_turns, self.max_turns_param)

    def _get_instructions(self) -> str:
        return (
            "Odd Occurrence Finder: In a hidden integer array, exactly one integer appears an odd number of times; all others appear an even number of times. Find that integer.\n"
            "Available actions (use the boxed format):\n"
            "- Describe the task: \\boxed{observe}\n"
            "- Get array length: \\boxed{len}\n"
            "- Access element at index i: \\boxed{get i}\n"
            "- XOR two integers a and b: \\boxed{xor a b}\n"
            "- Submit your final answer N: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Turn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例，确保恰有一个整数出现奇数次，其余均偶数次。"""
        L = max(1, int(self.array_length))
        k = max(1, int(self.num_constraints))
        vmax = max(2, int(self.value_range))

        # 确保 k 不超过数组长度约束（每个偶数元素至少出现2次）
        k = min(k, max(0, (L - 1) // 2))
        if k == 0:
            # 边界情况：数组长度为1时，只能包含一个奇数出现的元素
            odd_num = random.randint(1, vmax)
            arr = [odd_num]
            random.shuffle(arr)
            return {"array": arr, "odd_num": odd_num}

        # 选择奇数元素与其出现次数
        odd_num = random.randint(1, vmax)
        max_odd_count = L - 2 * k
        max_odd_count = max(1, max_odd_count)
        # 选择一个奇数次数（1..max_odd_count）
        candidate_counts = [c for c in range(1, max_odd_count + 1) if c % 2 == 1]
        odd_count = random.choice(candidate_counts)

        # 选择 k 个不同的偶数元素
        even_nums = set()
        while len(even_nums) < k:
            x = random.randint(1, vmax)
            if x != odd_num:
                even_nums.add(x)
        even_nums = list(even_nums)

        # 分配偶数元素的出现次数，总和为 L - odd_count，且每个至少2且为偶数
        remaining = L - odd_count
        # 初始化每个偶数元素的出现次数为2
        even_counts = [2 for _ in range(k)]
        remaining -= 2 * k

        # 将剩余的出现次数以2为步长随机分配给各偶数元素
        while remaining >= 2:
            idx = random.randint(0, k - 1)
            even_counts[idx] += 2
            remaining -= 2

        # 构建数组
        arr = [odd_num] * odd_count
        for num, cnt in zip(even_nums, even_counts):
            arr.extend([num] * cnt)
        random.shuffle(arr)

        return {"array": arr, "odd_num": odd_num}

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

        cmd = parsed["cmd"]
        args = parsed["args"]

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()

            elif cmd == "len":
                obs = self.GetArrayLength()

            elif cmd == "get":
                if len(args) != 1:
                    obs = "Invalid parameters for 'get'. Usage: \\boxed{get i}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        idx = int(args[0])
                    except Exception:
                        obs = "Invalid index. Provide an integer i."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs_val = self.GetElementAtIndex(idx)
                        if obs_val.startswith("Error:"):
                            obs = obs_val
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        else:
                            obs = obs_val

            elif cmd == "xor":
                if len(args) != 2:
                    obs = "Invalid parameters for 'xor'. Usage: \\boxed{xor a b}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        a = int(args[0])
                        b = int(args[1])
                        obs = self.XorNumbers(a, b)
                    except Exception:
                        obs = "Invalid integers for XOR."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True

            elif cmd == "answer":
                if len(args) != 1:
                    obs = "Invalid parameters for 'answer'. Usage: \\boxed{answer N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        ans = int(args[0])
                        msg = self.Done(ans)
                        obs = msg
                        # Done() sets reward internally, but GEM标准化为：正确=1.0，错误=-1.0
                        ref = self.get_ref_answer()
                        reward = 1.0 if ans == ref else -1.0
                        terminated = True
                    except Exception:
                        obs = "Invalid integer for final answer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Internal error: {str(e)}"
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

        # 解析命令和参数
        tokens = content.split()
        if not tokens:
            return None
        cmd = tokens[0].lower()
        args = tokens[1:]
        return {"cmd": cmd, "args": args}

    def sample_random_action(self) -> str:
        # 随机选择合法动作示例
        choices = [
            "\\boxed{observe}",
            "\\boxed{len}",
            "\\boxed{get 0}",
            "\\boxed{xor 1 2}",
            "\\boxed{answer 0}",
        ]
        return random.choice(choices)

    # ----------------------
    # 保留并转换原环境的辅助方法
    # ----------------------
    def Observe(self) -> str:
        return "There is an integer array where, except for one integer, all other integers appear an even number of times. Please find the integer that appears an odd number of times."

    def XorNumbers(self, a: int, b: int) -> str:
        return str(a ^ b)

    def GetArrayLength(self) -> str:
        return str(len(self.problem["array"]))

    def GetElementAtIndex(self, index: int) -> str:
        arr = self.problem["array"]
        if 0 <= index < len(arr):
            return str(arr[index])
        else:
            return "Error: Index out of range."

    def get_ref_answer(self) -> int:
        result = 0
        for num in self.problem["array"]:
            result ^= num
        return result

    def Done(self, answer: int) -> str:
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """使用内部信息自动求解并返回类似 Done 的结果消息"""
        result = 0
        for num in self.problem["array"]:
            result ^= num
        return self.Done(result)