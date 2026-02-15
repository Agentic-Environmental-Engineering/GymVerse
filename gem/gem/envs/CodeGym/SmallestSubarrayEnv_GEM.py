from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class SmallestSubarrayEnvGEM(Env):
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

        # 定义难度参数范围（根据原环境分析）
        # - array_length: 数组长度
        # - value_range: 数值范围（数组元素的最大值）
        # - threshold_pct: 阈值占数组总和的百分比（用于生成阈值）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 1000),
            "threshold_pct": (40, 90),
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance = {
            "array_length": 2,   # ±2 的方差
            "value_range": 25,   # ±25 的方差
            "threshold_pct": 5,  # ±5% 的方差
        }

        # 占位属性
        self.array_length: int = 0
        self.value_range: int = 0
        self.threshold_pct: int = 0

        # 问题实例
        self.arr: list[int] = []
        self.threshold: int = 0

        # 滑动窗口状态
        self.start: int = 0
        self.current_sum: int = 0
        self.min_length: float = float("inf")

        # 统计与终止状态
        self.turn_count: int = 0
        self._answered: bool = False
        self._final_message: str = ""

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
            "Smallest Subarray >= Threshold.\n"
            "Find the minimal length of a contiguous subarray with sum >= threshold.\n"
            "Available actions (use \\boxed{...}):\n"
            "- Initialize sliding window: \\boxed{init}\n"
            "- Get array length: \\boxed{len}\n"
            "- Get element at index i: \\boxed{get i}\n"
            "- Add value v to current sum: \\boxed{add v}\n"
            "- Compare current sum with threshold: \\boxed{cmp}\n"
            "- Update min length with end e and start s: \\boxed{update e s}\n"
            "- Subtract value v from current sum: \\boxed{sub v}\n"
            "- Increment pointer 'start': \\boxed{inc start}\n"
            "- Check current min length (or -1 if none): \\boxed{check}\n"
            "- Observe state: \\boxed{observe}\n"
            "- Submit final answer N: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array size: {len(self.arr)} | "
            f"Turn: {self.turn_count}/{self.max_turns} | "
            f"Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化环境状态
        self.arr = self.problem["arr"]
        self.threshold = self.problem["threshold"]
        self.start = 0
        self.current_sum = 0
        self.min_length = float("inf")

        self.turn_count = 0
        self._answered = False
        self._final_message = ""
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        length = self.array_length
        max_val = max(1, self.value_range)
        arr = [random.randint(1, max_val) for _ in range(length)]
        total = sum(arr)
        pct = self.threshold_pct
        threshold = int(round(total * pct / 100.0))
        # 确保阈值至少为1
        threshold = max(1, threshold)
        return {"arr": arr, "threshold": threshold}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            reward = LanguageGameReward.format_error_reward
            terminated = True
            truncated = False
            info = {"suffix": self.get_task_suffix()}
            # 超时检查（统一放在 step 结尾）
            if not terminated and self.turn_count >= self.max_turns:
                obs = f"{obs}\nReached max turns ({self.max_turns})."
                return obs, 0.0, True, True, info
            return obs, reward, terminated, truncated, info

        content = parsed["content"]
        tokens = content.strip().split()
        cmd = tokens[0].lower() if tokens else ""

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # 如果已经提交过答案，不再接受其他动作
        if self._answered and cmd != "answer":
            obs = "Answer already submitted. No further actions allowed."
            reward = LanguageGameReward.invalid_action_reward
            terminated = True
            info = {"suffix": self.get_task_suffix()}
            # 超时检查（统一放在 step 结尾）
            if not terminated and self.turn_count >= self.max_turns:
                obs = f"{obs}\nReached max turns ({self.max_turns})."
                return obs, 0.0, True, True, info
            return obs, reward, terminated, truncated, info

        try:
            if cmd == "init":
                obs = self.InitializeSlidingWindow()
                reward = 0.0

            elif cmd == "len":
                obs = self.GetArrayLength()
                reward = 0.0

            elif cmd == "get":
                if len(tokens) != 2 or not tokens[1].isdigit():
                    obs = "Invalid 'get' usage. Use: \\boxed{get i}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    index = int(tokens[1])
                    obs = self.GetElementAtIndex(index)
                    if obs.startswith("Error"):
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        reward = 0.0

            elif cmd == "add":
                if len(tokens) != 2 or not self._is_int(tokens[1]):
                    obs = "Invalid 'add' usage. Use: \\boxed{add v}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    value = int(tokens[1])
                    obs = self.AddToCurrentSum(value)
                    reward = 0.0

            elif cmd == "cmp":
                obs = self.CompareSumWithThreshold()
                reward = 0.0

            elif cmd == "update":
                if len(tokens) != 3 or not self._is_int(tokens[1]) or not self._is_int(tokens[2]):
                    obs = "Invalid 'update' usage. Use: \\boxed{update e s}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    end = int(tokens[1])
                    start = int(tokens[2])
                    if not (0 <= start < len(self.arr)) or not (0 <= end < len(self.arr)):
                        obs = "Error: 'end' or 'start' index out of bounds"
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.UpdateMinLength(end, start)
                        reward = 0.0

            elif cmd == "sub":
                if len(tokens) != 2 or not self._is_int(tokens[1]):
                    obs = "Invalid 'sub' usage. Use: \\boxed{sub v}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    value = int(tokens[1])
                    obs = self.SubtractFromCurrentSum(value)
                    reward = 0.0

            elif cmd == "inc":
                if len(tokens) != 2:
                    obs = "Invalid 'inc' usage. Use: \\boxed{inc start}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    pointer_name = tokens[1]
                    msg = self.IncrementPointer(pointer_name)
                    obs = msg
                    if msg.startswith("Error"):
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        reward = 0.0

            elif cmd == "check":
                obs = self.CheckMinLength()
                reward = 0.0

            elif cmd == "observe":
                obs = self.Observe()
                reward = 0.0

            elif cmd == "answer":
                if len(tokens) != 2 or not self._is_int(tokens[1]):
                    obs = "Invalid 'answer' usage. Use: \\boxed{answer N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    answer = int(tokens[1])
                    msg = self.Done(answer)
                    obs = msg
                    self._answered = True
                    # Done() sets correctness; we compute reward here
                    ref_answer = self.get_ref_answer()
                    reward = 1.0 if answer == ref_answer else -1.0
                    terminated = True

            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        info = {"suffix": self.get_task_suffix()}

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            return obs, 0.0, True, True, info

        return obs, reward, terminated, truncated, info

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
        actions = [
            "\\boxed{init}",
            "\\boxed{len}",
            "\\boxed{get 0}",
            "\\boxed{add 1}",
            "\\boxed{cmp}",
            "\\boxed{update 1 0}",
            "\\boxed{sub 1}",
            "\\boxed{inc start}",
            "\\boxed{check}",
            "\\boxed{observe}",
        ]
        return random.choice(actions)

    # -----------------------
    # 原环境的辅助方法（转换保留）
    # -----------------------

    def InitializeSlidingWindow(self) -> str:
        """
        Initialize sliding window parameters: start=0, current_sum=0, min_length=infinity.
        """
        self.start = 0
        self.current_sum = 0
        self.min_length = float("inf")
        return (
            f"Sliding window initialized: start={self.start}, "
            f"current_sum={self.current_sum}, min_length={self.min_length}"
        )

    def GetArrayLength(self) -> str:
        """
        Get the length of the array.
        """
        return str(len(self.arr))

    def GetElementAtIndex(self, index: int) -> str:
        """
        Get the element value at the specified index in the array.
        """
        if 0 <= index < len(self.arr):
            return str(self.arr[index])
        return "Error: Index out of bounds"

    def AddToCurrentSum(self, value: int) -> str:
        """
        Add the given value to the current sum.
        """
        self.current_sum += value
        return str(self.current_sum)

    def CompareSumWithThreshold(self) -> str:
        """
        Compare the current sum with the threshold.
        """
        return str(self.current_sum >= self.threshold)

    def UpdateMinLength(self, end: int, start: int) -> str:
        """
        Update the minimum subarray length.
        """
        current_window_length = end - start + 1
        if current_window_length < self.min_length:
            self.min_length = current_window_length
        return str(self.min_length)

    def SubtractFromCurrentSum(self, value: int) -> str:
        """
        Subtract the given value from the current sum.
        """
        self.current_sum -= value
        return str(self.current_sum)

    def IncrementPointer(self, pointer_name: str) -> str:
        """
        Increment the value of the specified pointer ("start" only).
        """
        if pointer_name == "start":
            self.start += 1
            return str(self.start)
        return "Error: Invalid pointer name"

    def CheckMinLength(self) -> str:
        """
        Check if the minimum length has been updated and return the result.
        """
        return str(self.min_length if self.min_length != float("inf") else -1)

    def Observe(self) -> str:
        """
        Return observation information of the current environment state.
        """
        return (
            f"Current state: start={self.start}, current_sum={self.current_sum}, "
            f"min_length={self.min_length}"
        )

    def Done(self, answer: int) -> str:
        """
        Verify whether the final answer is correct and return result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        self._final_message = msg
        return msg

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        n = len(self.arr)
        min_length = float("inf")
        current_sum = 0
        start = 0

        for end in range(n):
            current_sum += self.arr[end]
            while current_sum >= self.threshold:
                min_length = min(min_length, end - start + 1)
                current_sum -= self.arr[start]
                start += 1

        return int(min_length) if min_length != float("inf") else -1

    def solve(self) -> str:
        """
        Automatically call actions to complete the process and submit the answer.
        Returns the final verification result information.
        """
        # init
        self.step("\\boxed{init}")
        # get length
        obs, _, _, _, _ = self.step("\\boxed{len}")
        array_length = int(obs)
        end = 0
        while end < array_length:
            # get element
            obs, _, _, _, _ = self.step(f"\\boxed{{get {end}}}")
            element = int(obs)
            self.step(f"\\boxed{{add {element}}}")
            while True:
                cmp_obs, _, _, _, _ = self.step("\\boxed{cmp}")
                if cmp_obs == "True":
                    observe_info, _, _, _, _ = self.step("\\boxed{observe}")
                    # parse start from observe string
                    try:
                        start_str = observe_info.split("start=")[1].split(",")[0].strip()
                        start_idx = int(start_str)
                    except Exception:
                        start_idx = self.start
                    self.step(f"\\boxed{{update {end} {start_idx}}}")
                    # get arr[start]
                    start_element_obs, _, _, _, _ = self.step(f"\\boxed{{get {start_idx}}}")
                    if start_element_obs.startswith("Error"):
                        break
                    start_element = int(start_element_obs)
                    self.step(f"\\boxed{{sub {start_element}}}")
                    self.step("\\boxed{inc start}")
                else:
                    break
            end += 1
        answer_obs, _, _, _, _ = self.step("\\boxed{check}")
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {answer_obs}}}")
        return final_obs

    @staticmethod
    def _is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except Exception:
            return False