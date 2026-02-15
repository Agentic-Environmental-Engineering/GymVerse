from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxSubsegmentSumEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 6,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        # 如果 __init__ 提供了 max_turns，则优先使用；否则由难度参数控制
        self._max_turns_arg = max_turns

        # 定义难度参数范围（根据原环境分析）
        # - array_length: 数组长度
        # - value_range: 元素绝对值范围（元素分布在 [-value_range, value_range]）
        # - max_turns_budget: 步数限制预算（如未显式传入 max_turns，以该值为准）
        self.complexity_params = {
            "array_length": (5, 50),
            "value_range": (10, 1000),
            "max_turns_budget": (20, 200),
        }

        # 参数方差（用于微调随机性，启用 enable_param_randomization 时生效）
        self.param_variance = {
            "array_length": 2,
            "value_range": 50,
            "max_turns_budget": 10,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.value_range: int = 0
        self.max_turns_budget: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.step_count: int = 0  # 保留与原环境一致的计数名称
        self._done: bool = False
        self._reward: float = 0.0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.arr: list[int] = []
        self.n: int = 0

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

        # 应用步数限制：若 __init__ 提供了 max_turns，则优先使用；否则采用难度预算
        if self._max_turns_arg is not None:
            self.max_turns = int(self._max_turns_arg)
        else:
            self.max_turns = int(self.max_turns_budget)

    def _get_instructions(self) -> str:
        return (
            "Max Subsegment Sum (Kadane) Task:\n"
            "You are given a hidden integer array. Compute its maximum subarray sum.\n"
            "Available actions (use the last \\boxed{...} if multiple present):\n"
            "- Observe array info: \\boxed{observe}\n"
            "- Initialize values: \\boxed{init}\n"
            "- Get element by index (0-based): \\boxed{get I}\n"
            "- Update current max: \\boxed{update_current C N}\n"
            "- Update global max: \\boxed{update_global C G}\n"
            "- Submit final answer: \\boxed{answer A}\n"
            "Outputs for actions follow the original environment style (numbers in plain text).\n"
        )

    def get_task_suffix(self) -> str:
        n = self.problem.get("n", 0)
        return f"Array length: {n}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.arr = self.problem["arr"]
        self.n = self.problem["n"]

        self.turn_count = 0
        self.step_count = 0
        self._done = False
        self._reward = 0.0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        length = int(self.array_length)
        vr = int(self.value_range)

        # 生成包含正负数的数组；为简单起见，不强制约束分布
        arr = [random.randint(-vr, vr) for _ in range(length)]
        return {"arr": arr, "n": length}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        self.step_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            self._reward = LanguageGameReward.format_error_reward
            self._done = True
            # 超时检查放在结尾，先构造返回
            terminated = True
            truncated = False
            info = {"suffix": self.get_task_suffix()}
            # 超时检查（统一放在 step 结尾）
            if not terminated and self.turn_count >= self.max_turns:
                obs = f"{obs}\nReached max turns ({self.max_turns})."
                self._reward = 0.0
                return obs, self._reward, True, True, {"suffix": self.get_task_suffix()}
            return obs, self._reward, terminated, truncated, info

        content = parsed["content"]
        tokens = content.strip().split()
        cmd = tokens[0].lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()
                reward = 0.0
                terminated = False

            elif cmd == "init":
                obs = self.InitializeMaxValues()
                reward = 0.0
                terminated = False

            elif cmd == "get":
                if len(tokens) != 2:
                    obs = "Invalid get action format. Use: \\boxed{get I}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        index = int(tokens[1])
                        obs = self.GetArrayElement(index)
                        if obs.startswith("Error"):
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                        else:
                            reward = 0.0
                            terminated = False
                    except ValueError:
                        obs = "Invalid index for get."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True

            elif cmd == "update_current":
                if len(tokens) != 3:
                    obs = "Invalid update_current format. Use: \\boxed{update_current C N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        current_max = int(tokens[1])
                        num = int(tokens[2])
                        obs = self.UpdateCurrentMax(current_max, num)
                        reward = 0.0
                        terminated = False
                    except ValueError:
                        obs = "Invalid arguments for update_current."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True

            elif cmd == "update_global":
                if len(tokens) != 3:
                    obs = "Invalid update_global format. Use: \\boxed{update_global C G}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        current_max = int(tokens[1])
                        global_max = int(tokens[2])
                        obs = self.UpdateGlobalMax(current_max, global_max)
                        reward = 0.0
                        terminated = False
                    except ValueError:
                        obs = "Invalid arguments for update_global."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True

            elif cmd == "answer" or cmd == "done":
                if len(tokens) != 2:
                    obs = "Invalid answer format. Use: \\boxed{answer A}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        answer = int(tokens[1])
                        done_msg, correct = self.Done(answer, return_flag=True)
                        obs = done_msg
                        reward = 1.0 if correct else -1.0
                        terminated = True
                    except ValueError:
                        obs = "Invalid answer value."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
            else:
                obs = "Invalid action."
                reward = LanguageGameReward.invalid_action_reward
                terminated = True

        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        self._reward = reward
        self._done = terminated

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            self._reward = 0.0
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
        # 随机示例动作（不保证有效性）
        if self.n == 0:
            return "\\boxed{observe}"
        choice = random.choice(
            ["observe", "init", "get", "update_current", "update_global", "answer"]
        )
        if choice == "observe":
            return "\\boxed{observe}"
        elif choice == "init":
            return "\\boxed{init}"
        elif choice == "get":
            idx = random.randint(0, max(0, self.n - 1))
            return f"\\boxed{{get {idx}}}"
        elif choice == "update_current":
            c = random.randint(-self.value_range, self.value_range)
            n = random.randint(-self.value_range, self.value_range)
            return f"\\boxed{{update_current {c} {n}}}"
        elif choice == "update_global":
            c = random.randint(-self.value_range, self.value_range)
            g = random.randint(-self.value_range, self.value_range)
            return f"\\boxed{{update_global {c} {g}}}"
        else:
            ans = self.get_ref_answer()
            return f"\\boxed{{answer {ans}}}"

    # ======================
    # 保留并转换原环境的辅助方法
    # ======================

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self) -> float:
        return float(self._reward)

    def get_ref_answer(self) -> int:
        """Use the information in the environment to obtain the reference answer."""
        if self.n == 0:
            return 0
        max_current = max_global = self.arr[0]
        for num in self.arr[1:]:
            max_current = max(num, max_current + num)
            if max_current > max_global:
                max_global = max_current
        return max_global

    def InitializeMaxValues(self) -> str:
        """
        Initialize the current maximum subarray sum and the global maximum subarray sum
        as the first element of the array.
        Returns: "current_max,global_max"
        """
        if self.n == 0:
            return "0,0"
        initial_value = self.arr[0]
        return f"{initial_value},{initial_value}"

    def UpdateCurrentMax(self, current_max: int, num: int) -> str:
        """
        Update the current maximum subarray sum by max(num, current_max + num).
        Returns: str(new_current_max)
        """
        new_current_max = max(num, current_max + num)
        return str(new_current_max)

    def UpdateGlobalMax(self, current_max: int, global_max: int) -> str:
        """
        Update the global maximum subarray sum: max(global_max, current_max).
        Returns: str(new_global_max)
        """
        new_global_max = max(global_max, current_max)
        return str(new_global_max)

    def GetArrayElement(self, index: int) -> str:
        """
        Get the element at the specified index in the array.
        Returns: str(value) or error message
        """
        if 0 <= index < self.n:
            return str(self.arr[index])
        else:
            return "Error: Index out of range"

    def Observe(self) -> str:
        """
        Return the observation of the current environment, including array length and content.
        """
        return f"Array length: {self.n}, Array content: {self.arr}"

    def Done(self, answer: int, return_flag: bool = False) -> str | Tuple[str, bool]:
        """
        Verify whether the final answer is correct and return result information.
        Returns:
            str message (if return_flag=False)
            or (str message, bool correct) if return_flag=True
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        if return_flag:
            return msg, correct
        return msg + f", reward={self._reward}"

    def solve(self) -> str:
        """
        Automatically call all actions to complete the process and submit the answer for verification.
        Returns: The result information of the final answer verification.
        """
        # Observe
        observe_obs, _, _, _, _ = self.step("\\boxed{observe}")
        # Parse array length from observe output
        try:
            array_length_part = observe_obs.split("Array length: ")[1].split(",")[0]
            array_length = int(array_length_part)
        except Exception:
            array_length = self.n

        # Initialize
        init_obs, _, _, _, _ = self.step("\\boxed{init}")
        parts = init_obs.split(",")
        current_max = int(parts[0])
        global_max = int(parts[1])

        for index in range(1, array_length):
            get_obs, _, term, _, _ = self.step(f"\\boxed{{get {index}}}")
            if term and get_obs.startswith("Error"):
                # Invalid step; break early
                break
            num = int(get_obs)

            upd_cur_obs, _, _, _, _ = self.step(f"\\boxed{{update_current {current_max} {num}}}")
            current_max = int(upd_cur_obs)

            upd_glb_obs, _, _, _, _ = self.step(f"\\boxed{{update_global {current_max} {global_max}}}")
            global_max = int(upd_glb_obs)

        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {global_max}}}")
        return final_obs