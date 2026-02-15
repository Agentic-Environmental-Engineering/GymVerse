from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LongestConsecutiveOnesEnvGEM(Env):
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

        # 难度参数范围
        self.complexity_params = {
            "array_length": (5, 50),       # 数组长度
            "allowed_zeros": (0, 2),       # 允许窗口内的零个数
        }

        # 参数方差（可选）
        self.param_variance = {
            "array_length": 3,
            "allowed_zeros": 1,
        }

        # 占位属性
        self.array_length: int = 0
        self.allowed_zeros: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 算法状态变量
        self.nums: list[int] = []
        self.max_consecutive: int = 0
        self.num_zeros: int = 0
        self.left: int = 0
        self.right: int = 0

        # 其它
        self.problem: Dict[str, Any] = {}

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
            "Longest Consecutive Ones II (flip up to K zeros using sliding window).\n"
            "You can interact with the algorithm state using actions:\n"
            "- Initialize state: \\boxed{init}\n"
            "- Move right pointer: \\boxed{move} or \\boxed{move N}\n"
            "- Check and update zero count at current right: \\boxed{check}\n"
            "- Adjust left pointer when zero count too high: \\boxed{adjust}\n"
            "- Update max window length: \\boxed{update}\n"
            "- Observe current state: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            f"Allowed zeros (K) is determined by difficulty and equals {self.allowed_zeros}.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array length: {self.array_length}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            f"Pointers: left={self.left}, right={self.right}\n"
            f"Zeros in window={self.num_zeros}, max_consecutive={self.max_consecutive}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态
        self.nums = self.problem["nums"]
        self.max_consecutive = 0
        self.num_zeros = 0
        self.left = 0
        self.right = 0
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        nums = [random.randint(0, 1) for _ in range(self.array_length)]
        return {"nums": nums, "size": self.array_length, "allowed_zeros": self.allowed_zeros}

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
        tokens = content.split()
        cmd = tokens[0].lower() if tokens else ""

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "init":
                obs = self.InitializePointersAndCounters()

            elif cmd == "move":
                # Optional parameter: target index
                if len(tokens) >= 2:
                    try:
                        target = int(tokens[1])
                    except ValueError:
                        obs = "Invalid parameter for move. Expect integer index."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    # Set right to target (clamped)
                    target = max(0, min(target, len(self.nums) - 1)) if self.nums else 0
                    # Emulate original MoveRightPointer semantics via delta if possible
                    moved = self.MoveRightPointer(self.right if self.nums else 0)
                    # If we set explicitly, override
                    self.right = target
                    obs = str(self.right)
                else:
                    moved = self.MoveRightPointer(self.right if self.nums else 0)
                    obs = moved

            elif cmd == "check":
                current_right = self.right
                obs = self.CheckAndUpdateZeroCount(current_right)

            elif cmd == "adjust":
                obs = self.AdjustLeftPointer()

            elif cmd == "update":
                obs = self.UpdateMaxConsecutive(self.right, self.left)

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "answer":
                if len(tokens) < 2:
                    obs = "Missing answer value."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                try:
                    ans = int(tokens[1])
                except ValueError:
                    obs = "Invalid answer value (must be integer)."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                msg = self.Done(ans)
                obs = msg
                ref_answer = self.get_ref_answer()
                reward = 1.0 if ans == ref_answer else -1.0
                terminated = True

            else:
                obs = f"Invalid action: {content}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
        except Exception as e:
            obs = f"Error: {str(e)}"
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
        return "\\boxed{observe}"

    # =========================
    # 保留并转换原环境的辅助方法
    # =========================

    def get_ref_answer(self) -> int:
        """
        根据当前问题和允许的零数（allowed_zeros）计算参考答案。
        """
        max_consecutive = 0
        num_zeros = 0
        left = 0
        K = self.allowed_zeros

        for right in range(len(self.nums)):
            if self.nums[right] == 0:
                num_zeros += 1

            while num_zeros > K and left <= right:
                if self.nums[left] == 0:
                    num_zeros -= 1
                left += 1

            max_consecutive = max(max_consecutive, right - left + 1)

        return max_consecutive

    def InitializePointersAndCounters(self) -> str:
        """
        初始化算法指针和计数器。
        """
        self.max_consecutive = 0
        self.num_zeros = 0
        self.left = 0
        self.right = 0
        return (
            f"Initialized: max_consecutive={self.max_consecutive}, "
            f"num_zeros={self.num_zeros}, left={self.left}, right={self.right}"
        )

    def MoveRightPointer(self, current_right: int) -> str:
        """
        将右指针右移一位。
        """
        if not self.nums:
            self.right = 0
            return "0"
        new_right = current_right + 1
        # Clamp to valid range
        new_right = max(0, min(new_right, len(self.nums) - 1))
        self.right = new_right
        return str(new_right)

    def CheckAndUpdateZeroCount(self, current_right: int) -> str:
        """
        检查右指针位置元素是否为 0，更新零计数。
        """
        if 0 <= current_right < len(self.nums) and self.nums[current_right] == 0:
            self.num_zeros += 1
        return str(self.num_zeros)

    def AdjustLeftPointer(self) -> str:
        """
        当零计数超过允许值时，右移左指针直到零计数不超过允许值。
        """
        K = self.allowed_zeros
        while self.num_zeros > K and self.left <= self.right:
            if self.nums[self.left] == 0:
                self.num_zeros -= 1
            self.left += 1
        return f"left={self.left}, num_zeros={self.num_zeros}"

    def UpdateMaxConsecutive(self, current_right: int, current_left: int) -> str:
        """
        更新最长连续 1 的长度（允许翻转至多 K 个 0）。
        """
        window_length = current_right - current_left + 1
        if window_length > self.max_consecutive:
            self.max_consecutive = window_length
        return str(self.max_consecutive)

    def Observe(self) -> str:
        """
        返回当前环境状态，包括数组和算法参数。
        """
        state = {
            "nums": self.nums,
            "max_consecutive": self.max_consecutive,
            "num_zeros": self.num_zeros,
            "left": self.left,
            "right": self.right,
            "allowed_zeros": self.allowed_zeros,
        }
        return json.dumps(state)

    def Done(self, answer: int) -> str:
        """
        验证最终答案是否正确并返回结果信息。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg

    def solve(self) -> str:
        """
        自动调用所有动作完成流程，并提交答案进行验证。
        """
        # 初始化
        self.step("\\boxed{init}")
        # 观察以获取长度
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        observe_data = json.loads(obs)
        nums_length = len(observe_data.get("nums", []))

        # 使用滑动窗口进行求解
        # 我们将使用内部指针状态，并通过动作与环境交互
        while self.right < nums_length:
            # 检查当前右指针位置是否为零
            self.step("\\boxed{check}")

            # 根据零计数调整左指针
            if self.num_zeros > self.allowed_zeros:
                self.step("\\boxed{adjust}")
            else:
                # 可选：观察获取当前 left（不必须）
                self.step("\\boxed{observe}")

            # 更新最大窗口长度
            self.step("\\boxed{update}")

            # 移动右指针
            # 当 right 已经在末尾时，move 会保持在末尾，手动跳出
            prev_right = self.right
            self.step("\\boxed{move}")
            if self.right == prev_right and self.right == nums_length - 1:
                break

            # 如果右指针刚好在末尾，下一轮会跳出
            if self.right >= nums_length - 1:
                # 进行一次最后的处理（如果需要）
                pass

            # 如果达到末尾则停止
            if self.right >= nums_length:
                break

            # 防止无限循环
            if self.turn_count >= self.max_turns:
                break

        # 最终观察并提交答案
        final_obs, _, _, _, _ = self.step("\\boxed{observe}")
        final_state = json.loads(final_obs)
        max_length = final_state["max_consecutive"]

        answer_obs, reward, terminated, truncated, info = self.step(f"\\boxed{{answer {max_length}}}")
        return answer_obs