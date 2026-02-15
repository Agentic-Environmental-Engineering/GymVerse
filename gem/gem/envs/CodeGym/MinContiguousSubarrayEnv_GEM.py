from typing import Any, Dict, Optional, Tuple
import random
import re
from collections import defaultdict
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MinContiguousSubarrayEnvGEM(Env):
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

        # 难度参数范围（根据原环境分析）
        self.complexity_params = {
            "array_length": (5, 50),     # 数组长度 N
            "num_colors": (2, 8),        # 颜色种类 C
            "value_range": (3, 20),      # 值域大小（可选，主要用于标签空间与噪声控制）
            "turn_budget": (30, 200),    # 推荐的步数预算（不强制，供信息展示）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "array_length": 2,
            "num_colors": 1,
            "value_range": 3,
            "turn_budget": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.num_colors: int = 0
        self.value_range: int = 0
        self.turn_budget: int = 0

        # 原环境核心状态变量
        self.N: int = 0
        self.C: int = 0
        self.colors: list[int] = []

        # 滑动窗口变量
        self.left: int = 0
        self.right: int = 0
        self.color_count: Dict[int, int] = defaultdict(int)
        self.distinct_colors: int = 0
        self.min_length: float = float("inf")

        # 其他状态
        self.turn_count: int = 0
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

        # 额外一致性修正：确保 num_colors <= array_length
        if self.num_colors > self.array_length:
            self.num_colors = self.array_length
        # 最少两种颜色更有意义
        if self.num_colors < 2 and self.array_length >= 2:
            self.num_colors = 2

    def _get_instructions(self) -> str:
        return (
            "Min Contiguous Subarray (GEM): Find the minimal length of a contiguous subarray "
            "that contains all color IDs 1..C at least once.\n"
            "Available actions (use \\boxed{...}):\n"
            "- Move right pointer: \\boxed{move_right}\n"
            "- Move left pointer: \\boxed{move_left}\n"
            "- Record current window length (if includes all colors): \\boxed{record}\n"
            "- Get current minimum window length: \\boxed{get_min}\n"
            "- Check whether current window includes all colors: \\boxed{check}\n"
            "- Observe state: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "Goal: Submit the minimal window length as an integer.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"N={self.N}, C={self.C} | "
            f"Left={self.left}, Right={self.right}, Distinct={self.distinct_colors} | "
            f"Turn: {self.turn_count}/{self.max_turns} (recommended ≤{self.turn_budget})\n"
            "Enter an action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 将问题参数写入环境状态
        self.N = self.problem["N"]
        self.C = self.problem["C"]
        self.colors = self.problem["colors"][:]

        # 重置滑动窗口与统计
        self.left = 0
        self.right = 0
        self.color_count = defaultdict(int)
        self.distinct_colors = 0
        self.min_length = float("inf")

        # 重置回合控制
        self.turn_count = 0
        self._terminated = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        N = self.array_length
        C = max(1, min(self.num_colors, N))  # 安全限制

        # 确保所有颜色至少出现一次，保证答案存在
        base = list(range(1, C + 1))
        rest_len = max(0, N - C)
        rest = [random.randint(1, C) for _ in range(rest_len)]
        colors = base + rest
        random.shuffle(colors)

        return {"N": N, "C": C, "colors": colors}

    # ------------------ 原环境的辅助方法（转换保留） ------------------

    def get_ref_answer(self):
        """
        用滑动窗口算法计算参考答案（最短包含所有 C 种颜色的子数组长度）
        """
        color_count = defaultdict(int)
        distinct_colors = 0
        min_length = float("inf")
        left = 0

        for right in range(self.N):
            if color_count[self.colors[right]] == 0:
                distinct_colors += 1
            color_count[self.colors[right]] += 1

            while distinct_colors == self.C:
                min_length = min(min_length, right - left + 1)
                color_count[self.colors[left]] -= 1
                if color_count[self.colors[left]] == 0:
                    distinct_colors -= 1
                left += 1

        return min_length

    def MoveRightPointer(self) -> str:
        """
        向右移动右指针，并更新当前窗口的颜色计数。
        返回移动后右指针位置及其颜色信息。
        """
        if self.right < self.N:
            color = self.colors[self.right]
            if self.color_count[color] == 0:
                self.distinct_colors += 1
            self.color_count[color] += 1

            prev_right = self.right
            self.right += 1
            return f"Right pointer moved to position {prev_right}, color is {color}"
        return "Right pointer has reached the end, cannot move further"

    def MoveLeftPointer(self) -> str:
        """
        向右移动左指针，并更新当前窗口的颜色计数。
        返回移动后左指针位置及其颜色信息。
        """
        if self.left < self.right:
            color = self.colors[self.left]
            self.color_count[color] -= 1
            if self.color_count[color] == 0:
                self.distinct_colors -= 1

            prev_left = self.left
            self.left += 1
            return f"Left pointer moved to position {prev_left}, color is {color}"
        return "Left pointer has reached the right pointer position, cannot move further"

    def RecordWindowLength(self) -> str:
        """
        记录当前窗口长度并在包含所有颜色时更新最短窗口。
        """
        current_length = self.right - self.left
        if self.distinct_colors == self.C and current_length > 0:
            prev_min = self.min_length
            self.min_length = min(self.min_length, current_length)
            if self.min_length < prev_min:
                return f"Current window length is {current_length}, contains all colors, minimum window length updated to {self.min_length}"
            return f"Current window length is {current_length}, contains all colors, minimum window length remains {self.min_length}"
        return f"Current window length is {current_length}, does not contain all colors, minimum window length not updated"

    def GetMinWindowLength(self) -> str:
        """
        返回当前记录的最短窗口长度。
        """
        return str(self.min_length) if self.min_length != float("inf") else "inf"

    def CheckAllColorsPresent(self) -> str:
        """
        检查当前窗口是否包含所有颜色。
        返回格式: 'True,3' 或 'False,2'
        """
        return f"{self.distinct_colors == self.C},{self.distinct_colors}"

    def Observe(self) -> str:
        """
        返回当前状态的观测信息：左右指针位置与当前不同颜色数。
        """
        return (
            f"Left pointer position: {self.left}, "
            f"Right pointer position: {self.right}, "
            f"Current number of distinct colors: {self.distinct_colors}"
        )

    def Done(self, answer: int) -> str:
        """
        验证最终答案是否正确并返回结果信息。
        """
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        return (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )

    # ------------------ GEM 接口方法 ------------------

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._terminated:
            # 已终局的环境再次调用 step，视为无效动作
            obs = "Episode already terminated. Please reset."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        self.turn_count += 1
        parsed = self._parse_action(action)

        # 格式错误
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
        cmd = content["command"]
        arg = content.get("arg", None)

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd == "move_right":
                obs = self.MoveRightPointer()
            elif cmd == "move_left":
                obs = self.MoveLeftPointer()
            elif cmd == "record":
                obs = self.RecordWindowLength()
            elif cmd == "get_min":
                obs = self.GetMinWindowLength()
            elif cmd == "check":
                obs = self.CheckAllColorsPresent()
            elif cmd == "observe":
                obs = self.Observe()
            elif cmd == "answer":
                # 支持 'inf' 或整数
                if isinstance(arg, str) and arg.lower() == "inf":
                    submitted = float("inf")
                elif isinstance(arg, int):
                    submitted = arg
                else:
                    # 若是可转为整数的字符串
                    try:
                        submitted = int(arg)
                    except Exception:
                        obs = "Invalid answer format. Use integer N (or 'inf')."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                obs = self.Done(submitted)
                ref = self.get_ref_answer()
                correct = (submitted == ref)
                reward = 1.0 if correct else -1.0
                terminated = True
                self._terminated = True
            else:
                obs = f"Invalid action: {cmd}"
                return (
                    obs,
                    LanguageGameReward.invalid_action_reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
        except Exception as e:
            obs = f"Runtime error: {str(e)}"
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
            terminated = True
            truncated = True
            reward = 0.0
            self._terminated = True

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        """
        解析 \boxed{...} 中的动作文本。
        支持命令：
        - move_right | right | r
        - move_left  | left  | l
        - record
        - get_min
        - check
        - observe | obs
        - answer N | submit N
        """
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        content_lower = content.lower()

        # 拆分为指令和参数
        parts = content_lower.split()
        if not parts:
            return None

        cmd = parts[0]
        arg = None

        # 规范化命令
        if cmd in ["move_right", "right", "r"]:
            command = "move_right"
        elif cmd in ["move_left", "left", "l"]:
            command = "move_left"
        elif cmd == "record":
            command = "record"
        elif cmd == "get_min":
            command = "get_min"
        elif cmd == "check":
            command = "check"
        elif cmd in ["observe", "obs"]:
            command = "observe"
        elif cmd in ["answer", "submit"]:
            command = "answer"
            if len(parts) < 2:
                return {"content": {"command": "answer", "arg": None}}
            # 尝试解析参数为 int；也允许 'inf'
            arg_text = parts[1]
            if arg_text == "inf":
                arg = "inf"
            else:
                try:
                    arg = int(arg_text)
                except Exception:
                    arg = arg_text  # 留给 step 中做进一步校验
        else:
            # 未知命令
            command = cmd

        return {"content": {"command": command, "arg": arg}}

    def sample_random_action(self) -> str:
        # 随机选择一个非终止动作
        actions = ["observe", "check", "move_right", "record", "move_left", "get_min"]
        return f"\\boxed{{{random.choice(actions)}}}"