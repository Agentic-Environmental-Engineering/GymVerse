from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LongestTwoColorSubarrayEnvGEM(Env):
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

        # 难度参数范围：控制数组长度与颜色种类范围
        self.complexity_params = {
            "array_length": (5, 50),          # 颜色数组长度
            "color_value_range": (2, 10),     # 颜色取值范围 [1..color_value_range]
            "max_distinct_limit": (2, 2),     # 允许的最大不同颜色数（本题固定为2）
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "array_length": 3,          # ±3 的方差
            "color_value_range": 2,     # ±2 的方差
            "max_distinct_limit": 0,    # 固定为2，不随机
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.array_length: int = 0
        self.color_value_range: int = 0
        self.max_distinct_limit: int = 2

        # 问题实例
        self.problem: Dict[str, Any] = {}

        # 状态变量
        self.turn_count: int = 0

        # 内部求解状态（滑动窗口）
        self.n: int = 0
        self.colors: list = []
        self.left: int = 0
        self.right: int = 0
        self.max_length: int = 0
        self.color_count: Dict[int, int] = {}

        # 评估相关兼容属性
        self._reward: float = 0.0
        self._done: bool = False

        self.reset()

    # 兼容属性（保留）
    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        """
        支持从 CodeGym 风格 env_str 构造：
        例如：
        "LongestTwoColorSubarrayEnv@{\"n\": 7, \"colors\": [1,2,1,2,1,3,4]}"
        或
        "LongestTwoColorSubarrayEnvGEM@{\"n\": 7, \"colors\": [1,2,1,2,1,3,4]}"
        """
        prefix1 = "LongestTwoColorSubarrayEnv@"
        prefix2 = "LongestTwoColorSubarrayEnvGEM@"
        if env_str.startswith(prefix1):
            payload = env_str[len(prefix1):]
        elif env_str.startswith(prefix2):
            payload = env_str[len(prefix2):]
        else:
            return None
        try:
            import ast
            options = ast.literal_eval(payload)
        except Exception:
            return None
        env = LongestTwoColorSubarrayEnvGEM()
        env.set_problem(options)
        return env

    def set_problem(self, options: Dict[str, Any]):
        """设置固定问题实例（不受复杂度随机影响）"""
        self.n = int(options.get("n", 0))
        self.colors = list(options.get("colors", []))
        self.problem = {"colors": self.colors, "n": self.n}
        # 重置滑动窗口状态
        self.left = 0
        self.right = 0
        self.max_length = 0
        self.color_count = {}
        self.turn_count = 0
        self._reward = 0.0
        self._done = False

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
                    # 保证范围合法
                    actual_value = max(min_val, min(max_val, actual_value))

            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Longest Two-Color Subarray:\n"
            "Find the longest contiguous subarray that contains at most two distinct colors.\n"
            "Available actions (use boxed syntax):\n"
            "- Initialize sliding window: \\boxed{init}\n"
            "- Move right pointer to index r: \\boxed{move r}\n"
            "- Update color count for value c: \\boxed{count c}\n"
            "- Adjust left pointer when distinct colors > 2: \\boxed{adjust}\n"
            "- Calculate window length given left L and right R: \\boxed{length L R}\n"
            "- Update max length with current length N: \\boxed{update N}\n"
            "- Observe current state: \\boxed{observe}\n"
            "- Submit answer (final max length): \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Array length: {self.n}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter action."
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
        self.n = self.problem["n"]
        self.colors = self.problem["colors"]
        self.left = 0
        self.right = 0
        self.max_length = 0
        self.color_count = {}
        self.turn_count = 0
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.array_length
        # 颜色值范围为 [1..color_value_range]
        colors = [random.randint(1, self.color_value_range) for _ in range(n)]
        return {"colors": colors, "n": n}

    # 保留并转换原环境的辅助方法
    def get_ref_answer(self):
        """
        计算参考答案：最长的包含至多两种颜色的子数组长度。
        """
        if self.n == 0 or not self.colors:
            return 0

        left = 0
        max_length = 0
        color_count: Dict[int, int] = {}

        for right in range(self.n):
            color_count[self.colors[right]] = color_count.get(self.colors[right], 0) + 1

            while len(color_count) > 2:
                color_count[self.colors[left]] -= 1
                if color_count[self.colors[left]] == 0:
                    del color_count[self.colors[left]]
                left += 1

            max_length = max(max_length, right - left + 1)

        return max_length

    # 原动作方法（保留）
    def InitializeWindow(self):
        self.left = 0
        self.max_length = 0
        self.color_count = {}
        return f"Sliding window initialized: left=0, max_length=0, color_count={self.color_count}"

    def MoveRightPointer(self, right: int):
        self.right = right
        return str(self.right)

    def UpdateColorCount(self, color: int):
        self.color_count[color] = self.color_count.get(color, 0) + 1
        return str(self.color_count)

    def AdjustLeftPointer(self):
        while len(self.color_count) > 2:
            left_color = self.colors[self.left]
            self.color_count[left_color] -= 1
            if self.color_count[left_color] == 0:
                del self.color_count[left_color]
            self.left += 1
        return f"left={self.left}, color_count={self.color_count}"

    def CalculateWindowLength(self, left: int, right: int):
        length = right - left + 1
        return str(length)

    def UpdateMaxLength(self, current_length: int):
        self.max_length = max(self.max_length, current_length)
        return str(self.max_length)

    def Observe(self):
        return (
            f"Color array: {self.colors}, left={self.left}, right={self.right}, "
            f"max_length={self.max_length}, color_count={self.color_count}"
        )

    def Done(self, answer: int):
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg + f", reward={self._reward}"

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
        if len(tokens) == 0:
            obs = f"Format error at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "init":
                obs = self.InitializeWindow()

            elif cmd == "move":
                if len(tokens) < 2:
                    obs = "Error: 'move' requires index r."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                r = int(tokens[1])
                if r < 0 or r >= self.n:
                    obs = f"Invalid index r={r} for 'move'."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.MoveRightPointer(r)

            elif cmd == "count":
                if len(tokens) < 2:
                    obs = "Error: 'count' requires color value c."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                c = int(tokens[1])
                obs = self.UpdateColorCount(c)

            elif cmd == "adjust":
                obs = self.AdjustLeftPointer()

            elif cmd == "length":
                if len(tokens) < 3:
                    obs = "Error: 'length' requires left L and right R."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                L = int(tokens[1])
                R = int(tokens[2])
                if not (0 <= L <= R < self.n):
                    obs = f"Invalid L={L}, R={R}."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.CalculateWindowLength(L, R)

            elif cmd == "update":
                if len(tokens) < 2:
                    obs = "Error: 'update' requires current length N."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                N = int(tokens[1])
                obs = self.UpdateMaxLength(N)

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "answer":
                if len(tokens) < 2:
                    obs = "Error: 'answer' requires final length N."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans = int(tokens[1])
                obs = self.Done(ans)
                reward = self._reward
                terminated = True

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
            obs = f"Error: {str(e)}"
            return (
                obs,
                LanguageGameReward.format_error_reward,
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
        # 简单示例动作
        # 优先提供 observe 或 init
        if self.turn_count == 0:
            return "\\boxed{init}"
        return "\\boxed{observe}"

    # 转换版的 solve 方法（使用 GEM step 接口自动执行）
    def solve(self) -> str:
        # Initialize
        self.step("\\boxed{init}")
        observe_info, _, _, _, _ = self.step("\\boxed{observe}")
        # 解析数组
        try:
            # "Color array: [....], left=..., right=..., max_length=..., color_count=..."
            array_str = observe_info.split("Color array: ")[1].split(", left=")[0]
            color_array = eval(array_str)  # 安全性：输入为内部生成，受控
        except Exception:
            color_array = self.colors[:]
        n = len(color_array)

        for right in range(n):
            self.step(f"\\boxed{{move {right}}}")
            current_color = color_array[right]
            self.step(f"\\boxed{{count {current_color}}}")
            self.step("\\boxed{adjust}")
            adjusted_info, _, _, _, _ = self.step("\\boxed{observe}")
            try:
                left = int(adjusted_info.split("left=")[1].split(", right=")[0])
            except Exception:
                left = self.left
            length_obs, _, _, _, _ = self.step(f"\\boxed{{length {left} {right}}}")
            try:
                current_length = int(length_obs)
            except Exception:
                current_length = right - left + 1
            self.step(f"\\boxed{{update {current_length}}}")

        final_observe, _, _, _, _ = self.step("\\boxed{observe}")
        try:
            max_length = int(final_observe.split("max_length=")[1].split(", color_count=")[0])
        except Exception:
            max_length = self.max_length
        result_obs, reward, terminated, _, _ = self.step(f"\\boxed{{answer {max_length}}}")
        # 返回最终验证信息字符串（与原环境兼容）
        return result_obs