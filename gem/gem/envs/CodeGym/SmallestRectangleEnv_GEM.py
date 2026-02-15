from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class SmallestRectangleEnvGEM(Env):
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

        # 定义难度参数范围（根据原环境分析）
        # 影响难度的因素：点数（数组长度）、坐标取值范围（搜索空间）
        self.complexity_params = {
            "num_points": (5, 50),        # 点数量（数组长度）
            "value_range": (10, 10000),   # 坐标绝对值范围 [-value_range, value_range]
        }

        # 参数方差（用于微调随机性；enable_param_randomization=True 时生效）
        self.param_variance = {
            "num_points": 2,
            "value_range": 50,
        }

        # 占位属性
        self.num_points: int = 0
        self.value_range: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题数据
        self.points: list[Tuple[int, int]] = []

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
            "Smallest Rectangle: Given a set of 2D points, compute the area of the smallest axis-aligned rectangle that encloses all points.\n"
            "Available actions (use the latest \\boxed{...}):\n"
            "- Observe points: \\boxed{observe}\n"
            "- Find min x: \\boxed{find min_x}\n"
            "- Find max x: \\boxed{find max_x}\n"
            "- Find min y: \\boxed{find min_y}\n"
            "- Find max y: \\boxed{find max_y}\n"
            "- Calculate width: \\boxed{calc width min_x=NUM max_x=NUM}\n"
            "- Calculate height: \\boxed{calc height min_y=NUM max_y=NUM}\n"
            "- Calculate area: \\boxed{calc area width=NUM height=NUM}\n"
            "- Submit answer: \\boxed{answer NUM}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Points: {self.num_points}, Range: [-{self.value_range}, {self.value_range}]\n"
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
        self.points = self.problem["points"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        pts = []
        for _ in range(self.num_points):
            x = random.randint(-self.value_range, self.value_range)
            y = random.randint(-self.value_range, self.value_range)
            pts.append((x, y))
        return {"points": pts, "size": self.num_points}

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

        content = parsed["content"].strip().lower()

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if content == "observe":
                obs = self.Observe()

            elif content == "find min_x":
                obs = self.FindMinX()

            elif content == "find max_x":
                obs = self.FindMaxX()

            elif content == "find min_y":
                obs = self.FindMinY()

            elif content == "find max_y":
                obs = self.FindMaxY()

            elif content.startswith("calc width"):
                params = self._parse_kv_params(content)
                if "min_x" in params and "max_x" in params:
                    min_x = int(params["min_x"])
                    max_x = int(params["max_x"])
                    obs = self.CalculateWidth(min_x, max_x)
                else:
                    obs = "Error: 'min_x' or 'max_x' parameter is missing for calc width."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

            elif content.startswith("calc height"):
                params = self._parse_kv_params(content)
                if "min_y" in params and "max_y" in params:
                    min_y = int(params["min_y"])
                    max_y = int(params["max_y"])
                    obs = self.CalculateHeight(min_y, max_y)
                else:
                    obs = "Error: 'min_y' or 'max_y' parameter is missing for calc height."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

            elif content.startswith("calc area"):
                params = self._parse_kv_params(content)
                if "width" in params and "height" in params:
                    width = int(params["width"])
                    height = int(params["height"])
                    obs = self.CalculateArea(width, height)
                else:
                    obs = "Error: 'width' or 'height' parameter is missing for calc area."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True

            elif content.startswith("answer"):
                # answer NUM
                m = re.match(r"answer\s+(-?\d+)", content)
                if not m:
                    obs = "Format error: use \\boxed{answer NUM}."
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    area = int(m.group(1))
                    msg = self.Done(area)
                    obs = msg
                    # 依据结果设置奖励
                    if "Result: Correct" in msg:
                        reward = 1.0
                    else:
                        reward = -1.0
                    terminated = True

            else:
                obs = f"Invalid action: {content}"
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
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        return {"content": content}

    def _parse_kv_params(self, content: str) -> Dict[str, str]:
        """
        从命令内容中解析 key=value 参数对。
        例如: 'calc width min_x=1 max_x=4' -> {'min_x': '1', 'max_x': '4'}
        """
        params: Dict[str, str] = {}
        for k, v in re.findall(r"([a-zA-Z_]+)\s*=\s*(-?\d+)", content):
            params[k.lower()] = v
        return params

    def sample_random_action(self) -> str:
        return "\\boxed{observe}"

    # --------------------------
    # 保留并转换原环境的辅助方法
    # --------------------------
    def FindMinX(self) -> str:
        """
        Find the minimum x-coordinate among all points.
        Returns: str (e.g., '0')
        """
        if not self.points:
            return "0"
        min_x = min(point[0] for point in self.points)
        return str(min_x)

    def FindMaxX(self) -> str:
        """
        Find the maximum x-coordinate among all points.
        Returns: str (e.g., '4')
        """
        if not self.points:
            return "0"
        max_x = max(point[0] for point in self.points)
        return str(max_x)

    def FindMinY(self) -> str:
        """
        Find the minimum y-coordinate among all points.
        Returns: str (e.g., '0')
        """
        if not self.points:
            return "0"
        min_y = min(point[1] for point in self.points)
        return str(min_y)

    def FindMaxY(self) -> str:
        """
        Find the maximum y-coordinate among all points.
        Returns: str (e.g., '5')
        """
        if not self.points:
            return "0"
        max_y = max(point[1] for point in self.points)
        return str(max_y)

    def CalculateWidth(self, min_x: int, max_x: int) -> str:
        """
        Calculate the width: max_x - min_x.
        Returns: str
        """
        width = max_x - min_x
        return str(width)

    def CalculateHeight(self, min_y: int, max_y: int) -> str:
        """
        Calculate the height: max_y - min_y.
        Returns: str
        """
        height = max_y - min_y
        return str(height)

    def CalculateArea(self, width: int, height: int) -> str:
        """
        Calculate the area: width * height.
        Returns: str
        """
        area = width * height
        return str(area)

    def Observe(self) -> str:
        """
        Return the information of the point set in the current environment.
        Returns: str (list of points)
        """
        return str(self.points)

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        if not self.points:
            return 0

        min_x = min(point[0] for point in self.points)
        max_x = max(point[0] for point in self.points)
        min_y = min(point[1] for point in self.points)
        max_y = max(point[1] for point in self.points)

        width = max_x - min_x
        height = max_y - min_y

        return width * height

    def Done(self, area: int) -> str:
        """
        Verify whether the final answer is correct and return the result information.
        Returns: str message
        """
        ref_answer = self.get_ref_answer()
        correct = area == ref_answer
        msg = f"Your answer: {area}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        Automatically call actions to calculate the area of the smallest enclosing rectangle
        and submit the answer for verification. Returns final result message.
        """
        # Observe is optional; we go through the pipeline
        obs, _, _, _, _ = self.step("\\boxed{find min_x}")
        min_x = int(obs)
        obs, _, _, _, _ = self.step("\\boxed{find max_x}")
        max_x = int(obs)
        obs, _, _, _, _ = self.step(f"\\boxed{{calc width min_x={min_x} max_x={max_x}}}")
        width = int(obs)

        obs, _, _, _, _ = self.step("\\boxed{find min_y}")
        min_y = int(obs)
        obs, _, _, _, _ = self.step("\\boxed{find max_y}")
        max_y = int(obs)
        obs, _, _, _, _ = self.step(f"\\boxed{{calc height min_y={min_y} max_y={max_y}}}")
        height = int(obs)

        obs, _, _, _, _ = self.step(f"\\boxed{{calc area width={width} height={height}}}")
        area = int(obs)

        final_obs, reward, terminated, truncated, _ = self.step(f"\\boxed{{answer {area}}}")
        # Return the final observation string (message)
        return final_obs