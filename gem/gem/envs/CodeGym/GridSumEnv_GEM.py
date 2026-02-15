from typing import Any, Dict, Optional, Tuple, List
import random
import re
import ast
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class GridSumEnvGEM(Env):
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

        # 难度参数范围设计（针对网格大小、更新数量、数值范围、区域尺寸）
        self.complexity_params = {
            "grid_size": (2, 30),          # 方形网格边长
            "num_updates": (0, 100),       # 更新操作数量
            "value_max": (10, 10000),      # 元素最大值（最小值固定为 0）
            "region_max_dim": (1, 30),     # 区域最大尺寸（高/宽不超过该值）
        }

        # 参数方差（用于随机扰动）
        self.param_variance = {
            "grid_size": 1,
            "num_updates": 2,
            "value_max": 100,
            "region_max_dim": 1,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.grid_size: int = 0
        self.num_updates: int = 0
        self.value_max: int = 0
        self.region_max_dim: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题定义（grid/updates/region）
        self.grid: List[List[int]] = [[]]
        self.updates: List[Tuple[int, int, int]] = []
        self.region: Tuple[int, int, int, int] = (0, 0, 0, 0)

        # 用于从外部字符串强制设定问题实例（可选）
        self._forced_problem: Optional[Dict[str, Any]] = None

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

        # 保障依赖关系：region_max_dim 不应超过 grid_size
        self.region_max_dim = max(1, min(self.region_max_dim, self.grid_size))

        # 更新数量不超过总格子数
        total_cells = max(1, self.grid_size * self.grid_size)
        self.num_updates = max(0, min(self.num_updates, total_cells))

    def _get_instructions(self) -> str:
        return (
            "Grid Sum: Update cells and compute the sum over a rectangular region.\n"
            "Zero-based indices.\n"
            "Available actions:\n"
            "- Observe current state: \\boxed{observe}\n"
            "- Update a cell: \\boxed{update r c v}\n"
            "- Calculate region sum: \\boxed{sum r1 c1 r2 c2}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        rows = len(self.grid)
        cols = len(self.grid[0]) if rows > 0 else 0
        return (
            f"Grid: {rows}x{cols}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            f"Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        problem = self._generate_random_problem()
        self.grid = problem["grid"]
        self.updates = problem["updates"]
        self.region = problem["region"]

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        if self._forced_problem is not None:
            # 使用外部设定的实例（从 env_str 导入）
            return {
                "grid": [row[:] for row in self._forced_problem.get("grid", [[]])],
                "updates": list(self._forced_problem.get("updates", [])),
                "region": tuple(self._forced_problem.get("region", (0, 0, 0, 0))),
            }

        n = self.grid_size
        vmin = 0
        vmax = self.value_max

        # 生成网格
        grid = [[random.randint(vmin, vmax) for _ in range(n)] for _ in range(n)]

        # 生成更新列表（不重复位置）
        total_cells = n * n
        num_updates = min(self.num_updates, total_cells)
        positions = random.sample(range(total_cells), num_updates) if num_updates > 0 else []
        updates = []
        for pos in positions:
            r = pos // n
            c = pos % n
            val = random.randint(vmin, vmax)
            updates.append((r, c, val))

        # 生成区域（确保合法）
        max_h = self.region_max_dim
        max_w = self.region_max_dim
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        h = min(h, n)
        w = min(w, n)
        r1 = random.randint(0, n - h)
        c1 = random.randint(0, n - w)
        r2 = r1 + h - 1
        c2 = c1 + w - 1
        region = (r1, c1, r2, c2)

        return {"grid": grid, "updates": updates, "region": region}

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
        tokens = content.strip().split()
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
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd == "observe":
                obs = self.Observe()
                terminated = False
                reward = 0.0

            elif cmd == "update":
                if len(tokens) != 4:
                    obs = "Error: update requires exactly 3 integers: update r c v"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                r, c, v = int(tokens[1]), int(tokens[2]), int(tokens[3])
                obs = self.UpdateCell(r, c, v)
                # 出界等错误也视为无效动作
                if obs.startswith("Error:"):
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                terminated = False
                reward = 0.0

            elif cmd in ("sum", "calculate"):
                if len(tokens) != 5:
                    obs = "Error: sum requires exactly 4 integers: sum r1 c1 r2 c2"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                r1, c1, r2, c2 = int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4])
                obs = self.CalculateRegionSum(r1, c1, r2, c2)
                terminated = False
                reward = 0.0

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Error: answer requires exactly 1 integer: answer N"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans = int(tokens[1])
                msg = self.Done(ans)
                obs = msg
                # 依据正确与否返回奖励
                if "Result: Correct" in msg:
                    reward = 1.0
                else:
                    reward = -1.0
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
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        # 超时检查（放在 step 结尾）
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

    # --------------------------
    # 保留并转换原环境的辅助方法
    # --------------------------
    def get_ref_answer(self):
        """
        使用环境信息获得参考答案：在网格副本上应用 updates，再计算 region 的和。
        """
        # 副本避免修改原始网格
        temp_grid = [row.copy() for row in self.grid]

        # 应用所有更新
        for row, col, val in self.updates:
            if 0 <= row < len(temp_grid) and 0 <= col < len(temp_grid[0]):
                temp_grid[row][col] = val

        # 计算区域和
        row1, col1, row2, col2 = self.region
        total_sum = 0
        for i in range(row1, row2 + 1):
            for j in range(col1, col2 + 1):
                total_sum += temp_grid[i][j]
        return total_sum

    def UpdateCell(self, row: int, col: int, val: int):
        """
        更新网格指定位置的值。返回操作结果字符串。
        示例输出: "Updated cell (0, 1) to value 5"
        """
        if len(self.grid) == 0 or len(self.grid[0]) == 0:
            return "Error: Grid is empty"
        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0]):
            self.grid[row][col] = val
            return f"Updated cell ({row}, {col}) to value {val}"
        else:
            return f"Error: Cell ({row}, {col}) is out of grid bounds"

    def CalculateRegionSum(self, row1: int, col1: int, row2: int, col2: int):
        """
        计算指定矩形区域的所有单元格之和。返回和的字符串形式。
        示例输出: "13"
        """
        # 边界安全检查
        if row1 > row2 or col1 > col2:
            return "Error: Invalid region (row1>row2 or col1>col2)"
        if row1 < 0 or col1 < 0 or row2 >= len(self.grid) or col2 >= len(self.grid[0]):
            return "Error: Region out of bounds"
        total_sum = 0
        for i in range(row1, row2 + 1):
            for j in range(col1, col2 + 1):
                total_sum += self.grid[i][j]
        return str(total_sum)

    def Observe(self):
        """
        获取当前网格状态信息。
        示例输出:
        "Current grid: [[1, 5], [3, 4]], pending updates: [(0, 1, 5)], target region: (0, 0, 1, 1)"
        """
        return f"Current grid: {self.grid}, pending updates: {self.updates}, target region: {self.region}"

    def Done(self, answer):
        """
        校验最终答案是否正确并返回结果信息。
        示例输出:
        "Your answer: 13, Reference answer: 13, Result: Correct, reward=1"
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        result = "Correct" if correct else "Incorrect"
        reward_val = 1 if correct else 0
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {result}"
        return msg + f", reward={reward_val}"

    def solve(self) -> str:
        """
        自动解决流程：直接用参考答案计算并提交。
        返回最终答案的校验信息。
        """
        sum_result = self.get_ref_answer()
        return self.Done(sum_result)

    @staticmethod
    def from_env_str(env_str: str):
        """
        从原始环境字符串创建 GEM 环境实例。
        原始格式示例:
        GridSumEnv@{"grid": [[1,2],[3,4]], "updates": [(0,1,5)], "region": (0,0,1,1)}
        """
        prefix = "GridSumEnv@"
        if not env_str.startswith(prefix):
            return None
        try:
            options = ast.literal_eval(env_str.split("@", 1)[1])
            env = GridSumEnvGEM()
            env._forced_problem = {
                "grid": options.get("grid", [[]]),
                "updates": options.get("updates", []),
                "region": options.get("region", (0, 0, 0, 0)),
            }
            # 重置以应用强制问题
            env.reset()
            return env
        except Exception:
            return None