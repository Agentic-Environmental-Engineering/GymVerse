from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxApplesEnvGEM(Env):
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
            "grid_rows": (2, 15),         # 网格行数
            "grid_cols": (2, 15),         # 网格列数
            "cell_value_max": (5, 100),   # 单元格最大苹果数
        }

        # 参数方差（可选）
        self.param_variance = {
            "grid_rows": 1,
            "grid_cols": 1,
            "cell_value_max": 5,
        }

        # 占位属性
        self.grid_rows: int = 0
        self.grid_cols: int = 0
        self.cell_value_max: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 环境状态（与原环境保持一致的辅助属性）
        self.grid: Optional[list] = None
        self.dp: Optional[list] = None
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
            "Max Apples: Collect the maximum apples from top-left to bottom-right using DP.\n"
            "Grid cells contain positive integers. You can move only right or down.\n"
            "Available actions:\n"
            "- Initialize DP: \\boxed{init}\n"
            "- Fill first row: \\boxed{fill row}\n"
            "- Fill first column: \\boxed{fill col}\n"
            "- Fill DP cell (i, j): \\boxed{fill i j}  (i,j are 0-indexed; requires i>0 and j>0)\n"
            "- Get max apples: \\boxed{get}\n"
            "- Observe state: \\boxed{observe}\n"
            "- Submit answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        rows = self.grid_rows
        cols = self.grid_cols
        return f"Grid: {rows}x{cols}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.grid = self.problem["grid"]
        self.dp = None
        self._reward = 0.0
        self._done = False

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        rows = self.grid_rows
        cols = self.grid_cols
        max_val = self.cell_value_max
        grid = [[random.randint(1, max_val) for _ in range(cols)] for _ in range(rows)]
        return {"grid": grid, "rows": rows, "cols": cols}

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
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # 动作处理
        try:
            if tokens[0].lower() in ["init", "initialize", "initialize_dp"]:
                obs = self.InitializeDp()

            elif tokens[0].lower() == "fill":
                if len(tokens) == 2 and tokens[1].lower() in ["row", "first_row"]:
                    obs = self.FillFirstRow()
                elif len(tokens) == 2 and tokens[1].lower() in ["col", "column", "first_col", "first_column"]:
                    obs = self.FillFirstColumn()
                elif len(tokens) == 3:
                    try:
                        i = int(tokens[1])
                        j = int(tokens[2])
                    except ValueError:
                        obs = "Error: i and j must be integers."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    # 执行单元格填充
                    res = self.FillDpCell(i, j)
                    if res.startswith("Error"):
                        obs = res
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    obs = res
                else:
                    obs = "Error: invalid fill syntax."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

            elif tokens[0].lower() in ["get", "get_max", "getmax", "max"]:
                obs = self.GetMaxApples()

            elif tokens[0].lower() in ["observe", "status", "state"]:
                obs = self.Observe()

            elif tokens[0].lower() in ["answer", "submit"]:
                if len(tokens) != 2:
                    obs = "Error: answer requires a single integer parameter."
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
                    obs = "Error: answer must be an integer."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                obs = self.Done(ans)
                # 根据 Done 结果设置奖励和终止
                terminated = True
                reward = 1.0 if "Correct" in obs else -1.0

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
        # 提供一个合理的示例动作
        return "\\boxed{observe}"

    # -----------------------------
    # 保留原环境的辅助方法并转换
    # -----------------------------

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    def InitializeDp(self):
        r"""
        Initialize the dynamic programming (DP) array, set its size to be the same as the grid, and set the starting point value.
        """
        if not self.grid or not self.grid[0]:
            self.dp = []
            return "DP array initialized. Current DP array: []"

        n, m = len(self.grid), len(self.grid[0])
        self.dp = [[0] * m for _ in range(n)]
        if n > 0 and m > 0:
            self.dp[0][0] = self.grid[0][0]

        return f"DP array initialized. Current DP array: {str(self.dp)}"

    def FillFirstRow(self):
        r"""
        Fill the first row of the DP array, which can only be reached from the left cell.
        """
        if self.dp is None or len(self.dp) == 0 or len(self.dp[0]) == 0:
            return "Error: DP array not initialized or is empty"

        m = len(self.dp[0])
        for j in range(1, m):
            self.dp[0][j] = self.dp[0][j - 1] + self.grid[0][j]

        return f"First row of DP array filled. Current first row: {str(self.dp[0])}"

    def FillFirstColumn(self):
        r"""
        Fill the first column of the DP array, which can only be reached from the cell above.
        """
        if self.dp is None or len(self.dp) == 0:
            return "Error: DP array not initialized or is empty"

        n = len(self.dp)
        for i in range(1, n):
            self.dp[i][0] = self.dp[i - 1][0] + self.grid[i][0]

        first_column = [self.dp[i][0] for i in range(n)]
        return f"First column of DP array filled. Current first column: {str(first_column)}"

    def FillDpCell(self, i: int, j: int):
        r"""
        Fill the cell with coordinates (i, j) in the DP array.
        Requires i>0 and j>0 for valid transitions (from top or left).
        """
        if self.dp is None or i >= len(self.dp) or j >= len(self.dp[0]) or i < 0 or j < 0:
            return "Error: Index out of range or DP array not initialized"

        if i == 0 or j == 0:
            return "Error: i and j must be both > 0 for FillDpCell"

        self.dp[i][j] = max(self.dp[i - 1][j], self.dp[i][j - 1]) + self.grid[i][j]
        return f"DP[{i}][{j}] filled. Value: {self.dp[i][j]}"

    def GetMaxApples(self):
        r"""
        Get the value of the bottom-right cell in the DP array.
        """
        if self.dp is None or len(self.dp) == 0 or len(self.dp[0]) == 0:
            return "0"

        return str(self.dp[-1][-1])

    def Observe(self):
        r"""
        Return the current grid information and the state of the DP array.
        """
        dp_status = str(self.dp) if self.dp is not None else "uninitialized"
        return f"Grid: {str(self.grid)}. DP array: {dp_status}"

    def get_ref_answer(self):
        r"""
        Use the information in the environment to get the reference answer.
        """
        if not self.grid or not self.grid[0]:
            return 0

        n, m = len(self.grid), len(self.grid[0])
        dp = [[0] * m for _ in range(n)]

        dp[0][0] = self.grid[0][0]

        for i in range(1, n):
            dp[i][0] = dp[i - 1][0] + self.grid[i][0]

        for j in range(1, m):
            dp[0][j] = dp[0][j - 1] + self.grid[0][j]

        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + self.grid[i][j]

        return dp[-1][-1]

    def Done(self, answer):
        r"""
        Verify whether the final answer is correct and return the result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1.0 if correct else -1.0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self) -> str:
        r"""
        Automatically call all actions to complete the process, and submit the answer for verification.
        This uses GEM-style boxed actions.
        """
        # Initialize and fill edges
        self.step("\\boxed{init}")
        self.step("\\boxed{fill row}")
        self.step("\\boxed{fill col}")
        # Observe to get grid sizes
        self.step("\\boxed{observe}")
        n = len(self.grid)
        m = len(self.grid[0]) if n > 0 else 0
        # Fill interior cells
        for i in range(1, n):
            for j in range(1, m):
                self.step(f"\\boxed{{fill {i} {j}}}")
        # Get max and submit
        get_obs, _, _, _, _ = self.step("\\boxed{get}")
        try:
            max_apples = int(get_obs.strip())
        except Exception:
            max_apples = self.get_ref_answer()
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {max_apples}}}")
        return final_obs