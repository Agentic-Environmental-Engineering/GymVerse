from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class MaxGoldCoinsEnvGEM(Env):
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
            "num_rows": (2, 15),
            "num_cols": (2, 15),
            "cell_value_max": (5, 50),
        }

        # 参数方差
        self.param_variance = {
            "num_rows": 1,
            "num_cols": 1,
            "cell_value_max": 5,
        }

        # 占位属性
        self.num_rows: int = 0
        self.num_cols: int = 0
        self.cell_value_max: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题相关
        self.problem: Dict[str, Any] = {}
        self.grid: Optional[list] = None
        self.dp_table: Optional[list] = None

        # 终止与奖励状态
        self._terminated: bool = False
        self._last_reward: float = 0.0

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
            "Max Gold Coins: Compute maximum collectible gold moving right or down on a grid.\n"
            "Available actions (use \\boxed{...}):\n"
            "- Initialize DP table: \\boxed{init}\n"
            "- Fill first row: \\boxed{fill_row}\n"
            "- Fill first column: \\boxed{fill_col}\n"
            "- Fill DP cell (0-based indices, row>=1 and col>=1): \\boxed{fill r c}\n"
            "- Get max gold (bottom-right DP value): \\boxed{get}\n"
            "- Observe current state: \\boxed{observe}\n"
            "- Submit final answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        rows = self.num_rows
        cols = self.num_cols
        return f"Grid: {rows}x{cols}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # 仅影响实例生成，不影响难度参数

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.grid = self.problem.get("grid", [[]])

        self.dp_table = None
        self.turn_count = 0
        self._terminated = False
        self._last_reward = 0.0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        rows = self.num_rows
        cols = self.num_cols
        max_val = self.cell_value_max
        grid = [[random.randint(1, max_val) for _ in range(cols)] for _ in range(rows)]
        return {"grid": grid, "rows": rows, "cols": cols, "max_val": max_val}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        # 如果已经终止，后续动作视为无效
        if self._terminated:
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

        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            self._last_reward = LanguageGameReward.format_error_reward
            self._terminated = True
            return (
                obs,
                self._last_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        cmd = parsed.get("type")

        try:
            if cmd == "init":
                obs = self.InitializeDPTable()
            elif cmd == "fill_row":
                obs = self.FillFirstRow()
            elif cmd == "fill_col":
                obs = self.FillFirstColumn()
            elif cmd == "fill":
                row = parsed.get("row")
                col = parsed.get("col")
                if row is None or col is None:
                    obs = "Error: Missing coordinates for fill."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.FillDPTable(row, col)
                    # 若返回以 "Error:" 开头，视为失败
                    if obs.startswith("Error:"):
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
            elif cmd == "get":
                obs = self.GetMaxGold()
            elif cmd == "observe":
                obs = self.Observe()
            elif cmd == "answer":
                ans = parsed.get("answer")
                obs = self.Done(ans)
                # Done 方法内已进行判定，但这里需要设置奖励与终止标志
                correct = "Result: Correct" in obs
                reward = 1.0 if correct else -1.0
                terminated = True
                self._terminated = True
            else:
                obs = f"Invalid action: {parsed.get('raw', '')}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
        except Exception as e:
            obs = f"Error: {str(e)}"
            reward = LanguageGameReward.invalid_action_reward
            terminated = True

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            reward = 0.0
            terminated = True
            truncated = True
            self._terminated = True

        self._last_reward = reward
        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        raw = content
        tokens = content.split()
        if not tokens:
            return None

        cmd = tokens[0].lower()
        if cmd == "init":
            return {"type": "init", "raw": raw}
        if cmd == "fill_row":
            return {"type": "fill_row", "raw": raw}
        if cmd == "fill_col":
            return {"type": "fill_col", "raw": raw}
        if cmd == "fill":
            if len(tokens) >= 3:
                try:
                    row = int(tokens[1])
                    col = int(tokens[2])
                    return {"type": "fill", "row": row, "col": col, "raw": raw}
                except ValueError:
                    return {"type": "invalid", "raw": raw}
            else:
                return {"type": "invalid", "raw": raw}
        if cmd == "get":
            return {"type": "get", "raw": raw}
        if cmd == "observe":
            return {"type": "observe", "raw": raw}
        if cmd == "answer":
            if len(tokens) >= 2:
                try:
                    ans = int(tokens[1])
                    return {"type": "answer", "answer": ans, "raw": raw}
                except ValueError:
                    return {"type": "invalid", "raw": raw}
            else:
                return {"type": "invalid", "raw": raw}

        return {"type": "invalid", "raw": raw}

    def sample_random_action(self) -> str:
        # 简单策略：若未初始化，则 init；否则随机填充或观察或获取
        if self.dp_table is None:
            return "\\boxed{init}"
        choices = ["fill", "observe", "get"]
        choice = random.choice(choices)
        if choice == "fill":
            # 随机选择可填充的坐标
            r = random.randint(1, max(1, self.num_rows - 1))
            c = random.randint(1, max(1, self.num_cols - 1))
            return f"\\boxed{{fill {r} {c}}}"
        elif choice == "observe":
            return "\\boxed{observe}"
        else:
            return "\\boxed{get}"

    # 保留并转换原环境的辅助方法

    @property
    def finished(self) -> bool:
        return self._terminated

    @property
    def reward(self):
        return float(self._last_reward)

    def get_ref_answer(self):
        """
        使用环境信息获取参考答案。
        """
        if not self.grid or not self.grid[0]:
            return 0

        m, n = len(self.grid), len(self.grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = self.grid[0][0]

        for col in range(1, n):
            dp[0][col] = dp[0][col - 1] + self.grid[0][col]

        for row in range(1, m):
            dp[row][0] = dp[row - 1][0] + self.grid[row][0]

        for row in range(1, m):
            for col in range(1, n):
                dp[row][col] = max(dp[row - 1][col], dp[row][col - 1]) + self.grid[row][col]

        return dp[-1][-1]

    # 所有动作方法
    def InitializeDPTable(self):
        """
        初始化与 grid 同尺寸的 DP 表并设置左上角值。
        返回信息字符串。
        """
        if not self.grid or not self.grid[0]:
            self.dp_table = []
            return "DP table initialized successfully, size is 0x0"

        m, n = len(self.grid), len(self.grid[0])
        self.dp_table = [[0] * n for _ in range(m)]
        if m > 0 and n > 0:
            self.dp_table[0][0] = self.grid[0][0]
            return f"DP table initialized successfully, size is {m}x{n}, top-left value is {self.grid[0][0]}"
        return f"DP table initialized successfully, size is {m}x{n}"

    def FillFirstRow(self):
        """
        填充 DP 表的第一行。
        返回信息字符串。
        """
        if not self.dp_table or not self.dp_table[0]:
            return "Error: DP table not initialized or empty"

        n = len(self.dp_table[0])
        for col in range(1, n):
            self.dp_table[0][col] = self.dp_table[0][col - 1] + self.grid[0][col]

        return f"First row filled successfully, values are {self.dp_table[0]}"

    def FillFirstColumn(self):
        """
        填充 DP 表的第一列。
        返回信息字符串。
        """
        if not self.dp_table:
            return "Error: DP table not initialized"

        m = len(self.dp_table)
        for row in range(1, m):
            self.dp_table[row][0] = self.dp_table[row - 1][0] + self.grid[row][0]

        first_column = [self.dp_table[row][0] for row in range(m)]
        return f"First column filled successfully, values are {first_column}"

    def FillDPTable(self, row: int, col: int):
        """
        填充指定坐标的 DP 表单元（row, col），值为 max(上方, 左方) + 当前 grid 值。
        返回信息字符串。
        """
        if self.dp_table is None or not self.dp_table or not self.dp_table[0]:
            return "Error: DP table not initialized"
        if row < 0 or col < 0:
            return "Error: coordinates cannot be negative"
        # 要求 row>=1 和 col>=1 以避免越界访问上方或左方
        if row == 0 or col == 0:
            return "Error: row and col must be >= 1 to reference top/left cells"
        if row >= len(self.dp_table) or col >= len(self.dp_table[0]):
            return "Error: coordinates out of range"

        self.dp_table[row][col] = max(self.dp_table[row - 1][col], self.dp_table[row][col - 1]) + self.grid[row][col]
        return f"Filled successfully, value of cell ({row},{col}) is {self.dp_table[row][col]}"

    def GetMaxGold(self):
        """
        返回 DP 表右下角的值（最大黄金数）。
        """
        if not self.dp_table or not self.dp_table[0]:
            return "0"
        return str(self.dp_table[-1][-1])

    def Observe(self):
        """
        返回当前环境观测信息，包括 grid 和 dp_table。
        """
        return f"Grid: {self.grid}, DP table: {self.dp_table}"

    def Done(self, answer: int):
        """
        验证最终答案并返回结果信息。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._terminated = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={'1' if correct else '0'}"

    def solve(self) -> str:
        """
        自动执行完整流程并提交答案进行验证。
        返回最终验证结果信息字符串。
        """
        # 初始化
        self.step("\\boxed{init}")
        self.step("\\boxed{fill_row}")
        self.step("\\boxed{fill_col}")

        m = len(self.grid) if self.grid else 0
        n = len(self.grid[0]) if self.grid and self.grid[0] else 0

        # 填充其余 DP 表
        for row in range(1, m):
            for col in range(1, n):
                self.step(f"\\boxed{{fill {row} {col}}}")

        # 获取最大黄金值
        obs, _, _, _, _ = self.step("\\boxed{get}")
        try:
            max_gold = int(obs)
        except Exception:
            max_gold = self.get_ref_answer()

        # 提交答案
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {max_gold}}}")
        return final_obs