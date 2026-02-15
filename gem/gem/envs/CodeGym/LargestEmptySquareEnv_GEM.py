from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LargestEmptySquareEnvGEM(Env):
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

        # 难度参数范围（根据原环境分析）
        # 控制网格大小与障碍密度
        self.complexity_params = {
            "n_rows": (4, 25),              # 行数
            "n_cols": (4, 25),              # 列数
            "obstacle_percent": (5, 45),    # 障碍百分比（整数百分比）
        }

        # 参数方差（用于微调随机性）
        self.param_variance = {
            "n_rows": 1,
            "n_cols": 1,
            "obstacle_percent": 3,
        }

        # 占位属性
        self.n_rows: int = 0
        self.n_cols: int = 0
        self.obstacle_percent: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}
        self.grid: list = []
        self.dp_table: list = []

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
            "Largest Empty Square: Find the largest all-zero square in the grid.\n"
            "Grid cell values: 0 = empty, 1 = obstacle.\n"
            "Available actions:\n"
            "- Observe grid: \\boxed{observe}\n"
            "- Initialize DP table: \\boxed{init R C}\n"
            "- Update a DP cell: \\boxed{update I J V}\n"
            "- Find max square side from DP: \\boxed{findmax}\n"
            "- Submit final answer: \\boxed{answer N}\n"
            "Notes:\n"
            "- R, C are positive integers for DP table size.\n"
            "- I, J are zero-based indices in the DP table.\n"
            "- V is the value to set in DP cell.\n"
            "- N is the final largest square side length.\n"
        )

    def get_task_suffix(self) -> str:
        dp_inited = "yes" if (self.dp_table and len(self.dp_table) > 0) else "no"
        return (
            f"Grid: {self.n_rows}x{self.n_cols} | Turns: {self.turn_count}/{self.max_turns} | "
            f"DP initialized: {dp_inited}"
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
        self.grid = self.problem["grid"]
        self.dp_table = []  # 尚未初始化 DP 表

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

        # 根据难度参数随机生成问题实例

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        rows = self.n_rows
        cols = self.n_cols
        density = max(0.0, min(1.0, self.obstacle_percent / 100.0))

        grid = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                cell = 1 if random.random() < density else 0
                row.append(cell)
            grid.append(row)

        return {"grid": grid, "n": rows, "m": cols}

    # Actions adapted from original environment

    def Observe(self) -> str:
        """返回网格信息的 JSON 字符串"""
        observation = {
            "n": self.n_rows,
            "m": self.n_cols,
            "grid": self.grid,
        }
        return json.dumps(observation)

    def InitializeDPTable(self, rows: int, cols: int) -> str:
        """初始化 DP 表"""
        if rows <= 0 or cols <= 0:
            return f"Error: invalid DP size {rows}x{cols}"
        self.dp_table = [[0] * cols for _ in range(rows)]
        return f"DP table initialized, size {rows}x{cols}"

    def UpdateDPTableCell(self, i: int, j: int, value: int) -> str:
        """更新 DP 表单元"""
        if not self.dp_table or len(self.dp_table) == 0 or len(self.dp_table[0]) == 0:
            return "Error: DP table is not initialized"
        if i < 0 or j < 0:
            return f"Error: Cell ({i},{j}) is out of DP table range"
        if i >= len(self.dp_table) or j >= len(self.dp_table[0]):
            return f"Error: Cell ({i},{j}) is out of DP table range"
        self.dp_table[i][j] = value
        return f"DP table cell ({i},{j}) updated to {value}"

    def FindMaxSide(self) -> str:
        """在 DP 表中查找最大值"""
        if not self.dp_table:
            return "0"
        max_side = 0
        for row in self.dp_table:
            if row:
                current_max = max(row)
                if current_max > max_side:
                    max_side = current_max
        return str(max_side)

    def get_ref_answer(self) -> int:
        """根据当前网格计算参考答案（最大零方块边长）"""
        n = self.n_rows
        m = self.n_cols
        grid = self.grid
        if not grid or n == 0 or m == 0:
            return 0

        dp = [[0] * m for _ in range(n)]
        max_side = 0

        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    max_side = max(max_side, dp[i][j])
                else:
                    dp[i][j] = 0

        return max_side

    def Done(self, answer: int) -> str:
        """检查答案并返回结果信息（仅消息文本，不负责奖励/终止标志）"""
        ref_answer = self.get_ref_answer()
        correct = (answer == ref_answer)
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        # 格式错误
        if not parsed:
            obs = f"Format error at turn {self.turn_count}."
            reward = LanguageGameReward.format_error_reward
            terminated = True
            truncated = False
            info = {"suffix": self.get_task_suffix()}
            return obs, reward, terminated, truncated, info

        content = parsed["content"]
        tokens = content.strip().split()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            cmd = tokens[0].lower()

            if cmd == "observe":
                if len(tokens) != 1:
                    obs = "Error: observe takes no arguments."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.Observe()
                    reward = 0.0
                    terminated = False

            elif cmd == "init":
                if len(tokens) != 3:
                    obs = "Error: init requires 2 integers: R C."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    R = int(tokens[1])
                    C = int(tokens[2])
                    obs = self.InitializeDPTable(R, C)
                    if obs.startswith("Error"):
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        reward = 0.0
                        terminated = False

            elif cmd == "update":
                if len(tokens) != 4:
                    obs = "Error: update requires 3 integers: I J V."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    v = int(tokens[3])
                    obs = self.UpdateDPTableCell(i, j, v)
                    if obs.startswith("Error"):
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        reward = 0.0
                        terminated = False

            elif cmd == "findmax":
                if len(tokens) != 1:
                    obs = "Error: findmax takes no arguments."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.FindMaxSide()
                    reward = 0.0
                    terminated = False

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Error: answer requires 1 integer: N."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    ans = int(tokens[1])
                    result_msg = self.Done(ans)
                    correct = "Result: Correct" in result_msg
                    reward = 1.0 if correct else -1.0
                    terminated = True
                    obs = f"{result_msg}"

            else:
                obs = f"Invalid action: {cmd}"
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
        # 简单示例：观察或查找最大值
        choices = [
            "\\boxed{observe}",
            f"\\boxed{ { 'init ' + str(self.n_rows) + ' ' + str(self.n_cols) } }".replace(" ", "", 0),  # keep standard boxing
            "\\boxed{findmax}",
        ]
        return random.choice(choices)

    # 自动求解（保留并适配 GEM）
    def solve(self) -> str:
        """
        自动调用各动作完成流程并提交答案，返回最终验证信息文本。
        """
        # 1. 观察网格
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        grid_info = json.loads(obs)
        n = grid_info["n"]
        m = grid_info["m"]
        grid = grid_info["grid"]

        # 2. 初始化 DP 表
        self.step(f"\\boxed{{init {n} {m}}}")

        # 3. 计算 DP 值并逐步更新
        dp_table = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    if i == 0 or j == 0:
                        dp_table[i][j] = 1
                    else:
                        dp_table[i][j] = min(dp_table[i - 1][j], dp_table[i][j - 1], dp_table[i - 1][j - 1]) + 1
                    self.step(f"\\boxed{{update {i} {j} {dp_table[i][j]}}}")
                else:
                    dp_table[i][j] = 0
                    self.step(f"\\boxed{{update {i} {j} 0}}")

        # 4. 查找最大边长
        obs, _, _, _, _ = self.step("\\boxed{findmax}")
        try:
            max_side = int(obs.strip())
        except Exception:
            max_side = self.get_ref_answer()

        # 5. 提交答案
        obs, reward, terminated, truncated, _ = self.step(f"\\boxed{{answer {max_side}}}")
        return obs