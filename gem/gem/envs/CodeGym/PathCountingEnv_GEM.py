from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class PathCountingEnvGEM(Env):
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

        # 可选的预设棋盘（通过 from_env_str 或 **_ 提供）
        self.preset_board = _.get("board", None)

        # 定义难度参数范围（根据原环境：棋盘规模与障碍密度）
        self.complexity_params = {
            "rows": (3, 25),           # 棋盘行数
            "cols": (3, 25),           # 棋盘列数
            "obstacle_pct": (5, 45),   # 障碍物百分比（%）
        }

        # 参数方差（用于训练期微随机）
        self.param_variance = {
            "rows": 1,
            "cols": 1,
            "obstacle_pct": 3,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.rows: int = 0
        self.cols: int = 0
        self.obstacle_pct: int = 0

        # 原环境状态
        self.board: list = []
        self.N: int = 0
        self.M: int = 0
        self._reward: float = 0.0
        self._done: bool = False

        # GEM 状态变量
        self.turn_count: int = 0
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
            "Path Counting: Compute number of paths from (0,0) to (N-1,M-1) with obstacles.\n"
            "Grid cells contain 0 (empty) or 1 (obstacle). Only move right or down.\n"
            "Available actions:\n"
            "- Get dimensions: \\boxed{dims}\n"
            "- Check a cell: \\boxed{check i j}\n"
            "- Calculate DP value (neighbors): \\boxed{dp i j up left}\n"
            "- Observe: \\boxed{observe}\n"
            "- Submit answer: \\boxed{answer N}\n"
            "Notes:\n"
            "- dp i j up left returns '0' if cell (i,j) is obstacle, otherwise up+left.\n"
            "- Indices are 0-based. Ensure i in [0,N), j in [0,M).\n"
        )

    def get_task_suffix(self) -> str:
        return f"Grid: {self.N}x{self.M} | Turn: {self.turn_count}/{self.max_turns} | Enter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 同步原环境核心变量
        self.board = self.problem["board"]
        self.N = len(self.board) if self.board else 0
        self.M = len(self.board[0]) if self.N > 0 and self.board[0] else 0
        self._reward = 0.0
        self._done = False

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        if self.preset_board is not None:
            # 使用预设棋盘（确保起点与终点合法）
            board = [row[:] for row in self.preset_board]
            if board:
                n = len(board)
                m = len(board[0]) if n > 0 else 0
                if n > 0 and m > 0:
                    board[0][0] = 0
                    board[n - 1][m - 1] = 0
            return {"board": board}

        n = self.rows
        m = self.cols
        density = max(0.0, min(1.0, self.obstacle_pct / 100.0))

        board = []
        for i in range(n):
            row = []
            for j in range(m):
                # 初始置空，后续随机障碍
                row.append(0)
            board.append(row)

        # 随机填充障碍
        total_cells = n * m
        num_obstacles_target = int(round(total_cells * density))
        placed = 0
        all_positions = [(i, j) for i in range(n) for j in range(m)]
        random.shuffle(all_positions)
        for (i, j) in all_positions:
            if (i, j) in [(0, 0), (n - 1, m - 1)]:
                continue
            if placed >= num_obstacles_target:
                break
            board[i][j] = 1
            placed += 1

        # 确保起点与终点是空格
        if n > 0 and m > 0:
            board[0][0] = 0
            board[n - 1][m - 1] = 0

        return {"board": board}

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
        if not tokens:
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
            if cmd in ["dims", "getdims", "get"]:
                obs = self.GetBoardDimensions()

            elif cmd == "check":
                if len(tokens) != 3:
                    obs = "Invalid 'check' usage. Expect: \\boxed{check i j}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i = int(tokens[1])
                j = int(tokens[2])
                obs = self.CheckCell(i, j)
                if obs.startswith("Error"):
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

            elif cmd == "dp":
                if len(tokens) != 5:
                    obs = "Invalid 'dp' usage. Expect: \\boxed{dp i j up left}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                i = int(tokens[1])
                j = int(tokens[2])
                up_val = int(tokens[3])
                left_val = int(tokens[4])
                obs = self.CalculateDpValueNeighbors(i, j, up_val, left_val)
                if obs.startswith("Error"):
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

            elif cmd == "observe":
                obs = self.Observe()

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Invalid 'answer' usage. Expect: \\boxed{answer N}"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans = int(tokens[1])
                obs = self.Done(ans)
                # 依据校验结果设置奖励
                reward = 1.0 if self._reward == 1 else -1.0
                terminated = True

            else:
                obs = f"Unknown action: {cmd}"
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
        if self.N > 0 and self.M > 0:
            i = random.randint(0, self.N - 1)
            j = random.randint(0, self.M - 1)
            return f"\\boxed{ { 'check ' + str(i) + ' ' + str(j) } }"
        return "\\boxed{dims}"

    # ---------------------------
    # 原环境的辅助方法（已转换/保留）
    # ---------------------------

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    @staticmethod
    def from_env_str(env_str: str):
        prefix = "PathCountingEnvGEM@"
        if not env_str.startswith(prefix):
            return None
        # 解析选项
        try:
            options = json.loads(env_str.split("@", 1)[1])
        except Exception:
            try:
                # 尝试使用 ast 风格
                import ast
                options = ast.literal_eval(env_str.split("@", 1)[1])
            except Exception:
                options = {}
        board = options.get("board", None)
        seed = options.get("seed", None)
        complexity = options.get("complexity", 5)
        enable_param_randomization = options.get("enable_param_randomization", False)
        max_turns = options.get("max_turns", 100)
        env = PathCountingEnvGEM(
            complexity=complexity,
            enable_param_randomization=enable_param_randomization,
            max_turns=max_turns,
            board=board,
        )
        # 如果给了种子，主动 reset 一次
        env.reset(seed=seed)
        return env

    def GetBoardDimensions(self):
        """
        获取棋盘的行列数，返回 JSON 字符串。
        Example: '{"N": 3, "M": 3}'
        """
        return json.dumps({"N": self.N, "M": self.M})

    def CheckCell(self, i: int, j: int):
        """
        检查指定格子是否为障碍。
        返回: "1" 表示障碍, "0" 表示空。
        """
        if 0 <= i < self.N and 0 <= j < self.M:
            return str(self.board[i][j])
        return "Error: Invalid cell indices"

    def CalculateDpValue(self, i: int, j: int, dp_values: Optional[list] = None):
        """
        计算指定格子的 DP 值（到达该格子的路径数）。
        当提供 dp_values（二位列表）时，使用其上/左值计算。
        """
        if 0 <= i < self.N and 0 <= j < self.M:
            cell_value = self.board[i][j]
            if cell_value == 1:
                return "0"
            else:
                # 若未提供 dp_values，返回错误提示
                if dp_values is None:
                    return "Error: 'dp_values' is required"
                value = 0
                if i > 0:
                    value += dp_values[i - 1][j]
                if j > 0:
                    value += dp_values[i][j - 1]
                return str(value)
        return "Error: Invalid cell indices"

    def CalculateDpValueNeighbors(self, i: int, j: int, up_value: int, left_value: int):
        """
        使用邻居值（上/左）计算 DP 值的便捷方法，对应 GEM 文本动作：dp i j up left
        """
        if 0 <= i < self.N and 0 <= j < self.M:
            cell_value = self.board[i][j]
            if cell_value == 1:
                return "0"
            else:
                total = 0
                if i > 0:
                    total += max(0, int(up_value))
                if j > 0:
                    total += max(0, int(left_value))
                return str(total)
        return "Error: Invalid cell indices"

    def Observe(self):
        """
        返回当前状态的提示信息。
        """
        return (
            "Use dims to get board dimensions, check to query cells, and dp to compute neighbors-based DP values. "
            "Finally, submit the total paths with answer."
        )

    def Done(self, answer):
        """
        验证最终答案并返回结果信息。
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def get_ref_answer(self):
        """
        使用环境信息得到参考答案（标准 DP）。
        """
        if not self.board or self.N == 0 or self.M == 0:
            return 0

        if self.board[0][0] == 1 or self.board[self.N - 1][self.M - 1] == 1:
            return 0

        dp = [[0] * self.M for _ in range(self.N)]
        dp[0][0] = 1

        for i in range(self.N):
            for j in range(self.M):
                if self.board[i][j] == 1:
                    dp[i][j] = 0
                else:
                    if i > 0:
                        dp[i][j] += dp[i - 1][j]
                    if j > 0:
                        dp[i][j] += dp[i][j - 1]

        return dp[self.N - 1][self.M - 1]

    def solve(self) -> str:
        """
        自动调用动作完成计算流程，并提交答案验证。返回最终信息字符串。
        使用 GEM 文本动作接口（boxed）。
        """
        dims_obs, _, _, _, _ = self.step("\\boxed{dims}")
        try:
            dims = json.loads(dims_obs)
            N = dims["N"]
            M = dims["M"]
        except Exception:
            # 若解析失败，直接观察并尝试提交0作为兜底
            final_obs, _, _, _, _ = self.step("\\boxed{answer 0}")
            return final_obs

        # 起点或终点为障碍时答案为0
        start_val_obs, _, _, _, _ = self.step("\\boxed{check 0 0}")
        if start_val_obs == "1":
            final_obs, _, _, _, _ = self.step("\\boxed{answer 0}")
            return final_obs

        end_val_obs, _, _, _, _ = self.step(f"\\boxed{{check {N-1} {M-1}}}")
        if end_val_obs == "1":
            final_obs, _, _, _, _ = self.step("\\boxed{answer 0}")
            return final_obs

        # 计算 DP
        dp_values = [[0 for _ in range(M)] for _ in range(N)]
        dp_values[0][0] = 1

        # 第一行
        for j in range(1, M):
            cell_val_obs, _, _, _, _ = self.step(f"\\boxed{{check 0 {j}}}")
            if cell_val_obs == "1":
                dp_values[0][j] = 0
            else:
                dp_values[0][j] = dp_values[0][j - 1]

        # 第一列
        for i in range(1, N):
            cell_val_obs, _, _, _, _ = self.step(f"\\boxed{{check {i} 0}}")
            if cell_val_obs == "1":
                dp_values[i][0] = 0
            else:
                dp_values[i][0] = dp_values[i - 1][0]

        # 其余格子
        for i in range(1, N):
            for j in range(1, M):
                cell_val_obs, _, _, _, _ = self.step(f"\\boxed{{check {i} {j}}}")
                if cell_val_obs == "1":
                    dp_values[i][j] = 0
                else:
                    up = dp_values[i - 1][j]
                    left = dp_values[i][j - 1]
                    dp_obs, _, _, _, _ = self.step(f"\\boxed{{dp {i} {j} {up} {left}}}")
                    try:
                        dp_values[i][j] = int(dp_obs)
                    except Exception:
                        # 若 dp 解析失败，退回直接相加
                        dp_values[i][j] = up + left

        answer = dp_values[N - 1][M - 1]
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {answer}}}")
        return final_obs


# 简单自测（可选）
if __name__ == "__main__":
    # 预设棋盘测试（示例 1）
    test_board1 = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    env1 = PathCountingEnvGEM.from_env_str(f"PathCountingEnvGEM@{{\"board\": {test_board1}, \"max_turns\": 200}}")
    print("Test Case 1:")
    instr, info = env1.reset(seed=42)
    print(env1.solve())
    print("turn count:", env1.turn_count)

    # 预设棋盘测试（示例 2）
    test_board2 = [
        [0, 1],
        [0, 0]
    ]
    env2 = PathCountingEnvGEM.from_env_str(f"PathCountingEnvGEM@{{\"board\": {test_board2}, \"max_turns\": 200}}")
    print("\nTest Case 2:")
    instr, info = env2.reset(seed=42)
    print(env2.solve())
    print("turn count:", env2.turn_count)