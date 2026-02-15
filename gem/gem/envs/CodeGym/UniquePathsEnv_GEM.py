from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class UniquePathsEnvGEM(Env):
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

        # 难度参数范围：控制网格大小
        self.complexity_params = {
            "grid_rows": (2, 20),
            "grid_cols": (2, 20),
        }

        # 参数方差（用于 enable_param_randomization=True 时微调）
        self.param_variance = {
            "grid_rows": 2,
            "grid_cols": 2,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.grid_rows: int = 0
        self.grid_cols: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 运行时变量（问题实例与DP状态）
        self.problem: Dict[str, Any] = {}
        self.m: int = 0
        self.n: int = 0
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
            "Unique Paths (Grid DP): Compute number of unique paths from top-left to bottom-right.\n"
            "Available actions (use the last \\boxed{...} in your message):\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Initialize DP table: \\boxed{init M N}\n"
            "- Calculate a DP cell: \\boxed{calc i j}\n"
            "- Get bottom-right cell value: \\boxed{get}\n"
            "- Submit final answer: \\boxed{answer X}\n"
            "Indices i, j are 0-based. Boundary cells (i==0 or j==0) are initialized to 1.\n"
        )

    def get_task_suffix(self) -> str:
        dp_status = "initialized" if self.dp is not None else "not initialized"
        return (
            f"Grid size: {self.m}×{self.n}, DP {dp_status}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响问题实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.m = self.problem["m"]
        self.n = self.problem["n"]

        # 重置状态
        self.dp = None
        self._reward = 0.0
        self._done = False
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 允许在中心值附近抖动，以体现 seed 对实例的影响
        m_min = max(2, self.grid_rows - 1)
        m_max = self.grid_rows + 1
        n_min = max(2, self.grid_cols - 1)
        n_max = self.grid_cols + 1

        m = random.randint(m_min, m_max)
        n = random.randint(n_min, n_max)

        return {"m": m, "n": n}

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

        content = parsed["content"].strip()
        tokens = content.split()
        cmd = tokens[0].lower() if tokens else ""

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
                if len(tokens) != 3 or not (tokens[1].isdigit() and tokens[2].isdigit()):
                    obs = "Invalid init command. Usage: \\boxed{init M N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    m = int(tokens[1])
                    n = int(tokens[2])
                    obs = self.InitializeDP(m, n)
                    reward = 0.0
                    terminated = False

            elif cmd == "calc":
                if len(tokens) != 3 or not (tokens[1].isdigit() and tokens[2].isdigit()):
                    obs = "Invalid calc command. Usage: \\boxed{calc i j}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    obs = self.CalculateCell(i, j)
                    reward = 0.0
                    terminated = False

            elif cmd == "get":
                if len(tokens) != 1:
                    obs = "Invalid get command. Usage: \\boxed{get}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    obs = self.GetResult()
                    reward = 0.0
                    terminated = False

            elif cmd == "answer":
                if len(tokens) != 2 or not re.fullmatch(r"-?\d+", tokens[1]):
                    obs = "Invalid answer command. Usage: \\boxed{answer X}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    answer = int(tokens[1])
                    msg = self.Done(answer)
                    # 根据正确与否给奖励
                    reward = 1.0 if self._reward == 1 else -1.0
                    terminated = True
                    obs = msg

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
        # 如果尚未初始化DP，建议初始化
        if self.dp is None:
            return f"\\boxed{{init {self.m} {self.n}}}"
        # 若已初始化，随机计算一个非边界单元或观察/获取
        choices = []
        if self.m > 1 and self.n > 1:
            i = random.randint(1, self.m - 1)
            j = random.randint(1, self.n - 1)
            choices.append(f"\\boxed{{calc {i} {j}}}")
        choices.extend(["\\boxed{observe}", "\\boxed{get}"])
        return random.choice(choices)

    # -----------------------------
    # 保留原环境的辅助方法并转换
    # -----------------------------
    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        dp = [[1] * self.n for _ in range(self.m)]
        for i in range(1, self.m):
            for j in range(1, self.n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[self.m - 1][self.n - 1]

    def InitializeDP(self, m: int, n: int) -> str:
        """
        Initialize an m×n dynamic programming table with all cells initially set to 1.
        """
        # 更新网格大小到用户提供的值
        self.m = m
        self.n = n
        self.dp = [[1] * n for _ in range(m)]
        return f"Initialized a {m}×{n} dynamic programming table"

    def CalculateCell(self, i: int, j: int) -> str:
        """
        Calculate the number of paths for the cell at row i and column j.
        """
        if self.dp is None:
            return "Error: Please initialize the dynamic programming table first"

        if i < 0 or i >= len(self.dp) or j < 0 or j >= len(self.dp[0]):
            return "Error: Cell index out of range"

        if i == 0 or j == 0:
            return "The value of cell (0,0) is 1 (initial value of boundary cells is 1)"

        self.dp[i][j] = self.dp[i - 1][j] + self.dp[i][j - 1]
        return f"Calculated the value of cell ({i},{j}) as {self.dp[i][j]}"

    def GetResult(self) -> str:
        """
        Get the value of the bottom-right cell in the dynamic programming table.
        """
        if self.dp is None:
            return "Error: Please initialize the dynamic programming table first"

        m = len(self.dp)
        n = len(self.dp[0]) if m > 0 else 0
        return str(self.dp[m - 1][n - 1])

    def Observe(self) -> str:
        """
        Return the observation information of the current environment.
        """
        dp_status = "initialized" if self.dp is not None else "not initialized"
        return f"Grid size: {self.m}×{self.n}, dynamic programming table {dp_status}"

    def Done(self, answer: int) -> str:
        """
        Verify whether the final answer is correct and return the result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    def solve(self) -> str:
        """
        Automatically call all actions to complete the process and submit the answer.
        """
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        # Extract m and n from observation
        try:
            size_part = obs.split("Grid size: ")[1].split(",")[0]
            m_str, n_str = size_part.split("×")
            m_val = int(m_str.strip())
            n_val = int(n_str.strip())
        except Exception:
            # Fallback to problem definition if parsing fails
            m_val, n_val = self.m, self.n

        # Initialize DP
        self.step(f"\\boxed{{init {m_val} {n_val}}}")

        # Calculate DP cells
        for i in range(1, m_val):
            for j in range(1, n_val):
                self.step(f"\\boxed{{calc {i} {j}}}")

        # Get result
        res_obs, _, _, _, _ = self.step("\\boxed{get}")
        try:
            result_int = int(res_obs.strip())
        except Exception:
            # If failed to parse, compute reference directly
            result_int = self.get_ref_answer()

        # Submit answer
        final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {result_int}}}")
        return final_obs