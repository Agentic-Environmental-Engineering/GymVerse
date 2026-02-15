from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class GridPathCountEnvGEM(Env):
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
        # grid_size 对应原环境的 n；turn_limit 控制最大回合数（会覆盖 max_turns）
        self.complexity_params = {
            "grid_size": (1, 30),     # 网格大小 n: 1~30
            "turn_limit": (20, 200),  # 最大允许步数: 20~200
        }

        # 参数方差（启用随机化时使用）
        self.param_variance = {
            "grid_size": 1,
            "turn_limit": 5,
        }

        # 占位属性
        self.n: int = 1
        self.turn_count: int = 0

        # 环境内部状态
        self.dp_table: Optional[list] = None

        self.reset()

    def _apply_complexity_params(self):
        """根据 complexity 等级计算参数值"""
        normalized = min(1.0, (self.complexity - 1) / 9.0)  # [0, 1]

        computed: Dict[str, int] = {}
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value

            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    actual_value = max(min_val, min(max_val, actual_value))

            computed[param_name] = int(round(actual_value))

        # 应用到属性
        self.n = computed["grid_size"]
        # 难度控制最大步数（覆盖构造中的 max_turns）
        self.max_turns = computed["turn_limit"]

    def _get_instructions(self) -> str:
        return (
            "Grid Path Count: Compute number of paths on an (n+1)×(n+1) grid under rules.\n"
            "Rules:\n"
            "- DP table dp of size (n+1)×(n+1), dp[0][0] = 1.\n"
            "- For cell (i, j): dp[i][j] = (j>0 and j<=i ? dp[i][j-1] : 0) + (i>0 ? dp[i-1][j] : 0).\n"
            "Goal: Return dp[n][n].\n"
            "Available actions (wrap in \\boxed{...}):\n"
            "- Observe grid: \\boxed{observe}\n"
            "- Initialize table: \\boxed{init N}\n"
            "- Calculate cell: \\boxed{calc i j}\n"
            "- Get cell value: \\boxed{get i j}\n"
            "- Submit answer: \\boxed{answer V}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Grid: {self.n}×{self.n}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.dp_table = None
        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        # 对该任务，随机性主要体现在 n 的选择；这里附带一个占位数据以示接口一致性
        return {"grid_size": self.n}

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
        cmd = tokens[0].lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()
            elif cmd == "init":
                if len(tokens) != 2 or not tokens[1].isdigit():
                    obs = "Invalid 'init' usage. Expect: \\boxed{init N}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    size = int(tokens[1])
                    obs = self.InitializeDPTable(size)
            elif cmd == "calc":
                if len(tokens) != 3 or not tokens[1].isdigit() or not tokens[2].isdigit():
                    obs = "Invalid 'calc' usage. Expect: \\boxed{calc i j}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    obs = self.CalculateDPTableCell(i, j)
            elif cmd == "get":
                if len(tokens) != 3 or not tokens[1].isdigit() or not tokens[2].isdigit():
                    obs = "Invalid 'get' usage. Expect: \\boxed{get i j}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    val_str = self.GetDPTableCell(i, j)
                    if val_str.startswith("Error"):
                        obs = val_str
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = f"Value at ({i}, {j}): {val_str}"
            elif cmd == "answer":
                if len(tokens) != 2 or not tokens[1].isdigit():
                    obs = "Invalid 'answer' usage. Expect: \\boxed{answer V}"
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    answer_val = int(tokens[1])
                    result_msg, correct = self.Done(answer_val)
                    obs = result_msg
                    reward = 1.0 if correct else -1.0
                    terminated = True
            else:
                obs = f"Invalid action: {cmd}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
        except Exception as e:
            obs = f"Execution error: {str(e)}"
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
        if self.dp_table is None:
            return f"\\boxed{{init {self.n}}}"
        # 随机选择一个可计算的单元或观察
        choice = random.choice(["observe", "calc", "get"])
        if choice == "observe":
            return "\\boxed{observe}"
        i = random.randint(0, self.n)
        j = random.randint(0, self.n)
        if choice == "calc":
            return f"\\boxed{{calc {i} {j}}}"
        else:
            return f"\\boxed{{get {i} {j}}}"

    # ========= 辅助方法（保留原环境逻辑并转换） =========

    def get_ref_answer(self) -> int:
        """
        使用环境信息计算参考答案。
        """
        if self.n == 1:
            return 1

        dp = [[0] * (self.n + 1) for _ in range(self.n + 1)]
        dp[0][0] = 1

        for i in range(self.n + 1):
            for j in range(self.n + 1):
                if i + j > 0:
                    if j > 0 and j <= i:
                        dp[i][j] += dp[i][j - 1]
                    if i > 0:
                        dp[i][j] += dp[i - 1][j]

        return dp[self.n][self.n]

    def InitializeDPTable(self, size: int) -> str:
        r"""
        Initialize a (size+1)×(size+1) DP table and set dp[0][0] to 1.

        Args:
            size (int): The size parameter of the DP table; the actual table created is (size+1)×(size+1).

        Returns:
            str: A prompt message indicating successful initialization.

        Example Output:
            "DP table initialized successfully, size is 3×3"
        """
        self.dp_table = [[0] * (size + 1) for _ in range(size + 1)]
        self.dp_table[0][0] = 1
        return f"DP table initialized successfully, size is {(size+1)}×{(size+1)}"

    def GetDPTableCell(self, i: int, j: int) -> str:
        r"""
        Get the value of the cell at position (i, j) in the DP table.

        Args:
            i (int): The row index of the cell.
            j (int): The column index of the cell.

        Returns:
            str: The value of the cell at position (i, j) in the DP table.

        Example Output:
            "2"
        """
        if self.dp_table is None:
            return "Error: DP table has not been initialized, please call InitializeDPTable first"
        if 0 <= i < len(self.dp_table) and 0 <= j < len(self.dp_table[0]):
            return str(self.dp_table[i][j])
        else:
            return "Error: Index out of DP table range"

    def CalculateDPTableCell(self, i: int, j: int) -> str:
        r"""
        Calculate and update the value of the cell at position (i, j) in the DP table according to the rules.

        Args:
            i (int): The row index of the cell.
            j (int): The column index of the cell.

        Returns:
            str: The updated value of the cell at position (i, j) in the DP table.

        Example Output:
            "3"
        """
        if self.dp_table is None:
            return "Error: DP table has not been initialized, please call InitializeDPTable first"
        if 0 <= i < len(self.dp_table) and 0 <= j < len(self.dp_table[0]):
            value = 0
            if j > 0 and j <= i:
                value += self.dp_table[i][j - 1]
            if i > 0:
                value += self.dp_table[i - 1][j]
            self.dp_table[i][j] = value
            return str(value)
        else:
            return "Error: Index out of DP table range"

    def Observe(self) -> str:
        r"""
        Return the observation information of the current environment, including the grid size.

        Args:
            None

        Returns:
            str: The grid size information of the current environment.

        Example Output:
            "Current grid size is 2×2"
        """
        return f"Current grid size is {self.n}×{self.n}"

    def Done(self, answer: int) -> Tuple[str, bool]:
        r"""
        Verify whether the final answer is correct and return the result information.

        Args:
            answer (int): The answer submitted by the user.

        Returns:
            Tuple[str, bool]: (Result message, correctness flag)

        Example Output:
            ("Your answer: 2, Reference answer: 2, Result: Correct", True)
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, correct

    def solve(self) -> str:
        """
        Automatically call actions to complete the process and submit the answer for verification.

        Returns:
            str: The result information of the final answer verification.
        """
        # Observe
        observe_info = self.Observe()
        n = int(observe_info.split('×')[0].split('is ')[1])

        # Initialize DP
        self.InitializeDPTable(n)

        # Fill DP table
        for total_steps in range(0, 2 * n + 1):
            for j in range(0, min(total_steps + 1, n + 1)):
                i = total_steps - j
                if i < 0 or i > n:
                    continue
                if i == 0 and j == 0:
                    continue  # Initialized to 1
                self.CalculateDPTableCell(i, j)

        # Get answer and submit
        answer = int(self.GetDPTableCell(n, n))
        result_msg, _ = self.Done(answer)
        return result_msg