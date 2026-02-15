from typing import Any, Dict, Optional, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class LargestSquareEnvGEM(Env):
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
        # max_turns 将被难度控制覆盖
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数范围（根据问题规模和随机性）
        self.complexity_params = {
            "n_rows": (4, 30),           # 网格行数
            "n_cols": (4, 30),           # 网格列数
            "value_range": (2, 50),      # 网格数值范围（1..value_range）
            "same_value_bias": (80, 10), # 邻接取相同值的偏好（百分比，越大网格越同质，越容易）
            "max_turns_param": (20, 200) # 步数上限（随复杂度增长）
        }

        # 参数方差（启用随机化时微调）
        self.param_variance = {
            "n_rows": 1,
            "n_cols": 1,
            "value_range": 3,
            "same_value_bias": 5,
            "max_turns_param": 10,
        }

        # 占位属性
        self.n_rows: int = 0
        self.n_cols: int = 0
        self.value_range: int = 0
        self.same_value_bias: int = 0
        self.max_turns_param: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 问题数据
        self.n: int = 0
        self.m: int = 0
        self.grid: list = []
        self.dp_table: Optional[list] = None
        self.max_side: int = 0

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

        # 用难度参数覆盖 max_turns
        self.max_turns = int(self.max_turns_param)

    def _get_instructions(self) -> str:
        return (
            "Largest Square: Find the largest square area with equal values in the grid.\n"
            "You can interact with helper actions to compute the DP and submit the final answer.\n"
            "Grid is generated based on difficulty.\n"
            "Available actions (use LaTeX-style boxed commands):\n"
            "- Observe grid size: \\boxed{observe}\n"
            "- Initialize DP table: \\boxed{init}\n"
            "- Compute DP value at (i, j): \\boxed{compute i j}\n"
            "- Find current max side length: \\boxed{max}\n"
            "- Calculate area from side length S: \\boxed{area S}\n"
            "- Submit final area answer: \\boxed{answer N}\n"
            "Notes:\n"
            "- Indices i, j are 0-based.\n"
            "- Turn limit depends on difficulty.\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Grid: {self.n_rows}x{self.n_cols}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化状态变量
        self.n = self.n_rows
        self.m = self.n_cols
        self.grid = self.problem["grid"]
        self.dp_table = None
        self.max_side = 0

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        rows = self.n_rows
        cols = self.n_cols
        V = self.value_range
        bias = max(0, min(100, self.same_value_bias))  # 百分比

        grid = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                # 根据偏好，倾向于复制上方或左方的值，产生同质块
                if i == 0 and j == 0:
                    grid[i][j] = random.randint(1, V)
                else:
                    take_same = random.randint(1, 100) <= bias
                    candidates = []
                    if i > 0:
                        candidates.append(grid[i - 1][j])
                    if j > 0:
                        candidates.append(grid[i][j - 1])
                    if take_same and candidates:
                        grid[i][j] = random.choice(candidates)
                    else:
                        grid[i][j] = random.randint(1, V)
        return {"grid": grid}

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
        cmd = tokens[0].lower() if tokens else ""

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # 动作处理
        if cmd in ["observe"]:
            obs = self.Observe()
            reward = 0.0
            terminated = False

        elif cmd in ["init", "initialize"]:
            obs = self.InitializeDPTable()
            # 如果网格为空则认为失败
            if self.n == 0 or self.m == 0:
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                reward = 0.0
                terminated = False

        elif cmd in ["compute"]:
            if len(tokens) != 3:
                obs = "Error: compute requires two integers i j."
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                try:
                    i = int(tokens[1])
                    j = int(tokens[2])
                    msg = self.ComputeDPValue(i, j)
                    obs = msg
                    # 若出现错误消息（如未初始化或越界），判定为无效动作
                    if msg.startswith("Error:"):
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        reward = 0.0
                        terminated = False
                except Exception as e:
                    obs = f"Error: {str(e)}"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True

        elif cmd in ["max", "find_max", "findmax"]:
            msg = self.FindMaxSide()
            obs = msg
            if msg.startswith("Error:"):
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
            else:
                reward = 0.0
                terminated = False

        elif cmd in ["area"]:
            if len(tokens) != 2:
                obs = "Error: area requires one integer S (side length)."
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                try:
                    S = int(tokens[1])
                    obs = self.CalculateArea(S)
                    reward = 0.0
                    terminated = False
                except Exception as e:
                    obs = f"Error: {str(e)}"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True

        elif cmd in ["answer", "done", "submit"]:
            if len(tokens) != 2:
                obs = "Error: answer requires one integer N (area)."
                reward = LanguageGameReward.format_error_reward
                terminated = True
            else:
                try:
                    ans = int(tokens[1])
                    msg = self.Done(ans)
                    obs = msg
                    # 根据正确性设置奖励
                    if "Result: Correct" in msg:
                        reward = 1.0
                    else:
                        reward = -1.0
                    terminated = True
                except Exception as e:
                    obs = f"Error: {str(e)}"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True

        else:
            obs = f"Invalid action: {cmd}"
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
        # 随机采样一个动作（示例）
        choices = [
            "\\boxed{observe}",
            "\\boxed{init}",
            "\\boxed{compute 0 0}",
            "\\boxed{max}",
        ]
        return random.choice(choices)

    # ===== 以下为原环境的辅助方法（保留并适配） =====

    def get_ref_answer(self):
        """
        Use the information in the environment to get the reference answer.
        """
        if self.n == 0 or self.m == 0 or not self.grid:
            return 0

        dp = [[0] * self.m for _ in range(self.n)]

        max_side = 0
        for i in range(self.n):
            for j in range(self.m):
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    if self.grid[i][j] == self.grid[i - 1][j] == self.grid[i][j - 1] == self.grid[i - 1][j - 1]:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    else:
                        dp[i][j] = 1

                max_side = max(max_side, dp[i][j])

        return max_side * max_side

    def InitializeDPTable(self):
        """
        Initialize a DP table with the same size as the grid, all values initialized to 0.
        """
        if self.n == 0 or self.m == 0:
            self.dp_table = []
            return "DP table initialized successfully, dimensions: 0x0"

        self.dp_table = [[0] * self.m for _ in range(self.n)]
        self.max_side = 0
        return f"DP table initialized successfully, dimensions: {self.n}x{self.m}"

    def ComputeDPValue(self, i: int, j: int):
        """
        Calculate the value at position (i,j) in the DP table and update the maximum side length.
        """
        if self.dp_table is None:
            return "Error: DP table has not been initialized, please call InitializeDPTable first"

        if i < 0 or i >= self.n or j < 0 or j >= self.m:
            return "Error: Index out of range"

        if i == 0 or j == 0:
            self.dp_table[i][j] = 1
        else:
            if self.grid[i][j] == self.grid[i - 1][j] == self.grid[i][j - 1] == self.grid[i - 1][j - 1]:
                self.dp_table[i][j] = min(
                    self.dp_table[i - 1][j],
                    self.dp_table[i][j - 1],
                    self.dp_table[i - 1][j - 1],
                ) + 1
            else:
                self.dp_table[i][j] = 1

        if self.dp_table[i][j] > self.max_side:
            self.max_side = self.dp_table[i][j]

        return f"DP[{i}][{j}] = {self.dp_table[i][j]}, current maximum side length: {self.max_side}"

    def FindMaxSide(self):
        """
        Get the maximum side length value in the DP table.
        """
        if self.dp_table is None:
            return "Error: DP table has not been initialized, please call InitializeDPTable first"

        return str(self.max_side)

    def CalculateArea(self, side_length: int):
        """
        Calculate the area of the square based on the side length (the square of the side length).
        """
        return str(side_length * side_length)

    def Observe(self):
        """
        Return basic information about the current grid.
        """
        return f"Current grid dimensions: {self.n}x{self.m}"

    def Done(self, answer: int):
        """
        Verify whether the final answer is correct and return result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        Automatically perform actions to compute the answer and submit it (text-mode).
        """
        # Observe
        _ = self.step("\\boxed{observe}")
        # Initialize DP
        _ = self.step("\\boxed{init}")

        # Compute all DP entries
        for i in range(self.n):
            for j in range(self.m):
                terminated = self.step(f"\\boxed{{compute {i} {j}}}")[2]
                if terminated:
                    break
            else:
                continue
            break

        # Find max side and area
        obs_max = self.step("\\boxed{max}")[0]
        if obs_max.startswith("Error"):
            return obs_max

        try:
            max_side = int(obs_max)
        except Exception:
            return f"Error: failed to parse max side from '{obs_max}'"

        obs_area = self.step(f"\\boxed{{area {max_side}}}")[0]
        try:
            area_val = int(obs_area)
        except Exception:
            return f"Error: failed to parse area from '{obs_area}'"

        # Submit answer
        final_obs, reward, terminated, truncated, _ = self.step(f"\\boxed{{answer {area_val}}}")
        return f"{final_obs} (reward={reward}, terminated={terminated}, truncated={truncated})"