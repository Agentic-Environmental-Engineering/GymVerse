from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class SymmetricGridEnvGEM(Env):
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

        # 难度参数范围设计（根据原环境特性）
        # - num_rows/num_cols 控制网格维度
        # - value_range 控制值域大小
        # - forced_sym_lines 控制强制对称的行/列数量（增加存在对称性的概率）
        # - turn_allowance 控制允许的最大回合数（可受复杂度影响）
        self.complexity_params = {
            "num_rows": (3, 30),
            "num_cols": (3, 30),
            "value_range": (10, 10000),
            "forced_sym_lines": (0, 3),
            "turn_allowance": (20, 200),
        }

        # 参数方差（用于训练时增加多样性）
        self.param_variance = {
            "num_rows": 1,
            "num_cols": 1,
            "value_range": 50,
            "forced_sym_lines": 1,
            "turn_allowance": 10,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.num_rows: int = 0
        self.num_cols: int = 0
        self.value_range: int = 0
        self.forced_sym_lines: int = 0
        self.turn_allowance: int = 0

        # 状态变量
        self.turn_count: int = 0
        self.effective_max_turns: int = self.max_turns

        # 问题实例
        self.grid: list = []
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

        # 根据难度计算有效最大步数（若用户提供了 max_turns，上限取两者较小）
        self.effective_max_turns = min(self.max_turns, self.turn_allowance)

    def _get_instructions(self) -> str:
        return (
            "Symmetric Grid: Determine if any row or column in the grid is symmetric (palindromic).\n"
            "Actions (use 0-based indices for rows/columns):\n"
            "- Observe grid info: \\boxed{observe}\n"
            "- Get dimensions: \\boxed{dims}\n"
            "- Check row symmetry: \\boxed{row K} (K is row index)\n"
            "- Check column symmetry: \\boxed{col K} (K is column index)\n"
            "- Submit final answer (YES/NO): \\boxed{answer YES} or \\boxed{answer NO}\n"
        )

    def get_task_suffix(self) -> str:
        rows = len(self.grid)
        cols = len(self.grid[0]) if self.grid else 0
        return f"Grid: {rows}x{cols}\nTurn: {self.turn_count}/{self.effective_max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        rows = self.num_rows
        cols = self.num_cols
        # 初始化随机网格
        grid = [[random.randint(0, self.value_range) for _ in range(cols)] for _ in range(rows)]

        # 强制若干行/列为回文
        k = max(0, min(self.forced_sym_lines, rows + cols))
        chosen_rows = set()
        chosen_cols = set()
        for _ in range(k):
            # 随机决定强制行或列
            if random.random() < 0.5 and len(chosen_rows) < rows:
                r = random.randrange(rows)
                # 避免重复选择同一行
                tries = 0
                while r in chosen_rows and tries < 10:
                    r = random.randrange(rows)
                    tries += 1
                chosen_rows.add(r)
                half = cols // 2
                # 生成半边并镜像
                for j in range(half):
                    val = random.randint(0, self.value_range)
                    grid[r][j] = val
                    grid[r][cols - 1 - j] = val
                # 如果奇数列，中间元素随机
                if cols % 2 == 1:
                    mid = half
                    grid[r][mid] = random.randint(0, self.value_range)
            else:
                c = random.randrange(cols)
                # 避免重复选择同一列
                tries = 0
                while c in chosen_cols and tries < 10:
                    c = random.randrange(cols)
                    tries += 1
                chosen_cols.add(c)
                half = rows // 2
                # 生成半边并镜像
                for i in range(half):
                    val = random.randint(0, self.value_range)
                    grid[i][c] = val
                    grid[rows - 1 - i][c] = val
                # 如果奇数行，中间元素随机
                if rows % 2 == 1:
                    mid = half
                    grid[mid][c] = random.randint(0, self.value_range)

        self.grid = grid
        return {"grid": grid, "rows": rows, "cols": cols}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if not parsed:
            obs = f"Format error at turn {self.turn_count}. Use \\boxed{{...}}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        content = parsed["content"].strip()
        lower = content.lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        # 动作处理
        if lower == "observe":
            obs = self.Observe()
        elif lower in ("dims", "dimensions"):
            obs = self.GetGridDimensions()
        elif lower.startswith("row"):
            m = re.match(r"row\s+(\d+)$", lower)
            if not m:
                obs = "Invalid action: expected 'row K' with integer K."
                return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
            idx = int(m.group(1))
            res = self.CheckRowSymmetry(idx)
            obs = f"Row {idx} symmetric: {res}"
        elif lower.startswith("col") or lower.startswith("column"):
            m = re.match(r"(?:col|column)\s+(\d+)$", lower)
            if not m:
                obs = "Invalid action: expected 'col K' with integer K."
                return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
            idx = int(m.group(1))
            res = self.CheckColumnSymmetry(idx)
            obs = f"Column {idx} symmetric: {res}"
        elif lower.startswith("answer"):
            m = re.match(r"answer\s+([a-zA-Z]+)$", lower)
            if not m:
                obs = "Invalid action: expected 'answer YES' or 'answer NO'."
                return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
            ans = m.group(1).strip().upper()
            if ans not in ("YES", "NO"):
                obs = "Invalid action: expected 'answer YES' or 'answer NO'."
                return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
            # 提交答案并计算奖励
            obs = self.Done(ans)
            ref = self.get_ref_answer()
            correct = ans == ref
            reward = 1.0 if correct else -1.0
            terminated = True
        else:
            obs = f"Invalid action: '{content}'."
            return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}

        # 超时检查（统一放在 step 结尾）
        if not terminated and self.turn_count >= self.effective_max_turns:
            obs = f"{obs}\nReached max turns ({self.effective_max_turns})."
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
        # 随机示例动作
        actions = ["\\boxed{observe}", "\\boxed{dims}"]
        if self.grid and len(self.grid) > 0 and len(self.grid[0]) > 0:
            r = random.randrange(len(self.grid))
            c = random.randrange(len(self.grid[0]))
            actions.extend([f"\\boxed{{row {r}}}", f"\\boxed{{col {c}}}"])
        return random.choice(actions)

    # =========================
    # 原环境辅助方法（已转换）
    # =========================
    def get_ref_answer(self) -> str:
        """
        Use the information in the environment to get the reference answer.
        Return "YES" if any row or column is symmetric (palindromic), else "NO".
        """
        # Check for symmetric rows
        for row in self.grid:
            if row == row[::-1]:
                return "YES"

        # Check for symmetric columns
        if not self.grid:
            return "NO"

        num_columns = len(self.grid[0])
        for col in range(num_columns):
            column = [row[col] for row in self.grid]
            if column == column[::-1]:
                return "YES"

        return "NO"

    def CheckRowSymmetry(self, row_index: int) -> str:
        """
        Check if the specified row is symmetric (palindromic).
        Returns "True" if the row is symmetric, otherwise returns "False".
        """
        if row_index < 0 or row_index >= len(self.grid):
            return "False"
        row = self.grid[row_index]
        return str(row == row[::-1])

    def CheckColumnSymmetry(self, col_index: int) -> str:
        """
        Check if the specified column is symmetric (palindromic).
        Returns "True" if the column is symmetric, otherwise returns "False".
        """
        if not self.grid or col_index < 0 or col_index >= len(self.grid[0]):
            return "False"
        column = [row[col_index] for row in self.grid]
        return str(column == column[::-1])

    def GetGridDimensions(self) -> str:
        """
        Get the number of rows and columns of the grid.
        Returns a JSON string formatted as {"rows": n, "cols": m}.
        """
        rows = len(self.grid)
        cols = len(self.grid[0]) if self.grid else 0
        return json.dumps({"rows": rows, "cols": cols})

    def Observe(self) -> str:
        """
        Return basic information about the current grid.
        """
        rows = len(self.grid)
        cols = len(self.grid[0]) if self.grid else 0
        return f"The current grid is a {rows}x{cols} matrix. Please use relevant actions to check symmetry."

    def Done(self, answer: str) -> str:
        """
        Verify whether the final answer is correct and return result information.
        The answer should be "YES" or "NO".
        """
        ref_answer = self.get_ref_answer()
        correct = answer.strip().upper() == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def solve(self) -> str:
        """
        Automatically call actions to complete the process and submit the answer for verification.
        Returns the result information of the final answer verification.
        """
        # Get dimensions
        obs, _, _, _, _ = self.step("\\boxed{dims}")
        try:
            dim = json.loads(obs)
            rows = dim.get("rows", 0)
            cols = dim.get("cols", 0)
        except Exception:
            # Fallback to computed dimensions
            rows = len(self.grid)
            cols = len(self.grid[0]) if self.grid else 0

        # Check rows
        for r in range(rows):
            obs, _, term, _, _ = self.step(f"\\boxed{{row {r}}}")
            if "True" in obs:
                obs, reward, term, _, _ = self.step("\\boxed{answer YES}")
                return obs

        # Check columns
        for c in range(cols):
            obs, _, term, _, _ = self.step(f"\\boxed{{col {c}}}")
            if "True" in obs:
                obs, reward, term, _, _ = self.step("\\boxed{answer YES}")
                return obs

        obs, reward, term, _, _ = self.step("\\boxed{answer NO}")
        return obs