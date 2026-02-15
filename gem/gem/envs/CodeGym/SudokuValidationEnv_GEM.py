from typing import Any, Dict, Optional, Tuple
import random
import re
import json
import math
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class SudokuValidationEnvGEM(Env):
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

        # 难度参数范围（根据数独验证任务设计）
        # - grid_size: 网格边长（支持 4, 9, 16，插值后会自动就近映射）
        # - num_inconsistencies: 强制插入的违规数（越大越难）
        # - reveal_allowance: 允许 Observe 的次数上限
        # - max_turns: 每局最大步数（随难度线性插值）
        self.complexity_params: Dict[str, Tuple[int, int]] = {
            "grid_size": (4, 16),
            "num_inconsistencies": (0, 3),
            "reveal_allowance": (3, 10),
            "max_turns": (30, 200),
        }

        # 参数方差（可选，用于微调随机性）
        self.param_variance: Dict[str, int] = {
            "grid_size": 2,
            "num_inconsistencies": 1,
            "reveal_allowance": 1,
            "max_turns": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.grid_size: int = 9
        self.num_inconsistencies: int = 0
        self.reveal_allowance: int = 5

        # 状态变量
        self.turn_count: int = 0
        self.reveals_used: int = 0

        # 当前问题
        self.problem: Dict[str, Any] = {}
        self.matrix: Any = None  # 网格

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

        # grid_size 仅允许 {4, 9, 16}，将插值结果映射到最近的允许值
        self.grid_size = self._snap_grid_size(self.grid_size)
        # 将插值得到的 max_turns 应用于环境（覆盖 __init__ 的初始值）
        self.max_turns = int(self.max_turns)

    def _snap_grid_size(self, value: int) -> int:
        allowed = [4, 9, 16]
        closest = min(allowed, key=lambda x: abs(x - value))
        return closest

    def _get_instructions(self) -> str:
        return (
            "Sudoku Validation: Determine whether the given Sudoku is valid.\n"
            "A Sudoku is valid if each row, each column, and each subgrid contains no duplicate symbols (excluding '.').\n"
            "Available actions (use the last \\boxed{...} block as the command):\n"
            "- Observe the current grid: \\boxed{observe}\n"
            "- Check a row: \\boxed{check row i} (0-based index)\n"
            "- Check a column: \\boxed{check col j} (0-based index)\n"
            "- Check a subgrid: \\boxed{check subgrid r c} (0-based subgrid indices)\n"
            "- Submit final answer: \\boxed{answer true} or \\boxed{answer false}\n"
            "- Show help: \\boxed{help}\n"
        )

    def get_task_suffix(self) -> str:
        s = self._subgrid_size()
        return (
            f"Grid: {self.grid_size}x{self.grid_size}, Subgrid: {s}x{s}\n"
            f"Reveals: {self.reveals_used}/{self.reveal_allowance}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            "Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()
        self.matrix = self.problem["matrix"]

        self.turn_count = 0
        self.reveals_used = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _subgrid_size(self) -> int:
        s = int(round(math.sqrt(self.grid_size)))
        return s

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成数独实例（可能有效或无效）"""
        n = self.grid_size
        s = self._subgrid_size()
        assert s * s == n, "Grid size must be a perfect square."

        symbols = [self._symbol(k) for k in range(1, n + 1)]

        # 构造一个基础有效数独（行列和子宫格均满足）
        def pattern(r, c):
            return (s * (r % s) + r // s + c) % n

        # 随机化行、列、符号顺序，但保留数独结构有效性
        base_rows = [g * s + r for g in self._shuffle(range(s)) for r in self._shuffle(range(s))]
        base_cols = [g * s + c for g in self._shuffle(range(s)) for c in self._shuffle(range(s))]
        nums = self._shuffle(list(range(n)))

        full_grid = [[symbols[nums[pattern(r, c)]] for c in base_cols] for r in base_rows]

        # 随机去除部分格子为 '.' 以制造不完整网格（不影响有效性判断规则）
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        empty_ratio = 0.2 + 0.4 * normalized  # 难度越高空格越多（0.2 ~ 0.6）
        total_cells = n * n
        empties = int(round(total_cells * empty_ratio))
        positions = [(r, c) for r in range(n) for c in range(n)]
        random.shuffle(positions)
        grid = [row[:] for row in full_grid]
        for (r, c) in positions[:empties]:
            grid[r][c] = '.'

        # 插入违规（强制重复）以生成无效数独（如果 num_inconsistencies > 0）
        for _ in range(max(0, self.num_inconsistencies)):
            choice = random.choice(["row", "col", "subgrid"])
            if choice == "row":
                rr = random.randrange(n)
                c1, c2 = random.sample(range(n), 2)
                v = random.choice(symbols)
                grid[rr][c1] = v
                grid[rr][c2] = v
            elif choice == "col":
                cc = random.randrange(n)
                r1, r2 = random.sample(range(n), 2)
                v = random.choice(symbols)
                grid[r1][cc] = v
                grid[r2][cc] = v
            else:
                gr = random.randrange(s)
                gc = random.randrange(s)
                cells = [(gr * s + dr, gc * s + dc) for dr in range(s) for dc in range(s)]
                (r1, c1), (r2, c2) = random.sample(cells, 2)
                v = random.choice(symbols)
                grid[r1][c1] = v
                grid[r2][c2] = v

        return {"matrix": grid, "size": n, "subgrid_size": s}

    def _symbol(self, k: int) -> str:
        # 表示为 "1".."9","10".."16"（简单数字字符串）
        return str(k)

    def _shuffle(self, seq):
        seq = list(seq)
        random.shuffle(seq)
        return seq

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
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if len(tokens) == 0:
                obs = "Empty command."
                return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

            cmd = tokens[0].lower()

            if cmd == "help":
                obs = self._get_instructions()
                reward = 0.0
                terminated = False

            elif cmd == "observe":
                if self.reveals_used >= self.reveal_allowance:
                    obs = "No reveals remaining."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                self.reveals_used += 1
                obs = self.Observe()
                reward = 0.0
                terminated = False

            elif cmd == "check":
                if len(tokens) < 3:
                    obs = "Format error: expected 'check row i', 'check col j', or 'check subgrid r c'."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                subtype = tokens[1].lower()
                if subtype == "row":
                    if len(tokens) != 3:
                        obs = "Format error: 'check row i' requires exactly 1 index."
                        return (
                            obs,
                            LanguageGameReward.format_error_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    i = int(tokens[2])
                    if not (0 <= i < self.grid_size):
                        obs = f"Invalid row index: {i}."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    obs = self.CheckRow(i)
                    reward = 0.0
                    terminated = False

                elif subtype == "col":
                    if len(tokens) != 3:
                        obs = "Format error: 'check col j' requires exactly 1 index."
                        return (
                            obs,
                            LanguageGameReward.format_error_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    j = int(tokens[2])
                    if not (0 <= j < self.grid_size):
                        obs = f"Invalid column index: {j}."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    obs = self.CheckColumn(j)
                    reward = 0.0
                    terminated = False

                elif subtype == "subgrid":
                    if len(tokens) != 4:
                        obs = "Format error: 'check subgrid r c' requires exactly 2 indices."
                        return (
                            obs,
                            LanguageGameReward.format_error_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    r = int(tokens[2])
                    c = int(tokens[3])
                    s = self._subgrid_size()
                    if not (0 <= r < s and 0 <= c < s):
                        obs = f"Invalid subgrid indices: {r}, {c}."
                        return (
                            obs,
                            LanguageGameReward.invalid_action_reward,
                            True,
                            False,
                            {"suffix": self.get_task_suffix()},
                        )
                    obs = self.CheckSubgrid(r, c)
                    reward = 0.0
                    terminated = False

                else:
                    obs = f"Unknown check type: {subtype}."
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Format error: 'answer true|false' requires exactly 1 parameter."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                ans_token = tokens[1].lower()
                if ans_token not in ["true", "false"]:
                    obs = "Format error: answer must be 'true' or 'false'."
                    return (
                        obs,
                        LanguageGameReward.format_error_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                is_valid = (ans_token == "true")
                done_msg = self.Done(is_valid)
                # 奖励和终止逻辑
                ref = self.get_ref_answer()
                correct = (is_valid == ref)
                reward = 1.0 if correct else -1.0
                terminated = True
                obs = done_msg

            else:
                obs = f"Unknown command: {cmd}."
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
                LanguageGameReward.format_error_reward,
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
        return "\\boxed{check row 0}"

    # ----------------------- 辅助方法（保留并适配） -----------------------

    def get_ref_answer(self) -> bool:
        """
        Use the information in the environment to get the reference answer.
        Returns True if current matrix is a valid Sudoku (no duplicates in rows/cols/subgrids ignoring '.').
        """
        def is_valid_block(block):
            block = [num for num in block if num != '.']
            return len(block) == len(set(block))

        size = len(self.matrix)
        s = self._subgrid_size()

        # Check each row and each column
        for i in range(size):
            if not is_valid_block(self.matrix[i]) or not is_valid_block([self.matrix[r][i] for r in range(size)]):
                return False

        # Check each sub-grid
        for r in range(0, size, s):
            for c in range(0, size, s):
                block = [self.matrix[r + i][c + j] for i in range(s) for j in range(s)]
                if not is_valid_block(block):
                    return False

        return True

    def CheckRow(self, row_index: int) -> str:
        """
        Check if the specified row is valid (contains no duplicate numbers except '.').
        Returns "true" or "false".
        """
        row = self.matrix[row_index]
        block = [num for num in row if num != '.']
        is_valid = len(block) == len(set(block))
        return str(is_valid).lower()

    def CheckColumn(self, col_index: int) -> str:
        """
        Check if the specified column is valid (contains no duplicate numbers except '.').
        Returns "true" or "false".
        """
        column = [self.matrix[r][col_index] for r in range(len(self.matrix))]
        block = [num for num in column if num != '.']
        is_valid = len(block) == len(set(block))
        return str(is_valid).lower()

    def CheckSubgrid(self, subgrid_row: int, subgrid_col: int) -> str:
        """
        Check if the specified subgrid is valid (contains no duplicate numbers except '.').
        Returns "true" or "false".
        """
        s = self._subgrid_size()
        start_row = subgrid_row * s
        start_col = subgrid_col * s
        block = [self.matrix[start_row + i][start_col + j] for i in range(s) for j in range(s)]
        block = [num for num in block if num != '.']
        is_valid = len(block) == len(set(block))
        return str(is_valid).lower()

    def Observe(self) -> str:
        """
        Return the current Sudoku matrix as a JSON string.
        """
        return json.dumps(self.matrix)

    def Done(self, is_valid: bool) -> str:
        """
        Submit the final judgment result and verify if it is correct.
        Returns a string with correctness information (reward is handled by step()).
        """
        ref_answer = self.get_ref_answer()
        correct = (is_valid == ref_answer)
        msg = f"Your answer: {is_valid}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        # 保留原风格，附带 reward 信息（与 step 返回的 reward 一致但不作为判定依据）
        reward_txt = "1.0" if correct else "-1.0"
        return msg + f", reward={reward_txt}"

    def solve(self) -> str:
        """
        Automatically check all rows, columns, and subgrids, then submit the answer.
        Returns the final observation string from the 'answer' step.
        """
        n = self.grid_size
        s = self._subgrid_size()
        # 检查所有行
        for i in range(n):
            obs, _, terminated, _, _ = self.step(f"\\boxed{{check row {i}}}")
            if obs == "false":
                final_obs, _, _, _, _ = self.step("\\boxed{answer false}")
                return final_obs
            if terminated:
                # 若因错误或超时提前结束，直接返回
                return obs

        # 检查所有列
        for j in range(n):
            obs, _, terminated, _, _ = self.step(f"\\boxed{{check col {j}}}")
            if obs == "false":
                final_obs, _, _, _, _ = self.step("\\boxed{answer false}")
                return final_obs
            if terminated:
                return obs

        # 检查所有子宫格
        for r in range(s):
            for c in range(s):
                obs, _, terminated, _, _ = self.step(f"\\boxed{{check subgrid {r} {c}}}")
                if obs == "false":
                    final_obs, _, _, _, _ = self.step("\\boxed{answer false}")
                    return final_obs
                if terminated:
                    return obs

        final_obs, _, _, _, _ = self.step("\\boxed{answer true}")
        return final_obs