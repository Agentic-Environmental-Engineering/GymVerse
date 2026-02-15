from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from copy import deepcopy
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class GoldCollectionEnvGEM(Env):
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
        # - grid_rows/grid_cols: 网格行列
        # - value_max: 金币数值上限
        # - zero_ratio_pct: 0 值（不可访问/无金币）比例（百分比）
        self.complexity_params = {
            "grid_rows": (3, 10),
            "grid_cols": (3, 10),
            "value_max": (6, 50),
            "zero_ratio_pct": (30, 75),
        }

        # 参数方差（启用随机化时微调）
        self.param_variance = {
            "grid_rows": 1,
            "grid_cols": 1,
            "value_max": 5,
            "zero_ratio_pct": 10,
        }

        # 占位属性（由 _apply_complexity_params 填充）
        self.grid_rows: int = 0
        self.grid_cols: int = 0
        self.value_max: int = 0
        self.zero_ratio_pct: int = 0

        # 环境状态变量
        self.turn_count: int = 0

        # 来自原环境的运行时变量
        self.original_grid = []
        self.temp_grid = []
        self.rows = 0
        self.cols = 0
        self.start_x, self.start_y = -1, -1
        self.max_gold_collected = 0
        self._reward = 0.0
        self._done = False

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
            "Gold Collection: Find the maximum collectible gold in a grid by moving up/down/left/right without revisiting cells.\n"
            "Available actions (wrap one command in \\boxed{...}):\n"
            "- observe\n"
            "- set x y            # set start position\n"
            "- get x y            # get gold at cell\n"
            "- mark x y           # mark cell as visited (temporarily set to 0)\n"
            "- unmark x y gold    # restore cell with provided gold\n"
            "- explore x y        # show [up, down, left, right] gold around (x,y)\n"
            "- update amount      # update best tracked amount\n"
            "- answer amount      # submit final answer and finish\n"
        )

    def get_task_suffix(self) -> str:
        return f"Grid: {self.rows}x{self.cols} | Turn: {self.turn_count}/{self.max_turns} | MaxGoldSoFar: {self.max_gold_collected}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 初始化环境状态
        self.original_grid = deepcopy(self.problem["grid"])
        self.temp_grid = deepcopy(self.original_grid)
        self.rows = len(self.original_grid)
        self.cols = len(self.original_grid[0]) if self.rows > 0 else 0
        self.start_x, self.start_y = -1, -1
        self.max_gold_collected = 0
        self._reward = 0.0
        self._done = False
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self) -> Dict[str, Any]:
        """根据难度参数随机生成问题实例"""
        rows, cols = self.grid_rows, self.grid_cols
        zero_p = self.zero_ratio_pct / 100.0
        grid = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                if random.random() < zero_p:
                    row.append(0)
                else:
                    row.append(random.randint(1, self.value_max))
            grid.append(row)
        return {"grid": grid}

    # 解析动作（保留DungeonScout风格）
    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action:
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        return {"content": content}

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
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()

            elif cmd == "set":
                if len(tokens) != 3:
                    obs = "Error: set requires 2 integers: set x y"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                x, y = int(tokens[1]), int(tokens[2])
                obs = self.SetStartPosition(x, y)

            elif cmd == "mark":
                if len(tokens) != 3:
                    obs = "Error: mark requires 2 integers: mark x y"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                x, y = int(tokens[1]), int(tokens[2])
                obs = self.MarkVisited(x, y)

            elif cmd == "unmark":
                if len(tokens) != 4:
                    obs = "Error: unmark requires 3 integers: unmark x y gold"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                x, y, gold = int(tokens[1]), int(tokens[2]), int(tokens[3])
                obs = self.UnmarkVisited(x, y, gold)

            elif cmd == "get":
                if len(tokens) != 3:
                    obs = "Error: get requires 2 integers: get x y"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                x, y = int(tokens[1]), int(tokens[2])
                obs = self.GetGoldAt(x, y)

            elif cmd == "explore":
                if len(tokens) != 3:
                    obs = "Error: explore requires 2 integers: explore x y"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                x, y = int(tokens[1]), int(tokens[2])
                obs = self.ExploreDirection(x, y)

            elif cmd == "update":
                if len(tokens) != 2:
                    obs = "Error: update requires 1 integer: update amount"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                amount = int(tokens[1])
                obs = self.UpdateMaxGold(amount)

            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Error: answer requires 1 integer: answer amount"
                    return (
                        obs,
                        LanguageGameReward.invalid_action_reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
                amount = int(tokens[1])
                ref_answer = self.get_ref_answer()
                correct = amount == ref_answer
                # 调用辅助 Done 以生成一致的消息
                obs = self.Done(amount)
                reward = 1.0 if correct else -1.0
                terminated = True

            elif cmd == "help":
                obs = self._get_instructions()

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

    def sample_random_action(self) -> str:
        if self.rows > 0 and self.cols > 0:
            x = random.randint(0, max(0, self.rows - 1))
            y = random.randint(0, max(0, self.cols - 1))
            return f"\\boxed{{get {x} {y}}}"
        return "\\boxed{observe}"

    # =========================
    # 以下为原环境的辅助方法（保留并适配）
    # =========================

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def reward(self):
        return float(self._reward)

    # Reset 已适配 GEM 框架，不再使用原 options 形式
    # 参考答案（DFS）
    def get_ref_answer(self):
        rows, cols = len(self.original_grid), len(self.original_grid[0]) if self.original_grid else 0

        def dfs(x, y, grid_copy):
            if x < 0 or x >= rows or y < 0 or y >= cols or grid_copy[x][y] == 0:
                return 0

            current_gold = grid_copy[x][y]
            grid_copy[x][y] = 0  # Mark as visited

            max_gold = 0
            # Explore all 4 directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                max_gold = max(max_gold, dfs(new_x, new_y, grid_copy))

            grid_copy[x][y] = current_gold  # Unmark (backtrack)
            return current_gold + max_gold

        max_gold = 0
        for i in range(rows):
            for j in range(cols):
                if self.original_grid[i][j] != 0:
                    grid_copy = deepcopy(self.original_grid)
                    max_gold = max(max_gold, dfs(i, j, grid_copy))

        return max_gold

    # Action implementations
    def SetStartPosition(self, x: int, y: int) -> str:
        r"""
        Set the starting position of the current exploration.
        """
        if 0 <= x < self.rows and 0 <= y < self.cols:
            self.start_x, self.start_y = x, y
            return f"Start position set: ({x}, {y})"
        return f"Invalid start position: ({x}, {y})"

    def MarkVisited(self, x: int, y: int) -> str:
        r"""
        Mark the specified cell as visited (temporarily set to 0).
        """
        if 0 <= x < self.rows and 0 <= y < self.cols:
            gold = self.temp_grid[x][y]
            self.temp_grid[x][y] = 0  # Mark as visited
            return f"Marked cell ({x}, {y}) as visited, original gold amount: {gold}"
        return f"Invalid cell: ({x}, {y})"

    def UnmarkVisited(self, x: int, y: int, gold: int) -> str:
        r"""
        Unmark the specified cell (restore the original amount of gold).
        """
        if 0 <= x < self.rows and 0 <= y < self.cols:
            self.temp_grid[x][y] = gold
            return f"Restored gold amount for cell ({x}, {y}): {gold}"
        return f"Invalid cell: ({x}, {y})"

    def GetGoldAt(self, x: int, y: int) -> str:
        r"""
        Get the current amount of gold in the specified cell.
        """
        if 0 <= x < self.rows and 0 <= y < self.cols:
            return str(self.temp_grid[x][y])
        return "0"

    def ExploreDirection(self, x: int, y: int) -> str:
        r"""
        Explore the four directions from the specified position and return the amount of gold in each direction.
        Returns [up, down, left, right] in JSON string.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        gold_values = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                gold_values.append(self.temp_grid[nx][ny])
            else:
                gold_values.append(0)

        return json.dumps(gold_values)

    def UpdateMaxGold(self, amount: int) -> str:
        r"""
        Update the maximum amount of gold collected.
        """
        if amount > self.max_gold_collected:
            self.max_gold_collected = amount
            return f"Updated maximum gold amount to: {self.max_gold_collected}"
        return f"Current gold amount {amount} is not greater than maximum gold amount {self.max_gold_collected}"

    def Observe(self) -> str:
        r"""
        Return the observation information of the current environment, including the grid size and temporary grid status.
        """
        return f"Grid size: {self.rows}x{self.cols}, temporary grid status: {str(self.temp_grid)}"

    def Done(self, amount: int) -> str:
        r"""
        Submit the final answer and verify whether the collected amount of gold is correct.
        """
        ref_answer = self.get_ref_answer()
        correct = amount == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = f"Your answer: {amount}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg + f", reward={self._reward}"

    # Solve method to simulate agent behavior using GEM-style actions
    def solve(self):
        r"""
        Automatically call all actions in the environment to complete the full process,
        find the maximum collectible gold amount, and submit for verification.
        Returns:
            str: The result information of the final answer verification.
        """
        # Observe
        observe_result = self.step("\\boxed{observe}")[0]
        # Parse grid size from Observe output
        try:
            grid_size_str = observe_result.split(',')[0].split(': ')[1]
            rows, cols = map(int, grid_size_str.split('x'))
        except Exception:
            # fallback
            rows, cols = self.rows, self.cols

        max_gold = 0

        for x in range(rows):
            for y in range(cols):
                gold_at = self.step(f"\\boxed{{get {x} {y}}}")[0]
                try:
                    current_gold = int(gold_at)
                except Exception:
                    current_gold = 0
                if current_gold == 0:
                    continue  # No gold, skip

                self.step(f"\\boxed{{set {x} {y}}}")

                mark_result = self.step(f"\\boxed{{mark {x} {y}}}")[0]
                try:
                    original_gold = int(mark_result.split('original gold amount: ')[1])
                except Exception:
                    original_gold = current_gold

                def dfs(cx, cy):
                    dir_result = self.step(f"\\boxed{{explore {cx} {cy}}}")[0]
                    try:
                        directions = json.loads(dir_result)  # [up, down, left, right]
                    except Exception:
                        directions = [0, 0, 0, 0]
                    dir_coords = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]

                    max_path = 0
                    for i in range(4):
                        nx, ny = dir_coords[i]
                        if directions[i] > 0:
                            sub_mark_res = self.step(f"\\boxed{{mark {nx} {ny}}}")[0]
                            try:
                                sub_gold = int(sub_mark_res.split('original gold amount: ')[1])
                            except Exception:
                                sub_gold = directions[i]

                            current_path = sub_gold + dfs(nx, ny)
                            if current_path > max_path:
                                max_path = current_path

                            self.step(f"\\boxed{{unmark {nx} {ny} {sub_gold}}}")
                    return max_path

                current_max = current_gold + dfs(x, y)

                self.step(f"\\boxed{{unmark {x} {y} {original_gold}}}")

                if current_max > max_gold:
                    max_gold = current_max
                    self.step(f"\\boxed{{update {max_gold}}}")

        return self.step(f"\\boxed{{answer {max_gold}}}")[0]