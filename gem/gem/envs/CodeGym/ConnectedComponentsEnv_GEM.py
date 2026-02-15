from typing import Any, Dict, Optional, Tuple
import random
import re
import json
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class ConnectedComponentsEnvGEM(Env):
    def __init__(
        self,
        complexity: int = 3,  # 难度等级 1-10，默认中等
        enable_param_randomization: bool = False,  # 评估时关闭随机化
        max_turns: Optional[int] = 100,
        **_,  # 忽略其他参数
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # 难度参数：控制网格大小与密度
        self.complexity_params = {
            "n_dim": (3, 20),               # 行数
            "m_dim": (3, 20),               # 列数
            "ones_density_pct": (20, 70),   # '1' 的密度百分比
        }

        # 参数方差（可选）
        self.param_variance = {
            "n_dim": 1,
            "m_dim": 1,
            "ones_density_pct": 10,
        }

        # 占位属性（由 _apply_complexity_params() 设置）
        self.n_dim: int = 0
        self.m_dim: int = 0
        self.ones_density_pct: int = 0

        # 运行时状态
        self.turn_count: int = 0

        # 环境状态（问题实例）
        self.n: int = 0
        self.m: int = 0
        self.grid: Any = []
        self.visited: Any = []
        self.components: int = 0

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
            "Connected Components (GEM): Count the number of connected '1' regions in a grid (4-directional).\n"
            "Cells are '0' or '1'. You can inspect cells and mark them visited to explore.\n"
            "Available actions (wrap one command in \\boxed{...}):\n"
            "- Show grid JSON: \\boxed{grid}\n"
            "- Observe task: \\boxed{observe}\n"
            "- Check a cell: \\boxed{check x y}\n"
            "- Mark a cell visited: \\boxed{mark x y}\n"
            "- Get neighbors of a cell: \\boxed{neighbors x y}\n"
            "- Find next unvisited '1' starting from (sx, sy): \\boxed{find sx sy}\n"
            "- Increment component counter: \\boxed{inc}\n"
            "- Get current component count: \\boxed{count}\n"
            "- Submit final answer N: \\boxed{answer N}\n"
            "Note: Coordinates are zero-based: 0 <= x < n, 0 <= y < m."
        )

    def get_task_suffix(self) -> str:
        return (
            f"Grid: {self.n}x{self.m} | Components counted: {self.components} | "
            f"Turn: {self.turn_count}/{self.max_turns} | Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响实例生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 从生成的问题实例中设置环境状态
        self.n = self.problem["n"]
        self.m = self.problem["m"]
        self.grid = self.problem["grid"]
        self.visited = [[False for _ in range(self.m)] for _ in range(self.n)] if self.n > 0 and self.m > 0 else []
        self.components = 0

        self.turn_count = 0
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.n_dim
        m = self.m_dim
        density = max(0, min(100, self.ones_density_pct)) / 100.0

        grid = []
        any_one = False
        for _ in range(n):
            row = []
            for _ in range(m):
                val = '1' if random.random() < density else '0'
                if val == '1':
                    any_one = True
                row.append(val)
            grid.append("".join(row))

        # 确保至少有一个 '1'，避免过于空的实例
        if not any_one:
            rx = random.randint(0, n - 1)
            ry = random.randint(0, m - 1)
            row_list = list(grid[rx])
            row_list[ry] = '1'
            grid[rx] = "".join(row_list)

        return {"n": n, "m": m, "grid": grid}

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
        if len(tokens) == 0:
            obs = f"Invalid action at turn {self.turn_count}."
            return (
                obs,
                LanguageGameReward.invalid_action_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        cmd = tokens[0].lower()
        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        try:
            if cmd == "grid":
                obs = self.ObserveGrid()
            elif cmd == "observe":
                obs = self.Observe()
            elif cmd == "check":
                if len(tokens) != 3:
                    obs = "Usage: check x y"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                x, y = int(tokens[1]), int(tokens[2])
                obs = self.CheckCell(x, y)
            elif cmd == "mark":
                if len(tokens) != 3:
                    obs = "Usage: mark x y"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                x, y = int(tokens[1]), int(tokens[2])
                obs = self.MarkVisited(x, y)
            elif cmd == "neighbors":
                if len(tokens) != 3:
                    obs = "Usage: neighbors x y"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                x, y = int(tokens[1]), int(tokens[2])
                obs = self.GetNeighbors(x, y)
            elif cmd == "find":
                if len(tokens) != 3:
                    obs = "Usage: find start_x start_y"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                sx, sy = int(tokens[1]), int(tokens[2])
                obs = self.FindUnvisitedOne(sx, sy)
            elif cmd == "inc":
                obs = self.IncrementComponentCount()
            elif cmd == "count":
                obs = self.GetComponentCount()
            elif cmd == "answer":
                if len(tokens) != 2:
                    obs = "Usage: answer N"
                    return obs, LanguageGameReward.invalid_action_reward, True, False, {"suffix": self.get_task_suffix()}
                ans = int(tokens[1])
                obs, reward, terminated = self._handle_answer(ans)
            else:
                obs = f"Unknown command: {cmd}"
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
        # 简单示例：先查看网格
        return "\\boxed{grid}"

    # -------------------------
    # 以下为原环境的辅助方法（已适配）
    # -------------------------
    def ObserveGrid(self) -> str:
        grid_info = {
            "n": self.n,
            "m": self.m,
            "grid": self.grid
        }
        return json.dumps(grid_info)

    def MarkVisited(self, x: int, y: int) -> str:
        if 0 <= x < self.n and 0 <= y < self.m:
            self.visited[x][y] = True
            return f"Cell ({x},{y}) has been marked as visited"
        else:
            return f"Error: Coordinates ({x},{y}) are out of grid bounds"

    def CheckCell(self, x: int, y: int) -> str:
        if 0 <= x < self.n and 0 <= y < self.m:
            cell_info = {
                "value": self.grid[x][y],
                "visited": self.visited[x][y]
            }
            return json.dumps(cell_info)
        else:
            return json.dumps({"error": f"Coordinates ({x},{y}) are out of grid bounds"})

    def FindUnvisitedOne(self, start_x: int, start_y: int) -> str:
        for i in range(start_x, self.n):
            start_j = start_y if i == start_x else 0
            for j in range(start_j, self.m):
                if self.grid[i][j] == '1' and not self.visited[i][j]:
                    return json.dumps({"x": i, "y": j})

        for i in range(self.n):
            for j in range(self.m):
                if (i < start_x or (i == start_x and j < start_y)) and self.grid[i][j] == '1' and not self.visited[i][j]:
                    return json.dumps({"x": i, "y": j})

        return json.dumps({"x": -1, "y": -1})

    def GetNeighbors(self, x: int, y: int) -> str:
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.m:
                neighbors.append([nx, ny])

        return json.dumps(neighbors)

    def IncrementComponentCount(self) -> str:
        self.components += 1
        return str(self.components)

    def GetComponentCount(self) -> str:
        return str(self.components)

    def Observe(self) -> str:
        return "Please analyze the number of building groups in the grid"

    def get_ref_answer(self) -> int:
        temp_visited = [[False] * self.m for _ in range(self.n)]
        components = 0

        for i in range(self.n):
            for j in range(self.m):
                if self.grid[i][j] == '1' and not temp_visited[i][j]:
                    components += 1
                    temp_visited[i][j] = True
                    stack = [(i, j)]
                    while stack:
                        cx, cy = stack.pop()
                        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nx, ny = cx + dx, cy + dy
                            if (
                                0 <= nx < self.n
                                and 0 <= ny < self.m
                                and self.grid[nx][ny] == '1'
                                and not temp_visited[nx][ny]
                            ):
                                temp_visited[nx][ny] = True
                                stack.append((nx, ny))
        return components

    def Done(self, answer: int) -> str:
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg

    def _handle_answer(self, ans: int) -> Tuple[str, float, bool]:
        ref_answer = self.get_ref_answer()
        correct = (ans == ref_answer)
        obs = f"Your answer: {ans}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        reward = 1.0 if correct else -1.0
        terminated = True
        return obs, reward, terminated

    # 可选：自动求解方法，使用 GEM 风格动作
    def solve(self) -> str:
        # 简单 BFS/DFS 求解，返回提交结果文本
        # 读网格
        grid_info = json.loads(self.ObserveGrid())
        n = grid_info["n"]
        m = grid_info["m"]

        current_x, current_y = 0, 0

        # 重置 visited 与计数
        self.visited = [[False for _ in range(m)] for _ in range(n)]
        self.components = 0

        while True:
            next_cell = json.loads(self.FindUnvisitedOne(current_x, current_y))
            if next_cell["x"] == -1 and next_cell["y"] == -1:
                break

            self.IncrementComponentCount()
            queue = [(next_cell["x"], next_cell["y"])]
            self.MarkVisited(next_cell["x"], next_cell["y"])

            while queue:
                x, y = queue.pop(0)
                neighbors = json.loads(self.GetNeighbors(x, y))
                for nx, ny in neighbors:
                    if 0 <= nx < n and 0 <= ny < m:
                        cell_data = json.loads(self.CheckCell(nx, ny))
                        if cell_data.get("value") == '1' and not cell_data.get("visited"):
                            self.MarkVisited(nx, ny)
                            queue.append((nx, ny))

            current_x, current_y = next_cell["x"], next_cell["y"]

        component_count = int(self.GetComponentCount())
        msg, _, _ = self._handle_answer(component_count)
        return msg