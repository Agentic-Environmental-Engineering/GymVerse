from typing import Any, Dict, Optional, Tuple
import random
import re
from collections import deque
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class GridShortestPathEnvGEM(Env):
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
            "grid_rows": (2, 30),        # 行数
            "grid_cols": (2, 30),        # 列数
            "obstacle_pct": (5, 35),     # 障碍百分比（整数百分比）
            "turn_allowance": (30, 200), # 步数上限
        }

        # 参数方差（启用随机化时生效）
        self.param_variance = {
            "grid_rows": 2,
            "grid_cols": 2,
            "obstacle_pct": 5,
            "turn_allowance": 10,
        }

        # 占位属性
        self.grid_rows: int = 2
        self.grid_cols: int = 2
        self.obstacle_pct: int = 5
        self.turn_allowance: int = 100

        # 状态变量
        self.turn_count: int = 0

        # 问题生成相关
        self.grid: Optional[list] = None
        self.queue: Optional[deque] = None
        self.visited: Optional[set] = None
        self.current_node: Optional[Tuple[int, int, int]] = None

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

        # 用难度参数更新最大步数
        self.max_turns = int(self.turn_allowance)

    def _get_instructions(self) -> str:
        return (
            "Grid Shortest Path (BFS): Find the shortest path from (0,0) to (N-1,M-1) avoiding obstacles (#).\n"
            "Grid cells use '.' for free and '#' for obstacle.\n"
            "Available actions (use boxed syntax):\n"
            "- Observe grid: \\boxed{observe}\n"
            "- Initialize BFS: \\boxed{init}\n"
            "- Process next node: \\boxed{next}\n"
            "- Check neighbors: \\boxed{neighbors}\n"
            "- Get current node: \\boxed{current}\n"
            "- Get queue status: \\boxed{queue}\n"
            "- Submit answer: \\boxed{answer N}  (use -1 if no path)\n"
        )

    def get_task_suffix(self) -> str:
        queue_size = len(self.queue) if self.queue is not None else 0
        queue_empty = (len(self.queue) == 0) if self.queue is not None else True
        return (
            f"Grid: {self.grid_rows}x{self.grid_cols}, Obstacles~{self.obstacle_pct}% "
            f"Turn: {self.turn_count}/{self.max_turns} "
            f"Queue: empty={queue_empty}, size={queue_size}"
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
        self.queue = None
        self.visited = None
        self.current_node = None
        self.turn_count = 0

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        rows = self.grid_rows
        cols = self.grid_cols
        obstacle_probability = self.obstacle_pct / 100.0

        grid = []
        for r in range(rows):
            row_chars = []
            for c in range(cols):
                if random.random() < obstacle_probability:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            grid.append("".join(row_chars))

        # 保证起点和终点可进入
        if rows > 0 and cols > 0:
            start_row = list(grid[0])
            end_row = list(grid[rows - 1])
            start_row[0] = "."
            end_row[cols - 1] = "."
            grid[0] = "".join(start_row)
            grid[rows - 1] = "".join(end_row)

        # 存储到环境
        self.grid = grid

        return {"grid": grid, "N": rows, "M": cols}

    # ===== 原环境辅助方法（已适配） =====
    def Observe(self) -> str:
        """
        Obtain basic information about the grid.
        Returns JSON: {"N": int, "M": int, "grid": [str, ...]}
        """
        info = {
            "N": self.grid_rows,
            "M": self.grid_cols,
            "grid": self.grid,
        }
        return self._to_json(info)

    def InitializeBFS(self) -> str:
        """
        Initialize the BFS algorithm, setting up the queue and visited set.
        Returns a message string.
        """
        # Check if the start point is an obstacle
        if self.grid[0][0] == "#":
            self.queue = deque()
            self.visited = set()
            self.current_node = None
            return "BFS initialization failed, starting point (0,0) is an obstacle"

        self.queue = deque([(0, 0, 0)])  # (row, col, steps)
        self.visited = {(0, 0)}
        self.current_node = None
        return "BFS initialization successful, starting point (0,0) is reachable"

    def ProcessNextNode(self) -> str:
        """
        Dequeue the next node for processing and check if the end point is reached.
        Returns JSON: {"status": "queue_empty"|"processing"|"reached", ...}
        """
        if self.queue is None or not self.queue:
            return self._to_json({"status": "queue_empty"})

        self.current_node = self.queue.popleft()
        r, c, steps = self.current_node

        # Check if we reached the bottom-right corner
        if r == self.grid_rows - 1 and c == self.grid_cols - 1:
            return self._to_json(
                {"status": "reached", "steps": steps, "node": [r, c, steps]}
            )

        return self._to_json({"status": "processing", "node": [r, c, steps]})

    def CheckNeighbors(self) -> str:
        """
        Check the four-directional neighbors of the current node and add valid neighbors to the queue.
        Returns JSON: {"added_neighbors": int, "total_queue_size": int} or error JSON.
        """
        if self.current_node is None:
            return self._to_json(
                {"error": "no_current_node", "message": "Please call next (ProcessNextNode) first"}
            )

        r, c, steps = self.current_node
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        added_count = 0

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < self.grid_rows
                and 0 <= nc < self.grid_cols
                and (nr, nc) not in self.visited
                and self.grid[nr][nc] == "."
            ):
                self.visited.add((nr, nc))
                self.queue.append((nr, nc, steps + 1))
                added_count += 1

        return self._to_json(
            {"added_neighbors": added_count, "total_queue_size": len(self.queue)}
        )

    def GetCurrentNode(self) -> str:
        """
        Obtain information about the current node being processed.
        Returns "(r, c, steps)" or "None"
        """
        if self.current_node is None:
            return "None"
        r, c, steps = self.current_node
        return f"({r}, {c}, {steps})"

    def GetQueueStatus(self) -> str:
        """
        Obtain the current status of the queue, including whether it is empty and the size.
        Returns JSON: {"queue_empty": bool, "queue_size": int}
        """
        status = {
            "queue_empty": (len(self.queue) == 0) if self.queue else True,
            "queue_size": len(self.queue) if self.queue else 0,
        }
        return self._to_json(status)

    def get_ref_answer(self) -> int:
        """
        Compute the shortest path length using BFS.
        Returns the number of steps or -1 if no path exists.
        """
        if self.grid[0][0] == "#" or self.grid[self.grid_rows - 1][self.grid_cols - 1] == "#":
            return -1

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        queue = deque([(0, 0, 0)])
        visited = {(0, 0)}

        while queue:
            r, c, steps = queue.popleft()

            if r == self.grid_rows - 1 and c == self.grid_cols - 1:
                return steps

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.grid_rows
                    and 0 <= nc < self.grid_cols
                    and (nr, nc) not in visited
                    and self.grid[nr][nc] == "."
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc, steps + 1))

        return -1

    def solve(self) -> str:
        """
        Automatically execute actions to find the shortest path and submit the answer.
        Returns the final observation string of answer submission.
        """
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        obs, _, _, _, _ = self.step("\\boxed{init}")

        while True:
            queue_status_str, _, terminated, _, _ = self.step("\\boxed{queue}")
            if terminated:
                return queue_status_str
            # parse queue status
            status = self._from_json(queue_status_str) or {}
            if status.get("queue_empty", True):
                final_obs, _, _, _, _ = self.step("\\boxed{answer -1}")
                return final_obs

            process_result_str, _, terminated, _, _ = self.step("\\boxed{next}")
            if terminated:
                return process_result_str
            process_data = self._from_json(process_result_str) or {}
            if process_data.get("status") == "reached":
                current_node_str, _, terminated2, _, _ = self.step("\\boxed{current}")
                if terminated2:
                    return current_node_str
                try:
                    # "(r, c, steps)"
                    steps = int(current_node_str.strip().split(",")[2].strip().strip(")"))
                except Exception:
                    # fallback to ProcessNextNode's steps
                    steps = int(process_data.get("steps", -1))
                final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {steps}}}")
                return final_obs

            # Otherwise, expand neighbors
            obs, _, terminated3, _, _ = self.step("\\boxed{neighbors}")
            if terminated3:
                return obs

    # ===== GEM 交互方法 =====
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
        cmd = tokens[0].lower() if tokens else ""

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        if cmd == "observe":
            obs = self.Observe()

        elif cmd == "init":
            obs = self.InitializeBFS()

        elif cmd == "next":
            obs = self.ProcessNextNode()

        elif cmd == "neighbors":
            obs = self.CheckNeighbors()

        elif cmd == "current":
            obs = self.GetCurrentNode()

        elif cmd == "queue":
            obs = self.GetQueueStatus()

        elif cmd == "answer":
            if len(tokens) < 2:
                obs = "Invalid answer format. Use: \\boxed{answer N}"
                reward = LanguageGameReward.invalid_action_reward
                terminated = True
                truncated = False
            else:
                try:
                    answer_val = int(tokens[1])
                except Exception:
                    obs = "Invalid answer value. Must be an integer."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                    truncated = False
                else:
                    ref_answer = self.get_ref_answer()
                    correct = (answer_val == ref_answer)
                    result_str = "Correct" if correct else "Incorrect"
                    obs = f"Your answer: {answer_val}, Reference answer: {ref_answer}, Result: {result_str}"
                    reward = 1.0 if correct else -1.0
                    terminated = True
                    truncated = False
        else:
            obs = f"Invalid action '{cmd}'."
            reward = LanguageGameReward.invalid_action_reward
            terminated = True
            truncated = False

        # 超时检查（放在 step 结尾）
        if not terminated and self.turn_count >= self.max_turns:
            obs = f"{obs}\nReached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        return {"content": content}

    def sample_random_action(self) -> str:
        # 初始随机动作，通常先观察或初始化
        return "\\boxed{observe}"

    # ===== 工具方法 =====
    def _to_json(self, obj: Dict[str, Any]) -> str:
        # 简单 JSON 序列化（避免引入额外依赖）
        # 注意：obj 中不应含有不可序列化对象
        import json

        return json.dumps(obj, separators=(",", ":"))

    def _from_json(self, s: str) -> Optional[Dict[str, Any]]:
        import json

        try:
            return json.loads(s)
        except Exception:
            return None