from typing import Any, Dict, Optional, Tuple
import random
import re
import sys
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class HamiltonianCycleEnvGEM(Env):
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
        # - num_cities: 城市数量（越多越难）
        # - density_percent: 边密度百分比（越高越接近完全图）
        # - max_weight: 边权重最大值（越大搜索空间越广）
        self.complexity_params = {
            "num_cities": (5, 15),
            "density_percent": (40, 100),
            "max_weight": (10, 1000),
        }

        # 参数方差（启用随机化时用于微调）
        self.param_variance = {
            "num_cities": 1,
            "density_percent": 10,
            "max_weight": 50,
        }

        # 占位属性
        self.num_cities: int = 0
        self.density_percent: int = 0
        self.max_weight: int = 0

        # 状态变量
        self.turn_count: int = 0

        # 任务相关状态
        self.n: int = 0
        self.m: int = 0
        self.roads: list = []
        self.adjacency_matrix: Optional[list] = None
        self.dp_table: Optional[list] = None

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
            "Hamiltonian Cycle (TSP variant): Find the minimum cycle starting and ending at city 0.\n"
            "Cities are indexed 0..N-1 internally (roads use 1-based indices). Use DP over subsets.\n"
            "Available actions:\n"
            "- Observe environment: \\boxed{observe}\n"
            "- Create adjacency: \\boxed{create}\n"
            "- Initialize DP: \\boxed{init}\n"
            "- Update DP: \\boxed{update mask=<int> u=<int> v=<int>}\n"
            "- Calculate minimum cycle: \\boxed{calc}\n"
            "- Submit answer: \\boxed{answer N}\n"
        )

    def get_task_suffix(self) -> str:
        return f"Cities: {self.n}, Roads: {self.m}\nTurn: {self.turn_count}/{self.max_turns}\nEnter action."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 重置内部状态
        self.turn_count = 0
        self.adjacency_matrix = None
        self.dp_table = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.num_cities
        total_pairs = n * (n - 1) // 2
        target_edges = max(n - 1, min(total_pairs, int(round(total_pairs * self.density_percent / 100.0))))

        # 生成一个随机生成树以确保连通
        roads = []
        nodes = list(range(1, n + 1))
        parent = nodes[0]
        for child in nodes[1:]:
            w = random.randint(1, self.max_weight)
            roads.append((parent, child, w))
            parent = random.choice(nodes[:nodes.index(child) + 1])

        # 剩余可选边集合
        existing = set()
        for u, v, _ in roads:
            if u > v:
                u, v = v, u
            existing.add((u, v))
        candidates = []
        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                if (u, v) not in existing:
                    candidates.append((u, v))

        remaining = max(0, target_edges - len(roads))
        random.shuffle(candidates)
        for i in range(min(remaining, len(candidates))):
            u, v = candidates[i]
            w = random.randint(1, self.max_weight)
            roads.append((u, v, w))

        self.n = n
        self.m = len(roads)
        self.roads = roads
        return {"n": self.n, "m": self.m, "roads": self.roads}

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
        cmd = tokens[0].lower()

        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if cmd == "observe":
                obs = self.Observe()
            elif cmd == "create":
                obs = self.CreateAdjacencyMatrix()
            elif cmd == "init":
                obs = self.InitializeDPTable()
            elif cmd == "update":
                # parse parameters mask=<int> u=<int> v=<int>
                params = self._parse_kv_params(content)
                if params is None or not {"mask", "u", "v"}.issubset(params.keys()):
                    obs = "Invalid action: update requires mask=<int> u=<int> v=<int>."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        mask = int(params["mask"])
                        u = int(params["u"])
                        v = int(params["v"])
                    except Exception:
                        obs = "Invalid action: parameters must be integers."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        obs = self.UpdateDPTable(mask, u, v)
            elif cmd == "calc":
                obs = self.CalculateMinimumCycle()
            elif cmd == "answer":
                if len(tokens) < 2:
                    obs = "Invalid action: answer requires an integer."
                    reward = LanguageGameReward.invalid_action_reward
                    terminated = True
                else:
                    try:
                        answer_val = int(tokens[1])
                    except Exception:
                        obs = "Invalid action: answer requires an integer."
                        reward = LanguageGameReward.invalid_action_reward
                        terminated = True
                    else:
                        msg, correct = self.Done(answer_val)
                        obs = msg
                        reward = 1.0 if correct else -1.0
                        terminated = True
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

    def _parse_kv_params(self, content: str) -> Optional[Dict[str, str]]:
        # Parse key=value pairs from content after command token
        parts = content.strip().split()
        if len(parts) <= 1:
            return {}
        params_str = " ".join(parts[1:])
        # Split by spaces, key=value
        pairs = re.findall(r"(\w+)\s*=\s*([-\w]+)", params_str)
        return {k: v for k, v in pairs} if pairs else {}

    def sample_random_action(self) -> str:
        if self.n > 0:
            return "\\boxed{observe}"
        return "\\boxed{observe}"

    # =========================
    # 原环境的辅助方法（转换为内部方法）
    # =========================
    def CreateAdjacencyMatrix(self) -> str:
        """
        Create an adjacency matrix between cities, representing the travel time between cities.
        """
        self.adjacency_matrix = [[sys.maxsize] * self.n for _ in range(self.n)]
        for u, v, w in self.roads:
            self.adjacency_matrix[u - 1][v - 1] = w
            self.adjacency_matrix[v - 1][u - 1] = w
        return "Adjacency matrix created successfully"

    def InitializeDPTable(self) -> str:
        """
        Initialize the dynamic programming table and set the initial state.
        """
        if self.adjacency_matrix is None:
            return "Error: Please create the adjacency matrix first"

        self.dp_table = [[sys.maxsize] * self.n for _ in range(1 << self.n)]
        self.dp_table[1][0] = 0  # Starting from city 0
        return "DP table initialized successfully"

    def UpdateDPTable(self, mask: int, u: int, v: int) -> str:
        """
        Update the dynamic programming table and attempt to travel from city u to city v.
        """
        if self.adjacency_matrix is None:
            return "Error: Please create the adjacency matrix first"

        if self.dp_table is None:
            return "Error: Please initialize the DP table first"

        if (
            mask & (1 << u)
            and (mask & (1 << v) == 0)
            and self.adjacency_matrix[u][v] != sys.maxsize
        ):
            new_mask = mask | (1 << v)
            new_value = min(
                self.dp_table[new_mask][v],
                self.dp_table[mask][u] + self.adjacency_matrix[u][v],
            )

            if new_value != self.dp_table[new_mask][v]:
                self.dp_table[new_mask][v] = new_value
                return f"DP table updated successfully: dp[{new_mask}][{v}] = {new_value}"
            else:
                return f"DP table not updated: dp[{new_mask}][{v}] is already the minimum value"
        else:
            return "Unable to update DP table: update conditions not met"

    def CalculateMinimumCycle(self) -> str:
        """
        Calculate the minimum Hamiltonian cycle returning to the starting point.
        """
        if self.adjacency_matrix is None:
            return "Error: Please create the adjacency matrix first"

        if self.dp_table is None:
            return "Error: Please initialize and update the DP table first"

        answer = sys.maxsize
        full_mask = (1 << self.n) - 1

        for i in range(1, self.n):
            if self.adjacency_matrix[i][0] != sys.maxsize:
                answer = min(
                    answer,
                    self.dp_table[full_mask][i] + self.adjacency_matrix[i][0],
                )

        return str(-1 if answer == sys.maxsize else answer)

    def Observe(self) -> str:
        """
        Return the number of cities and road information in the current environment.
        """
        return f"Number of cities: {self.n}, Number of roads: {self.m}, Road information: {self.roads}"

    def Done(self, answer: int) -> Tuple[str, bool]:
        """
        Verify whether the final answer is correct and return the result information.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        msg = f"Your answer: {answer}, Reference answer: {ref_answer}, Result: {'Correct' if correct else 'Incorrect'}"
        return msg, correct

    def get_ref_answer(self) -> int:
        """
        Use the information in the environment to get the reference answer.
        """
        matrix = [[sys.maxsize] * self.n for _ in range(self.n)]

        for u, v, w in self.roads:
            matrix[u - 1][v - 1] = w
            matrix[v - 1][u - 1] = w

        dp = [[sys.maxsize] * self.n for _ in range(1 << self.n)]
        dp[1][0] = 0  # starting from city 0

        for mask in range(1 << self.n):
            for u in range(self.n):
                if mask & (1 << u):
                    for v in range(self.n):
                        if (mask & (1 << v)) == 0 and matrix[u][v] != sys.maxsize:
                            next_mask = mask | (1 << v)
                            dp[next_mask][v] = min(
                                dp[next_mask][v], dp[mask][u] + matrix[u][v]
                            )

        answer = sys.maxsize
        for i in range(1, self.n):
            if matrix[i][0] != sys.maxsize:
                answer = min(answer, dp[(1 << self.n) - 1][i] + matrix[i][0])

        return -1 if answer == sys.maxsize else answer

    def solve(self) -> str:
        """
        Automatically call all actions to complete the process, and submit the answer for verification.
        """
        # Create adjacency and initialize DP
        self.step("\\boxed{create}")
        self.step("\\boxed{init}")

        # Extract n from observation
        obs, _, _, _, _ = self.step("\\boxed{observe}")
        city_str = "Number of cities: "
        n_start = obs.find(city_str) + len(city_str)
        n_end = obs.find(",", n_start)
        n = int(obs[n_start:n_end])

        # Update DP
        for mask in range(1, 1 << n):
            for u in range(n):
                if (mask & (1 << u)) == 0:
                    continue
                for v in range(n):
                    if (mask & (1 << v)) != 0:
                        continue
                    self.step(f"\\boxed{{update mask={mask} u={u} v={v}}}")

        # Calculate minimum cycle and submit answer
        min_cycle_str, _, _, _, _ = self.step("\\boxed{calc}")
        min_cycle = int(min_cycle_str) if min_cycle_str != "Error: Please initialize and update the DP table first" and min_cycle_str != "Error: Please create the adjacency matrix first" else -1
        final_obs, reward, terminated, truncated, _ = self.step(f"\\boxed{{answer {min_cycle}}}")
        return final_obs