from typing import Any, Dict, Optional, Tuple
import random
import re
import heapq
from gem.core import Env
from gem.utils.constants import LanguageGameReward


class ShortestPathEnvGEM(Env):
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
        # - num_locations: 图中的节点数量
        # - avg_degree: 平均度（近似控制边数量）
        # - max_weight: 边权重的最大值
        self.complexity_params = {
            "num_locations": (5, 50),
            "avg_degree": (2, 10),
            "max_weight": (10, 100),
        }

        # 参数方差（enable_param_randomization=True 时生效）
        self.param_variance = {
            "num_locations": 3,
            "avg_degree": 1,
            "max_weight": 10,
        }

        # 占位属性（将由 _apply_complexity_params 填充）
        self.num_locations: int = 0
        self.avg_degree: int = 0
        self.max_weight: int = 0

        # 问题实例
        self.problem: Dict[str, Any] = {}

        # 算法状态变量
        self.graph: Dict[int, list] = {}
        self.priority_queue: list = []
        self.distances: Dict[int, float] = {}
        self.processed_nodes: set = set()

        # 评估状态
        self.turn_count: int = 0
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

        # 约束 avg_degree 不超过 num_locations-1
        if self.num_locations > 0:
            self.avg_degree = max(1, min(self.avg_degree, self.num_locations - 1))

    def _get_instructions(self) -> str:
        return (
            "Shortest Path (Dijkstra): Find shortest distance from start to treasure.\n"
            "Nodes are labeled 1..N. Undirected weighted edges.\n"
            "Available actions (wrap one command in \\boxed{...}):\n"
            "- Build graph: \\boxed{build}\n"
            "- Initialize Dijkstra: \\boxed{init}\n"
            "- Process next node: \\boxed{process}\n"
            "- Check treasure status: \\boxed{check}\n"
            "- Observe state: \\boxed{observe}\n"
            "- Submit answer (integer distance or IMPOSSIBLE): \\boxed{answer X}\n"
        )

    def get_task_suffix(self) -> str:
        num_paths = self.problem.get("num_paths", 0)
        start = self.problem.get("start", "?")
        treasure = self.problem.get("treasure", "?")
        return (
            f"Nodes: {self.num_locations}, Edges: {num_paths}\n"
            f"Start: {start}, Treasure: {treasure}\n"
            f"Processed: {len(self.processed_nodes)}, Queue: {len(self.priority_queue)}\n"
            f"Turn: {self.turn_count}/{self.max_turns}\n"
            f"Enter action."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)  # seed 只影响随机生成

        # 应用难度参数（不受 seed 影响）
        self._apply_complexity_params()

        # 使用难度参数生成问题实例（受 seed 影响）
        self.problem = self._generate_random_problem()

        # 重置状态
        self.graph = {}
        self.priority_queue = []
        self.distances = {}
        self.processed_nodes = set()

        self.turn_count = 0
        self._reward = 0.0
        self._done = False

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _generate_random_problem(self):
        """根据难度参数随机生成问题实例"""
        n = self.num_locations
        # 目标边数（无向图只计一次）
        target_edges = max(n - 1, int(n * self.avg_degree // 2))
        max_possible = n * (n - 1) // 2
        target_edges = min(max_possible, target_edges)

        edges_set = set()
        edges = []

        # 可选：先生成一棵随机生成树以保证连通性
        nodes = list(range(1, n + 1))
        random.shuffle(nodes)
        for i in range(1, n):
            u = nodes[i - 1]
            v = nodes[i]
            pair = (min(u, v), max(u, v))
            if pair not in edges_set:
                edges_set.add(pair)
                w = random.randint(1, self.max_weight)
                edges.append((u, v, w))

        # 添加其他随机边直到达到目标
        while len(edges) < target_edges:
            u = random.randint(1, n)
            v = random.randint(1, n)
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            if pair in edges_set:
                continue
            edges_set.add(pair)
            w = random.randint(1, self.max_weight)
            edges.append((u, v, w))

        start = random.randint(1, n)
        treasure = random.randint(1, n)
        # 确保起点和宝物不同
        if treasure == start:
            treasure = start % n + 1

        return {
            "num_locations": n,
            "num_paths": len(edges),
            "edges": edges,
            "start": start,
            "treasure": treasure,
        }

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

        content = parsed["content"].strip().lower()
        obs = ""
        reward = 0.0
        terminated = False
        truncated = False

        try:
            if content == "build":
                obs = self.BuildGraph()
            elif content == "init":
                obs = self.InitializeDijkstra()
            elif content == "process":
                obs = self.ProcessNextNode()
            elif content == "check":
                obs = self.CheckTreasureStatus()
            elif content == "observe":
                obs = self.Observe()
            elif content.startswith("answer"):
                # 解析答案：整数或 IMPOSSIBLE
                m = re.match(r"^answer\s+(.+)$", content, re.IGNORECASE)
                if not m:
                    obs = f"Invalid answer format: {content}"
                    reward = LanguageGameReward.format_error_reward
                    terminated = True
                else:
                    ans_token = m.group(1).strip().upper()
                    if ans_token == "IMPOSSIBLE":
                        answer = "IMPOSSIBLE"
                    else:
                        if not ans_token.isdigit():
                            obs = f"Invalid answer token: {ans_token}"
                            reward = LanguageGameReward.invalid_action_reward
                            terminated = True
                            # 超时检查在结尾
                            suffix_info = {"suffix": self.get_task_suffix()}
                            if not terminated and self.turn_count >= self.max_turns:
                                obs = f"{obs}\nReached max turns ({self.max_turns})."
                                return obs, 0.0, True, True, suffix_info
                            return obs, reward, terminated, truncated, suffix_info
                        answer = int(ans_token)
                    obs = self.Done(answer)
                    # 根据 Done 结果设置奖励与终止
                    reward = 1.0 if self._reward == 1 else -1.0
                    terminated = True
            else:
                obs = f"Invalid action: {content}"
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
        # 随机示例动作
        return "\\boxed{observe}"

    # -------------------------------
    # 辅助方法（从原环境转换保留）
    # -------------------------------
    def BuildGraph(self) -> str:
        """
        Construct a graph represented by an adjacency list based on the edge information in the environment.

        Returns:
            str: Basic information of the constructed graph.
        """
        self.num_locations = self.problem["num_locations"]
        self.num_paths = self.problem["num_paths"]
        self.edges = self.problem["edges"]
        self.start = self.problem["start"]
        self.treasure = self.problem["treasure"]

        self.graph = {i: [] for i in range(1, self.num_locations + 1)}
        for u, v, w in self.edges:
            self.graph[u].append((v, w))
            self.graph[v].append((u, w))
        return f"Graph constructed, containing {self.num_locations} nodes and {self.num_paths} edges"

    def InitializeDijkstra(self) -> str:
        """
        Initialize the priority queue and distance table required for Dijkstra's algorithm.

        Returns:
            str: Prompt message indicating successful initialization.
        """
        if self.num_locations == 0 or not self.edges:
            # 若未构建图，允许初始化，但 distances 只设置 start
            self.num_locations = self.problem["num_locations"]
            self.start = self.problem["start"]
            self.treasure = self.problem["treasure"]
            if not self.graph:
                self.graph = {i: [] for i in range(1, self.num_locations + 1)}

        self.priority_queue = [(0, self.start)]
        self.distances = {i: float("inf") for i in range(1, self.num_locations + 1)}
        self.distances[self.start] = 0
        self.processed_nodes = set()
        return f"Dijkstra's algorithm initialized, starting node is {self.start}"

    def ProcessNextNode(self) -> str:
        """
        Extract the next node from the priority queue for processing and update the distances of its neighbors.

        Returns:
            str: Processing result, including the currently processed node and updated neighbor information.
        """
        if not self.priority_queue:
            return "Priority queue is empty, cannot process more nodes"

        current_distance, current_node = heapq.heappop(self.priority_queue)

        if current_node in self.processed_nodes:
            return f"Node {current_node} has already been processed, skipping"

        self.processed_nodes.add(current_node)
        updated_neighbors = []

        if current_node in self.graph:
            for neighbor, weight in self.graph[current_node]:
                distance = current_distance + weight
                if distance < self.distances[neighbor]:
                    self.distances[neighbor] = distance
                    heapq.heappush(self.priority_queue, (distance, neighbor))
                    updated_neighbors.append(f"{neighbor}(distance {distance})")

        if updated_neighbors:
            return f"Processed node {current_node}: updated neighbors {', '.join(updated_neighbors)}"
        else:
            return f"Processed node {current_node}: no neighbors updated"

    def CheckTreasureStatus(self) -> str:
        """
        Check if the treasure location has been found or if a path to it is still possible.

        Returns:
            str: Status information of the treasure location.
        """
        if self.treasure in self.processed_nodes:
            return (
                f"Current shortest distance to treasure location {self.treasure} "
                f"is {self.distances[self.treasure]}"
            )
        elif not self.priority_queue:
            return "Priority queue is empty, unable to reach treasure location"
        else:
            current_known = (
                self.distances[self.treasure]
                if self.distances.get(self.treasure, float("inf")) != float("inf")
                else "unknown"
            )
            return (
                f"Treasure location {self.treasure} not processed yet, "
                f"currently known shortest distance is {current_known}, "
                f"there are {len(self.priority_queue)} nodes waiting in the queue"
            )

    def Observe(self) -> str:
        """
        Obtain the current state information of the environment, including the number of processed nodes
        and the status of the priority queue.

        Returns:
            str: Description of the current environment state.
        """
        return (
            f"Processed {len(self.processed_nodes)} nodes, "
            f"{len(self.priority_queue)} nodes waiting in the priority queue"
        )

    def Done(self, answer: Any) -> str:
        """
        Submit the final answer and verify its correctness.

        Args:
            answer (Union[int, str]): The submitted shortest path time, or "IMPOSSIBLE" indicating unreachable.

        Returns:
            str: Result information, including correctness and reward details.
        """
        ref_answer = self.get_ref_answer()
        correct = answer == ref_answer
        self._reward = 1 if correct else 0
        self._done = True
        msg = (
            f"Your answer: {answer}, Reference answer: {ref_answer}, "
            f"Result: {'Correct' if correct else 'Incorrect'}"
        )
        return msg

    def get_ref_answer(self) -> Any:
        """
        Compute the reference shortest path using Dijkstra's algorithm.
        """
        n = self.problem["num_locations"]
        edges = self.problem["edges"]
        start = self.problem["start"]
        treasure = self.problem["treasure"]

        graph = {i: [] for i in range(1, n + 1)}
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))

        pq = [(0, start)]
        dist = {i: float("inf") for i in range(1, n + 1)}
        dist[start] = 0

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_node == treasure:
                return current_distance
            if current_distance > dist[current_node]:
                continue
            for neighbor, weight in graph[current_node]:
                distance = current_distance + weight
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return "IMPOSSIBLE"

    def solve(self) -> str:
        """
        Automatically calls all actions in the environment to complete the shortest path calculation
        and submits the answer for verification.

        Returns:
            str: The result information of the final answer verification.
        """
        # Build and init
        self.step("\\boxed{build}")
        self.step("\\boxed{init}")
        # Iterate until we can determine answer
        while True:
            status, _, term, trunc, _ = self.step("\\boxed{check}")
            if term:
                # If terminated by format/invalid or answer, break
                if "Result:" in status or "Format error" in status or "Invalid" in status:
                    return status
            if "Priority queue is empty, unable to reach treasure location" in status:
                answer = "IMPOSSIBLE"
                final_obs, _, _, _, _ = self.step("\\boxed{answer IMPOSSIBLE}")
                return final_obs
            elif "Current shortest distance to treasure location" in status and "is " in status:
                try:
                    # Extract the integer after "is "
                    ans_str = status.split("is ")[1].strip()
                    # Remove potential trailing content
                    ans_token = ans_str.split()[0]
                    answer_val = int(ans_token)
                except Exception:
                    answer_val = "IMPOSSIBLE"
                final_obs, _, _, _, _ = self.step(f"\\boxed{{answer {answer_val}}}")
                return final_obs
            # Otherwise keep processing
            self.step("\\boxed{process}")