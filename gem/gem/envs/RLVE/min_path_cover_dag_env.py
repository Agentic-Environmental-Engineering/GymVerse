from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from collections import deque
import re
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinPathCover_DAGEnv(Env):
    """Environment for a DAG path cover minimization task (single-turn Q&A).

    Task:
      - You are given a directed acyclic graph (DAG) with N vertices labeled 1..N.
      - All edges must be covered by a set of directed paths, and each path must start from vertex 1.
      - The cost of a path is the sum of its edge weights.
      - The objective is to minimize the total cost across all chosen paths.

    The environment generates a random DAG that satisfies:
      - The graph is acyclic.
      - Vertex 1 can reach every other vertex.

    The agent must output the minimum possible total cost as a single integer inside \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 12,
        max_generation_attempts: int = 1000,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            N: If provided, use this fixed number of vertices (must be >= 4).
            min_N: Minimum number of vertices when sampling N randomly.
            max_N: Maximum number of vertices when sampling N randomly.
            max_generation_attempts: Maximum attempts to generate a valid DAG.
        """
        super().__init__()
        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N
        self.max_generation_attempts = max_generation_attempts

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.total_cost_sum_upper_bound: int = 0

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a directed acyclic graph (DAG) and must cover all edges using a set of paths, "
            "each path starting at vertex 1. The cost of a path is the sum of its edge weights, and the goal "
            "is to minimize the total cost across all chosen paths.\n"
            "Please provide your final answer (the minimal total cost) in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            assert self.fixed_N >= 4, "N should be greater than or equal to 4"
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, max(self.min_N, self.max_N))
            assert N >= 4, "N should be greater than or equal to 4"

        # Generate a valid DAG where 1 can reach all other vertices
        edges: List[Tuple[int, int, int]] = []
        for _ in range(self.max_generation_attempts):
            candidate_edges: List[Tuple[int, int, int]] = []
            topological_order = list(range(1, N + 1))
            random.shuffle(topological_order[1:])  # Keep 1 as the first vertex

            for i in range(1, N):
                t = topological_order[i]
                # Each vertex t gets edges from a non-empty subset of previous vertices
                prev_vertices = topological_order[:i]
                k = random.randint(1, i)
                for s in random.sample(prev_vertices, k):
                    w = random.randint(1, N * (N - 1))
                    candidate_edges.append((s, t, w))

            # Ensure no duplicate edges
            if len(candidate_edges) != len(set((s, t) for s, t, _ in candidate_edges)):
                continue

            # Build and validate DAG
            G = networkx.DiGraph()
            G.add_weighted_edges_from(candidate_edges)
            if not networkx.is_directed_acyclic_graph(G):
                continue
            if not all(networkx.has_path(G, 1, v) for v in range(2, N + 1)):
                continue

            edges = candidate_edges
            break

        if not edges:
            raise RuntimeError("Failed to generate a valid DAG within the maximum attempts.")

        # Compute the reference (minimal total cost) using the original algorithm
        reference = self._compute_min_total_cost(N, edges)

        # Store state
        self.N = N
        self.edges = edges
        self.reference_answer = reference
        self.total_cost_sum_upper_bound = sum(w for _, _, w in edges)

        # Build problem statement
        edge_lines = "\n".join(f"({s}, {t}, {w})" for s, t, w in edges)
        self.current_problem = (
            f"You are given a directed acyclic graph (DAG) with {N} vertices labeled from 1 to {N}. "
            f"The graph contains the following directed edges (s, t, w), meaning there is an edge from s to t with weight w. "
            f"It is guaranteed that vertex 1 can reach all other vertices:\n{edge_lines}\n\n"
            f"Find a set of directed paths such that each path starts from vertex 1 and every edge in the graph is covered by at least one path. "
            f"Minimize the total weight of all paths, where the weight of a path is the sum of the weights of its edges.\n\n"
            f"Output Format: Provide a single integer — the minimum possible total weight — in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the agent's answer and return the result."""
        # Extract boxed answer
        extracted = self._parse_answer(action)
        if extracted is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem"}

        # Validate numeric answer
        try:
            user_answer = int(extracted)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "num_edges": len(self.edges),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # Use a heuristic bound for random guessing
        upper = max(1, self.total_cost_sum_upper_bound)
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"

    def _compute_min_total_cost(self, N: int, edges: List[Tuple[int, int, int]]) -> int:
        """Compute the minimal total cost using the original algorithm."""
        # Prepare arrays (1-based indexing with extra nodes)
        A = [0] * (N + 4)  # 1..N, rT=N+1, vS=N+2, vT=N+3
        edges_data: List[List[Tuple[int, int]]] = [[] for _ in range(N + 4)]
        total_cost_sum = 0
        M = len(edges)

        # Populate edges_data and A as in the original code
        for s, t, w in edges:
            edges_data[s].append((t, w))
            A[t] += 1
            A[s] -= 1
            total_cost_sum += w

        # INF sentinel based on input
        INF = total_cost_sum + M + 5

        size = N + 4  # indices used: 1..N, rT=N+1, vS=N+2, vT=N+3
        Graph: List[List["Edge"]] = [[] for _ in range(size)]

        class Edge:
            __slots__ = ("to", "cap", "cost", "rev")

            def __init__(self, to: int, cap: int, cost: int, rev: int):
                self.to = to
                self.cap = cap
                self.cost = cost
                self.rev = rev

        def add_edge(u: int, v: int, cap: int, cost: int) -> None:
            Graph[u].append(Edge(v, cap, cost, len(Graph[v])))
            Graph[v].append(Edge(u, 0, -cost, len(Graph[u]) - 1))

        rS = 1
        rT = N + 1
        vS = N + 2
        vT = N + 3

        # Build graph
        for i in range(1, N + 1):
            for (u_to, cost_w) in edges_data[i]:
                add_edge(i, u_to, INF - 1, cost_w)

        for i in range(2, N + 1):
            add_edge(i, rT, INF, 0)

        for i in range(1, N + 1):
            if A[i] > 0:
                add_edge(vS, i, A[i], 0)
            elif A[i] < 0:
                add_edge(i, vT, -A[i], 0)

        add_edge(rT, rS, INF, 0)

        S = vS
        T = vT

        Dist = [0] * size
        Cur = [0] * size
        InQ = [False] * size
        Vis = [False] * size

        # ret starts as the sum of all edge costs
        ret = total_cost_sum

        def spfa() -> bool:
            for i in range(size):
                Dist[i] = INF
                InQ[i] = False
            Dist[S] = 0
            q = deque([S])
            InQ[S] = True
            while q:
                u = q.popleft()
                InQ[u] = False
                for e in Graph[u]:
                    if e.cap > 0 and Dist[e.to] > Dist[u] + e.cost:
                        Dist[e.to] = Dist[u] + e.cost
                        if not InQ[e.to]:
                            InQ[e.to] = True
                            q.append(e.to)
            return Dist[T] < INF

        def dfs(x: int, f: int) -> int:
            nonlocal ret
            if x == T:
                return f
            Vis[x] = True
            flow = 0
            i = Cur[x]
            while i < len(Graph[x]) and flow < f:
                Cur[x] = i
                e = Graph[x][i]
                v = e.to
                if (not Vis[v]) and e.cap > 0 and Dist[v] == Dist[x] + e.cost:
                    pushed = dfs(v, min(e.cap, f - flow))
                    if pushed:
                        ret += pushed * e.cost
                        e.cap -= pushed
                        Graph[v][e.rev].cap += pushed
                        flow += pushed
                i += 1
            Vis[x] = False
            return flow

        def dinic() -> int:
            total = 0
            while spfa():
                for i in range(size):
                    Cur[i] = 0
                    Vis[i] = False
                while True:
                    pushed = dfs(S, INF)
                    if pushed == 0:
                        break
                    total += pushed
            return total

        dinic()
        assert ret > 0, "Reference answer must be positive"
        return ret