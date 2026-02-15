import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SpyNetworkEnv(Env):
    """Directed graph coverage with minimum cost - single turn Q&A environment."""

    def __init__(
        self,
        N: int = 10,
        edge_density: float = 0.3,
        dominated_probability: float = 0.5,
        **kwargs
    ):
        """
        Initialize the SpyNetworkEnv instance.

        Parameters:
        - N: number of vertices (N >= 3)
        - edge_density: probability factor for edge sampling in [0.0, 1.0]
        - dominated_probability: probability a vertex is "dominated", affects edge generation bias
        """
        super().__init__()
        self.N = N
        self.edge_density = edge_density
        self.dominated_probability = dominated_probability

        # Internal state for current problem
        self.current_problem: Optional[str] = None
        self.edges: List[Tuple[int, int]] = []
        self.costs: List[int] = []
        self.gold_answer: Optional[int] = None
        self.reference_vertices: List[int] = []

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a directed graph selection problem.\n"
            "Your task is to select a subset of vertices such that every vertex is reachable from at least one selected vertex.\n"
            "Your goal is to minimize the total cost of the selected vertices.\n"
            "Please provide your answer in \\boxed{...} format, where the content is a space-separated list of vertex indices.\n"
            "Example: \\boxed{0 1 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Parameter validation
        assert isinstance(self.N, int) and self.N >= 3, "N should be an integer >= 3"
        assert 0.0 <= float(self.edge_density) <= 1.0, "edge_density should be between 0.0 and 1.0"
        assert 0.0 <= float(self.dominated_probability) <= 1.0, "dominated_probability should be between 0.0 and 1.0"

        N = self.N
        edge_density = float(self.edge_density)
        dominated_probability = float(self.dominated_probability)

        # Generate biased edge candidate set
        dominated = [random.random() < dominated_probability for _ in range(N)]
        all_edges = [(s, t) for s in range(N) for t in range(N) if s != t and (not dominated[s] or dominated[t])]
        edges = random.sample(all_edges, min(len(all_edges), int(edge_density * N * (N - 1))))
        random.shuffle(edges)

        # Validate edges
        assert len(edges) == len(set(edges)), "edges should be unique"
        for s, t in edges:
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"

        # Generate costs
        costs = [random.randint(1, N) for _ in range(N)]

        # Build adjacency
        adj: List[List[int]] = [[] for _ in range(N)]
        for s, t in edges:
            adj[s].append(t)

        # Tarjan's algorithm for SCCs
        scc_id = [0] * N
        pre = [0] * N
        low = [0] * N
        stack: List[int] = []
        in_stack = [False] * N
        scc_count = 0
        dfs_clock = 0

        def tarjan(u: int) -> None:
            nonlocal dfs_clock, scc_count
            dfs_clock += 1
            pre[u] = dfs_clock
            low[u] = dfs_clock
            stack.append(u)
            in_stack[u] = True

            for v in adj[u]:
                if pre[v] == 0:
                    tarjan(v)
                    low[u] = min(low[u], low[v])
                elif in_stack[v]:
                    low[u] = min(low[u], pre[v])

            if low[u] == pre[u]:
                while True:
                    x = stack.pop()
                    in_stack[x] = False
                    scc_id[x] = scc_count
                    if x == u:
                        break
                scc_count += 1

        for i in range(N):
            if pre[i] == 0:
                tarjan(i)

        # Identify SCCs with zero in-degree in SCC DAG
        scc_in_degree = [False] * scc_count
        for u in range(N):
            for v in adj[u]:
                if scc_id[u] != scc_id[v]:
                    scc_in_degree[scc_id[v]] = True

        # For each SCC, find minimum cost vertex
        min_costs: List[Optional[int]] = [None] * scc_count
        min_vertices: List[Optional[int]] = [None] * scc_count
        for i, c in enumerate(costs):
            s_id = scc_id[i]
            if min_costs[s_id] is None or c < min_costs[s_id]:
                min_costs[s_id] = c
                min_vertices[s_id] = i

        ref_vertices = [min_vertices[s] for s in range(scc_count) if not scc_in_degree[s]]
        assert all(v is not None for v in ref_vertices)
        ref_vertices = [int(v) for v in ref_vertices]  # type: ignore

        gold_cost = sum(costs[v] for v in ref_vertices)
        assert gold_cost == sum(min_costs[s] for s in range(scc_count) if not scc_in_degree[s])  # type: ignore
        assert gold_cost > 0, "gold_answer should be greater than 0"

        # Save to state
        self.edges = edges
        self.costs = costs
        self.gold_answer = gold_cost
        self.reference_vertices = ref_vertices

        # Build problem prompt
        edges_str = "\n".join(f"({s}, {t})" for s, t in edges)
        costs_str = "\n".join(f"c[{i}]={costs[i]}" for i in range(N))
        example_cost = costs[0] + costs[1] + costs[N - 1]

        problem = (
            f"You are given a directed graph with {N} vertices, labeled from 0 to {N - 1}.\n\n"
            f"The graph contains the following directed edges. Each edge is represented as a tuple (s, t), "
            f"meaning there is a directed edge from vertex s to vertex t:\n{edges_str}\n\n"
            f"Each vertex i has an associated cost c[i], given as follows:\n{costs_str}\n\n"
            f"Your task is to select a subset of vertices s_1, s_2, ..., s_k such that:\n"
            f"- Every vertex in the graph is reachable (i.e., there exists a path ending at that vertex) starting from at least one of the selected vertices.\n"
            f"- Your goal is to minimize the total cost of the selected vertices: c[s_1] + c[s_2] + ... + c[s_k].\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single line containing the selected vertices separated by spaces, "
            f"enclosed in \\boxed{{...}}.\n"
            f"Example: \\boxed{{0 1 {N - 1}}} (this means the selected vertices are 0, 1, and {N - 1}, "
            f"and the total cost is c[0] + c[1] + c[{N - 1}] = {costs[0]} + {costs[1]} + {costs[N - 1]} = {example_cost})."
        )

        self.current_problem = problem
        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure we have a generated problem
        if self.gold_answer is None or self.current_problem is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Parse space-separated integers
        tokens = boxed_content.strip().split()
        if len(tokens) == 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "reason": "empty_selection"}

        try:
            selected_vertices: List[int] = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "reason": "non_integer_vertex"}

        # Validate selection
        if len(selected_vertices) != len(set(selected_vertices)):
            info = {
                "correct": False,
                "error": "invalid_solution",
                "reason": "duplicate_vertices",
                "user_vertices": selected_vertices,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        N = self.N
        for v in selected_vertices:
            if not (0 <= v < N):
                info = {
                    "correct": False,
                    "error": "invalid_solution",
                    "reason": "vertex_out_of_range",
                    "user_vertices": selected_vertices,
                }
                return TERMINAL_STATE, 0.0, True, False, info

        # Check coverage by DFS from selected vertices
        adj: List[List[int]] = [[] for _ in range(N)]
        for s, t in self.edges:
            adj[s].append(t)

        visited = [False] * N

        def dfs(u: int) -> None:
            if visited[u]:
                return
            visited[u] = True
            for w in adj[u]:
                dfs(w)

        for v in selected_vertices:
            dfs(v)

        coverage_ok = all(visited)
        total_cost = sum(self.costs[v] for v in selected_vertices)

        is_optimal_cost = (total_cost == self.gold_answer)
        is_correct = coverage_ok and is_optimal_cost

        info = {
            "correct": is_correct,
            "coverage_ok": coverage_ok,
            "user_cost": total_cost,
            "gold_cost": self.gold_answer,
            "user_vertices": selected_vertices,
            "reference_vertices": self.reference_vertices,
        }

        reward: float = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (not necessarily correct)."""
        # Randomly select some vertices
        if self.N <= 0:
            return r"\boxed{}"
        k = random.randint(1, max(1, min(self.N, 3)))
        verts = sorted(random.sample(range(self.N), k))
        content = " ".join(map(str, verts))
        return f"\\boxed{{{content}}}"