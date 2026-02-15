from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Vertex_KCenterEnv(Env):
    """Vertex K-Center problem environment - single-turn Q&A.

    The task is to select K distinct vertices in an undirected connected weighted graph
    such that the maximum distance from any vertex to its nearest selected vertex is minimized.
    The answer must be provided in \\boxed{...} format containing K integers separated by spaces.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        edge_density: Optional[float] = None,
        min_N: int = 3,
        max_N: int = 12,
        **kwargs
    ):
        """
        Initialize the Vertex_KCenterEnv.

        Parameters:
        - N: Optional fixed number of vertices. If None, a random N in [min_N, max_N] is chosen on reset.
        - edge_density: Optional target edge density in [0.0, 1.0]. If None, sampled uniformly on reset.
        - min_N: Minimum number of vertices if N is not provided. Must be >= 3.
        - max_N: Maximum number of vertices if N is not provided. Must be >= min_N.

        Note:
        - Rewards are fixed: correct=1.0, wrong=0.0, format_error=-0.1.
        """
        super().__init__()
        assert isinstance(min_N, int) and isinstance(max_N, int), "min_N and max_N must be integers"
        assert min_N >= 3, "min_N should be greater than or equal to 3"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"
        if N is not None:
            assert isinstance(N, int), "N must be an integer"
            assert N >= 3, "N should be greater than or equal to 3"
        if edge_density is not None:
            assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        self.fixed_N = N
        self.fixed_edge_density = edge_density
        self.min_N = min_N
        self.max_N = max_N

        # Runtime state
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.Floyd: List[List[int]] = []
        self.gold_answer: Optional[int] = None
        self.reference_answer_list: List[int] = []
        self.reference_answer_str: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task description."""
        return (
            "You are solving a Vertex K-Center problem on an undirected connected weighted graph.\n"
            "Choose K distinct vertices to minimize the largest shortest-path distance of any vertex to the nearest chosen vertex.\n"
            "Please provide your answer in \\boxed{v1 v2 ... vK} format, where v1..vK are integers.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N and edge density
        N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"

        if self.fixed_edge_density is not None:
            edge_density = self.fixed_edge_density
            assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"
        else:
            edge_density = random.random()

        # Random K
        K = random.randint(1, N - 1)

        # Build a connected random graph
        edges: List[Tuple[int, int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            edges.append((u, v, random.randint(1, N)))

        # Target number of edges by density
        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges:
            existing_pairs = set((u, v) for u, v, _ in edges)
            all_pairs = set((u, v) for u in range(N) for v in range(u + 1, N))
            remaining_edges = list(all_pairs - existing_pairs)
            random.shuffle(remaining_edges)
            remaining_edges = remaining_edges[: max(0, num_edges - len(edges))]
            for u, v in remaining_edges:
                edges.append((u, v, random.randint(1, N)))
        random.shuffle(edges)

        # Floyd-Warshall initialization
        INF = N * N
        Floyd = [[INF] * N for _ in range(N)]
        for i in range(N):
            Floyd[i][i] = 0
        for u, v, w in edges:
            assert 0 <= u < v < N
            Floyd[u][v] = min(Floyd[u][v], w)
            Floyd[v][u] = min(Floyd[v][u], w)
        assert len(edges) == len(set((u, v) for u, v, _ in edges)), "edges should be unique"

        # Floyd-Warshall algorithm
        for k in range(N):
            for i in range(N):
                dik = Floyd[i][k]
                if dik == INF:
                    continue
                for j in range(N):
                    val = dik + Floyd[k][j]
                    if val < Floyd[i][j]:
                        Floyd[i][j] = val

        # Compute optimal solution (gold) via DFS over combinations
        gold_answer = INF
        reference_solution: List[int] = []
        solution: List[int] = []
        solution_dist: List[int] = [INF] * N

        def dfs(u: int) -> None:
            nonlocal solution, solution_dist, gold_answer, reference_solution
            if len(solution) + (N - u) < K:
                return
            if u == N:
                assert len(solution) == K, "solution should have exactly K elements"
                current_answer = max(solution_dist)
                if current_answer < gold_answer:
                    gold_answer = current_answer
                    reference_solution = solution.copy()
                return
            # Skip u
            dfs(u + 1)
            # Include u
            if len(solution) < K:
                solution.append(u)
                cache_solution_dist = solution_dist.copy()
                for v in range(N):
                    if Floyd[u][v] < solution_dist[v]:
                        solution_dist[v] = Floyd[u][v]
                dfs(u + 1)
                solution_dist = cache_solution_dist
                solution.pop()

        dfs(0)
        assert gold_answer > 0, "gold_answer should be positive for K <= N-1 in a connected graph"
        reference_answer_str = " ".join(map(str, reference_solution))

        # Store state
        self.N = N
        self.K = K
        self.edges = edges
        self.Floyd = Floyd
        self.gold_answer = gold_answer
        self.reference_answer_list = reference_solution
        self.reference_answer_str = reference_answer_str

        # Build problem statement
        edges_str = "\n".join(f"({u}, {v}, {w})" for u, v, w in edges)
        self.current_problem = (
            f"You are given an undirected connected graph with {N} vertices, labeled from 0 to {N - 1}.\n\n"
            f"The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w), "
            f"meaning an undirected edge connecting vertex u to vertex v with weight w:\n"
            f"{edges_str}\n\n"
            f"Please select a set of {K} distinct vertices. Try your best to minimize the largest distance of any vertex "
            f"in the graph to its closest vertex in the selected set; the distance between two vertices u and v is defined "
            f"as the sum of the weights of the edges in the shortest path connecting them.\n\n"
            f"Output Format: Your final answer should be K integers in \\boxed{{v1 v2 ... v{K}}}, separated by spaces."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure environment is initialized
        if self.N is None or self.K is None or self.Floyd is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Parse integers
        try:
            parts = boxed_content.strip().split()
            selected_vertices = list(map(int, parts))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer_format"}

        # Validate selection
        info: dict[str, Any] = {
            "N": self.N,
            "K": self.K,
            "selected_vertices": selected_vertices,
            "reference_answer": self.reference_answer_str,
            "gold_answer": self.gold_answer,
        }

        if len(selected_vertices) != len(set(selected_vertices)):
            info["error"] = "duplicate_vertices"
            return TERMINAL_STATE, 0.0, True, False, info
        if len(selected_vertices) != self.K:
            info["error"] = "wrong_number_of_vertices"
            return TERMINAL_STATE, 0.0, True, False, info
        if not all(isinstance(u, int) and 0 <= u < self.N for u in selected_vertices):
            info["error"] = "vertex_out_of_range"
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute max distance to nearest selected vertex
        answer_value = 0
        for u in range(self.N):
            dist = self.Floyd[u][selected_vertices[0]]
            for sv in selected_vertices[1:]:
                if self.Floyd[u][sv] < dist:
                    dist = self.Floyd[u][sv]
            if dist > answer_value:
                answer_value = dist

        info["user_answer_value"] = answer_value
        is_optimal = (answer_value == self.gold_answer)
        reward = 1.0 if is_optimal else 0.0
        info["correct"] = is_optimal

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action in \\boxed{...} format."""
        if self.N is None or self.K is None:
            # Fallback to a random size if reset not called
            N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
            K = random.randint(1, N - 1)
            vertices = random.sample(range(N), K)
        else:
            vertices = random.sample(range(self.N), self.K)
        return "\\boxed{" + " ".join(map(str, vertices)) + "}"