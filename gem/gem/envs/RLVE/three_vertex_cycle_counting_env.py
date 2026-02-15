from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ThreeVertexCycleCountingEnv(Env):
    """Single-turn environment for counting three-vertex cycles (triangles) in an undirected graph."""

    def __init__(
        self,
        N: int = 6,
        edge_ratio: float = 1.5,
        **kwargs,
    ):
        """
        Initialize the environment.

        Args:
            N: Number of vertices in the graph (must be >= 4).
            edge_ratio: Controls the number of edges sampled as int(edge_ratio * N), clipped to [1, N*(N-1)/2].
        """
        super().__init__()
        self.N: int = N
        self.edge_ratio: float = edge_ratio

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.edges: List[tuple[int, int]] = []

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a graph counting problem.\n"
            "Given an undirected graph, count the number of distinct three-vertex cycles (triangles).\n"
            "A triangle is a set of three vertices where every pair is connected by an edge.\n"
            "The order of vertices does not matter; triangles are distinct if their vertex sets differ.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Parameter validation
        assert isinstance(self.N, int) and self.N >= 4, "N should be an integer >= 4"
        assert isinstance(self.edge_ratio, (int, float)) and self.edge_ratio >= 0, "edge_ratio should be non-negative"

        # Generate edges
        all_edges = [(u, v) for u in range(self.N) for v in range(u + 1, self.N)]
        total_possible = len(all_edges)
        sample_size = max(1, min(total_possible, int(self.edge_ratio * self.N)))
        self.edges = random.sample(all_edges, sample_size)
        random.shuffle(self.edges)

        # Validate edges
        for u, v in self.edges:
            assert 0 <= u < v < self.N, "Edge vertices must satisfy 0 <= u < v < N"
        assert len(self.edges) == len(set(self.edges)), "Edges should be unique"

        # Compute reference answer (number of triangles)
        self.reference_answer = self._count_triangles(self.N, self.edges)

        # Build problem statement
        edges_str = "\n".join(f"({u}, {v})" for (u, v) in self.edges)
        self.current_problem = (
            f"You are given an undirected graph with {self.N} vertices, labeled from 0 to {self.N - 1}.\n"
            f"The graph contains the following undirected edges:\n{edges_str}\n\n"
            "Please count the number of distinct three-vertex cycles in the graph "
            "(the order of vertices in the cycle does not matter, and cycles are considered distinct "
            "if they have different sets of vertices).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by parsing and verifying the submitted answer."""
        # Parse answer from boxed format
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and score
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        is_correct = (self.reference_answer is not None) and (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "edges": self.edges,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _count_triangles(N: int, edges: List[tuple[int, int]]) -> int:
        """Count the number of triangles using degree-based oriented adjacency."""
        degree = [0] * N
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1

        # Build adjacency lists with edges directed from lower-degree to higher-degree endpoint
        adj: List[List[int]] = [[] for _ in range(N)]
        for u, v in edges:
            a, b = u, v
            if degree[a] > degree[b] or (degree[a] == degree[b] and a > b):
                a, b = b, a
            adj[a].append(b)

        # Count triangles
        vis = [False] * N
        ans = 0
        for i in range(N):
            # Mark neighbors of i
            for j in adj[i]:
                vis[j] = True
            # For each two-hop path i -> j -> k, check if k is also a neighbor of i
            for j in adj[i]:
                for k in adj[j]:
                    if vis[k]:
                        ans += 1
            # Unmark
            for j in adj[i]:
                vis[j] = False

        return ans

    def sample_random_action(self) -> str:
        """Sample a random action in boxed format."""
        # The maximum possible number of triangles is C(N, 3)
        max_triangles = self.N * (self.N - 1) * (self.N - 2) // 6
        random_answer = random.randint(0, max_triangles)
        return f"\\boxed{{{random_answer}}}"