from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaximumCliqueEnv(Env):
    """Maximum Clique problem environment - single-turn Q&A."""

    prompt_template = (
        "You are given an undirected graph with {N} vertices, labeled from 0 to {N_minus_1}.\n"
        "The graph contains the following undirected edges:\n"
        "{edges}\n\n"
        "Your task is to select a subset of vertices v1, v2, ..., vk such that:\n"
        "- 0 ≤ v1, v2, ..., vk < {N} and all selected vertices are distinct.\n"
        "- The selected vertices form a clique — that is, every pair of distinct selected vertices is connected by at least one edge.\n"
        "- Your goal is to maximize the number of selected vertices k.\n\n"
        "Output Format:\n"
        "Your final answer should be a single line containing the selected vertex indices v1 v2 ... vk, separated by spaces, wrapped in \\boxed{{...}}.\n"
        "Example: \\boxed{{0 2 3}} (do NOT include commas).\n"
    )

    def __init__(self, N: int = 8, edge_density: float = 0.5, **kwargs):
        super().__init__()
        self.N = N
        self.edge_density = edge_density

        # Problem state
        self.edges: List[Tuple[int, int]] = []
        self.adjacent_bits: List[int] = []
        self.reference_clique_vertices: List[int] = []
        self.gold_size: int = 0
        self.current_problem: Optional[str] = None

        # Validate parameters
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 2, "N should be greater than or equal to 2"
        assert 0.0 <= self.edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

    def _get_instructions(self) -> str:
        """Return general instructions for the task."""
        return (
            "You are solving a Maximum Clique problem on an undirected graph.\n"
            "Please provide your final answer in \\boxed{...} format, where the content is a list of vertex indices separated by spaces (no commas).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        edge_density = self.edge_density

        # Generate edges
        all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        m = int(edge_density * len(all_pairs))
        self.edges = random.sample(all_pairs, m)
        random.shuffle(self.edges)

        # Sanity checks
        for u, v in self.edges:
            assert 0 <= u < v < N
        assert len(self.edges) == len(set(self.edges)), "edges should be unique"

        # Build adjacency bitset
        self.adjacent_bits = [0] * N
        for u, v in self.edges:
            self.adjacent_bits[u] |= 1 << v
            self.adjacent_bits[v] |= 1 << u

        # Compute a maximum clique using DFS with pruning
        best: List[int] = []
        clique: List[int] = []

        def dfs(u: int, allowed_set: int) -> None:
            nonlocal best, clique
            # Upper bound pruning: even if we select all remaining vertices, cannot beat current best
            if len(clique) + (N - u) <= len(best):
                return
            if u == N:
                # At this point, len(clique) must be greater than len(best) due to the pruning condition
                best = clique.copy()
                return
            # Try including u if it is allowed
            if allowed_set & (1 << u):
                clique.append(u)
                dfs(u + 1, allowed_set & self.adjacent_bits[u])
                clique.pop()
            # Try excluding u
            dfs(u + 1, allowed_set)

        dfs(0, (1 << N) - 1)

        self.reference_clique_vertices = best
        self.gold_size = len(best)

        # Build problem prompt
        edges_text = "\n".join(f"({u}, {v})" for (u, v) in self.edges)
        self.current_problem = self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges=edges_text,
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer and return the result."""
        content = self._parse_answer(action)
        if content is None:
            # Format error: no \\boxed{...} found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse space-separated integers; allow empty content as empty clique
        content = content.strip()
        if content == "":
            user_vertices: List[int] = []
        else:
            parts = content.split()
            try:
                user_vertices = list(map(int, parts))
            except ValueError:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer_format"}

        # Validate solution
        N = self.N
        edges_set = set(self.edges)  # undirected edges stored with u < v

        # Distinctness
        if len(user_vertices) != len(set(user_vertices)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "duplicate_vertices", "valid": False}

        # Range check
        for v in user_vertices:
            if not (0 <= v < N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "out_of_range_vertex", "valid": False}

        # Clique check
        is_clique = True
        for i in range(len(user_vertices)):
            for j in range(i + 1, len(user_vertices)):
                u, v = user_vertices[i], user_vertices[j]
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in edges_set:
                    is_clique = False
                    break
            if not is_clique:
                break

        if not is_clique:
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_a_clique", "valid": False}

        # Check optimality
        user_size = len(user_vertices)
        is_correct = (user_size == self.gold_size)

        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "valid": True,
            "user_vertices": user_vertices,
            "user_size": user_size,
            "gold_size": self.gold_size,
            "reference_example_vertices": self.reference_clique_vertices,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns the last match if multiple exist."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random subset of vertices (not necessarily a clique)."""
        k = random.randint(0, self.N)
        subset = sorted(random.sample(range(self.N), k))
        content = " ".join(map(str, subset))
        return f"\\boxed{{{content}}}"