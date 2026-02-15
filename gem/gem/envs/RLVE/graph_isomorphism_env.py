import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GraphIsomorphismEnv(Env):
    """Graph Isomorphism environment - single-turn Q&A.

    The task is to find a permutation p mapping vertices of G1 to vertices of G2
    such that edges are preserved exactly (i.e., the graphs are isomorphic under p).
    The answer must be provided inside \\boxed{...} in the format: p(0) p(1) ... p(N-1).
    """

    def __init__(
        self,
        N: int = 6,
        edge_density: float = 0.5,
        **kwargs: Any,
    ):
        """Initialize the environment with graph size and edge density.

        Args:
            N: Number of vertices in each graph (must be >= 3).
            edge_density: Density of edges for the undirected graphs (0.0 < density <= 1.0),
                the number of edges will be int(edge_density * N * (N - 1) / 2) and must be > 0.
            **kwargs: Extra keyword arguments (ignored).

        Raises:
            AssertionError: If parameter validation fails.
        """
        super().__init__()
        # Parameter validation as in original environment
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"
        assert 0.0 < edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"
        assert int(edge_density * N * (N - 1) / 2) > 0, "edge_density too small for N; must yield at least one edge"

        self.N: int = N
        self.edge_density: float = edge_density

        # Runtime state
        self.G1_edges: List[Tuple[int, int]] = []
        self.G2_edges: List[Tuple[int, int]] = []
        self.mapping: List[int] = []
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a Graph Isomorphism problem.\n"
            "Please provide your final permutation as space-separated integers inside \\boxed{...}.\n"
            "Example: \\boxed{3 0 2 1}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new graph isomorphism instance.

        Args:
            seed: Optional random seed.

        Returns:
            observation: The problem description string (instructions + instance).
            info: Additional info dict (empty for this environment).
        """
        super().reset(seed)

        N = self.N
        # Number of undirected edges to sample
        M = int(self.edge_density * N * (N - 1) / 2)
        assert M > 0

        # Generate G1 edges by sampling from all undirected pairs (u < v)
        all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        G1_edges: List[Tuple[int, int]] = random.sample(all_pairs, M)
        random.shuffle(G1_edges)

        # Generate a random permutation mapping
        mapping = list(range(N))
        random.shuffle(mapping)

        # Apply mapping to obtain G2 edges; normalize each edge as (min, max)
        G2_edges: List[Tuple[int, int]] = []
        for u, v in G1_edges:
            mu, mv = mapping[u], mapping[v]
            if mu > mv:
                mu, mv = mv, mu
            G2_edges.append((mu, mv))
        random.shuffle(G2_edges)

        # Sanity checks: edges are valid and unique
        for edges in (G1_edges, G2_edges):
            for u, v in edges:
                assert 0 <= u < v < N
            assert len(edges) == len(set(edges)), "Edges should be unique"

        # Store state
        self.G1_edges = G1_edges
        self.G2_edges = G2_edges
        self.mapping = mapping
        self.reference_answer = " ".join(map(str, mapping))

        # Build problem prompt
        G1_edges_str = "\n".join(f"({u}, {v})" for u, v in G1_edges)
        G2_edges_str = "\n".join(f"({u}, {v})" for u, v in G2_edges)
        reversed_perm = " ".join(map(str, range(N - 1, -1, -1)))

        problem = (
            f"You are given two undirected graphs, G1 and G2, each with {N} vertices labeled from 0 to {N - 1}. "
            f"Both graphs contain exactly {len(G1_edges)} undirected edges.\n\n"
            f"- Graph G1 has the following (undirected) edge set E1:\n{G1_edges_str}\n\n"
            f"- Graph G2 has the following (undirected) edge set E2:\n{G2_edges_str}\n\n"
            "Your task is to find a bijection (i.e., a permutation) p from the vertices of G1 to the vertices of G2 "
            "such that: For every edge (u, v) in E1, the edge (p(u), p(v)) exists in E2, and vice versa.\n\n"
            f"Output Format: Your final answer should be a single line containing the permutation p(0), p(1), ..., p({N - 1}), "
            f"separated by spaces, and placed inside \\boxed{{...}}. Example: \\boxed{{{reversed_perm}}}."
        )

        self.current_problem = problem
        obs = self._get_instructions() + problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the user's answer.

        Args:
            action: The agent's response text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE since this is single-turn.
            reward: 1.0 for correct, 0.0 for wrong/invalid, -0.1 for format error.
            terminated: True (single-turn).
            truncated: False.
            info: Additional information about correctness and answers.
        """
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse permutation from the boxed content
        user_perm: List[int] = []
        try:
            tokens = boxed.strip().split()
            user_perm = [int(tok) for tok in tokens]
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer_format"}

        N = self.N
        # Validate permutation structure
        if len(user_perm) != N:
            info = {
                "error": "invalid_permutation_length",
                "expected_length": N,
                "got_length": len(user_perm),
                "reference_answer": self.reference_answer,
                "user_answer": boxed,
            }
            return TERMINAL_STATE, 0.0, True, False, info
        if len(set(user_perm)) != N or not all(0 <= i < N for i in user_perm):
            info = {
                "error": "invalid_permutation_values",
                "reference_answer": self.reference_answer,
                "user_answer": boxed,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check isomorphism: map G1 edges via the permutation and compare to G2 edges set
        new_G2_edges = set()
        for u, v in self.G1_edges:
            mu, mv = user_perm[u], user_perm[v]
            if mu > mv:
                mu, mv = mv, mu
            new_G2_edges.add((mu, mv))

        target_G2_edges = set(self.G2_edges)
        is_correct = (new_G2_edges == target_G2_edges)
        reward = 1.0 if is_correct else 0.0

        # Additional diagnostics
        overlap = len(new_G2_edges & target_G2_edges)
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": boxed,
            "overlap": overlap,
            "total_edges": len(self.G2_edges),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random permutation action wrapped in \\boxed{...}."""
        perm = list(range(self.N))
        random.shuffle(perm)
        return f"\\boxed{{{' '.join(map(str, perm))}}}"