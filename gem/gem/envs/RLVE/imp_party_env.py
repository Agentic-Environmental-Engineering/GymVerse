from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ImpPartyEnv(Env):
    """Imperfect Party problem environment (single-turn Q&A).

    Task:
      - You are given an undirected graph with 3 × N vertices labeled 0..(3N-1).
      - It is guaranteed that the graph contains a clique of size 2 × N.
      - Your goal is to output any clique of size N (indices separated by spaces).

    Answer format:
      - Your final answer must be enclosed in \\boxed{...}, e.g., \\boxed{0 1 2}.
    """

    def __init__(self, N: int, **kwargs):
        """Initialize the environment.

        Args:
            N: Problem parameter. Must satisfy N >= 3.
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N

        # Stateful variables populated in reset()
        self.current_problem: Optional[str] = None
        self.edges: List[Tuple[int, int]] = []
        self._constructed_clique: List[int] = []  # for generation only

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an undirected graph with 3 × N vertices labeled from 0 to 3N-1.\n"
            "It is guaranteed that the graph contains a clique of size 2 × N.\n"
            "Your task is to output any clique of size N:\n"
            "- Provide exactly N distinct vertex indices in the range [0, 3N-1].\n"
            "- The chosen vertices must form a clique (every pair is connected by an edge).\n"
            "Output Format: Put the N indices separated by single spaces inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        three_n = 3 * N

        # Construct a hidden clique of size 2N
        constructed_clique = random.sample(range(three_n), 2 * N)
        edges: List[Tuple[int, int]] = []

        # Add all edges inside the constructed clique
        for i, u in enumerate(constructed_clique):
            for v in constructed_clique[i + 1 :]:
                uu, vv = (u, v) if u < v else (v, u)
                edges.append((uu, vv))

        # Add some random edges between the clique and the remaining vertices
        not_in_constructed = list(set(range(three_n)) - set(constructed_clique))
        cross_candidates = [
            (min(u, v), max(u, v)) for u in constructed_clique for v in not_in_constructed
        ]
        k = random.randint(0, len(cross_candidates))
        edges += random.sample(cross_candidates, k)

        random.shuffle(edges)

        # Validations as in the original environment
        for u, v in edges:
            assert 0 <= u < v < three_n, "edges should be within the range of 0 to 3N-1 with u < v"
        assert len(edges) == len(set(edges)), "edges should be unique"

        # Save to state
        self.edges = edges
        self._constructed_clique = constructed_clique

        # Build the problem prompt
        edges_str = "\n".join(f"({u}, {v})" for (u, v) in self.edges)
        self.current_problem = (
            f"You are given an undirected graph with 3 × {N} vertices, labeled from 0 to {three_n - 1}.\n"
            f"The graph contains the following undirected edges:\n{edges_str}\n\n"
            f"It is guaranteed that the graph contains a clique of size 2 × {N}.\n"
            f"Your task is to find any clique of size {N} in the graph.\n"
            f"Output the indices of the selected {N} vertices, separated by spaces, inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the outcome."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: missing or malformed \\boxed{...}
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to parse a list of integers separated by spaces
        try:
            tokens = boxed_content.strip().split()
            answer_vertices = list(map(int, tokens))
        except Exception:
            # Content of box is not a list of integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.N
        three_n = 3 * N
        info: dict[str, Any] = {}

        # Validate basic constraints
        if len(answer_vertices) != N:
            info["error"] = "invalid_solution_wrong_length"
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(isinstance(x, int) and 0 <= x < three_n for x in answer_vertices):
            info["error"] = "invalid_solution_out_of_range"
            return TERMINAL_STATE, 0.0, True, False, info

        if len(set(answer_vertices)) != len(answer_vertices):
            info["error"] = "invalid_solution_duplicates"
            return TERMINAL_STATE, 0.0, True, False, info

        # Check clique property
        edges_set = set(self.edges)
        satisfied = 0
        for i, u in enumerate(answer_vertices):
            for v in answer_vertices[i + 1 :]:
                uu, vv = (u, v) if u < v else (v, u)
                if (uu, vv) in edges_set:
                    satisfied += 1

        total_pairs = N * (N - 1) // 2
        is_clique = (satisfied == total_pairs)

        reward: float = 1.0 if is_clique else 0.0
        info.update(
            {
                "correct": is_clique,
                "satisfied_edges": satisfied,
                "total_pairs": total_pairs,
                "user_answer": answer_vertices,
                "N": N,
                "num_edges": len(self.edges),
            }
        )

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract content inside \\boxed{...}."""
        import re

        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: pick N distinct vertices uniformly at random."""
        vertices = random.sample(range(3 * self.N), self.N)
        content = " ".join(map(str, vertices))
        return f"\\boxed{{{content}}}"