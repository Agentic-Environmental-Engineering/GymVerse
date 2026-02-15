import heapq
import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class HamiltonianPathExistenceEnv(Env):
    """
    GEM environment for finding a Hamiltonian path in a directed graph.
    Single-turn Q&A: the agent must output a permutation of all vertices that forms
    a valid directed path using only the provided edges.
    """

    def __init__(
        self,
        N: int = 5,
        edge_density: float = 0.3,
        # Preserved but unused reward configuration parameters from the original RLVE environment
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(existing/all)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 5.0,
        **kwargs,
    ):
        super().__init__()
        self.N = N
        self.edge_density = edge_density

        # Preserved parameters (not used for reward calculation in GEM)
        self.wrong_format = wrong_format
        self.invalid_solution = invalid_solution
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # Episode state
        self.graph_edges: List[Tuple[int, int]] = []
        self.reference_solution: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a directed graph and must find a Hamiltonian path.\n"
            "A Hamiltonian path visits every vertex exactly once, following directed edges.\n"
            "Format your final answer as a single line path inside \\boxed{...}, with vertices separated by spaces.\n"
            "Example: \\boxed{0 2 1}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new graph instance."""
        super().reset(seed)

        # Validate parameters (preserving original validation logic)
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 3, "N should be greater than or equal to 3"
        assert 0.0 <= self.edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        N = self.N
        edge_density = self.edge_density

        # Construct a guaranteed Hamiltonian path by shuffling all vertices
        constructed_path = list(range(N))
        random.shuffle(constructed_path)
        self.reference_solution = " ".join(map(str, constructed_path))

        edges: List[Tuple[int, int]] = []
        for s, t in zip(constructed_path, constructed_path[1:]):
            edges.append((s, t))

        # Add additional edges to reach the target density if needed
        num_edges_target = int(edge_density * N * (N - 1))
        if len(edges) < num_edges_target:
            all_possible = {(s, t) for s in range(N) for t in range(N) if s != t}
            existing = set(edges)
            remaining = list(all_possible - existing)
            to_add = min(len(remaining), num_edges_target - len(edges))
            if to_add > 0:
                edges.extend(random.sample(remaining, to_add))

        random.shuffle(edges)

        # Sanity checks (preserving original assertions)
        assert len(edges) == len(set(edges)), "edges should be unique"
        for s, t in edges:
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"

        self.graph_edges = edges

        # Build the problem statement
        edges_str = "\n".join(f"({s}, {t})" for s, t in edges)
        problem = (
            f"You are given a directed graph with {N} vertices, labeled from 0 to {N - 1}.\n"
            f"The graph contains the following directed edges. Each edge is a tuple (s, t), "
            f"meaning there is a directed edge from vertex s to vertex t:\n"
            f"{edges_str}\n\n"
            f"Please find a path p_1, p_2, ..., p_{N} such that the path visits every vertex exactly once "
            f"(revisiting vertices is NOT allowed).\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single line containing the path in order, separated by spaces, "
            f"and wrapped in \\boxed{{...}}.\n"
            f"Example: \\boxed{{0 2 1}}"
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted path."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the path as integers separated by spaces
        try:
            tokens = boxed_content.strip().split()
            path = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate path structure
        N = self.N
        edges_set = set(self.graph_edges)

        # Check length and permutation of vertices
        is_permutation = (len(path) == N and set(path) == set(range(N)))

        # Count how many consecutive edges exist in the provided graph
        existing = 0
        if is_permutation:
            for s, t in zip(path, path[1:]):
                if (s, t) in edges_set:
                    existing += 1

        # Determine correctness: all consecutive edges must exist and it must be a permutation
        is_correct = bool(is_permutation and existing == (N - 1))

        reward: float = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "user_path": path,
            "existing_edges_in_path": existing,
            "N": N,
            "edges": list(self.graph_edges),
            "reference_solution": self.reference_solution,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random permutation of vertices wrapped in \\boxed{...}."""
        perm = list(range(self.N))
        random.shuffle(perm)
        ans = " ".join(map(str, perm))
        return f"\\boxed{{{ans}}}"