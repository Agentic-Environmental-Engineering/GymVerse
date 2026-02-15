import random
from typing import Any, Optional, SupportsFloat, Tuple, Dict
import numpy as np
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ShortestPathCountConstructionEnv(Env):
    """Environment for constructing an undirected simple graph whose number of shortest paths
    between vertex 1 and vertex 2 equals a given K. Single-turn Q&A.
    """

    prompt_template = (
        "Please construct a simple undirected graph with N vertices, such that the number of "
        "shortest paths between vertex 1 and vertex 2 is {K}. Since there are multiple valid "
        "graphs satisfying the condition, you can output any of them.\n"
        "{N_constraint}\n\n"
        "Please strictly follow the output format without additional content:\n"
        "1. The first line must contain an integer N.\n"
        "2. The next N lines each contain a string of length N, representing the adjacency matrix G "
        "with N rows and N columns. Each element of the matrix must be 'N' or 'Y'. If Gij is 'Y', then "
        "graph G has an edge connecting vertex i and vertex j. Consider the graph vertices are numbered "
        "from 1 to N. The graph must be undirected and simple: Gii = 'N' and Gij = Gji must hold. And "
        "there must be at least one path between vertex 1 and vertex 2.\n\n"
        "Output Format: Return your ENTIRE output (the integer N followed by N lines of the adjacency matrix) "
        "wrapped inside a single \\boxed{{...}} block. For example:\n"
        "\\boxed{{\n"
        "3\n"
        "NYY\n"
        "YNY\n"
        "YYN\n"
        "}}"
    )

    def __init__(
        self,
        max_k: int = 1000,
        # Keep original reward-related parameters for configuration compatibility (not used for GEM scoring)
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 10.0,
        trivial_solution_penalty: float = -0.5,
        **kwargs: Any,
    ):
        """Initialize the environment.

        Args:
            max_k: Maximum K used to sample the target number of shortest paths (uniformly from [3, max_k]).
            wrong_format: Kept for compatibility with the original environment (not used in GEM reward).
            rewarding_strategy: Kept for compatibility (not used in GEM reward).
            rewarding_weight: Kept for compatibility (not used in GEM reward).
            rewarding_beta: Kept for compatibility (not used in GEM reward).
            trivial_solution_penalty: Kept for compatibility (not used in GEM reward).
        """
        super().__init__()
        if max_k < 3:
            raise ValueError("max_k should be greater than or equal to 3")
        self.max_k: int = max_k

        # Keep original reward configuration in case needed for analysis/debug
        self.rewards_config: Dict[str, Any] = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
            "trivial_solution_penalty": trivial_solution_penalty,
        }

        # State placeholders
        self.k: Optional[int] = None
        self.n_constraint: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are constructing an undirected simple graph to match a target number of shortest paths.\n"
            "Please provide your entire output inside a single \\boxed{...} block.\n"
            "Inside the box, the first line must be N, followed by N lines of an N×N adjacency matrix using only 'Y' and 'N'.\n"
            "The graph must be undirected (symmetric adjacency), simple (no self-loops), and there must be at least one path from vertex 1 to vertex 2.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Sample K in [3, max_k]
        self.k = random.randint(3, self.max_k)

        # Compute optional N constraint for larger K
        self.n_constraint = None
        if self.k >= 12:
            # Same logic as the original environment
            bits_len = len(bin(self.k)[2:])
            self.n_constraint = min(((bits_len * 3 + 1) + 1) * 2, self.k + 2)

        # Build problem description
        if self.k >= 12 and self.n_constraint is not None:
            n_constraint_text = f"Please ensure that the number of vertices N is fewer than {self.n_constraint}."
        else:
            n_constraint_text = (
                f"Please try your best to avoid constructing a trivial solution with N = {self.k} + 2 "
                f"(by just putting {self.k} intermediate vertices between vertex 1 and vertex 2)."
            )

        self.current_problem = self.prompt_template.format(
            K=self.k,
            N_constraint=n_constraint_text,
        )
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted graph.

        Returns:
            TERMINAL_STATE, reward, terminated=True, truncated=False, info
        """
        # Extract boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "missing_boxed_block"}

        # Parse N and adjacency matrix from boxed content
        parsed = self._parse_graph_answer(boxed)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "parse_failure"}

        n, adjacency_matrix = parsed

        # Basic sanity check (must have at least vertices 1 and 2)
        if n < 2:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "n_too_small"}

        # If K >= 12 and n_constraint is defined, enforce the constraint strictly
        if self.k is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_error", "detail": "k_not_set"}

        if self.k >= 12 and self.n_constraint is not None and n >= self.n_constraint:
            info = {
                "correct": False,
                "reason": "constraint_violation",
                "n": n,
                "n_constraint": self.n_constraint,
                "target_k": self.k,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute the number of shortest paths between vertex 1 and vertex 2
        real_k = int(self._count_shortest_paths(n, adjacency_matrix))

        is_correct = (real_k == self.k)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "target_k": self.k,
            "computed_k": real_k,
            "n": n,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract content inside \\boxed{...}."""
        import re

        # This regex captures the innermost boxed content; choose the last occurrence if multiple
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if not matches:
            return None
        # Return the last boxed content trimmed
        return matches[-1].strip()

    def _parse_graph_answer(self, content: str) -> Optional[Tuple[int, np.ndarray]]:
        """Parse N and adjacency matrix from the boxed content.

        Expected format (exactly N lines after the first line):
        N
        <row1>
        ...
        <rowN>

        Each row must be a string of length N with only 'Y' or 'N'.
        The matrix must be symmetric and have zero diagonal.
        """
        try:
            content = content.strip()
            first_newline = content.find("\n")
            if first_newline == -1:
                return None

            n_str = content[:first_newline].strip()
            n = int(n_str)
            rest = content[first_newline + 1 :]

            # Require exactly N lines in the remainder
            if sum(1 for c in rest if c == "\n") != n - 1:
                return None

            lines = rest.splitlines()
            if len(lines) != n:
                return None

            adjacency_matrix = np.ndarray((n, n), dtype=int)
            for i in range(n):
                row = lines[i].strip()
                if len(row) != n:
                    return None
                for j in range(n):
                    ch = row[j]
                    if ch not in ("N", "Y"):
                        return None
                    adjacency_matrix[i, j] = 1 if ch == "Y" else 0

            # Validate undirected simple graph: symmetric, zero diagonal
            if not np.all(adjacency_matrix == adjacency_matrix.T):
                return None
            if not np.all(np.diag(adjacency_matrix) == 0):
                return None

            return n, adjacency_matrix
        except Exception:
            return None

    def _count_shortest_paths(self, n: int, adjacency_matrix: np.ndarray) -> int:
        """Count the number of shortest paths between vertex 1 and vertex 2 using matrix-vector multiplication."""
        start_node_idx = 0
        end_node_idx = 1

        current_paths_vec = np.zeros(n, dtype=int)
        current_paths_vec[start_node_idx] = 1

        # Enumerate the shortest path length
        for _k in range(1, n):
            next_paths_vec = adjacency_matrix @ current_paths_vec

            # Check if there is a path to the end node at this length
            if next_paths_vec[end_node_idx] > 0:
                return int(next_paths_vec[end_node_idx])

            # Update the vector for the next iteration
            current_paths_vec = next_paths_vec

        # No path from start to end
        return 0

    def sample_random_action(self) -> str:
        """Sample a random (likely invalid) action formatted in \\boxed{...}."""
        # Generate a small random adjacency matrix to demonstrate formatting
        n = random.randint(2, 6)
        A = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                val = random.randint(0, 1)
                A[i][j] = val
                A[j][i] = val
            A[i][i] = 0

        lines = ["".join("Y" if A[i][j] == 1 else "N" for j in range(n)) for i in range(n)]
        content = str(n) + "\n" + "\n".join(lines)
        return f"\\boxed{{\n{content}\n}}"