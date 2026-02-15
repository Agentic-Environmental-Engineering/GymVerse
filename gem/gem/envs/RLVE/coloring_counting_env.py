import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ColoringCountingEnv(Env):
    """Environment for the graph coloring counting problem - single-turn Q&A."""

    def __init__(
        self,
        min_n: int = 2,
        max_n: int = 12,
        fixed_n: Optional[int] = None,
        edge_density: Optional[float] = None,
        min_edge_density: float = 0.0,
        max_edge_density: float = 1.0,
        **kwargs
    ):
        """
        Initialize the ColoringCountingEnv instance.

        Parameters:
        - min_n: minimum number of vertices (inclusive), must be >= 2
        - max_n: maximum number of vertices (inclusive), must be >= min_n
        - fixed_n: if provided, use this exact number of vertices
        - edge_density: if provided, use this exact edge density in [0.0, 1.0]
        - min_edge_density: minimum edge density for random sampling (inclusive)
        - max_edge_density: maximum edge density for random sampling (inclusive)
        """
        super().__init__()
        # Parameter validation
        if min_n < 2:
            raise ValueError("min_n should be greater than or equal to 2")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        if not (0.0 <= min_edge_density <= 1.0):
            raise ValueError("min_edge_density should be between 0.0 and 1.0")
        if not (0.0 <= max_edge_density <= 1.0):
            raise ValueError("max_edge_density should be between 0.0 and 1.0")
        if min_edge_density > max_edge_density:
            raise ValueError("min_edge_density should be less than or equal to max_edge_density")
        if fixed_n is not None and fixed_n < 2:
            raise ValueError("fixed_n should be greater than or equal to 2")
        if edge_density is not None and not (0.0 <= edge_density <= 1.0):
            raise ValueError("edge_density should be between 0.0 and 1.0")

        self.min_n = min_n
        self.max_n = max_n
        self.fixed_n = fixed_n
        self.edge_density = edge_density
        self.min_edge_density = min_edge_density
        self.max_edge_density = max_edge_density

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.edges: Optional[List[Tuple[int, int]]] = None
        self.R: Optional[Tuple[int, ...]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a graph coloring counting problem.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Set random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)

        # Determine parameters N and edge_density
        N = self.fixed_n if self.fixed_n is not None else random.randint(self.min_n, self.max_n)
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")
        self.N = N

        if self.edge_density is not None:
            edge_density = self.edge_density
        else:
            edge_density = random.uniform(self.min_edge_density, self.max_edge_density)
        if not (0.0 <= edge_density <= 1.0):
            raise ValueError("edge_density should be between 0.0 and 1.0")

        # Generate edges
        all_possible_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        num_edges = int(edge_density * N * (N - 1) / 2)
        edges = random.sample(all_possible_edges, num_edges)
        random.shuffle(edges)

        # Validate and compute degrees
        Deg = [0] * N
        for (u, v) in edges:
            if not (0 <= u < v < N):
                raise ValueError("Edge indices out of range or not in sorted order (u < v)")
            Deg[u] += 1
            Deg[v] += 1
        if len(edges) != len(set(edges)):
            raise ValueError("Edges should be unique")
        self.edges = edges

        # Generate R values
        R = tuple(random.randint(Deg[u], 2 * Deg[u]) for u in range(N))
        self.R = R

        # Prepare sorting by R to optimize DP
        nodes = list(enumerate(R))
        nodes.sort(key=lambda x: x[1])
        sorted_R = [r for _, r in nodes]
        orig_to_sorted = [0] * N
        for new_idx, (orig_idx, _) in enumerate(nodes):
            orig_to_sorted[orig_idx] = new_idx

        # Build adjacency matrix using sorted indices
        G = [[False] * N for _ in range(N)]
        for u, v in edges:
            su = orig_to_sorted[u]
            sv = orig_to_sorted[v]
            G[su][sv] = True
            G[sv][su] = True

        # Precompute Can[S]: whether subset S is an independent set
        total_S = 1 << N
        Can = [True] * total_S
        for S in range(total_S):
            for u in range(N):
                if not (S >> u) & 1:
                    continue
                for v in range(u + 1, N):
                    if (S >> v) & 1 and G[u][v]:
                        Can[S] = False
                        break
                if not Can[S]:
                    break

        # DP over subsets
        F = [[0] * (N + 1) for _ in range(total_S)]
        F[total_S - 1][0] = 1

        for S in range(total_S - 1, 0, -1):
            # Find minimum index present in S
            Min = None
            for i in range(N):
                if (S >> i) & 1:
                    Min = i
                    break
            if Min is None:
                continue
            max_k = min(sorted_R[Min], N - 1)
            for k in range(max_k + 1):
                ways = F[S][k]
                if ways == 0:
                    continue
                W = S & ~(1 << Min)
                T = W
                while True:
                    if Can[T | (1 << Min)]:
                        new_S = W & ~T
                        F[new_S][k + 1] += ways * (sorted_R[Min] + 1 - k)
                    if T == 0:
                        break
                    T = (T - 1) & W

        reference_answer = sum(F[0][k] * k for k in range(1, N + 1))
        if reference_answer <= 0:
            # This should not happen for valid instances per original logic
            raise ValueError("Reference answer should be positive")
        self.reference_answer = reference_answer

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        R_str = "\n".join(f"R[{u}]={Ru}" for u, Ru in enumerate(R))
        problem_text = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}. "
            f"The graph contains the following undirected edges:\n{edges_str}\n\n"
            f"You are also given an array R of length {N}, where R[u] denotes the maximum allowed color for vertex u:\n{R_str}\n\n"
            f"A coloring assigns an integer C[u] to each vertex u, satisfying the following conditions:\n"
            f"- 0 <= C[u] <= R[u] for all vertices u\n"
            f"- For every edge (u, v), C[u] ≠ C[v] (i.e., adjacent vertices must have different colors)\n\n"
            f"The value of a valid coloring is the number of distinct colors used (i.e., the count of unique values among C[0], C[1], ..., C[{N - 1}]). "
            f"Please compute the total value of all valid colorings.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        info = {}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the answer."""
        # Parse the boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate answer as integer
        try:
            user_answer = int(answer_text.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Compare with reference
        if self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_reference"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "edges": self.edges,
            "R": self.R,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        max_val = 1
        if self.reference_answer is not None:
            max_val = max(1, self.reference_answer * 2)
        random_answer = random.randint(0, max_val)
        return f"\\boxed{{{random_answer}}}"