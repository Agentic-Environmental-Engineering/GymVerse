import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CycleCountingEnv(Env):
    """Undirected simple cycle counting environment - single-turn Q&A.

    This environment generates a random undirected graph with N vertices and a given edge density.
    The task is to count the number of simple cycles (length >= 3) in the graph, where cycles are
    considered equivalent if they contain the same set of edges (ignoring direction and starting point).
    """

    def __init__(
        self,
        N_min: int = 3,
        N_max: int = 12,
        edge_density_min: float = 0.0,
        edge_density_max: float = 1.0,
        **kwargs
    ):
        """Initialize the environment with configurable difficulty parameters.

        Args:
            N_min: Minimum number of vertices N (inclusive). Must be >= 2.
            N_max: Maximum number of vertices N (inclusive). Must be >= N_min.
            edge_density_min: Minimum edge density in [0.0, 1.0].
            edge_density_max: Maximum edge density in [0.0, 1.0].
        """
        super().__init__()
        # Validate ranges
        if N_min < 2:
            raise ValueError("N_min should be greater than or equal to 2")
        if N_max < N_min:
            raise ValueError("N_max should be greater than or equal to N_min")
        if not (0.0 <= edge_density_min <= 1.0):
            raise ValueError("edge_density_min should be between 0.0 and 1.0")
        if not (0.0 <= edge_density_max <= 1.0):
            raise ValueError("edge_density_max should be between 0.0 and 1.0")
        if edge_density_max < edge_density_min:
            raise ValueError("edge_density_max should be greater than or equal to edge_density_min")

        self.N_min = N_min
        self.N_max = N_max
        self.edge_density_min = edge_density_min
        self.edge_density_max = edge_density_max

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.edges: Optional[List[tuple[int, int]]] = None

        # Prompt template
        self.prompt_template = (
            "You are given an undirected graph with {N} vertices, labeled from 0 to {N_minus_1}. "
            "The graph contains the following undirected edges:\n"
            "{edges}\n\n"
            "Please count the number of simple cycles in the graph. A simple cycle is a cycle with at least 3 vertices, "
            "with no repeated vertices or edges.\n"
            "Two cycles are considered equivalent if they consist of the same set of edges, regardless of the order or starting point; "
            "for example, the cycles (0, 1, 2, 3) and (1, 0, 3, 2) are identical, while (0, 1, 2, 3) and (1, 0, 2, 3) are NOT.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a graph cycle counting problem.\n"
            "Your task is to compute the number of simple cycles in the given undirected graph.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample parameters
        N = random.randint(self.N_min, self.N_max)
        edge_density = random.uniform(self.edge_density_min, self.edge_density_max)

        # Validate parameters (preserving original logic)
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")
        if not (0.0 <= edge_density <= 1.0):
            raise ValueError("edge_density should be between 0.0 and 1.0")

        # Generate undirected edges uniformly at random according to density
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        m_total = len(all_edges)
        k = int(edge_density * m_total)
        edges = random.sample(all_edges, k)
        random.shuffle(edges)

        # Compute reference answer using DP (Codeforces 11D style)
        reference_answer = self._count_simple_cycles(N, edges)

        # Build problem statement
        edges_str = "\n".join(f"({u}, {v})" for (u, v) in edges)
        self.current_problem = self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges=edges_str,
        )

        # Store internal state
        self.N = N
        self.edges = edges
        self.reference_answer = reference_answer

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided answer and return the result.

        Returns:
            observation: TERMINAL_STATE (single-turn)
            reward: 1.0 if correct, 0.0 if incorrect, -0.1 if format error
            terminated: True
            truncated: False
            info: Additional info including correctness and reference answer
        """
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(answer_text.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate against reference
        assert self.reference_answer is not None, "Reference answer is not set. Call reset() before step()."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "edges": self.edges,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the model's output."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random non-negative integer) in boxed format."""
        # Heuristic random guess: value between 0 and number of edges (not meaningful, but valid format)
        guess = random.randint(0, 100)
        return f"\\boxed{{{guess}}}"

    @staticmethod
    def _count_simple_cycles(N: int, edges: List[tuple[int, int]]) -> int:
        """Count the number of simple cycles using DP over subsets.

        This replicates the core logic from Codeforces 11D solution approach:
        dpF[S][end]: number of paths starting from the minimum-index vertex in S and ending at 'end',
        visiting exactly the set S. Cycles are counted when an edge closes back to the minimum-index vertex.
        """
        # Validate uniqueness and bounds
        assert len(edges) == len(set(edges)), "edges should be unique"
        adjacent = [[False] * N for _ in range(N)]
        for u, v in edges:
            assert 0 <= u < v < N, "Edge endpoints out of range or not ordered"
            adjacent[u][v] = adjacent[v][u] = True

        # DP initialization
        dpF = [[0] * N for _ in range(1 << N)]
        for end in range(N):
            dpF[1 << end][end] = 1

        answer = 0
        # Iterate over all non-empty subsets
        for S in range(1, 1 << N):
            # Find the lowest-index vertex in S
            lowindex = 0
            while (1 << lowindex) != (S & -S):
                lowindex += 1

            nowS = S
            # Iterate over possible end vertices in S
            while nowS:
                end = 0
                while (1 << end) != (nowS & -nowS):
                    end += 1
                nowS ^= (1 << end)

                if dpF[S][end] == 0:
                    continue

                # If there is an edge back to the lowest-index vertex and path length >= 2 (excluding start and end),
                # then a cycle is found.
                if adjacent[end][lowindex]:
                    if S - (1 << lowindex) - (1 << end) > 0:
                        answer += dpF[S][end]

                # Try to extend the path to a new vertex 'next' not in S
                nowR = ((1 << N) - 1) - S
                while nowR:
                    nxt = 0
                    while (1 << nxt) != (nowR & -nowR):
                        nxt += 1
                    nowR ^= (1 << nxt)

                    # Ensure nxt is not in S
                    if S & (1 << nxt):
                        raise AssertionError("next should not be in S")

                    # To avoid duplicates, only allow extending to vertices with index >= lowindex
                    if nxt < lowindex:
                        continue
                    if not adjacent[end][nxt]:
                        continue

                    dpF[S | (1 << nxt)][nxt] += dpF[S][end]

        # Each cycle is counted twice (once in each direction)
        return answer // 2