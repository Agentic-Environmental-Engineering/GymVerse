import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumChromaticNumberEnv(Env):
    """Minimum Chromatic Number environment - single-turn Q&A.

    The task is to color vertices of an undirected graph using non-negative integers
    such that adjacent vertices have different colors and the total number of distinct colors
    is minimized. The answer must be provided in \\boxed{...} format containing space-separated
    colors for vertices 0..N-1.
    """

    def __init__(
        self,
        # Problem generation parameters
        N: Optional[int] = None,
        min_n: int = 2,
        max_n: int = 10,
        edge_density: Optional[float] = None,
        edge_density_min: float = 0.0,
        edge_density_max: float = 1.0,
        # Legacy reward-related parameters (preserved for compatibility, not used)
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 5.0,
        **kwargs
    ):
        super().__init__()
        # Store problem generation configuration
        self.fixed_N = N
        self.min_n = min_n
        self.max_n = max_n
        self.fixed_edge_density = edge_density
        self.edge_density_min = edge_density_min
        self.edge_density_max = edge_density_max

        # Preserve legacy parameters (not used in GEM reward scheme)
        self.legacy_rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # Internal state
        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int]] = []
        self.adjacent_bits: List[int] = []
        self.reference_colors: Optional[List[int]] = None
        self.reference_answer_string: Optional[str] = None
        self.gold_chromatic_number: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an undirected graph coloring task.\n"
            "Assign a non-negative integer color to each vertex so that adjacent vertices have different colors,\n"
            "and minimize the total number of distinct colors used.\n"
            "Output Format: Provide your colors for vertices 0..N-1 as space-separated integers inside \\boxed{...}.\n"
            "Example: \\boxed{0 1 0 2}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            if self.fixed_N < 2:
                raise ValueError("N should be greater than or equal to 2")
            self.N = self.fixed_N
        else:
            if self.min_n < 2:
                raise ValueError("min_n should be greater than or equal to 2")
            if self.min_n > self.max_n:
                raise ValueError("min_n should be less than or equal to max_n")
            self.N = random.randint(self.min_n, self.max_n)

        assert self.N is not None
        N = self.N

        # Determine edge density
        if self.fixed_edge_density is not None:
            if not (0.0 <= self.fixed_edge_density <= 1.0):
                raise ValueError("edge_density should be between 0.0 and 1.0")
            edge_density = self.fixed_edge_density
        else:
            if not (0.0 <= self.edge_density_min <= 1.0) or not (0.0 <= self.edge_density_max <= 1.0):
                raise ValueError("edge_density_min and edge_density_max should be within [0.0, 1.0]")
            if self.edge_density_min > self.edge_density_max:
                raise ValueError("edge_density_min should be less than or equal to edge_density_max")
            edge_density = random.uniform(self.edge_density_min, self.edge_density_max)

        # Generate edges
        all_possible_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        num_edges = int(edge_density * N * (N - 1) / 2)
        num_edges = max(0, min(num_edges, len(all_possible_edges)))
        edges = random.sample(all_possible_edges, num_edges)
        random.shuffle(edges)

        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"

        self.edges = edges

        # Build adjacency bitsets
        adjacent = [0] * N
        for u, v in edges:
            adjacent[u] |= 1 << v
            adjacent[v] |= 1 << u
        self.adjacent_bits = adjacent

        # Compute minimal chromatic number and a corresponding coloring
        colors: List[Optional[int]] = [None] * N
        color2set: List[int] = [0] * N
        best_colors: List[int] = list(range(N))  # default reference
        best_num_colors: int = N  # upper bound

        def dfs(u: int, max_color: int) -> None:
            nonlocal best_colors, best_num_colors
            # Prune if already worse or equal to current best
            if max_color + 1 >= best_num_colors:
                return
            # Completed coloring
            if u == N:
                best_colors = [int(c) for c in colors]  # type: ignore
                best_num_colors = max_color + 1
                return
            # Try available colors including introducing a new one
            for color in range(max_color + 2):
                if (color2set[color] & adjacent[u]) == 0:
                    colors[u] = color
                    color2set[color] |= 1 << u
                    dfs(u + 1, max(max_color, color))
                    color2set[color] &= ~(1 << u)

        dfs(0, -1)

        self.reference_colors = best_colors
        self.gold_chromatic_number = best_num_colors
        self.reference_answer_string = " ".join(map(str, best_colors))

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v})" for u, v in self.edges)
        problem_prompt = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}.\n"
            f"The graph contains the following undirected edges:\n{edges_str}\n\n"
            "Your task is to assign a non-negative integer color to each vertex, represented as "
            f"c[0], c[1], ..., c[{N - 1}], such that:\n"
            "- For every edge (u, v) in the graph, c[u] != c[v].\n"
            "- The total number of distinct colors used is minimized.\n\n"
            "Output Format: Your final answer should be a single line containing the color of each vertex in order, "
            "separated by spaces, inside \\boxed{...}.\n"
            "Example: \\boxed{0 1 0 2}"
        )

        self.current_problem = problem_prompt
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the submitted answer."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse colors
        try:
            parts = boxed_content.strip().split()
            user_colors = [int(x) for x in parts]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate solution
        N = self.N if self.N is not None else 0
        if len(user_colors) != N:
            info = {"error": "length_mismatch", "expected_length": N, "received_length": len(user_colors)}
            return TERMINAL_STATE, 0.0, True, False, info

        # Non-negative colors check
        if any(c < 0 for c in user_colors):
            return TERMINAL_STATE, 0.0, True, False, {"error": "negative_color"}

        # Adjacent vertices must have different colors
        for u, v in self.edges:
            if user_colors[u] == user_colors[v]:
                return TERMINAL_STATE, 0.0, True, False, {"error": "adjacent_conflict", "edge": (u, v)}

        # Chromatic number check
        user_distinct = len(set(user_colors))
        gold = self.gold_chromatic_number if self.gold_chromatic_number is not None else user_distinct
        is_correct = (user_distinct == gold)

        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "gold_chromatic_number": gold,
            "user_distinct_colors": user_distinct,
            "reference_answer": self.reference_answer_string,
            "N": N,
            "edges": self.edges,
            "user_colors": user_colors,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: random colors (not guaranteed to be valid)."""
        if self.N is None:
            return r"\boxed{}"
        random_colors = [str(random.randint(0, self.N - 1)) for _ in range(self.N)]
        return f"\\boxed{{{' '.join(random_colors)}}}"