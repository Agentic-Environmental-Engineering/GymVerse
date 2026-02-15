from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumHarmoniousChromaticNumberEnv(Env):
    """Environment for the Minimum Harmonious Chromatic Number problem - single-turn Q&A."""

    def __init__(
        self,
        N: int = 8,
        edge_density: float = 0.3,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 5.0,
        **kwargs
    ):
        """
        Initialize the MinimumHarmoniousChromaticNumberEnv instance.

        Parameters:
        - N: number of vertices in the graph (must be >= 2).
        - edge_density: density of edges (between 0.0 and 1.0).
        - The remaining parameters are preserved from the original environment for compatibility,
          but this GEM environment uses a fixed reward scheme:
          correct answer: 1.0, wrong answer: 0.0, format error: -0.1.
        """
        super().__init__()
        self.N: int = N
        self.edge_density: float = edge_density

        # Preserve original reward configuration fields for compatibility (not used in GEM reward calculation)
        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # Runtime state
        self.current_problem: Optional[str] = None
        self.edges: List[Tuple[int, int]] = []
        self.reference_assignment_list: List[int] = []
        self.reference_assignment_str: str = ""
        self.gold_answer: int = 0

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a graph coloring problem: Minimum Harmonious Chromatic Number.\n"
            "Provide your final answer inside \\boxed{...}.\n"
            "Inside the box, output the color of each vertex in order: c[0] c[1] ... c[N-1], separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Parameter validation
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 2, "N should be greater than or equal to 2"
        assert isinstance(self.edge_density, float) or isinstance(self.edge_density, int), "edge_density must be a float"
        assert 0.0 <= float(self.edge_density) <= 1.0, "edge_density should be between 0.0 and 1.0"

        N = self.N
        edge_density = float(self.edge_density)

        # Generate edges based on edge_density
        all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        m = int(edge_density * N * (N - 1) / 2)
        m = max(0, min(m, len(all_pairs)))
        edges = random.sample(all_pairs, m)
        random.shuffle(edges)

        # Validate edges
        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"

        self.edges = edges

        # Compute a reference minimal harmonious coloring via DFS (as in the original environment)
        # Default best solution is distinct colors for every vertex
        best_assignment: List[int] = list(range(N))
        best_color_count: int = N

        # Build adjacency info for pruning and constraints
        adjacent_bitmask: List[int] = [0] * N
        smaller_adjacents: List[List[int]] = [[] for _ in range(N)]
        for u, v in edges:
            adjacent_bitmask[u] |= 1 << v
            adjacent_bitmask[v] |= 1 << u
            # Only check adjacency pairs where neighbor index is smaller than current index
            smaller_adjacents[max(u, v)].append(min(u, v))

        colors: List[Optional[int]] = [None] * N
        color2set: List[int] = [0] * N  # bitset of vertices for each color
        color_adjacent: List[List[bool]] = [[False] * N for _ in range(N)]  # color pair adjacency occurrence

        def dfs(u: int, max_color: int) -> None:
            nonlocal best_assignment, best_color_count, colors, color2set, color_adjacent
            # Prune if we cannot beat current best
            if max_color + 1 >= best_color_count:
                return
            if u == N:
                # Found a better solution
                best_assignment = [int(c) for c in colors]  # type: ignore
                best_color_count = max_color + 1
                return
            # Try colors from 0..(max_color+1)
            for color in range((max_color + 1) + 1):
                # Adjacent vertices must have different colors
                if (color2set[color] & adjacent_bitmask[u]) == 0:
                    colors[u] = color

                    # Copy color_adjacent matrix to safely test harmonious constraint
                    new_color_adjacent = [row.copy() for row in color_adjacent]

                    invalid = False
                    for v in smaller_adjacents[u]:
                        cu, cv = colors[u], colors[v]  # type: ignore
                        assert cu != cv, "Adjacent vertices should have different colors"
                        color_u, color_v = (cu, cv) if cu < cv else (cv, cu)
                        if new_color_adjacent[color_u][color_v]:
                            invalid = True
                            break
                        new_color_adjacent[color_u][color_v] = True

                    if not invalid:
                        color2set[color] |= (1 << u)
                        old_color_adjacent = color_adjacent
                        color_adjacent = new_color_adjacent
                        dfs(u + 1, max(max_color, color))
                        color_adjacent = old_color_adjacent
                        color2set[color] &= ~(1 << u)

        dfs(0, -1)

        self.reference_assignment_list = best_assignment
        self.reference_assignment_str = " ".join(map(str, best_assignment))
        self.gold_answer = best_color_count

        # Build the problem prompt
        edges_str = "\n".join(f"({u}, {v})" for u, v in self.edges)
        self.current_problem = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}.\n"
            f"The graph contains the following undirected edges:\n{edges_str}\n\n"
            "Your task is to assign a non-negative integer color to each vertex, represented as "
            f"c[0], c[1], ..., c[{N - 1}], such that:\n"
            "- For every edge (u, v) in the graph, c[u] != c[v] — adjacent vertices must have different colors.\n"
            "- For every pair of two distinct used colors x and y, there exists at most one edge (u, v) such that "
            "c[u] = x and c[v] = y (harmonious coloring).\n"
            "- The total number of distinct colors used is minimized.\n\n"
            "Output Format:\n"
            "Your final answer should be the colors in order, separated by spaces, wrapped in \\boxed{...}.\n"
            "Example: \\boxed{0 1 0 2}\n"
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer."""
        content = self._parse_answer(action)
        if content is None:
            # Format error: no \\boxed{...} detected
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from the boxed content
        try:
            tokens = content.strip().split()
            colors = list(map(int, tokens))
        except Exception:
            # Inside the box is not a valid space-separated list of integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.N
        # Validate length
        if len(colors) != N:
            info = {
                "error": "invalid_solution_length",
                "expected_length": N,
                "provided_length": len(colors),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate harmonious coloring constraints
        adjacent_color_pairs = set()
        for u, v in self.edges:
            if colors[u] == colors[v]:
                return TERMINAL_STATE, 0.0, True, False, {"error": "adjacent_same_color"}

            color_u, color_v = (colors[u], colors[v])
            if color_u > color_v:
                color_u, color_v = color_v, color_u
            pair = (color_u, color_v)
            if pair in adjacent_color_pairs:
                return TERMINAL_STATE, 0.0, True, False, {"error": "non_harmonious_pair_reused"}
            adjacent_color_pairs.add(pair)

        used_colors = len(set(colors))
        is_optimal = (used_colors == self.gold_answer)
        reward = 1.0 if is_optimal else 0.0

        info = {
            "correct": is_optimal,
            "valid_coloring": True,
            "used_colors": used_colors,
            "min_colors": self.gold_answer,
            "reference_min_coloring": self.reference_assignment_str,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random coloring wrapped in \\boxed{...}."""
        colors = [str(random.randint(0, max(0, self.N - 1))) for _ in range(self.N)]
        return f"\\boxed{{{' '.join(colors)}}}"