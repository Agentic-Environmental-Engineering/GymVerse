import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaximumAchromaticNumberEnv(Env):
    """Maximum Achromatic Number environment - single-turn Q&A.

    The task is to color the vertices of an undirected graph with non-negative integers such that:
    - Adjacent vertices have different colors (proper coloring).
    - For every pair of distinct used colors x and y, there exists at least one edge (u, v) such that c[u] = x and c[v] = y (complete coloring).
    - The number of distinct colors is maximized (achromatic number).

    The answer must be provided in \\boxed{...} format containing space-separated integers c[0] c[1] ... c[N-1].
    """

    def __init__(
        self,
        N: int = 8,
        edge_density: float = 0.5,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 5.0,
        **kwargs
    ):
        super().__init__()
        self.N = N
        self.edge_density = edge_density

        # These parameters are kept to preserve original defaults and compatibility,
        # but are not used in GEM scoring as per conversion requirements.
        self.rewards_config = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self.edges: List[Tuple[int, int]] = []
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a maximum achromatic number coloring problem on an undirected graph.\n"
            "Output Format: The final answer must be provided in \\boxed{...}, containing the colors\n"
            "c[0], c[1], ..., c[N-1] separated by spaces (non-negative integers).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Validate parameters
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 2, "N should be greater than or equal to 2"
        assert 0.0 <= self.edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        # Generate edges according to edge density
        all_edges = [(u, v) for u in range(self.N) for v in range(u + 1, self.N)]
        m = int(self.edge_density * self.N * (self.N - 1) / 2)
        self.edges = random.sample(all_edges, m)
        random.shuffle(self.edges)

        for u, v in self.edges:
            assert 0 <= u < v < self.N
        assert len(self.edges) == len(set(self.edges)), "edges should be unique"

        # Compute achromatic number and a reference coloring using DFS
        self.reference_answer, self.gold_answer = self._compute_reference_and_gold(self.N, self.edges)

        # Build problem text
        edges_text = "\n".join(f"({u}, {v})" for u, v in self.edges)
        self.current_problem = (
            f"You are given an undirected graph with {self.N} vertices, labeled from 0 to {self.N - 1}.\n"
            f"The graph contains the following undirected edges:\n{edges_text}\n\n"
            "Your task is to assign a non-negative integer color to each vertex, represented as c[0], c[1], ..., c[N-1], such that:\n"
            "- For every edge (u, v) in the graph, c[u] ≠ c[v] — adjacent vertices must have different colors.\n"
            "- For every pair of two distinct used colors x and y, there exists at least one edge (u, v) such that c[u] = x and c[v] = y (complete coloring).\n"
            "- The total number of distinct colors used is maximized.\n\n"
            "Output Format: Your final answer should be the color of each vertex in order c[0], c[1], ..., c[N-1], separated by spaces, "
            "placed inside \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.N,
            "edges": self.edges,
            "gold_answer": self.gold_answer,
            "reference_answer": self.reference_answer,
        }
        return obs, info

    def _compute_reference_and_gold(
        self, N: int, edges: List[Tuple[int, int]]
    ) -> Tuple[str, int]:
        """Compute a maximum complete proper coloring using DFS and return (reference_answer_str, gold_answer_int)."""
        # Build adjacency bitsets
        adjacent = [0] * N
        for u, v in edges:
            adjacent[u] |= 1 << v
            adjacent[v] |= 1 << u

        colors = [None] * N  # type: ignore
        color2set = [0] * N
        best_colors: Optional[List[int]] = None
        best_k: int = 0

        def DFS(u: int, max_color: int) -> None:
            nonlocal best_colors, best_k
            # Pruning: even if we color all remaining vertices with new colors,
            # we cannot beat current best.
            if (max_color + 1) + (N - u) <= best_k:
                return
            if u == N:
                # Validate complete coloring among used colors
                color_adjacent = [[False] * (max_color + 1) for _ in range(max_color + 1)]
                satisfied_color_pair_num = 0
                for a, b in edges:
                    ca, cb = colors[a], colors[b]
                    assert ca is not None and cb is not None
                    cu, cv = min(ca, cb), max(ca, cb)
                    assert cu != cv, "Adjacent vertices should have different colors"
                    if not color_adjacent[cu][cv]:
                        color_adjacent[cu][cv] = True
                        satisfied_color_pair_num += 1
                assert satisfied_color_pair_num <= (max_color + 1) * max_color // 2, \
                    "The number of satisfied color pairs should not exceed the maximum possible pairs"
                if satisfied_color_pair_num == (max_color + 1) * max_color // 2:
                    best_colors = colors.copy()  # type: ignore
                    best_k = max_color + 1
                return

            # Try assigning available colors including a new color (max_color + 1)
            for color in range((max_color + 1) + 1):
                if (color2set[color] & adjacent[u]) == 0:
                    colors[u] = color
                    color2set[color] += 1 << u
                    DFS(u + 1, max(max_color, color))
                    color2set[color] -= 1 << u
                    colors[u] = None  # backtrack

        DFS(0, -1)

        # Convert best_colors to string
        if best_colors is None:
            # Fallback: assign all zeros (valid only when no edges). Achromatic number is 1.
            best_colors = [0] * N
            best_k = 1

        reference_answer_str = " ".join(map(str, best_colors))
        return reference_answer_str, best_k

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and evaluate the submitted coloring."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from boxed content
        try:
            colors = list(map(int, boxed_content.strip().split()))
        except ValueError:
            # Content is not a valid sequence of integers
            info = {"error": "invalid_answer"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate length
        if len(colors) != self.N:
            info = {"error": "invalid_length", "expected_length": self.N, "got_length": len(colors)}
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate proper coloring and collect adjacent color pairs
        adjacent_color_pairs = set()
        for u, v in self.edges:
            if colors[u] == colors[v]:
                info = {"error": "adjacent_same_color", "edge": (u, v)}
                return TERMINAL_STATE, 0.0, True, False, info
            pair = (min(colors[u], colors[v]), max(colors[u], colors[v]))
            adjacent_color_pairs.add(pair)

        distinct_colors = set(colors)
        max_possible_pairs = len(distinct_colors) * (len(distinct_colors) - 1) // 2
        if len(adjacent_color_pairs) < max_possible_pairs:
            info = {"error": "incomplete_coloring", "pairs_satisfied": len(adjacent_color_pairs), "pairs_required": max_possible_pairs}
            return TERMINAL_STATE, 0.0, True, False, info

        # Check if the number of distinct colors equals the achromatic number (gold answer)
        gold = int(self.gold_answer if self.gold_answer is not None else 0)
        is_correct = (len(distinct_colors) == gold)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "gold_answer": gold,
            "user_answer": colors,
            "N": self.N,
            "edges": self.edges,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random coloring action."""
        # Random colors in range [0, N-1]
        colors = [str(random.randint(0, self.N - 1)) for _ in range(self.N)]
        return f"\\boxed{{{' '.join(colors)}}}"