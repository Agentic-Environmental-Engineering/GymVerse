import random
import re
from typing import Any, List, Optional, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PrimeGraph_MinimumChromaticNumberEnv(Env):
    """Prime-difference graph coloring environment (single-turn Q&A).

    Given N vertices labeled 1..N, connect u and v if |u - v| is a prime number.
    The task is to assign a non-negative integer color to each vertex such that:
    - Adjacent vertices have different colors.
    - The number of distinct colors used is minimized.

    The agent must output a single line of N integers separated by spaces, wrapped in \\boxed{...}.
    Correct solution (valid coloring using the minimal number of colors for this instance): reward 1.0.
    Wrong solution: reward 0.0.
    Format error (no \\boxed{...}): reward -0.1.
    """

    def __init__(self, max_n: int = 100, **kwargs) -> None:
        """
        Initialize the environment.

        Args:
            max_n: Maximum value of N to sample for each episode. Must be >= 3.
        """
        super().__init__()
        assert max_n >= 3, "max_n should be greater than or equal to 3"
        self.max_n: int = max_n

        # Problem-specific state (set in reset)
        self.N: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_coloring_list: Optional[List[int]] = None
        self.reference_coloring_str: Optional[str] = None
        self.gold_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general instructions for the task."""
        return (
            "You are solving a graph coloring problem.\n"
            "Please provide your final answer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample N
        N = random.randint(3, self.max_n)
        self.N = N

        # Construct problem description
        self.current_problem = (
            f"You are given an undirected graph with {N} vertices labeled from 1 to {N}. "
            "Two vertices u and v are connected by an edge if and only if the absolute difference |u - v| is a prime number.\n\n"
            "Your task is to assign a non-negative integer color to each vertex, represented as c[1], c[2], ..., c[{N}], such that:\n"
            "- For every edge (u, v), c[u] != c[v].\n"
            "- The total number of distinct colors used is minimized.\n\n"
            "Output Format: Your final answer should be a single line containing the colors of all vertices in order:\n"
            "c[1] c[2] ... c[N]\n"
            "The entire line must be wrapped in \\boxed{...}."
        ).replace("{N}", str(N))

        # Create a reference coloring to determine an expected minimal color count ("gold")
        if N <= 6:
            # Reference coloring as in the original environment
            ref_list = [((i + 1) // 2) for i in range(1, N + 1)]
        else:
            # Reference coloring as in the original environment
            ref_list = [i & 3 for i in range(1, N + 1)]

        self.reference_coloring_list = ref_list
        self.reference_coloring_str = " ".join(map(str, ref_list))
        self.gold_answer = len(set(ref_list))

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        """Validate the proposed coloring and return reward."""
        # Parse answer from \\boxed{...}
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the boxed content into a list of integers
        try:
            tokens = boxed.strip().split()
            colors_raw = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length and graph coloring constraints
        if self.N is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error"}

        if len(colors_raw) != self.N:
            info = {
                "error": "invalid_solution_length",
                "expected_length": self.N,
                "received_length": len(colors_raw),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Convert to 1-indexed array as in the original logic
        colors = [-1] + colors_raw  # sentinel at index 0

        # Sieve of Eratosthenes to find primes up to N
        N = self.N
        is_prime = [True] * (N + 1)
        if N >= 0:
            is_prime[0] = False
        if N >= 1:
            is_prime[1] = False
        primes: List[int] = []
        for i in range(2, N + 1):
            if is_prime[i]:
                primes.append(i)
                for j in range(i * i, N + 1, i):
                    is_prime[j] = False

        # Check adjacency constraints: for each prime difference p, adjacent vertices must differ
        for p in primes:
            for i in range(1, N - p + 1):
                if colors[i] == colors[i + p]:
                    info = {
                        "error": "edge_conflict",
                        "conflict_edge": (i, i + p),
                        "difference_is_prime": p,
                    }
                    return TERMINAL_STATE, 0.0, True, False, info

        # Compute number of distinct colors used by the user
        user_color_count = len(set(colors[1:]))

        # Determine correctness: must use exactly minimal number of colors (gold)
        is_correct = (user_color_count == self.gold_answer)

        info = {
            "correct": is_correct,
            "user_distinct_colors": user_color_count,
            "gold_distinct_colors": self.gold_answer,
            "N": self.N,
        }

        reward = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...}."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        if self.N is None or self.gold_answer is None:
            # Default fallback if called before reset
            n = 5
            random_colors = [random.randint(0, 3) for _ in range(n)]
        else:
            # Use up to gold_answer distinct colors to increase chance of optimality
            n = self.N
            max_color = max(0, self.gold_answer - 1)
            random_colors = [random.randint(0, max_color) for _ in range(n)]
        return f"\\boxed{{{' '.join(map(str, random_colors))}}}"