from typing import Any, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MostNumEdge_NonSelfIsomorphismEnv(Env):
    """
    Environment for the problem:
    Given N labeled vertices, consider simple undirected graphs that are asymmetric
    (the only automorphism is the identity). What is the maximum number of edges such a graph can have?
    Single-turn Q&A environment.
    """

    def __init__(
        self,
        max_n: int = 60,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: Maximum allowed N for the problem instance. Must be >= 6.
        """
        super().__init__()
        if max_n < 6:
            raise ValueError("max_n should be greater than or equal to 6")
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_n: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an extremal graph theory problem about asymmetric graphs.\n"
            "Provide your final answer as a single integer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameter N
        N = random.randint(6, self.max_n)
        self.current_n = N

        # Compute the reference answer using the original algorithm
        self.reference_answer = self._compute_reference_answer(N)

        # Build the problem prompt
        problem = (
            f"Consider a simple undirected graph G on {N} labeled vertices 1 to {N}. "
            "We say G is asymmetric if the only bijection (permutation) p of the vertices that preserves all edges "
            "(i.e., (u, v) is an edge iff (p(u), p(v)) is an edge) is the identity permutation. "
            f"What is the maximum number of edges an asymmetric graph G on {N} labeled vertices can have?\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the outcome."""
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "n": self.current_n,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        # Heuristic range for random sampling
        if self.current_n is not None:
            upper = self.current_n * (self.current_n - 1) // 2
            guess = random.randint(0, max(upper, 1))
        else:
            guess = random.randint(0, 1000)
        return f"\\boxed{{{guess}}}"

    def _compute_reference_answer(self, N: int) -> int:
        """Compute the reference answer using the original algorithm."""
        def C(n: int, m: int) -> int:
            if 0 > m or m > n:
                return 0
            ans = 1
            for i in range(m):
                ans = ans * (n - i) // (i + 1)
            return ans

        # Note: f and h are intentionally aliased as in the original code
        f = h = [0 for _ in range(0, N + 1)]
        g = [[0 for _ in range(0, N + 1)] for _ in range(0, N + 1)]
        g[0][0] = 1

        for i in range(1, N + 1):
            h[i] = g[i - 1][i - 1]
            for j in range(0, N + 1):
                for k in range(j // i + 1):
                    g[i][j] += C(h[i], k) * g[i - 1][j - i * k]

        for i in range(1, N + 1):
            f[i] = g[(i - 1) // 2][i - 1]
            if i % 2 == 0:
                f[i] += C(g[i // 2 - 1][i // 2 - 1], 2)

        res = N * (N - 1) // 2 - N
        original_N = N
        for i in range(1, original_N + 1):
            cnt = min(N // i, f[i])
            res += cnt
            N -= i * cnt

        return res