from typing import Any, Optional, SupportsFloat, Tuple
import random
from math import gcd
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DistinctEdgeColoredCompleteGraphCountingEnv(Env):
    """Environment for counting distinct M-colored complete undirected graphs up to isomorphism (single-turn Q&A)."""

    MOD_CHOICES = (666623333, 998244353, 10 ** 9 + 7)

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 12,
        M: Optional[int] = None,
        modulo_choices: Tuple[int, ...] = MOD_CHOICES,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed number of vertices. If None, N is sampled uniformly from [min_n, max_n].
        - min_n: Minimum N when sampling.
        - max_n: Maximum N when sampling.
        - M: Optional fixed number of colors. If None, M is sampled uniformly from [2, N*(N-1)//2].
        - modulo_choices: Tuple of possible modulo values to choose from.
        """
        super().__init__()
        self.fixed_N = N
        self.min_n = min_n
        self.max_n = max_n
        self.fixed_M = M
        self.modulo_choices = modulo_choices

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_N: Optional[int] = None
        self.current_M: Optional[int] = None
        self.current_MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem: counting distinct edge-colored complete undirected graphs.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            if self.fixed_N < 3:
                raise ValueError("N should be greater than or equal to 3.")
            self.current_N = self.fixed_N
        else:
            if self.min_n < 3:
                raise ValueError("min_n should be greater than or equal to 3.")
            if self.max_n < self.min_n:
                raise ValueError("max_n should be greater than or equal to min_n.")
            self.current_N = random.randint(self.min_n, self.max_n)

        N = self.current_N

        # Determine M
        max_colors = N * (N - 1) // 2
        if self.fixed_M is not None:
            if not (2 <= self.fixed_M <= max_colors):
                raise ValueError(f"M should be in range [2, {max_colors}] for N={N}.")
            self.current_M = self.fixed_M
        else:
            self.current_M = random.randint(2, max_colors)

        # Determine MOD
        if not self.modulo_choices:
            raise ValueError("modulo_choices must contain at least one modulus.")
        self.current_MOD = random.choice(self.modulo_choices)

        # Build problem prompt
        self.current_problem = (
            f"Consider all complete undirected graphs on vertices 1, 2, ..., {N}, "
            f"where each edge is assigned a color from {self.current_M} colors (labeled from 1 to {self.current_M}). "
            f"Two such graphs G and G' are considered the same if there exists a permutation p of the vertices "
            f"such that for every unordered pair (u, v), the color of edge (u, v) in G equals the color of edge "
            f"(p(u), p(v)) in G'. What is the number of distinct graphs under this equivalence "
            f"(i.e., the number of non-isomorphic {self.current_M}-colored complete graphs on {N} vertices)? "
            f"Output the result modulo {self.current_MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._count_distinct_graphs(N, self.current_M, self.current_MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _count_distinct_graphs(self, N: int, M: int, MOD: int) -> int:
        """Compute the number of distinct M-colored complete graphs on N vertices modulo MOD."""
        # Modular exponentiation
        def qPow(b: int, e: int) -> int:
            a = 1
            b %= MOD
            while e:
                if e & 1:
                    a = (a * b) % MOD
                b = (b * b) % MOD
                e >>= 1
            return a

        # Precompute modular inverses, factorials, and inverse factorials up to N
        Inv = [0] * (N + 1)
        Fac = [0] * (N + 1)
        iFac = [0] * (N + 1)

        def Init(limit: int) -> None:
            Inv[1] = 1
            for i in range(2, limit + 1):
                Inv[i] = (MOD - MOD // i) * Inv[MOD % i] % MOD
            Fac[0] = 1
            iFac[0] = 1
            for i in range(1, limit + 1):
                Fac[i] = (Fac[i - 1] * i) % MOD
                iFac[i] = (iFac[i - 1] * Inv[i]) % MOD

        Sum = 0
        stk = [0]  # Sentinel to mimic C++ global zero-initialized array
        t = 0
        n1 = 0
        n2 = 1

        def DFS(s: int, mx: int, c: int) -> None:
            nonlocal Sum, t, n1, n2
            if s == 0:
                Sum = (Sum + qPow(M, n1) * n2) % MOD
                return
            a = n1
            b = n2
            for i in range(1, mx + 1):
                stk.append(i)
                t += 1
                n1 = a + i // 2
                for j in range(1, t):
                    n1 += gcd(stk[j], i)
                n2 = b * Inv[i] % MOD
                if i == stk[t - 1]:
                    n2 = n2 * Fac[c] % MOD * iFac[c + 1] % MOD
                DFS(s - i, min(s - i, i), c + 1 if i == stk[t - 1] else 1)
                t -= 1
                stk.pop()

        Init(N)
        DFS(N, N, 0)
        return Sum

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the submitted answer."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)

        if answer_str is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate answer is an integer
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate range: answer should be in [0, MOD)
        if self.current_MOD is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_error", "message": "MOD not set"}
        if not (0 <= user_answer < self.current_MOD):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "out_of_range": True,
                "N": self.current_N,
                "M": self.current_M,
                "MOD": self.current_MOD,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check correctness
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_N,
            "M": self.current_M,
            "MOD": self.current_MOD,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract answer from \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        if self.current_MOD is None:
            # Default to a generic random integer if MOD is not yet set
            random_answer = random.randint(0, 1000)
        else:
            random_answer = random.randint(0, self.current_MOD - 1)
        return f"\\boxed{{{random_answer}}}"