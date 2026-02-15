from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Cube_FixedLocalMaximumCountingEnv(Env):
    """Environment for counting dominant cells in a 3D grid with modular probability output."""

    def __init__(
        self,
        max_n_m_l: int = 10,
        modulo: int = 998244353,
        n: Optional[int] = None,
        m: Optional[int] = None,
        l: Optional[int] = None,
        k: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
            max_n_m_l: Upper bound for N, M, L (inclusive). Must be >= 2.
            modulo: Prime modulus for computing the answer.
            n, m, l: Optional fixed dimensions for the grid. If provided, must be in [2, max_n_m_l].
            k: Optional fixed number of dominant cells to compute probability for. If provided, must be in [2, min(n, m, l)].

        Note:
            This is a single-turn environment. The agent must provide the answer in \\boxed{...} format.
        """
        super().__init__()
        if max_n_m_l < 2:
            raise ValueError("max_n_m_l should be greater than or equal to 2")
        self.max_n_m_l = max_n_m_l
        self.modulo = modulo

        # Optional fixed parameters
        self.fixed_n = n
        self.fixed_m = m
        self.fixed_l = l
        self.fixed_k = k

        # Validate fixed parameters if provided
        if self.fixed_n is not None and not (2 <= self.fixed_n <= self.max_n_m_l):
            raise ValueError("n must be within [2, max_n_m_l]")
        if self.fixed_m is not None and not (2 <= self.fixed_m <= self.max_n_m_l):
            raise ValueError("m must be within [2, max_n_m_l]")
        if self.fixed_l is not None and not (2 <= self.fixed_l <= self.max_n_m_l):
            raise ValueError("l must be within [2, max_n_m_l]")
        if self.fixed_k is not None:
            if self.fixed_n is None or self.fixed_m is None or self.fixed_l is None:
                raise ValueError("k is fixed but n, m, l are not all fixed; cannot validate k range.")
            min_dim = min(self.fixed_n, self.fixed_m, self.fixed_l)
            if not (2 <= self.fixed_k <= min_dim):
                raise ValueError("k must be within [2, min(n, m, l)]")

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.L: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions for the agent."""
        return (
            "You are solving a probability problem on a 3D grid.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate or use fixed parameters
        if self.fixed_n is not None:
            self.N = self.fixed_n
        else:
            self.N = random.randint(2, self.max_n_m_l)

        if self.fixed_m is not None:
            self.M = self.fixed_m
        else:
            self.M = random.randint(2, self.max_n_m_l)

        if self.fixed_l is not None:
            self.L = self.fixed_l
        else:
            self.L = random.randint(2, self.max_n_m_l)

        min_dim = min(self.N, self.M, self.L)

        if self.fixed_k is not None:
            self.K = self.fixed_k
        else:
            self.K = random.randint(2, min_dim)

        # Build problem prompt
        total = self.N * self.M * self.L
        self.current_problem = (
            f"You are given a 3D grid of size {self.N} × {self.M} × {self.L}. "
            f"Each cell will be filled with a unique number from 1 to {total} "
            f"(where {total} = {self.N} × {self.M} × {self.L}). The numbers are assigned randomly and uniformly — "
            f"every permutation of the {total} numbers over the grid is equally likely. "
            f"A cell is called dominant if its value is strictly greater than all other cells that share at least one coordinate "
            f"(i.e., same x, y, or z index). Please compute the probability that exactly {self.K} dominant cells exist after filling the grid.\n\n"
            f"Output Format: Your final answer should be a single integer equal to the required probability modulo {self.modulo}, "
            f"written in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_answer(self.N, self.M, self.L, self.K, self.modulo)

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.N,
            "M": self.M,
            "L": self.L,
            "K": self.K,
            "modulo": self.modulo,
        }
        return obs, info

    def _compute_answer(self, N: int, M: int, L: int, K: int, MOD: int) -> int:
        """Compute the probability modulo MOD using the provided algorithm."""
        def inv_list(n: int) -> list[int]:
            """Compute modular inverses of 1..n under MOD."""
            invs = [0] * (n + 1)
            invs[1] = 1
            for i in range(2, n + 1):
                invs[i] = (-(MOD // i) * invs[MOD % i]) % MOD
            return invs

        def modinv(x: int) -> int:
            """Modular inverse of x under MOD (MOD is prime)."""
            return pow(x, MOD - 2, MOD)

        Q = min(N, M, L)
        invs = inv_list(Q)

        def R(x: int) -> int:
            """R(x) = (N-x)*(M-x)*(L-x) mod MOD."""
            return ((N - x) % MOD) * ((M - x) % MOD) % MOD * ((L - x) % MOD) % MOD

        vals = [0] * (Q + 1)
        iprod = [0] * (Q + 1)
        iprod[0] = 1

        R0 = R(0)
        for i in range(1, Q + 1):
            vals[i] = (R0 - R(i)) % MOD
            iprod[i] = iprod[i - 1] * vals[i] % MOD

        inv_total = modinv(iprod[Q])
        for i in range(Q, 0, -1):
            prev = iprod[i - 1]
            iprod[i] = inv_total * prev % MOD
            inv_total = inv_total * vals[i] % MOD

        ans = 0
        C = 0
        S = 1
        for i in range(1, Q + 1):
            S = S * R(i - 1) % MOD * iprod[i] % MOD
            if i == K:
                C = 1
            elif i > K:
                C = (-C * i * invs[i - K]) % MOD
            ans = (ans + C * S) % MOD

        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the agent's answer."""
        if self.reference_answer is None:
            # Environment not properly reset
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Parse the answer from \boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if not (0 <= user_answer < self.modulo):
            # Out-of-range answer
            return TERMINAL_STATE, 0.0, True, False, {"error": "range_error"}

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "L": self.L,
            "K": self.K,
            "modulo": self.modulo,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from \\boxed{...} in the provided text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action (answer) for testing or exploration."""
        random_answer = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{random_answer}}}"