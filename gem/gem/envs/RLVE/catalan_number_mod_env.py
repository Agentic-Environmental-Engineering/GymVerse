from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CatalanNumberModEnv(Env):
    """Environment for computing Catalan numbers modulo a given value - single-turn Q&A.

    The task: Given N and MOD, compute the number of valid permutations defined as:
    - Odd-indexed elements are strictly increasing
    - Even-indexed elements are strictly increasing
    - Each adjacent pair (odd, even) is strictly increasing

    This count equals the N-th Catalan number. Output should be modulo MOD.
    """

    def __init__(
        self,
        max_n: int = 1000,
        max_mod: int = 1000000000,
        **kwargs
    ):
        super().__init__()
        self.max_n = max_n
        self.max_mod = max_mod

        # Internal state for the current problem
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a combinatorics problem related to Catalan numbers.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate parameters
        if self.max_n < 2:
            raise ValueError("max_n should be greater than or equal to 2")
        if self.max_mod < 2:
            raise ValueError("max_mod should be greater than or equal to 2")

        # Generate problem parameters
        N = random.randint(2, self.max_n)
        MOD = random.randint(2, self.max_mod)
        self.N = N
        self.MOD = MOD

        # Build the problem prompt
        problem_statement = (
            f"We define a valid permutation of the integers from 1 to 2×{N} (i.e., a permutation A[1], A[2], ..., A[2×{N}]) "
            f"that satisfies all of the following conditions:\n"
            f"- A[1] < A[3] < ... < A[2×{N} - 1] (all elements at odd indices form a strictly increasing sequence)\n"
            f"- A[2] < A[4] < ... < A[2×{N}] (all elements at even indices form a strictly increasing sequence)\n"
            f"- For all i = 1 to {N}, A[2i - 1] < A[2i] (each adjacent pair forms an increasing pair)\n\n"
            f"Please compute the total number of such valid permutations. Output the result modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem_statement

        # Compute reference answer using prime factorization and modular multiplication
        self.reference_answer = self._compute_catalan_mod(N, MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and terminate."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error: no boxed answer found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to convert the boxed content to integer
        try:
            user_answer = int(boxed_content.strip())
        except ValueError:
            # Content inside boxed is not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check as in original environment: answer must be in [0, MOD)
        if not (0 <= user_answer < (self.MOD if self.MOD is not None else 1)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "out_of_range"}

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "MOD": self.MOD,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} format."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action (answer)."""
        modulo = self.MOD if self.MOD is not None else self.max_mod
        random_answer = random.randint(0, modulo - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_catalan_mod(self, n: int, mod: int) -> int:
        """Compute the nth Catalan number modulo mod using prime factorization.

        Catalan(n) = C(2n, n) / (n + 1).
        We compute the product:
          numerator: (n+2) * (n+3) * ... * (2n)
          denominator: 1 * 2 * ... * n
        Then factorize and multiply primes^exponent modulo mod.
        """
        limit = 2 * n

        # Linear sieve to compute smallest prime factor (spf) up to 2n
        spf = [0] * (limit + 1)
        primes = []
        for i in range(2, limit + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
            for p in primes:
                ip = i * p
                if p > spf[i] or ip > limit:
                    break
                spf[ip] = p

        # cnt[i] holds the exponent contribution of i in the product:
        # numerator: (n+2)*(n+3)*...*(2n)
        # denominator: 1*2*...*n
        cnt = [0] * (limit + 1)
        # subtract denominator
        for i in range(1, n + 1):
            cnt[i] = -1
        # add numerator (skip n+1, which is neither in numerator nor denominator here)
        for i in range(n + 2, limit + 1):
            cnt[i] = 1

        # Propagate counts down to prime factors
        for i in range(limit, 1, -1):
            if spf[i] < i:
                c = cnt[i]
                if c:
                    cnt[spf[i]] += c
                    cnt[i // spf[i]] += c

        # Multiply out primes^cnt[p] mod MOD
        result = 1
        for p in primes:
            exp = cnt[p]
            if exp:
                # exp should be non-negative for Catalan numbers (they are integers)
                result = (result * pow(p, exp, mod)) % mod

        return result