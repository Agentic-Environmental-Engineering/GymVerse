import random
import re
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumProductDivisorNumEnv(Env):
    """Environment for computing the sum of the number of divisors of i * j over ranges.

    Task:
      Compute S = sum_{i=1..N} sum_{j=1..M} d(i * j),
      where d(x) is the number of distinct divisors of integer x.

    Single-turn Q&A environment:
      - reset() generates a new problem and returns the prompt.
      - step(action) validates the boxed answer and returns the reward.
    """

    def __init__(
        self,
        max_n_m: int = 5000,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            max_n_m: Upper bound for both N and M (inclusive). Must be >= 2.
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        self.max_n_m: int = max_n_m

        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a number theory problem about divisor counts.\n"
            "Given positive integers N and M, compute the sum of d(i * j) over all pairs (i, j)\n"
            "such that 1 ≤ i ≤ N and 1 ≤ j ≤ M, where d(x) is the number of distinct divisors of x.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Build problem description
        self.current_problem = (
            f"Please compute sum(d(i * j)) for all pairs (i, j) such that 1 ≤ i ≤ {self.N} and 1 ≤ j ≤ {self.M}. "
            "Here, d(x) denotes the number of distinct divisors of integer x, and d(i * j) is the number of divisors "
            "of the product of i and j.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(self.N, self.M)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the user's answer.

        Returns:
            TERMINAL_STATE as observation since this is a single-turn environment.
            Reward: 1.0 for correct, 0.0 for incorrect, -0.1 for format error.
        """
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate integer answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random boxed integer)."""
        random_answer = random.randint(1, 100)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, N: int, M: int) -> int:
        """Compute the reference answer using Möbius inversion with harmonic blocking."""
        max_val = max(N, M)
        mu_pref, s = self._precompute(max_val)

        # Ensure N ≤ M for the optimized solver
        n, m = (N, M) if N <= M else (M, N)
        return self._solve_case(n, m, mu_pref, s)

    def _precompute(self, max_val: int) -> Tuple[list[int], list[int]]:
        """Precompute:
        - mu_pref[x] = sum_{k=1..x} μ(k) (Möbius prefix sum, 0-indexed storage, meaningful for indices 0..max_val)
        - s[x]       = sum_{k=1..x} floor(x/k) for each x in 1..max_val
        """
        # Linear sieve for Möbius function μ
        mu = [0] * (max_val + 1)
        mu[1] = 1
        is_composite = [False] * (max_val + 1)
        primes: list[int] = []

        for i in range(2, max_val + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                ip = i * p
                if ip > max_val:
                    break
                is_composite[ip] = True
                if i % p == 0:
                    mu[ip] = 0
                    break
                mu[ip] = -mu[i]

        # Convert μ to its prefix sum in-place
        for i in range(1, max_val + 1):
            mu[i] += mu[i - 1]

        # Precompute s[x] = sum_{k=1..x} floor(x/k) for each 1..max_val
        s = [0] * (max_val + 1)
        for x in range(1, max_val + 1):
            res = 0
            i = 1
            while i <= x:
                j = x // (x // i)
                res += (j - i + 1) * (x // i)
                i = j + 1
            s[x] = res

        return mu, s

    def _solve_case(self, N: int, M: int, mu_pref: list[int], s: list[int]) -> int:
        """Compute sum_{i=1..N} sum_{j=1..M} d(i*j) using Möbius trick with harmonic blocking.
        Requires N ≤ M.
        """
        ans = 0
        i = 1
        while i <= N:
            j = min(N // (N // i), M // (M // i))
            ans += (mu_pref[j] - mu_pref[i - 1]) * s[N // i] * s[M // i]
            i = j + 1
        return ans