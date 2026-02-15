from typing import Any, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GCDOne_CountingEnv(Env):
    """Environment for counting pairs (x, y) with gcd(x, y) = 1, single-turn Q&A.

    The task:
    Given integers N and M, count the number of pairs (x, y) such that
    1 ≤ x ≤ N, 1 ≤ y ≤ M, and gcd(x, y) = 1.

    Answer format:
    The agent must output the final answer in \\boxed{...} format.
    """

    def __init__(
        self,
        max_n_m: int = 1000000,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            max_n_m: The maximum value for N and M (inclusive). Must be >= 2.
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        self.max_n_m: int = max_n_m

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a number theory counting problem.\n"
            "Task: Count the number of pairs (x, y) where 1 ≤ x ≤ N, 1 ≤ y ≤ M, and gcd(x, y) = 1.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate N and M in [2, max_n_m]
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Build the problem statement
        self.current_problem = (
            f"How many pairs (x, y) satisfy gcd(x, y) being exactly 1, "
            f"where 1 ≤ x ≤ {self.N} and 1 ≤ y ≤ {self.M}? Here, gcd(x, y) denotes the greatest common divisor of integers x and y.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute the reference answer using Möbius inversion
        self.reference_answer = self._count_coprime_pairs(self.N, self.M)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        # Parse \\boxed{...}
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not properly initialized. Call reset() first."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last occurrence of \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _count_coprime_pairs(self, n: int, m: int) -> int:
        """Count pairs (x, y) with gcd(x, y) = 1 for 1 ≤ x ≤ n, 1 ≤ y ≤ m."""
        limit = min(n, m)
        if limit < 1:
            return 0

        # Linear sieve for Möbius function mu[1..limit]
        mu = [0] * (limit + 1)
        mu[1] = 1
        is_composite = [False] * (limit + 1)
        primes = []

        for i in range(2, limit + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                ip = i * p
                if ip > limit:
                    break
                is_composite[ip] = True
                if i % p == 0:
                    mu[ip] = 0
                    break
                else:
                    mu[ip] = -mu[i]

        # Prefix sums of mu
        prefix = [0] * (limit + 1)
        for i in range(1, limit + 1):
            prefix[i] = prefix[i - 1] + mu[i]

        # Summation with block decomposition:
        # sum_{d=1..limit} mu[d] * floor(n/d) * floor(m/d)
        ans = 0
        l = 1
        while l <= limit:
            an = n // l
            am = m // l
            r = min(n // an, m // am)
            ans += (prefix[r] - prefix[l - 1]) * an * am
            l = r + 1

        assert ans > 0, "Reference answer should be positive for n, m >= 2."
        return ans

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # If N and M are available, sample within a broad plausible range
        if self.N is not None and self.M is not None:
            upper = max(1, self.N * self.M)
            random_answer = random.randint(0, upper)
        else:
            random_answer = random.randint(0, 1000)
        return f"\\boxed{{{random_answer}}}"