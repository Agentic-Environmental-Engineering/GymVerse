from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GCDPrime_CountingEnv(Env):
    """Environment for counting pairs (x, y) with gcd(x, y) being a prime number."""

    def __init__(
        self,
        max_n_m: int = 100000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n_m: The maximum value for N and M (inclusive). Must be >= 2.
        """
        super().__init__()
        if max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")
        self.max_n_m = max_n_m

        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a number theory counting problem.\n"
            "Task: Count the number of pairs (x, y) such that gcd(x, y) is a prime number.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters N and M
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Build problem prompt
        self.current_problem = (
            f"How many pairs (x, y) satisfy gcd(x, y) being a prime number, "
            f"where 1 ≤ x ≤ {self.N} and 1 ≤ y ≤ {self.M}? Here, gcd(x, y) denotes "
            f"the greatest common divisor of integers x and y.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_answer(self.N, self.M)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            # Not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Compare with reference answer
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
        """Extract the answer inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_answer(self, N: int, M: int) -> int:
        """Compute the number of pairs (x, y) with gcd(x, y) being a prime number."""
        lim = min(N, M)
        if lim < 2:
            return 0

        # Linear sieve to compute Möbius function up to lim and collect primes
        mu = [0] * (lim + 1)
        mu[1] = 1
        is_composite = [False] * (lim + 1)
        primes: list[int] = []

        for i in range(2, lim + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                ip = i * p
                if ip > lim:
                    break
                is_composite[ip] = True
                if i % p == 0:
                    break
                else:
                    mu[ip] = -mu[i]

        # Construct f where f[n] = sum_{p prime, p|n} mu[n/p]
        f = [0] * (lim + 1)
        for p in primes:
            for j in range(1, lim // p + 1):
                f[j * p] += mu[j]

        # Prefix sums of f
        prefix = [0] * (lim + 1)
        s = 0
        for i in range(1, lim + 1):
            s += f[i]
            prefix[i] = s

        # Summation using harmonic ranges
        ans = 0
        l = 1
        while l <= N and l <= M:
            an = N // l
            am = M // l
            r = min(N // an, M // am)
            ans += (prefix[r] - prefix[l - 1]) * an * am
            l = r + 1

        return ans

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        # As a random guess, use a non-negative integer
        random_answer = random.randint(0, (self.N or 1) * (self.M or 1))
        return f"\\boxed{{{random_answer}}}"