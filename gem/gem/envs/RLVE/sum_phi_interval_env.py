from typing import Any, Optional, SupportsFloat, Tuple
import math
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumPHIIntervalEnv(Env):
    """Environment for summing the count of non-coprime integers within an interval."""

    prompt_template = (
        "Define F(x) as the number of integers in the range [1, x] that are not coprime to x. "
        "Please output the sum of F(i) for all integers i in the range [{L}, {R}] (inclusive)."
    )

    def __init__(
        self,
        max_delta: int = 1000,
        **kwargs
    ):
        """
        Initialize the SumPHIIntervalEnv.

        Parameters:
            max_delta (int): Controls the size of the interval. L is chosen in [1, max_delta^2],
                             and R = L + random integer in [1, max_delta].
        """
        super().__init__()
        self.max_delta = max_delta
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.L: Optional[int] = None
        self.R: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory problem involving Euler's totient function.\n"
            "Let F(x) be the number of integers in [1, x] that are not coprime to x.\n"
            "Your task is to compute sum_{i=L..R} F(i).\n\n"
            "Answer Format: Provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Parameter validation
        assert self.max_delta >= 1, "max_delta should be greater than or equal to 1"

        # Generate problem parameters
        self.L = random.randint(1, self.max_delta ** 2)
        self.R = self.L + random.randint(1, self.max_delta)

        # Build problem prompt
        self.current_problem = self.prompt_template.format(L=self.L, R=self.R) + (
            "\n\nOutput Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_sum_non_coprimes(self.L, self.R)
        assert self.reference_answer is not None and self.reference_answer > 0, "The reference answer should be greater than 0"

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the answer."""
        boxed_answer = self._parse_answer(action)

        if boxed_answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(boxed_answer)
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "L": self.L,
            "R": self.R,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the provided text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_sum_non_coprimes(self, L: int, R: int) -> int:
        """
        Compute sum_{n=L..R} (n - phi(n)) using a segmented sieve approach.
        """
        limit = math.isqrt(R)
        is_prime = [True] * (limit + 1)
        primes = []
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
                if i * i <= limit:
                    for j in range(i * i, limit + 1, i):
                        is_prime[j] = False

        size = R - L + 1
        A = [L + i for i in range(size)]  # will become phi(L+i)
        B = [L + i for i in range(size)]  # copy to strip prime factors

        for p in primes:
            if p * p > R:
                break
            start = ((L + p - 1) // p) * p
            for x in range(start, R + 1, p):
                idx = x - L
                A[idx] //= p
                A[idx] *= (p - 1)
                while B[idx] % p == 0:
                    B[idx] //= p

        ans = 0
        for i in range(size):
            if B[i] > 1:
                A[i] //= B[i]
                A[i] *= (B[i] - 1)
            ans += (L + i) - A[i]

        return ans

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # As a heuristic, sample a random number in a plausible range
        # The sum of non-coprimes is at most sum_{n=L..R} (n - 1) ~ O((R-L+1)*R)
        # Here we simply sample a random integer in a conservative range.
        if self.L is not None and self.R is not None:
            conservative_upper = max(1, (self.R - self.L + 1) * (self.R // 2))
        else:
            conservative_upper = 1000
        random_answer = random.randint(0, conservative_upper)
        return f"\\boxed{{{random_answer}}}"