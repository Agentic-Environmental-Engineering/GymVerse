import math
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class AddMultiple_Divisible_CountingEnv(Env):
    """Environment for counting pairs (a, b) such that a × b is divisible by a + b."""

    def __init__(
        self,
        max_n: int = 1000000,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs,
    ):
        """
        Initialize the environment with parameter controls.

        Parameters:
        - max_n: Upper bound for N (must be >= 6).
        - wrong_format: Legacy parameter from RLVE, default -1.0 (unused in GEM).
        - rewarding_strategy: Legacy parameter from RLVE (unused in GEM).
        - rewarding_weight: Legacy parameter from RLVE (unused in GEM).
        - rewarding_beta: Legacy parameter from RLVE (unused in GEM).
        """
        super().__init__()
        self.max_n = max_n
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None

        # Legacy reward configuration preserved for compatibility, not used in GEM reward calculation
        self.rewards = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # Validate parameters similar to original environment logic
        assert self.max_n >= 6, "max_n should be greater than or equal to 6"

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory counting problem.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate the problem parameter N
        self.N = random.randint(6, self.max_n)

        # Build problem statement
        self.current_problem = (
            f"Please compute the number of pairs (a, b) such that:\n"
            f"- 1 ≤ a < b ≤ {self.N}\n"
            f"- a × b is divisible by a + b\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute the reference answer using the original algorithm
        self.reference_answer = self._compute_answer(self.N)
        assert self.reference_answer > 0, "Answer should be greater than 0"

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {"N": self.N}
        return obs, info

    def _compute_answer(self, N: int) -> int:
        """Compute the number of valid pairs using the original algorithm."""
        def calc(x: int, y: int) -> int:
            """
            Compute sum_{k = x+1..2*x-1} floor(y / k) by grouping k's with the same quotient.
            """
            if y == 0:
                return 0
            a = 0
            z = x << 1
            i = x + 1
            while i < z:
                q = y // i
                if q == 0:
                    break
                j = min(y // q, z - 1)
                a += (j - i + 1) * q
                i = j + 1
            return a

        m = math.isqrt(N)

        mu = [0] * (m + 1)
        mu[1] = 1
        is_comp = [False] * (m + 1)
        primes = []

        for i in range(2, m + 1):
            if not is_comp[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                ip = i * p
                if ip > m:
                    break
                is_comp[ip] = True
                if i % p == 0:
                    mu[ip] = 0
                    break
                else:
                    mu[ip] = -mu[i]

        ans = 0
        for i in range(1, m + 1):
            if mu[i] == 0:
                continue
            ii = i * i
            top = m // i
            for j in range(1, top + 1):
                y = N // (ii * j)
                ans += mu[i] * calc(j, y)
        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step to verify the submitted answer."""
        # Parse the boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric format
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Enforce original environment's non-positive answer check
        if user_answer <= 0:
            return TERMINAL_STATE, -0.1, True, False, {"error": "non_positive_answer"}

        assert self.reference_answer is not None, "Reference answer not computed"
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
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
        """Sample a random action in \\boxed{...} format."""
        # Random guess; could be any non-negative integer
        random_answer = random.randint(1, max(1, self.N or 100))
        return f"\\boxed{{{random_answer}}}"