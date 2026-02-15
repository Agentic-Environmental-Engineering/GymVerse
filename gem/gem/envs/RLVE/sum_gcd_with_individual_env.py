import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumGCDWithIndividualEnv(Env):
    """Environment for computing the sum of GCD(i, N) for 1 ≤ i ≤ N - single-turn Q&A."""

    def __init__(
        self,
        max_n: int = 1000000,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_n: Upper bound for N, must be >= 4.
        - wrong_format: Preserved parameter from original RLVE environment (not used in reward calculation).
        - rewarding_strategy: Preserved parameter from original RLVE environment (not used in reward calculation).
        - rewarding_weight: Preserved parameter from original RLVE environment (not used in reward calculation).
        - rewarding_beta: Preserved parameter from original RLVE environment (not used in reward calculation).
        """
        super().__init__()
        assert max_n >= 4, "max_n should be greater than or equal to 4"
        self.max_n = max_n

        # Preserved parameters (not used in GEM reward settings but kept for compatibility)
        self.wrong_format = wrong_format
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # State attributes
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_n: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory problem.\n"
            "Task: Compute the sum of GCD(i, N) for all integers i such that 1 ≤ i ≤ N.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameter
        N = random.randint(4, self.max_n)
        self.current_n = N

        # Build problem prompt
        self.current_problem = (
            f"Please compute the sum of GCD(i, {N}) for all i such that 1 ≤ i ≤ {N}. "
            f"Here, GCD(i, j) denotes the greatest common divisor of integers i and j.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute the reference answer
        self.reference_answer = self._compute_sum_gcd(N)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_sum_gcd(self, n: int) -> int:
        """
        Compute sum_{i=1..n} gcd(i, n) using prime factorization-based formula.

        This implementation mirrors the logic from the original RLVE environment.
        """
        t = n
        ans = n
        i = 2
        # Iterate over potential prime factors up to sqrt(t), updating t as we go
        while i * i <= t:
            if t % i == 0:
                b = 0
                # Count multiplicity of prime factor i in t
                while t % i == 0:
                    b += 1
                    t //= i
                # Incorporate factor i with exponent b into ans
                ans //= i
                ans *= (b * i - b + i)
            i += 1

        # If there's a prime factor left greater than sqrt(n)
        if t > 1:
            ans //= t
            ans *= (t + t - 1)

        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the provided answer."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)

        if answer_text is None:
            # Format error: missing or invalid \\boxed{...}
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            # Answer is not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Compare with reference
        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_n,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by producing a random integer in boxed format."""
        random_answer = random.randint(0, max(1, (self.current_n or 10)))
        return f"\\boxed{{{random_answer}}}"