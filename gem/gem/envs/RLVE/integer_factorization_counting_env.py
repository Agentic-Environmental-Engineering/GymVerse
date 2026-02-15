from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class IntegerFactorizationCountingEnv(Env):
    """Environment for counting the number of ways to factorize N into multiple distinct integers > 1."""

    def __init__(
        self,
        max_n: int = 100000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: The maximum value of N to sample. Must be >= 4.
        """
        super().__init__()
        assert max_n >= 4, "max_n should be greater than or equal to 4"
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_n: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "Task: Count the number of ways to factorize a given integer N into multiple (i.e., more than 1) "
            "distinct positive integers greater than 1 such that their product is exactly N. "
            "The order of factors does not matter.\n"
            "For example, 688 = 2 × 4 × 86 = 2 × 8 × 43 = 2 × 344 = 4 × 172 = 8 × 86 = 16 × 43, "
            "so there are 6 valid ways in total.\n\n"
            "Output Format: Provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample problem parameter
        N = random.randint(4, self.max_n)
        self.current_n = N

        # Build problem statement
        self.current_problem = (
            f"Count the number of ways to factorize {N} into multiple (more than 1) distinct positive integers "
            f"greater than 1 such that their product is {N}. The order of factors does not matter.\n\n"
            f"Output Format: Provide your final answer in \\boxed{{...}}."
        )

        # Compute reference answer using the original algorithm
        self.reference_answer = self._count_factorizations(N)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and score
        try:
            user_answer = int(parsed)
        except ValueError:
            # Not an integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_n,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _count_factorizations(self, N: int) -> int:
        """
        Count the number of ways to express N as a product of multiple distinct integers > 1.
        The order of factors does not matter.
        """
        # 1. Enumerate divisors of N
        divs = []
        i = 1
        while i * i <= N:
            if N % i == 0:
                divs.append(i)
                if i != N // i:
                    divs.append(N // i)
            i += 1
        divs.sort()
        total = len(divs)

        # 2. Map each divisor to its index
        idx = {d: i for i, d in enumerate(divs)}

        # 3. dp[i] = number of ways to get product = divs[i] using distinct divisors seen so far
        dp = [0] * total
        dp[0] = 1  # one way to make 1 (the empty product)

        # 4. For each divisor x = divs[j] (skip 1), update dp in place from high i down to j
        for j in range(1, total):
            xj = divs[j]
            for k in range(total - 1, j - 1, -1):
                di = divs[k]
                if di % xj == 0:
                    dp[k] += dp[idx[di // xj]]

        # 5. dp[total - 1] counts also the trivial factorization [N] → subtract 1
        return dp[total - 1] - 1

    def sample_random_action(self) -> str:
        """Sample a random action in boxed format."""
        # Randomly guess a non-negative integer
        guess = random.randint(0, max(1, (self.reference_answer or 0) * 2 + 5))
        return f"\\boxed{{{guess}}}"