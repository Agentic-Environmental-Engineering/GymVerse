from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxMultiplicationFixedSumEnv(Env):
    """Environment for the maximum product of positive integers with a fixed sum - single-turn QA."""

    def __init__(
        self,
        max_n: int = 100,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: Maximum value for N. Must be >= 10.
        """
        super().__init__()
        self.max_n = max_n
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_n: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a combinatorial optimization problem.\n"
            "Task: Given a positive integer N, find the maximum product of positive integers (not necessarily distinct) whose sum is exactly N.\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate parameters
        assert self.max_n >= 10, "max_n should be greater than or equal to 10"

        # Generate problem instance
        N = random.randint(4, self.max_n)
        self.current_n = N

        # Compute reference answer using optimal partition into 3s (and possibly a 4 or 2)
        self.reference_answer = self._max_product_fixed_sum(N)

        # Build problem description
        self.current_problem = (
            f"Can you tell me the maximum product of positive integers (not necessarily distinct) whose sum is exactly {N}?\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and terminate."""
        # Extract the boxed answer
        extracted = self._parse_answer(action)
        if extracted is None:
            # Format error (no valid \boxed{...} found)
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate that the extracted content is an integer
        try:
            user_answer = int(extracted.strip())
        except ValueError:
            # Content inside the box is not a valid integer -> treat as format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "invalid_answer"}

        # Compare with the reference answer
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
        """Extract the content of the last \\boxed{...} occurrence from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _max_product_fixed_sum(self, n: int) -> int:
        """
        Compute the maximum product of positive integers (not necessarily distinct)
        whose sum equals n using the well-known partition strategy into 3s.
        """
        if n % 3 == 0:
            return 3 ** (n // 3)
        elif n % 3 == 1:
            return (3 ** ((n - 4) // 3)) * 4
        else:
            return (3 ** ((n - 2) // 3)) * 2

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        # Sample a small random integer as a placeholder guess
        random_answer = random.randint(0, 1000)
        return f"\\boxed{{{random_answer}}}"