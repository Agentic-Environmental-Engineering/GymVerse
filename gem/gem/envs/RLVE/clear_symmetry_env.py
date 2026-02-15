import random
import math
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ClearSymmetryEnv(Env):
    """Environment for the 'Clear Symmetry' matrix problem - single-turn Q&A."""

    def __init__(self, max_x: int = 1000000, **kwargs):
        """
        Initialize the ClearSymmetryEnv instance.

        Parameters:
            max_x (int): Maximum value for x to control problem difficulty. Must be >= 1.
        """
        super().__init__()
        assert max_x >= 1, "max_x should be greater than or equal to 1"
        self.max_x = max_x

        self.current_problem: Optional[str] = None
        self.current_x: Optional[int] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a matrix symmetry and clarity problem.\n"
            "Please provide your final answer as a single integer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameter
        x = random.randint(1, self.max_x)
        self.current_x = x

        # Build problem prompt
        self.current_problem = (
            "Consider a square matrix A with side n consisting of zeros and ones. "
            "There are n rows numbered from 1 to n from top to bottom and n columns numbered from 1 to n from left to right in this matrix. "
            "We denote the element of the matrix located at the intersection of the i-th row and the j-th column as A(i, j).\n\n"
            "A matrix A is called clear if no two cells containing ones have a common side.\n"
            "A matrix A is called symmetrical if it matches the matrices formed from it by a horizontal and/or a vertical reflection. "
            "Formally, for each pair (i, j) (1 ≤ i, j ≤ n) both of the following conditions must be met: "
            "A(i, j) = A(n - i + 1, j) and A(i, j) = A(i, n - j + 1).\n"
            "The sharpness of matrix A is defined as the number of ones in it.\n\n"
            f"Given integer x = {x}, your task is to find the smallest positive integer n such that there exists a clear symmetrical matrix A with side n and sharpness x.\n\n"
            "Output Format: Provide only the integer n in \\boxed{...} without any other text."
        )

        # Compute reference answer based on the original algorithm
        self.reference_answer = self._compute_min_n(x)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the answer."""
        # Parse the answer from \\boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            # Format error: no boxed content found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate the parsed answer
        try:
            user_answer = int(parsed)
        except ValueError:
            # Answer is not a valid integer
            info = {
                "error": "invalid_answer",
                "reference_answer": self.reference_answer,
                "user_answer": parsed,
                "x": self.current_x,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "x": self.current_x,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} format in the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_min_n(self, x: int) -> int:
        """
        Compute the smallest positive integer n such that there exists
        a clear symmetrical matrix A of side n with sharpness x.

        This follows the logic from the referenced solution:
        - Special case: if x == 3, return 5.
        - Otherwise, n = ceil(sqrt(2*x - 1)), and adjust to the next odd number.
        """
        if x == 3:
            return 5
        n = math.ceil(math.sqrt(2 * x - 1))
        return n + 1 - n % 2

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        # Randomly guess an integer around the computed range; use a broader range for diversity.
        random_answer = random.randint(1, max(10, (self.reference_answer or 10) * 2))
        return f"\\boxed{{{random_answer}}}"