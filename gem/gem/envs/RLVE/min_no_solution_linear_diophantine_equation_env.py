import math
import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinNoSolutionLinearDiophantineEquationEnv(Env):
    """Environment for finding the largest non-negative integer z such that Ax + By = z has no non-negative integer solutions."""

    def __init__(self, max_a_b: int = 50, **kwargs) -> None:
        """
        Initialize the environment.

        Args:
            max_a_b: The maximum value for coefficients A and B (inclusive). Must be >= 3.
        """
        super().__init__()
        self.max_a_b = max_a_b
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.A: Optional[int] = None
        self.B: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory problem related to linear Diophantine equations.\n"
            "Given positive integers A and B, consider the equation A*x + B*y = z with x, y >= 0 being integers.\n"
            "Find the largest non-negative integer z that cannot be represented as A*x + B*y with x, y being non-negative integers.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)
        assert self.max_a_b >= 3, "A and B should be greater than or equal to 3"

        # Generate coprime A and B
        while True:
            A = random.randint(2, self.max_a_b)
            B = random.randint(2, self.max_a_b)
            if math.gcd(A, B) == 1:
                break

        self.A = A
        self.B = B

        # Reference answer (Frobenius number for two coprime integers)
        self.reference_answer = A * B - A - B
        assert self.reference_answer is not None and self.reference_answer > 0

        # Build the problem statement
        self.current_problem = (
            f"Consider the equation {A}x + {B}y = z. "
            f"Find the largest non-negative integer z ≥ 0 such that the equation has no non-negative integer solutions (x, y).\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer."""
        answer_str = self._parse_answer(action)

        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "A": self.A,
            "B": self.B,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random guess)."""
        # Use a heuristic range for sampling based on current A and B if available
        if self.A is not None and self.B is not None:
            upper = max(self.A * self.B, 1)
        else:
            upper = max(self.max_a_b * self.max_a_b, 1)
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"