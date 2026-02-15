from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Sum_DivisorNumEnv(Env):
    """Environment for computing the sum of the number of divisors over a range [L, R]."""

    def __init__(
        self,
        max_r: int = 1000000,
        **kwargs
    ):
        """
        Initialize the Sum_DivisorNumEnv instance.

        Parameters:
        - max_r: The maximum value for R in the range [L, R]. Must be >= 2.
        """
        super().__init__()
        if max_r < 2:
            raise ValueError("max_r should be greater than or equal to 2")
        self.max_r = max_r

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.L: Optional[int] = None
        self.R: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving divisor-count summation problems.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameters
        R = random.randint(2, self.max_r)
        L = random.randint(1, R)
        self.L = L
        self.R = R

        # Compute reference answer using the efficient summatory function
        self.reference_answer = self._sum_divisor_count_prefix(R) - self._sum_divisor_count_prefix(L - 1)

        # Build problem description
        self.current_problem = (
            f"Please compute sum(d(i)) for all integers i such that {L} ≤ i ≤ {R}. "
            f"Here, d(i) denotes the number of positive divisors of the integer i.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {"L": L, "R": R}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)

        if answer_text is None:
            # Format error: no boxed answer found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and check correctness
        try:
            user_answer = int(answer_text.strip())
            is_correct = (user_answer == self.reference_answer)
            reward = 1.0 if is_correct else 0.0
        except ValueError:
            # Answer is not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "L": self.L,
            "R": self.R
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _sum_divisor_count_prefix(self, n: int) -> int:
        """
        Compute S(n) = sum_{i=1..n} floor(n / i), which equals sum of d(k) for k=1..n.
        Uses the O(sqrt(n)) grouping technique.
        """
        if n <= 0:
            return 0
        total = 0
        l = 1
        while l <= n:
            val = n // l
            r = n // val
            total += val * (r - l + 1)
            l = r + 1
        return total

    def sample_random_action(self) -> str:
        """Sample a random answer formatted in \\boxed{...}."""
        upper = self.reference_answer if self.reference_answer is not None else self.max_r
        random_answer = random.randint(0, max(1, upper))
        return f"\\boxed{{{random_answer}}}"