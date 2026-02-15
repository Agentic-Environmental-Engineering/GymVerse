from typing import Any, Optional, SupportsFloat, Tuple
import random
import bisect
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumFibonacciRepresentationEnv(Env):
    """Environment for computing the minimum number of Fibonacci numbers (added or subtracted)
    needed to represent a given positive integer K. Single-turn Q&A.
    """

    def __init__(
        self,
        max_k: int = 1000,
        min_k: int = 4,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            max_k: Maximum possible value for K. Must be >= 10.
            min_k: Minimum possible value for K. Default is 4 (as in the original environment).
        """
        super().__init__()
        assert max_k >= 10, "max_k should be greater than or equal to 10"
        assert min_k >= 4, "min_k should be greater than or equal to 4"
        assert min_k <= max_k, "min_k should be less than or equal to max_k"

        self.max_k = max_k
        self.min_k = min_k

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_k: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: You can represent any positive integer by adding or subtracting Fibonacci numbers.\n"
            "We use the Fibonacci-like sequence starting with 1, 2, and then each next term is the sum of the previous two: 1, 2, 3, 5, 8, 13, 21, ...\n"
            "For a given integer K, compute the minimum number of Fibonacci numbers required (with + or - allowed) to represent K.\n"
            "Examples:\n"
            "- 10 = 5 + 5 → uses 2 Fibonacci numbers\n"
            "- 19 = 21 - 2 → uses 2 Fibonacci numbers\n"
            "- 17 = 13 + 5 - 1 → uses 3 Fibonacci numbers\n"
            "- 1070 = 987 + 89 - 5 - 1 → uses 4 Fibonacci numbers\n\n"
            "Output Format: Provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Args:
            seed: Optional random seed.

        Returns:
            observation: The problem statement including instructions.
            info: Additional information dictionary (empty for this environment).
        """
        super().reset(seed)

        # Sample K
        K = random.randint(self.min_k, self.max_k)
        self.current_k = K

        # Build the problem statement
        self.current_problem = (
            f"Define Fibonacci numbers as the sequence: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ... "
            f"You can represent any positive integer by adding or subtracting Fibonacci numbers.\n"
            f"Please compute the minimum number of Fibonacci numbers needed (added or subtracted) to represent the number {K}.\n"
            f"Output a single integer in \\boxed{{...}} — the minimum number of Fibonacci numbers used."
        )

        # Compute the reference answer
        self.reference_answer = self._compute_min_fib_terms(K)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer.

        Args:
            action: The agent's answer text, expected in \\boxed{...} format.

        Returns:
            observation: TERMINAL_STATE for single-turn setup.
            reward: 1.0 if correct; 0.0 if incorrect; -0.1 if format error.
            terminated: True (single-turn environment).
            truncated: False.
            info: Dictionary with validation details.
        """
        # Parse boxed answer
        answer_text = self._parse_answer(action)

        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
            is_correct = (user_answer == self.reference_answer)
            reward = 1.0 if is_correct else 0.0
        except ValueError:
            # Content inside boxed is not an integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "K": self.current_k,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the given text.

        Args:
            text: The input text.

        Returns:
            The content inside the last \\boxed{...} if present, else None.
        """
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_min_fib_terms(self, k: int) -> int:
        """Compute the minimal number of Fibonacci numbers (with + or -) to represent k.

        The Fibonacci-like sequence is defined as: 1, 2, 3, 5, 8, ...
        This method uses a greedy strategy of moving towards the nearest Fibonacci number.

        Args:
            k: The target integer.

        Returns:
            The minimal count of Fibonacci numbers needed.
        """
        # Build Fibonacci-like sequence up to just above k
        F = [1, 2]
        while F[-1] <= k:
            F.append(F[-2] + F[-1])

        res = 0
        n = k
        while n:
            res += 1
            idx = bisect.bisect_right(F, n)
            larger = F[idx]
            smaller = F[idx - 1]
            n = min(larger - n, n - smaller)
        return res

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        random_answer = random.randint(1, 10)
        return f"\\boxed{{{random_answer}}}"