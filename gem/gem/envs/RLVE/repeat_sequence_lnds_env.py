from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RepeatSequenceLNDSEnv(Env):
    """Environment for the Longest Non-Decreasing Subsequence problem on a repeated sequence."""

    def __init__(
        self,
        n: int = 5,
        max_t: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - n: Length of the initial pattern (must be at least 2).
        - max_t: Maximum number of repetitions T (must be at least 2). Actual T is sampled uniformly from [2, max_t].
        """
        super().__init__()
        self.n: int = n
        self.max_t: int = max_t

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Parameters for the current instance
        self.T: Optional[int] = None
        self.a: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a longest non-decreasing subsequence problem on a repeated sequence.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Validate parameters
        assert isinstance(self.n, int) and self.n >= 2, "n must be an integer and at least 2"
        assert isinstance(self.max_t, int) and self.max_t >= 2, "max_t must be an integer and at least 2"

        # Sample T
        self.T = random.randint(2, self.max_t)

        # Generate the initial array of length n with values in [1, n]
        self.a = [random.randint(1, self.n) for _ in range(self.n)]

        # Compute the reference answer
        self.reference_answer = self._calculate_longest_nds(self.a, self.n, self.T)

        # Build the problem description
        total_length = self.n * self.T
        example_text = (
            "For example, if the initial pattern is [1, 3, 2] and it repeats 2 times, "
            "the full array would be [1, 3, 2, 1, 3, 2]."
        )
        self.current_problem = (
            f"You are given an array that repeats every {self.n} elements. "
            f"The initial pattern is: {self.a}. This pattern repeats {self.T} times, "
            f"creating a total array length of {total_length}.\n\n"
            f"{example_text}\n\n"
            "Find the length of the longest non-decreasing subsequence (not necessarily contiguous) in this repeated array.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def _calculate_longest_nds(self, a: List[int], n: int, T: int) -> int:
        """
        Calculate the longest non-decreasing subsequence using the provided algorithm.
        Source: https://codeforces.com/contest/582/submission/282761264
        """
        max_val = max(a)
        s = [0] * (max_val + 1)
        d = [0] * (max_val + 1)

        # Count frequencies in the base pattern
        for val in a:
            d[val] += 1

        # Process up to min(T, 2 * n) repetitions
        for val in a * min(T, 2 * n):
            s[val] = max(s[:val + 1]) + 1

        # Extend with full repetitions of the most frequent element if applicable
        return max(s) + max((T - n * 2) * max(d), 0)

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided answer and terminate the episode."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "n": self.n,
            "T": self.T,
            "a": self.a,
            "max_t": self.max_t
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
        """Sample a random action by generating a plausible boxed integer."""
        # The LNDS length is at most the total number of elements n * T
        upper = self.n * (self.T if self.T is not None else max(2, self.max_t))
        random_answer = random.randint(0, max(1, upper))
        return f"\\boxed{{{random_answer}}}"