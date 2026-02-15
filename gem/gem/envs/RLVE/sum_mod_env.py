from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumMODEnv(Env):
    """Environment for computing sum of (N mod i) × (M mod j) over all pairs (i, j) with i != j."""

    def __init__(self, max_n_m: int = 1000000, **kwargs) -> None:
        """
        Initialize the SumMODEnv environment.

        Args:
            max_n_m: The maximum value for N and M (must be >= 3).
        """
        super().__init__()
        if max_n_m < 3:
            raise ValueError("max_n_m should be greater than or equal to 3")
        self.max_n_m: int = max_n_m

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory summation problem.\n"
            "Please provide your final answer inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem and compute the reference answer."""
        super().reset(seed)

        # Generate parameters
        self.N = random.randint(3, self.max_n_m)
        self.M = random.randint(3, self.max_n_m)

        # Build problem prompt
        self.current_problem = (
            f"Please compute the sum of (N mod i) × (M mod j) over all pairs of integers (i, j) such that:\n"
            f"- 1 ≤ i ≤ {self.N}\n"
            f"- 1 ≤ j ≤ {self.M}\n"
            f"- i ≠ j\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._solve(self.N, self.M)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the outcome."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)

        if answer_str is None:
            # Format error: missing or invalid boxed answer
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate integer
        try:
            user_answer = int(answer_str)
        except ValueError:
            # Not an integer, considered invalid answer (no format penalty)
            info = {
                "error": "invalid_answer",
                "reference_answer": self.reference_answer,
                "user_answer": answer_str,
                "N": self.N,
                "M": self.M,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Negative answers are considered format errors to align with original logic
        if user_answer < 0:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error_negative"}

        assert self.reference_answer is not None
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        random_answer = random.randint(0, 10**6)
        return f"\\boxed{{{random_answer}}}"

    @staticmethod
    def _sum1(l: int, r: int) -> int:
        """Compute sum_{k=l..r} k."""
        return (l + r) * (r - l + 1) // 2

    @staticmethod
    def _sum2(x: int) -> int:
        """Compute sum_{k=1..x} k^2."""
        return x * (x + 1) * (2 * x + 1) // 6

    def _calc(self, n: int) -> int:
        """Compute sum_{i=1..n} (n mod i) in O(sqrt(n)) time."""
        res, l = 0, 1
        while l <= n:
            q = n // l
            r = n // q
            res += n * (r - l + 1) - self._sum1(l, r) * q
            l = r + 1
        return res

    def _solve(self, n: int, m: int) -> int:
        """
        Compute sum of (n mod i) × (m mod j) over all 1 ≤ i ≤ n, 1 ≤ j ≤ m, i ≠ j.
        This preserves the original algorithmic logic from the RLVE environment.
        """
        if n > m:
            n, m = m, n

        ans = self._calc(n) * self._calc(m)

        l = 1
        while l <= n:
            nd, md = n // l, m // l
            r = min(n // nd, m // md)

            cnt = r - l + 1
            SUM = n * m * cnt
            Sum = nd * md * (self._sum2(r) - self._sum2(l - 1))
            SUMK = (nd * m + md * n) * self._sum1(l, r)
            ans -= (SUM + Sum - SUMK)
            l = r + 1

        return ans