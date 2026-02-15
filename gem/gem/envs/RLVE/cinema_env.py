from typing import Any, Optional, SupportsFloat, Tuple
import math
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CinemaEnv(Env):
    """Cinema seating probability environment - single-turn Q&A."""

    def __init__(
        self,
        max_n_k: int = 1000,
        **kwargs
    ):
        """
        Initialize the CinemaEnv instance.

        Parameters:
        - max_n_k: The maximum value for N and K. Must be >= 2.
        """
        super().__init__()
        assert max_n_k >= 2, "max_n_k should be greater than or equal to 2"
        self.max_n_k = max_n_k

        # Internal state
        self.n: Optional[int] = None
        self.k: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[Tuple[int, int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a probability problem about seating in a cinema.\n"
            "Please provide your final answer as two integers 'A B' inside \\boxed{...},\n"
            "where A/B is the probability in reduced form (i.e., gcd(A, B) = 1).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameters
        self.n = random.randint(2, self.max_n_k)
        self.k = random.randint(self.n, self.max_n_k)
        assert self.n <= self.k, "N should be less than or equal to K"

        # Build problem statement
        self.current_problem = (
            f"There are {self.n} people entering a cinema and {self.k} numbered seats labeled from 1 to {self.k}.\n\n"
            f"Each person, in order from 1 to {self.n}, independently picks a random integer L from 1 to {self.k}, uniformly at random.\n"
            f"- If seat L is unoccupied, they take it.\n"
            f"- If it's taken, they try seat L + 1, then L + 2, ..., up to seat {self.k}, until they find a free seat.\n"
            f"- If all seats from L to {self.k} are occupied, the person must stand.\n\n"
            f"Please compute the probability that all {self.n} people get a seat (i.e., no one ends up standing).\n"
            f"Output Format: Provide your final answer as a reduced fraction 'A B' inside \\boxed{{...}}, where A/B is the probability and gcd(A, B) = 1."
        )

        # Compute reference answer as reduced fraction (A, B)
        ans_num = ((self.k + 1) ** (self.n - 1)) * (self.k - self.n + 1)
        ans_den = self.k ** self.n
        g = math.gcd(ans_num, ans_den)
        ans_num //= g
        ans_den //= g

        self.gold_answer = (ans_num, ans_den)
        self.reference_answer = f"{ans_num} {ans_den}"

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error: no boxed answer found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse two integers A and B from boxed content
        parts = boxed_content.strip().split()
        if len(parts) != 2:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            a = int(parts[0])
            b = int(parts[1])
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Verify correctness
        assert self.gold_answer is not None, "Gold answer not computed. Call reset() before step()."
        is_correct = (a, b) == self.gold_answer
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": (a, b),
            "N": self.n,
            "K": self.k,
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

    def sample_random_action(self) -> str:
        """Sample a random action in the expected \\boxed{A B} format."""
        # Random small integers to form a plausible fraction
        a = random.randint(0, 10**6)
        b = random.randint(1, 10**6)
        return f"\\boxed{{{a} {b}}}"