from typing import Any, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumPseudoEuclideanEnv(Env):
    """Sum of f(i, j) over a range problem environment - single-turn Q&A.

    The task is based on a pseudo-Euclidean recursive function:
        f(a, b):
            if a == b: return 0
            if a > b:  return f(a - b, 2*b) + 1
            else:      return f(2*a, b - a) + 1

    If the function enters an infinite loop, its return value is treated as 0.
    The environment asks for the sum of f(i, j) for 1 <= i <= N and 1 <= j <= N,
    where N is randomly chosen in reset().

    The final answer must be provided in \\boxed{...} format.
    """

    def __init__(
        self,
        max_n: int = 1_000_000,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        """
        Initialize the SumPseudoEuclideanEnv instance.

        Parameters:
        - max_n: maximum N used in problem generation (must be >= 5)
        - wrong_format, rewarding_strategy, rewarding_weight, rewarding_beta:
          preserved parameters from the original environment (not used for scoring here)
        """
        super().__init__()
        self.max_n = max_n
        assert self.max_n >= 5, "max_n should be greater than or equal to 5"

        # Preserved reward-related parameters (not used in GEM scoring as per requirements)
        self.rewards = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory problem involving a pseudo-Euclidean recursion.\n"
            "Please provide your final answer as a single integer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation (str): Combined instructions and problem statement.
            info (dict): Additional information (empty for this environment).
        """
        super().reset(seed)

        # Generate problem parameter N
        N = random.randint(5, self.max_n)
        self.current_N = N

        # Build problem statement
        self.current_problem = (
            "Consider the function f(a, b) defined in Python as follows:\n"
            "def f(a: int, b: int) -> int:\n"
            "    if a == b:\n"
            "        return 0\n"
            "    if a > b:\n"
            "        return f(a - b, b + b) + 1\n"
            "    else:\n"
            "        return f(a + a, b - a) + 1\n\n"
            "If the function enters an infinite loop, we treat its return value as 0.\n"
            f"Tell me the sum of f(i, j) over all pairs (i, j) such that 1 <= i <= {N} and 1 <= j <= {N}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer using the preserved algorithm
        self.reference_answer = self._solve(N)
        assert self.reference_answer is not None and self.reference_answer > 0, "Reference answer should be greater than 0"

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _solve(self, N: int) -> int:
        """Compute the sum using the number-theory decomposition from the original environment."""
        def count_odds(x: int, y: int) -> int:
            length = y - x + 1
            if (length & 1) and (x & 1):
                return (length >> 1) + 1
            else:
                return length >> 1

        def block_sum(l: int, k: int, N: int) -> int:
            total = 0
            while l <= k:
                lg = l.bit_length() - 1
                r = min((1 << (lg + 1)) - 1, k)
                total += lg * (N // l) * count_odds(l, r)
                l = r + 1
            return total

        ans = 0
        l = 1
        while l <= N:
            v = N // l
            r = N // v
            ans += block_sum(l, r, N)
            l = r + 1
        return ans * 2

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the submitted answer.

        Returns:
            observation (str): TERMINAL_STATE for single-turn environments.
            reward (float): 1.0 if correct, 0.0 if wrong, -0.1 for format error.
            terminated (bool): True, as this is a single-turn environment.
            truncated (bool): False, no truncation in single turn.
            info (dict): Contains correctness and reference/user answers or error details.
        """
        # Parse \\boxed{...} answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the boxed answer from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        upper = self.reference_answer if self.reference_answer is not None else self.max_n
        upper = max(10, min(upper * 2, 1_000_000_000))
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"