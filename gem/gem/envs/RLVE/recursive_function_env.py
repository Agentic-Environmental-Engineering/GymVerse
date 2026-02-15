import random
import re
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RecursiveFunctionEnv(Env):
    """Recursive function problem environment - single-turn Q&A.

    The function f(m, n) is defined as:
    1. If m = 0, then f(m, n) = n + 1.
    2. If m > 0 and n = 0, then f(m, n) = f(m - 1, 1).
    3. If m > 0 and n > 0, then
       f(m, n) = f(m // 2, f(m // 2, n // 2)) + f(m // 2, f(m // 2, n - 1)).
       Here, // denotes integer division.

    The task is to compute f(M, N) for randomly generated M and N.
    """

    def __init__(
        self,
        max_m_n: int = 10,
        correct_reward: float = 1.0,
        wrong_answer_reward: float = 0.0,
        format_error_reward: float = -0.1,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            max_m_n: Maximum value for both M and N (inclusive). Must be >= 1.
            correct_reward: Reward for a correct answer (default 1.0).
            wrong_answer_reward: Reward for an incorrect answer (default 0.0).
            format_error_reward: Reward for a formatting error (default -0.1).
        """
        super().__init__()
        assert max_m_n >= 1, "max_m_n should be greater than or equal to 1"

        # Parameters controlling problem generation and rewards
        self.max_m_n = max_m_n
        self.correct_reward = correct_reward
        self.wrong_answer_reward = wrong_answer_reward
        self.format_error_reward = format_error_reward

        # State variables
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.M: Optional[int] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a recursive function problem.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Args:
            seed: Optional random seed to make generation deterministic.

        Returns:
            A tuple of observation string and an info dict.
        """
        super().reset(seed)

        # Generate problem parameters
        self.M = random.randint(1, self.max_m_n)
        self.N = random.randint(1, self.max_m_n)

        # Build problem prompt
        self.current_problem = (
            "Define a function f(m, n) as follows:\n"
            "1. If m = 0, then f(m, n) = n + 1.\n"
            "2. If m > 0 and n = 0, then f(m, n) = f(m - 1, 1).\n"
            "3. If m > 0 and n > 0, then f(m, n) = f(m // 2, f(m // 2, n // 2)) "
            "+ f(m // 2, f(m // 2, n - 1)). Here, `//` denotes integer division.\n\n"
            f"Please compute the value of f({self.M}, {self.N}).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer using memoized recursion
        self.reference_answer = self._compute_reference(self.M, self.N)

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "M": self.M,
            "N": self.N,
        }
        return obs, info

    def _compute_reference(self, m: int, n: int) -> int:
        """Compute the reference answer using the specified recursive function."""
        memo: dict[tuple[int, int], int] = {}

        def ack(x: int, y: int) -> int:
            if x == 0:
                return y + 1
            if (x, y) not in memo:
                if y == 0:
                    memo[(x, y)] = ack(x - 1, 1)
                else:
                    memo[(x, y)] = ack(x // 2, ack(x // 2, y // 2)) + ack(x // 2, ack(x // 2, y - 1))
            return memo[(x, y)]

        return ack(m, n)

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step by verifying the submitted answer.

        Args:
            action: The agent's answer text, expected in \\boxed{...} format.

        Returns:
            A tuple containing:
            - observation: TERMINAL_STATE (since this is single-turn)
            - reward: float reward based on correctness/format
            - terminated: True (single-turn environment)
            - truncated: False
            - info: dict with verification details
        """
        # Parse the boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        # Verify that the parsed answer is an integer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, self.wrong_answer_reward, True, False, {"error": "invalid_answer"}

        # Compare with reference answer
        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = self.correct_reward if is_correct else self.wrong_answer_reward

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "M": self.M,
            "N": self.N,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer answer) in boxed format."""
        random_answer = random.randint(0, 100)
        return f"\\boxed{{{random_answer}}}"