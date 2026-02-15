from typing import Any, Optional, SupportsFloat, Tuple, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NumberPartitionCountingEnv(Env):
    """Environment for counting the number of distinct partitions of N into exactly K positive integers (order does not matter).
    
    Single-turn question-answer environment.
    """

    def __init__(self, max_n: int = 100, **kwargs) -> None:
        """Initialize the environment.

        Args:
            max_n: Maximum value for N (N will be sampled uniformly from [1, max_n]).
        """
        super().__init__()
        if max_n < 1:
            raise ValueError("max_n should be greater than or equal to 1")
        self.max_n: int = max_n

        # State variables
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number partition counting problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Args:
            seed: Random seed.

        Returns:
            observation: The problem description.
            info: Additional info dict (empty).
        """
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(1, self.max_n)
        K = random.randint(1, N)

        # Compute reference answer using dynamic programming
        reference_answer = self._count_partitions_exact_k(N, K)

        self.N = N
        self.K = K
        self.reference_answer = reference_answer

        # Build problem statement
        self.current_problem = (
            f"You are given a positive integer N = {N}. Your task is to divide it into exactly K = {K} non-empty positive integers such that:\n"
            "- The sum of the K parts is exactly N,\n"
            "- The order does not matter — partitions that are permutations of the same multiset are considered the same,\n"
            "- All parts must be strictly positive integers (no zero).\n\n"
            "Determine how many distinct ways there are to partition the number N into K such parts.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the provided answer.

        Args:
            action: The agent's answer text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE (single-turn).
            reward: 1.0 if correct, 0.0 if incorrect, -0.1 if format error.
            terminated: Always True (single-turn).
            truncated: Always False.
            info: Additional info including correctness and reference answer.
        """
        answer_str = self._parse_answer(action)

        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not reset or reference answer missing."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}.

        Args:
            text: The agent's response text.

        Returns:
            The last captured string inside \\boxed{...}, or None if not found.
        """
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _count_partitions_exact_k(self, N: int, K: int) -> int:
        """Count the number of partitions of N into exactly K positive integers (order does not matter).

        This matches the original DP logic from the RLVE environment.
        """
        if K < 0 or N < 0:
            return 0
        if K == 0:
            return 1 if N == 0 else 0
        if K > N:
            return 0

        dpF = [[0 for _ in range(K + 1)] for _ in range(N + 1)]
        # Set base conditions as in the original environment
        dpF[0][0] = 1
        for i in range(1, N + 1):
            dpF[i][1] = 1
            dpF[i][0] = 1

        for i in range(2, N + 1):
            for x in range(2, K + 1):
                if i > x:
                    dpF[i][x] = dpF[i - 1][x - 1] + dpF[i - x][x]
                else:
                    dpF[i][x] = dpF[i - 1][x - 1]

        return dpF[N][K]

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # If reference answer is available, sample near it; otherwise sample a small random integer
        upper = max(10, (self.reference_answer or 10) * 2 + 5)
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"