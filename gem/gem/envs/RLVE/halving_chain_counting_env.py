from typing import Any, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class HalvingChainCountingEnv(Env):
    """
    Halving Chain Counting problem environment - single-turn Q&A.

    Task:
    - Given an integer N, count the number of valid sequences constructed as follows:
      1) A sequence containing only the single number N is valid.
      2) Given any valid sequence, you can append a positive integer to the end, but
         the new number must be at most half of the last number in the current sequence.

    Answer format:
    - The final answer must be a single integer in \\boxed{...} format.
    """

    def __init__(
        self,
        max_n: int = 1000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: Maximum possible value for N (must be >= 1).
        """
        super().__init__()
        if max_n < 1:
            raise ValueError("max_n should be greater than or equal to 1")
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving halving chain counting problems.\n"
            "Please provide your final answer as a single integer enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Args:
            seed: Optional random seed.

        Returns:
            observation: The problem description as a string.
            info: An empty info dict.
        """
        super().reset(seed)

        # Generate N
        self.N = random.randint(1, self.max_n)

        # Build problem statement
        self.current_problem = (
            "Construct sequences based on the following rules:\n\n"
            f"1. A sequence that contains only a single number {self.N} is considered a valid sequence.\n"
            "2. Given any valid sequence, you can create a new valid sequence by appending a positive integer to the end — "
            "but the new number must be at most half of the last number in the current sequence (i.e., ≤ last_element / 2).\n\n"
            "Your task is to determine how many distinct valid sequences can be constructed following these rules.\n\n"
            "Output Format:\n"
            "Your answer should be a single integer in \\boxed{...}.\n"
            "Example: \\boxed{10}"
        )

        # Compute reference answer using dynamic programming
        self.reference_answer = self._count_sequences(self.N)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer.

        Args:
            action: The agent's answer text.

        Returns:
            observation: TERMINAL_STATE (since the environment is single-turn).
            reward: 1.0 if correct, 0.0 if wrong, -0.1 if format error.
            terminated: Always True.
            truncated: Always False.
            info: Additional information about the result.
        """
        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(answer_str.strip())
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Non-positive integers are considered format error (invalid answer domain)
        if user_answer <= 0:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Compare with reference
        is_correct = (self.reference_answer == user_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _count_sequences(self, n: int) -> int:
        """Dynamic programming to count valid sequences for a given n."""
        dp = [0] * (n + 1)
        for x in range(1, n + 1):
            dp[x] = 1
            # Sum over all y <= x // 2
            for y in range(1, x // 2 + 1):
                dp[x] += dp[y]
        return dp[n]

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # Random positive integer; not necessarily correct
        random_answer = random.randint(1, max(2, (self.reference_answer or 10) * 2))
        return f"\\boxed{{{random_answer}}}"