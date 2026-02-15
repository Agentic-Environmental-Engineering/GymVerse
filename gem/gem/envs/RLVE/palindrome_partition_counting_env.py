from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PalindromePartitionCountingEnv(Env):
    """Environment for counting palindrome partitionings of a binary string - single turn Q&A.

    The task: Given a binary string S, count the number of ways to partition S into non-empty palindromic substrings.
    The agent must output the final integer answer in \\boxed{...} format.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        max_n: int = 30,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
        - N: Optional fixed length of the string S. If provided, must be >= 2.
        - max_n: Maximum length for random generation when N is not provided (>= 2).
        - wrong_format, rewarding_strategy, rewarding_weight, rewarding_beta:
          Preserved from the original environment as configurable parameters, but not used for reward calculation here,
          since GEM requires fixed reward settings.
        """
        super().__init__()
        if N is not None and N < 2:
            raise ValueError("N should be greater than or equal to 2")
        if max_n < 2:
            raise ValueError("max_n should be greater than or equal to 2")

        self.N = N
        self.max_n = max_n

        # Preserved parameters from the original environment (not used for reward calculation)
        self.wrong_format = wrong_format
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # State variables
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_string: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a palindrome partition counting problem.\n"
            "Given a binary string S, count the number of ways to partition S into non-empty palindromic substrings.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem, and return the observation."""
        super().reset(seed)

        # Determine length N
        if self.N is not None:
            N = self.N
        else:
            N = random.randint(2, self.max_n)

        # Generate a random binary string with bias for zeros
        zero_probability = random.randint(1, 9) / 10.0
        S = "".join("01"[random.random() < zero_probability] for _ in range(N))

        # Compute the number of palindrome partitions using DP
        # dpF[i] = number of ways to partition S[:i] into palindromic substrings
        dpF = [1] + [0] * N
        for i in range(1, N + 1):
            for j in range(i):
                substr = S[j:i]
                if substr == substr[::-1]:
                    dpF[i] += dpF[j]

        self.current_string = S
        self.reference_answer = dpF[N]

        # Build problem statement
        self.current_problem = (
            f"Please count the number of ways to partition the string '{S}' into non-empty palindromic substrings.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {"string": S, "n": N}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Additional validity: counts should be positive integers
        if user_answer <= 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "non_positive_answer"}

        # Check correctness
        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "string": self.current_string,
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
        """Sample a random action (random integer answer in boxed format)."""
        # Sample a random integer; upper bound chosen arbitrarily
        random_answer = random.randint(1, 1000)
        return f"\\boxed{{{random_answer}}}"