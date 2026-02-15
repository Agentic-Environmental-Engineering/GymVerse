import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SubsetSumSequenceEnv(Env):
    """Environment for the sequence formed by finite sums of distinct powers of K - single-turn QA.

    The sequence is constructed by taking all finite sums of distinct powers of K and sorting them
    in increasing order (1-based indexing). The task is to compute the N-th term of this sequence.
    """

    def __init__(
        self,
        max_k: int = 1000,
        max_n: int = 1000000,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
            max_k: The maximum value for K (must be >= 2).
            max_n: The maximum value for N (must be >= 1).
        """
        super().__init__()
        if max_k < 2:
            raise ValueError("max_k should be greater than or equal to 2")
        if max_n < 1:
            raise ValueError("max_n should be greater than or equal to 1")

        self.max_k = max_k
        self.max_n = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.K: Optional[int] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a sequence problem defined by finite sums of distinct powers of K.\n"
            "Please provide your final answer in \\boxed{...} format. The boxed content should be a single decimal integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: The instruction string followed by the problem statement.
            info: An empty dict.
        """
        super().reset(seed)

        # Generate problem parameters
        self.N = random.randint(1, self.max_n)
        self.K = random.randint(2, self.max_k)

        # Build problem prompt with example terms
        term_0 = 1
        term_1 = self.K
        term_2 = 1 + self.K
        term_3 = self.K ** 2

        prompt = (
            f"Consider all powers of {self.K}, and all finite sums of distinct powers of {self.K}. "
            f"Collect these numbers and sort them in increasing order (starting from index 1) to form a sequence:\n"
            f"{term_0}, {term_1}, {term_2}, {term_3}, ...\n\n"
            f"Your task is to compute the value of the {self.N}-th term in this sequence (1-based indexing), "
            f"and output it in decimal (base 10).\n\n"
            f"Output Format: Your final answer should be a single decimal number in \\boxed{{...}}.\n"
            f"Example: \\boxed{{{self.K}}}\n"
        )
        self.current_problem = prompt

        # Compute the reference answer by mapping N's binary digits to powers of K
        self.reference_answer = self._compute_reference_answer(self.K, self.N)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, k: int, n: int) -> int:
        """Compute the N-th term of the sequence using the original algorithm."""
        ans = 0
        base = 1
        while n:
            if n & 1:
                ans += base
            n //= 2
            base *= k
        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step to verify the submitted answer.

        Returns:
            observation: TERMINAL_STATE
            reward: 1.0 if correct, 0.0 if incorrect, -0.1 if format error
            terminated: True (single-turn environment)
            truncated: False
            info: Additional information dictionary
        """
        # Parse answer from \boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate if the parsed content is an integer
        try:
            user_answer = int(parsed.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Additional checks following the original verification logic
        if user_answer <= 0:
            info = {
                "error": "non_positive",
                "user_answer": user_answer,
                "reference_answer": self.reference_answer,
                "K": self.K,
                "N": self.N,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check that the number contains only digits 0 and 1 in base K
        if not self._valid_base_k_binary(user_answer, self.K):
            info = {
                "error": "invalid_representation",
                "user_answer": user_answer,
                "reference_answer": self.reference_answer,
                "K": self.K,
                "N": self.N,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compare with reference answer
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "user_answer": user_answer,
            "reference_answer": self.reference_answer,
            "K": self.K,
            "N": self.N,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _valid_base_k_binary(self, num: int, k: int) -> bool:
        """Check whether num in base-k representation contains only digits 0 and 1."""
        if num < 0:
            return False
        while num:
            digit = num % k
            if digit not in (0, 1):
                return False
            num //= k
        return True

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action in \\boxed{...} format.

        Generates a random number that is a sum of distinct powers of K to satisfy the representation check.
        """
        k = self.K if self.K is not None else 2
        # Choose a random length for the sum of powers
        length = random.randint(1, 10)
        # Randomly include each power from 0 to length-1
        value = 0
        for i in range(length):
            if random.choice([True, False]):
                value += k ** i
        # Ensure it's positive
        if value <= 0:
            value = 1
        return f"\\boxed{{{value}}}"