import random
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NegativeBaseEnv(Env):
    """Negative base representation environment - single turn Q&A.

    Task:
        Convert a given decimal integer N (N != 0) into base -R (R >= 2),
        using digits 0..R-1, and output the digits from most significant
        to least significant, separated by spaces, inside \\boxed{...}.
    """

    def __init__(
        self,
        max_n: int = 1_000_000,
        max_r: int = 10,
        **kwargs: Any,
    ):
        """Initialize the environment.

        Args:
            max_n: Maximum absolute value for N (N is sampled from [-max_n, max_n] excluding 0). Must be >= 1.
            max_r: Maximum base magnitude R (R is sampled from [2, max_r]). Must be >= 2.
            **kwargs: Extra arguments reserved for future use.

        Raises:
            AssertionError: If max_n < 1 or max_r < 2.
        """
        super().__init__()
        assert max_n >= 1, "max_n should be greater than or equal to 1"
        assert max_r >= 2, "max_r should be greater than or equal to 2"
        self.max_n = max_n
        self.max_r = max_r

        # Internal state for the current episode
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[List[int]] = None
        self.N: Optional[int] = None
        self.R: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving negative base representation problems.\n"
            "We can represent integers using a negative base system with base -R, where R is a positive integer greater than 1. "
            "In this system, the digits used are from 0 to R - 1 (in decimal).\n"
            "For example, the decimal number -15 can be represented as 110001 in base -2, since:\n"
            "1×(-2)^5 + 1×(-2)^4 + 0×(-2)^3 + 0×(-2)^2 + 0×(-2)^1 + 1×(-2)^0 = -15.\n\n"
            "Answer format:\n"
            "- Provide your final answer inside \\boxed{...}.\n"
            "- Inside the box, list the digits (in decimal) from most significant to least significant, separated by single spaces.\n"
            "Example: \\boxed{3 0 1} means 3 * (-R)^2 + 0 * (-R)^1 + 1 * (-R)^0.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Args:
            seed: Optional random seed.

        Returns:
            observation: The task instructions and the specific problem statement.
            info: Additional information dictionary (empty for this environment).
        """
        super().reset(seed)

        # Sample N (non-zero) and R
        N = 0
        while N == 0:
            N = random.randint(-self.max_n, self.max_n)
        R = random.randint(2, self.max_r)

        # Compute the gold answer: digits for base -R representation
        digits = self._convert_to_negative_base(N, -R)
        self._validate_representation(N, R, digits)

        # Store state
        self.N = N
        self.R = R
        self.gold_answer = digits
        self.reference_answer = " ".join(map(str, digits))

        # Build problem description
        problem = (
            f"We can represent integers using a negative base system with base -R.\n"
            f"Convert the decimal number {N} into base -{R}, and output its digits (in decimal) from most significant to least significant.\n\n"
            f"Output Format:\n"
            f"Your final answer should be the digits separated by spaces inside \\boxed{{...}}.\n"
            f"For example: \\boxed{{{R - 1} 0 1}} means {R - 1} * (-{R})^2 + 0 * (-{R})^1 + 1 * (-{R})^0 in decimal."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take one step by submitting an answer.

        Args:
            action: The model's output text, which should contain \\boxed{...} with the answer.

        Returns:
            observation: TERMINAL_STATE (single-turn task).
            reward: 1.0 if correct; 0.0 if wrong; -0.1 if format error.
            terminated: True (single-turn task).
            truncated: False.
            info: Additional info including correctness and reference answer.
        """
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse digits from boxed content
        tokens = boxed.strip().split()
        try:
            user_digits = [int(tok) for tok in tokens]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate against gold answer (exact match)
        is_correct = (self.gold_answer is not None and user_digits == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, user_digits)),
            "N": self.N,
            "R": self.R,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re

        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _convert_to_negative_base(n: int, r: int) -> List[int]:
        """Convert integer n to base r (where r is negative), returning the digit list (MSD to LSD).

        This uses a recursive algorithm ensuring digits are non-negative and less than |r|.
        """
        if n == 0:
            return []
        m = n % r
        if m < 0:
            m -= r
            n += r
        return NegativeBaseEnv._convert_to_negative_base(n // r, r) + [m]

    @staticmethod
    def _validate_representation(n: int, R: int, digits: List[int]) -> None:
        """Validate that the provided digits reconstruct n in base -R."""
        assert R >= 2, "R must be at least 2"
        assert len(digits) > 0, "Representation should not be empty for non-zero n"

        total = 0
        base = -R
        for d in digits:
            assert 0 <= d < R, f"Digit {d} is out of range [0, {R - 1}]"
            total = total * base + d
        assert total == n, f"Reconstruction failed: {total} != {n}"

    def sample_random_action(self) -> str:
        """Sample a random action in valid format (may or may not be correct)."""
        # Provide the correct answer with 50% probability; otherwise random digits
        if self.reference_answer is not None and random.random() < 0.5:
            return f"\\boxed{{{self.reference_answer}}}"

        # Random digits: choose a random length around the gold length (if available), else 1..8
        if self.gold_answer is not None:
            length = max(1, len(self.gold_answer) + random.choice([-1, 0, 1]))
            R = self.R if self.R is not None else 10
        else:
            length = random.randint(1, 8)
            R = 10

        digits = [str(random.randint(0, max(1, R - 1))) for _ in range(length)]
        return f"\\boxed{{{ ' '.join(digits) }}}"