from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DeltaMinPopcountEnv(Env):
    """Single-turn environment for computing the minimum popcount of n XOR (n + d) for a given binary d."""

    def __init__(
        self,
        digit_num: int = 1,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            digit_num: The number of bits of the binary string d (must be >= 1).
        """
        super().__init__()
        assert isinstance(digit_num, int), "digit_num must be an integer"
        assert digit_num >= 1, "digit_num should be greater than or equal to 1"
        self.digit_num = digit_num

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.binary_string: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving bitwise XOR popcount minimization problems.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate a binary string of length digit_num, starting with '1'
        self.binary_string = "1" + "".join(str(random.randint(0, 1)) for _ in range(self.digit_num - 1))

        # Compute the reference answer using the original algorithm
        S = self.binary_string[::-1] + "00"
        cur = 0
        ans = 0
        for i in range(len(S) - 1):
            x = int(S[i])
            if x != cur:
                ans += 1
                cur = 1 if S[i + 1] == "1" else 0

        self.reference_answer = ans

        # Build the problem statement
        self.current_problem = (
            "Define popcount(x) as the number of 1s in the binary representation of a non-negative integer x. "
            "For example, popcount(5) = 2 because (5)_10 = (101)_2.\n\n"
            f"You are given a binary number d = ({self.binary_string})_2 (i.e., the base-2 representation of a decimal integer d). "
            "Please compute the minimum value of popcount(n XOR (n + d)) over all non-negative integers n, "
            "where XOR denotes the bitwise exclusive OR operation.\n\n"
            "Output Format: Your final answer should be a single base-10 integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        if self.reference_answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "no_problem"}

        parsed = self._parse_answer(action)
        if parsed is None:
            # Format error: no \\boxed{...} found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(parsed)
        except ValueError:
            # Not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "binary_string": self.binary_string,
        }

        return TERMINAL_STATE, reward, True, False, info

    def sample_random_action(self) -> str:
        """Sample a random action (answer) in \\boxed{...} format."""
        # The minimum popcount is typically small; sample a small integer as a guess
        random_answer = random.randint(0, self.digit_num + 2)
        return f"\\boxed{{{random_answer}}}"