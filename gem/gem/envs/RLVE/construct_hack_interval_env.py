import random
import re
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ConstructHackIntervalEnv(Env):
    """Environment for constructing an interval [L, R] such that the sum of digit sums over the interval is divisible by MOD."""

    def __init__(
        self,
        max_mod: int = 1000000,
        **kwargs
    ):
        """
        Initialize the ConstructHackIntervalEnv instance.

        Args:
            max_mod: The maximum value for MOD (must be >= 1).
        """
        super().__init__()
        if max_mod < 1:
            raise ValueError("max_mod should be greater than or equal to 1")
        self.max_mod: int = max_mod

        self.MOD: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the general task instructions."""
        return (
            "You are solving a constructive number theory problem.\n"
            "Provide your final answer in \\boxed{L R} format, where L and R are two positive integers separated by a space.\n"
            "Do not include any additional text outside the boxed answer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate MOD parameter
        self.MOD = random.randint(1, self.max_mod)

        # Build the problem description
        self.current_problem = (
            "Let's define f(x) as the sum of digits in the decimal representation of number x "
            "(for example, f(1234) = 1 + 2 + 3 + 4). Please construct an interval [L, R], such that "
            f"the sum of f(x) for all x in the interval is divisible by {self.MOD}.\n"
            "Note that L and R should be both positive integers, L should be less than or equal to R, "
            f"and R should be less than or equal to 10 * {self.MOD}.\n\n"
            "Output Format: Your final answer should be two integers in \\boxed{L R}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        parsed = self._parse_answer(action)
        if parsed is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        L, R = parsed
        MOD = self.MOD if self.MOD is not None else 1

        # Validate constraints
        if not (1 <= L <= R and R <= 10 * MOD):
            info = {
                "error": "invalid_solution",
                "L": L,
                "R": R,
                "modulo": MOD,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute digit sum for [L, R]
        digit_sum = self._count_digit_sum(L, R)
        is_correct = (digit_sum % MOD == 0)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "L": L,
            "R": R,
            "digit_sum": digit_sum,
            "modulo": MOD,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[Tuple[int, int]]:
        """Extract two integers L and R from \\boxed{...} format."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if not matches:
            return None
        content = matches[-1].strip()
        parts = content.split()
        if len(parts) != 2:
            return None
        try:
            L, R = int(parts[0]), int(parts[1])
            return L, R
        except Exception:
            return None

    def _count_digit_sum(self, L: int, R: int) -> int:
        """Compute the total sum of digits for all numbers in the interval [L, R]."""

        def count_digits_up_to(n: int) -> int:
            """
            Count the sum of digits of all numbers in the interval [0, n].
            """
            if n < 0:
                return 0
            if n < 10:
                return sum(range(1, n + 1))

            digits = len(str(n))
            total = 0
            first_digit = int(str(n)[0])
            remaining = int(str(n)[1:]) if len(str(n)) > 1 else 0

            # Contribution from numbers with fewer digits and from full blocks by first_digit
            total += (digits - 1) * 45 * (10 ** (digits - 2)) * first_digit
            total += first_digit * (first_digit - 1) // 2 * (10 ** (digits - 1))

            # Contribution from remaining part
            total += count_digits_up_to(remaining) + first_digit * (remaining + 1)

            return total

        return count_digits_up_to(R) - count_digits_up_to(L - 1)

    def sample_random_action(self) -> str:
        """Sample a random (not necessarily correct) action in boxed format."""
        MOD = self.MOD if self.MOD is not None else max(1, self.max_mod // 2)
        R = random.randint(1, 10 * MOD)
        L = random.randint(1, R)
        return f"\\boxed{{{L} {R}}}"