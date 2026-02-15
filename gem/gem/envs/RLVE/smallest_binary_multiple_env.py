import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SmallestBinaryMultipleEnv(Env):
    """Environment for finding the smallest positive integer B such that A × B has only digits 0 and 1."""

    def __init__(
        self,
        max_a: int = 1000000,
        a: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_a: Upper bound for random A generation (must be >= 2).
        - a: Optional fixed value of A. If provided, must satisfy 2 <= a <= max_a.
        """
        super().__init__()
        if max_a < 2:
            raise ValueError("max_a should be greater than or equal to 2")
        self.max_a = max_a
        self.fixed_a = a
        if self.fixed_a is not None and not (2 <= self.fixed_a <= self.max_a):
            raise ValueError("a must be between 2 and max_a inclusive")

        self.current_problem: Optional[str] = None
        self.A: Optional[int] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Find the smallest positive integer B such that A × B contains only digits 0 and 1 in decimal.\n"
            "Output Format: Provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Choose A
        if self.fixed_a is not None:
            self.A = self.fixed_a
        else:
            self.A = random.randint(2, self.max_a)

        # Build problem statement
        self.current_problem = (
            f"Find the smallest positive integer B such that the product {self.A} × B contains only digits "
            f"0 and 1 in its decimal representation.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._solve_smallest_binary_multiple(self.A)

        obs = self._get_instructions() + self.current_problem
        return obs, {"A": self.A}

    def _solve_smallest_binary_multiple(self, A: int) -> int:
        """
        Solve for the smallest positive integer B such that A × B uses only digits 0 and 1.
        The algorithm performs a BFS over decimal digit positions, constructing multiples with digits {0,1}.
        """
        dp = {0: 0}
        cur_value = 1          # 10^k (a single '1' at the current digit position)
        cur_mod = 1 % A        # (10^k) mod A

        while True:
            new_states = []

            for remainder, value in dp.items():
                candidate = value + cur_value     # turn the current digit from 0 to 1
                new_remainder = (remainder + cur_mod) % A

                if new_remainder == 0:
                    # candidate is the first multiple of A that uses only 0/1 digits
                    B = candidate // A
                    return B

                if new_remainder not in dp:
                    new_states.append((new_remainder, candidate))

            for r, v in new_states:
                dp[r] = v

            cur_value *= 10
            cur_mod = (cur_mod * 10) % A

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the submitted answer."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check positivity
        if user_answer <= 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "non_positive"}

        # Check that A * B uses only digits 0 and 1
        A = self.A if self.A is not None else 0
        AB = A * user_answer
        if AB == 0:
            # This can only happen if user_answer == 0, which is disallowed above, but keep safeguard
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        valid_digits = True
        while AB:
            if AB % 10 not in (0, 1):
                valid_digits = False
                break
            AB //= 10

        if not valid_digits:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_digits"}

        # Compare with reference answer
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "A": self.A,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random positive integer answer)."""
        random_answer = random.randint(1, max(2, self.max_a))
        return f"\\boxed{{{random_answer}}}"