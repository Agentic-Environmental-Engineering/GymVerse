from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PowerCycleEnv(Env):
    """Environment for analyzing cycles of the last K digits of powers of an integer N."""

    def __init__(self, digit_num: int = 6, **kwargs):
        """
        Initialize the PowerCycleEnv instance.

        Parameters:
        - digit_num: Maximum number of digits for N (N is in [1, 10^digit_num - 1]).
                     Also used as the upper bound for K (K in [1, digit_num]).
        """
        super().__init__()
        assert digit_num >= 1, "digit_num should be greater than or equal to 1"
        self.digit_num = digit_num

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are analyzing cycles of the last K digits (base-10) of positive powers of an integer N.\n"
            "Important Notes:\n"
            "1. If a power of N has fewer than K digits, consider the missing leading digits as 0 (pad with zeros from the left).\n"
            "2. If the cycle length is L, it means for every positive integer a, the last K digits of N^a are the same as those of N^(a+L).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem."""
        super().reset(seed)

        # Generate N, K and ensure the reference answer is valid (> 0)
        while True:
            self.N = random.randint(1, 10 ** self.digit_num - 1)
            self.K = random.randint(1, self.digit_num)
            self.reference_answer = self._solve_cycle(self.N, self.K)
            if self.reference_answer is not None and self.reference_answer > 0:
                break

        # Build the problem prompt
        self.current_problem = (
            "It is well known that the last digit of positive powers of 2 follows a repeating pattern: "
            "2, 4, 8, 6, 2, 4, 8, 6, ... . We say that the last digit of powers of 2 has a cycle length of 4.\n\n"
            f"Now, analyze positive powers of the integer N = {self.N} and determine whether the last K = {self.K} digits "
            "form a repeating cycle. If so, what is the minimum cycle length?\n\n"
            "Notes:\n"
            "1. If a power of N has fewer than K digits, pad with leading zeros to length K.\n"
            "2. If the cycle length is L, then for every positive integer a, the last K digits of N^a are the same as those of N^(a+L).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {"N": self.N, "K": self.K}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        # Extract the boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and score
        try:
            user_answer = int(answer_text.strip())
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
        except ValueError:
            # Answer is not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        random_answer = random.randint(1, 100)
        return f"\\boxed{{{random_answer}}}"

    def _solve_cycle(self, S: int, K: int) -> Optional[int]:
        """
        Compute the minimum cycle length for the last K digits of positive powers of S.
        Returns:
        - The cycle length as a positive integer if a cycle exists.
        - None if no valid cycle is found (original code used -1; here we return None).
        """
        mod = 10 ** K
        t = S % mod

        last = t
        ans = 1
        n_val = t

        for i in range(1, K + 1):
            _last = 1
            flag = False
            for j in range(1, 11):
                n_val = (n_val * last) % mod
                _last = (_last * last) % mod
                if n_val % (10 ** i) == t % (10 ** i):
                    multiplier = j if j < 10 else 10
                    ans *= multiplier
                    flag = True
                    break
            if not flag:
                return None
            n_val = t
            last = _last

        return ans