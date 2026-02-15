from typing import Any, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NextPalindromicEnv(Env):
    """Environment for finding the smallest palindromic number greater than N - single-turn Q&A."""

    def __init__(
        self,
        digit_num: int = 3,
        **kwargs
    ):
        """
        Initialize the NextPalindromicEnv.

        Parameters:
            digit_num: Number of digits to control the upper bound for N (1 to 10^digit_num - 1).
        """
        super().__init__()
        assert digit_num >= 1, "digit_num should be greater than or equal to 1"
        self.digit_num = digit_num

        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving palindromic number problems.\n"
            "Given an integer N, find the smallest palindromic number strictly greater than N.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate N within the specified digit range
        self.N = random.randint(1, 10 ** self.digit_num - 1)

        # Build problem statement
        self.current_problem = (
            f"Given N = {self.N}, find the smallest palindromic number that is strictly greater than N.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = int(self._next_palindrome(str(self.N)))

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the submitted answer."""
        # Parse boxed answer
        raw_answer = self._parse_answer(action)
        if raw_answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(raw_answer)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate conditions
        assert self.N is not None and self.reference_answer is not None, "Environment must be reset before stepping."

        is_palindrome = str(user_answer) == str(user_answer)[::-1]
        is_greater = user_answer > self.N
        is_correct = user_answer == self.reference_answer

        # Determine reward and error info
        reward: float = 1.0 if is_correct else 0.0
        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "is_palindrome": is_palindrome,
            "is_greater_than_N": is_greater,
        }

        if not is_correct:
            if not is_palindrome:
                info["error"] = "not_palindromic"
            elif not is_greater:
                info["error"] = "not_greater_than_N"
            else:
                info["error"] = "wrong_answer"

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the provided text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _next_palindrome(self, s: str) -> str:
        """
        Compute the smallest palindromic number strictly greater than the number represented by string s.
        """
        l = len(s)

        # Special case: all '9's -> next palindrome is 1 followed by zeros and ending with 1 (length increases by 1)
        if all(ch == '9' for ch in s):
            return '1' + '0' * (l - 1) + '1'

        # Build initial palindrome by mirroring left half to right half
        ans = list(s)
        for i in range(l // 2):
            ans[l - 1 - i] = ans[i]

        pal = ''.join(ans)
        # If this palindrome is already greater than the original, return it
        if pal > s:
            return pal

        # Otherwise, increment the middle and propagate carry
        mid = (l - 1) // 2
        i = mid
        # Move left through the middle until a non-'9' digit is found, setting '9's to '0'
        while i >= 0 and ans[i] == '9':
            ans[i] = '0'
            i -= 1

        # Increment the first non-'9' digit
        ans[i] = str(int(ans[i]) + 1)
        # Mirror the incremented digit to the other side
        ans[l - 1 - i] = ans[i]

        # Mirror the rest of the left half to the right half to form a valid palindrome
        for j in range(l // 2):
            ans[l - 1 - j] = ans[j]

        return ''.join(ans)

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # Generate a random number; not necessarily correct
        random_answer = random.randint(self.N + 1 if self.N is not None else 1, 10 ** self.digit_num + 100)
        return f"\\boxed{{{random_answer}}}"