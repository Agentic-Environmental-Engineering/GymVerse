from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PalindromicSubstringNumberCountingEnv(Env):
    """Environment for counting numbers that contain palindromic substrings of length greater than 1 within a given range."""

    def __init__(
        self,
        max_r: int = 1000000,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_r: The maximum possible value for R when generating problems. Must be >= 20.
        """
        super().__init__()
        if max_r < 20:
            raise ValueError("max_r should be greater than or equal to 20")
        self.max_r = max_r

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.L: Optional[int] = None
        self.R: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the instructions for the task."""
        return (
            "You are solving a counting problem about palindromic substrings in numbers.\n"
            "We treat every positive integer as a string of digits (without leading zeros). "
            "A number is called a 'good number' if it contains at least one palindromic substring of length greater than 1.\n\n"
            "Examples:\n"
            "- 101 is a good number because it contains the substring '101'.\n"
            "- 110 is a good number because it contains the substring '11'.\n"
            "- 102 and 1201 are not good numbers because they do not contain any palindromic substring of length greater than 1.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
            "\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate parameters L and R
        R = random.randint(20, self.max_r)
        L = random.randint(1, R - 1)
        self.L, self.R = L, R

        # Build the problem statement
        self.current_problem = (
            f"Please count how many good numbers exist in the range [{L}, {R}] (inclusive).\n"
        )

        # Compute the reference answer using digit DP
        def str_minus_one(s: str) -> str:
            # Subtract 1 from a positive decimal string s
            lst = list(s)
            i = len(lst) - 1
            # borrow until we find a non-zero digit
            while i >= 0 and lst[i] == '0':
                lst[i] = '9'
                i -= 1
            if i >= 0:
                lst[i] = str(int(lst[i]) - 1)
            # strip leading zeros (but leave one zero if result is 0)
            if lst and lst[0] == '0':
                j = 0
                while j < len(lst) - 1 and lst[j] == '0':
                    j += 1
                lst = lst[j:]
            return ''.join(lst) if lst else '0'

        def solve_for(bound_str: str) -> int:
            # Count "good" numbers in [0, bound_str]
            n = len(bound_str)
            # d[1] = least significant digit, ..., d[n] = most significant
            d = [0] * (n + 1)
            for i, ch in enumerate(reversed(bound_str), start=1):
                d[i] = int(ch)

            # dp cache: f[x][num][pre][lovely][lead][prelead], initialized to -1
            f = [[[[[[ -1 for _ in range(2)]
                        for _ in range(2)]
                        for _ in range(2)]
                    for _ in range(10)]
                    for _ in range(10)]
                    for _ in range(n + 1)]

            def dfs(x: int, num: int, pre: int, lovely: bool,
                    lead: bool, prelead: bool, top: bool) -> int:
                # base case: all digits placed
                if x == 0:
                    return 1 if lovely else 0

                # use cache when not tight
                if not top:
                    cached = f[x][num][pre][int(lovely)][int(lead)][int(prelead)]
                    if cached != -1:
                        return cached

                bound = d[x] if top else 9
                total = 0

                for digit in range(bound + 1):
                    # check for palindrome substrings of length 2 or 3
                    is_lovely = lovely \
                        or ((not lead) and digit == num) \
                        or ((not prelead) and digit == pre)
                    next_lead = lead and (digit == 0)
                    next_prelead = lead
                    next_top = top and (digit == bound)

                    total += dfs(x - 1, digit, num,
                                 is_lovely, next_lead,
                                 next_prelead, next_top)

                if not top:
                    f[x][num][pre][int(lovely)][int(lead)][int(prelead)] = total

                return total

            # start from position n, with no previous digits placed
            return dfs(n, 0, 0, False, True, True, True)

        L_str, R_str = str(L), str(R)
        L_minus_one = str_minus_one(L_str)
        self.reference_answer = solve_for(R_str) - solve_for(L_minus_one)

        obs = self._get_instructions() + self.current_problem
        return obs, {"L": L, "R": R}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process one action (the answer) and return the result."""
        # Parse the answer from \\boxed{...}
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare the answer
        try:
            user_answer = int(boxed)
            is_correct = (user_answer == self.reference_answer)
            reward = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "invalid_answer_format"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "L": self.L,
            "R": self.R
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action for testing purposes."""
        # Heuristic random guess: a number within [0, R - L + 1]
        span = 0
        if self.L is not None and self.R is not None and self.R >= self.L:
            span = max(0, self.R - self.L + 1)
        random_answer = random.randint(0, max(1, span))
        return f"\\boxed{{{random_answer}}}"