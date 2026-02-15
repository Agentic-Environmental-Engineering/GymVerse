import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CapitalCityEffectEnv(Env):
    """
    Capital City Effect environment (single-turn QA) in GEM format.

    Task:
    Define f(x) for a positive integer x (in base-10) as follows:
    - Split x into segments, where each segment is a maximal substring consisting of the same digit.
    - For each segment, compute digit × (length of segment)^2.
    - f(x) is the sum of these values over all segments.

    Given L and R, compute the sum of f(x) for all integers x in [L, R].

    Answer format: Provide your final answer as a single integer in \\boxed{...}.
    """

    def __init__(
        self,
        max_r: int = 1000000,
        # The following parameters existed in the original RLVE environment but are not used for rewards here.
        # They are retained to preserve parameterization compatibility.
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        super().__init__()
        self.max_r = max_r
        if self.max_r < 20:
            raise ValueError("max_r should be greater than or equal to 20")

        # Retained for compatibility (not used in GEM reward logic)
        self.wrong_format = wrong_format
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.L: Optional[int] = None
        self.R: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a digit-segmentation summation problem.\n"
            "For a positive integer x in base-10, define f(x) as follows:\n"
            "- Split x into maximal segments of the same digit.\n"
            "- For each segment, compute digit × (length of segment)^2.\n"
            "- f(x) is the sum over all segments.\n"
            "Your task: Given L and R, compute sum_{x=L}^{R} f(x).\n"
            "Please provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem instance."""
        super().reset(seed)

        # Generate L and R
        R = random.randint(20, self.max_r)
        L = random.randint(1, R)
        self.L, self.R = L, R

        # Build problem statement
        self.current_problem = (
            "Let’s define f(x) as follows, where x is a positive integer in its base-10 representation:\n"
            "- Divide x into segments, where each segment is a maximal substring consisting of the same digit.\n"
            "- For each segment, compute digit × (length of segment)^2.\n"
            "- Then, f(x) is the sum over all segments.\n"
            f"For example, f(2334222) = 2×1^2 + 3×2^2 + 4×1^2 + 2×3^2 = 2 + 12 + 4 + 18 = 36.\n\n"
            f"Please compute the sum of f(x) for all integers x in the range [{L}, {R}] (inclusive).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer using the original digit DP approach
        self.reference_answer = self._cumulative_f(R) - self._cumulative_f(L - 1)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided answer and return the result."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "L": self.L,
            "R": self.R,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the latest \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer) wrapped in \\boxed{...}."""
        # If reference answer exists, sample around it; otherwise random small integer
        if self.reference_answer is not None:
            guess = self.reference_answer + random.randint(-5, 5)
        else:
            guess = random.randint(0, 100)
        return f"\\boxed{{{guess}}}"

    def _cumulative_f(self, x: int) -> int:
        """
        Compute sum_{n=0}^{x} f(n) using a digit DP approach equivalent to the original implementation.
        This method matches the original RLVE logic: solve(R) - solve(L - 1).
        """
        if x < 0:
            return 0
        digits = list(map(int, str(x)))
        n = len(digits)

        # Memoization for non-tight states: key = (pos, last, length, sum_), value = total sum
        dp: dict[tuple[int, int, int, int], int] = {}

        def dfs(pos: int, last: int, length: int, sum_: int, tight: bool) -> int:
            # If all digits are placed, add the final segment's contribution
            if pos == n:
                return sum_ + (length * length * last if last != -1 else 0)

            # Memoization only when not tight
            if not tight:
                key = (pos, last, length, sum_)
                if key in dp:
                    return dp[key]

            maxd = digits[pos] if tight else 9
            ans = 0
            for d in range(maxd + 1):
                if d == last:
                    # Extend current segment
                    new_sum = sum_
                    new_len = length + 1
                else:
                    # Close off previous segment (if any) and start a new one
                    closed = (length * length * last) if last != -1 else 0
                    new_sum = sum_ + closed
                    new_len = 1
                ans += dfs(pos + 1, d, new_len, new_sum, tight and d == maxd)

            if not tight:
                dp[(pos, last, length, sum_)] = ans
            return ans

        return dfs(0, -1, 0, 0, True)