import math
import random
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DecreasingDigitCountingEnv(Env):
    """Environment for the Decreasing Digit Counting problem in base 2^K (single-turn Q&A)."""

    def __init__(
        self,
        max_k: int = 10,
        max_w: int = 10000,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
            max_k (int): Maximum allowed K (K >= 2).
            max_w (int): Maximum allowed W (W >= 1).
        """
        super().__init__()
        if max_k < 2:
            raise ValueError("max_k should be greater than or equal to 2")
        if max_w < 1:
            raise ValueError("max_w should be greater than or equal to 1")

        self.max_k = max_k
        self.max_w = max_w

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.K: Optional[int] = None
        self.W: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a counting problem for numbers represented in base 2^K.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate parameters K and W
        K = random.randint(2, self.max_k)
        high_w = min(self.max_w, K * (1 << K))
        if K + 1 <= high_w:
            W = random.randint(K + 1, high_w)
        else:
            W = self.max_w

        self.K = K
        self.W = W

        # Build the problem prompt
        power_2_K = 2 ** K
        problem_text = (
            f"Let R be a number in base 2^{K} = {power_2_K}, satisfying the following conditions:\n"
            f"- R must be at least a 2-digit number in base 2^{K} (leading zeros are ignored; "
            f"i.e., we don’t count numbers like 01 or 0005).\n"
            f"- When viewed as a number in base 2^{K}, each digit of R, except for the last one, "
            f"must be strictly less than its immediate right neighbor. (Digits are read from left to right, "
            f"with the leftmost digit being the most significant — following natural reading order.)\n"
            f"- When R is converted to its binary representation, the total number of bits (ignoring leading zeros) "
            f"must not exceed {W}.\n\n"
            f"Your task is to determine how many distinct valid values of R satisfy all the above conditions.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}.\n"
        )

        self.current_problem = problem_text

        # Compute the reference answer using the original algorithm
        r0 = W % K
        m_max = W // K + (1 if r0 != 0 else 0)
        if m_max < 2:
            answer = 0
        else:
            max_val = (1 << K) - 1
            total = 0
            for m in range(2, m_max + 1):
                if m > max_val:
                    continue
                if m < m_max or (m == m_max and r0 == 0):
                    total += math.comb(max_val, m)
                else:
                    max_high = (1 << r0) - 1
                    for i in range(1, max_high + 1):
                        ni = max_val - i
                        mi = m - 1
                        if ni >= mi:
                            total += math.comb(ni, mi)
            answer = total

        self.reference_answer = answer

        obs = self._get_instructions() + self.current_problem
        return obs, {"K": K, "W": W, "power_2_K": power_2_K}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step by validating the provided answer."""
        # Parse the answer from \\boxed{...}
        answer_text = self._parse_answer(action)

        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate the answer
        try:
            user_answer = int(answer_text)
            is_correct = (user_answer == self.reference_answer)
            reward = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "K": self.K,
            "W": self.W,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} format."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        random_answer = random.randint(0, max(1, (self.reference_answer or 100)))
        return f"\\boxed{{{random_answer}}}"