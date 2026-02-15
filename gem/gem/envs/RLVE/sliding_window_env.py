import random
import re
from collections import deque
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SlidingWindowEnv(Env):
    """Sliding Window Minimum environment - single-turn Q&A.

    The task: Given a list of N integers and a window size K, compute the minimum
    of each contiguous subarray of length K, from left to right. The answer should
    be provided as space-separated integers inside \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 1000,
        value_low_divisor: int = 20,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
        - N: If provided, use this fixed N (must be >= 3). If None, N is sampled in [min_n, max_n].
        - min_n: Minimum N when sampling.
        - max_n: Maximum N when sampling.
        - value_low_divisor: Controls the lower bound of element values as -(N // value_low_divisor).
        """
        super().__init__()
        self.fixed_N = N
        self.min_n = min_n
        self.max_n = max_n
        self.value_low_divisor = value_low_divisor

        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.A: Optional[List[int]] = None

        self.current_problem: Optional[str] = None
        self.reference_answer_list: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a sliding window minimum problem.\n"
            "Please provide your final answer as space-separated integers inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = int(self.fixed_N)
        else:
            N = random.randint(self.min_n, self.max_n)

        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Generate K and sequence A
        K = random.randint(2, N - 1)
        low_bound = -(N // self.value_low_divisor) if self.value_low_divisor > 0 else -N
        A = [random.randint(low_bound, N) for _ in range(N)]

        # Compute sliding window minimums using a deque
        min_deque: deque[int] = deque()  # stores indices with increasing values of A
        mins: List[int] = []
        for i in range(N):
            # Remove indices out of window
            while min_deque and min_deque[0] <= i - K:
                min_deque.popleft()
            # Maintain increasing deque
            while min_deque and A[min_deque[-1]] > A[i]:
                min_deque.pop()
            min_deque.append(i)
            # Append minimum for the window ending at i
            if i >= K - 1:
                mins.append(A[min_deque[0]])

        if len(mins) != N - K + 1:
            raise AssertionError("The length of gold_answer should be N - K + 1")

        # Store state
        self.N = N
        self.K = K
        self.A = A
        self.reference_answer_list = mins
        self.reference_answer_str = " ".join(map(str, mins))

        # Build problem prompt
        self.current_problem = (
            f"You are given the following list of {N} numbers: {' '.join(map(str, A))}\n"
            f"Please find the minimum value in each contiguous subarray of size {K} "
            f"(there are {N - K + 1} such subarrays in total).\n\n"
            f"Output Format: Provide the minimum values from left to right as space-separated integers inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the answer."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert content to list of integers
        try:
            user_answer_list = list(map(int, boxed_content.strip().split()))
        except ValueError:
            # Content not convertible to integers
            info = {
                "error": "invalid_answer",
                "reference_answer": self.reference_answer_str,
                "user_answer_raw": boxed_content,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate length and correctness
        expected_len = (self.N - self.K + 1) if (self.N is not None and self.K is not None) else None
        is_length_ok = (expected_len is None) or (len(user_answer_list) == expected_len)
        is_correct = is_length_ok and (user_answer_list == self.reference_answer_list)

        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "reference_answer_list": self.reference_answer_list,
            "user_answer_list": user_answer_list,
            "N": self.N,
            "K": self.K,
            "A": self.A,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: random space-separated integers inside \\boxed{...}."""
        if self.N is None or self.K is None:
            # Fallback if called before reset
            random_values = [str(random.randint(-10, 10)) for _ in range(5)]
        else:
            length = self.N - self.K + 1
            low_bound = -(self.N // self.value_low_divisor) if self.value_low_divisor > 0 else -self.N
            high_bound = self.N
            random_values = [str(random.randint(low_bound, high_bound)) for _ in range(length)]
        return f"\\boxed{{{' '.join(random_values)}}}"