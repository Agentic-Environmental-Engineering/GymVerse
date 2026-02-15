import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BoundedMeanSubarrayCountingEnv(Env):
    """Environment for counting subarrays whose mean is greater than or equal to K.

    This is a single-turn question-answer environment. The agent receives a problem
    statement asking for the number of nonempty contiguous subarrays of A whose mean
    is >= K. The answer must be provided in \\boxed{...} format.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 2,
        max_n: int = 1000,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed array length. If provided, must be >= 2.
        - min_n: Minimum array length (used when N is not fixed). Must be >= 2.
        - max_n: Maximum array length (used when N is not fixed). Must be >= min_n.
        """
        super().__init__()
        if N is not None:
            assert N >= 2, "N should be greater than or equal to 2"
        assert min_n >= 2, "min_n should be greater than or equal to 2"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"

        self.N_fixed: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an array counting problem.\n"
            "Given an integer array A and an integer K, count the number of nonempty contiguous subarrays\n"
            "whose mean is greater than or equal to K.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine array length N
        self.N = self.N_fixed if self.N_fixed is not None else random.randint(self.min_n, self.max_n)
        assert self.N is not None and self.N >= 2, "N should be greater than or equal to 2"

        # Generate array A with elements in [0, N]
        self.A = [random.randint(0, self.N) for _ in range(self.N)]
        # Choose K between min(A) and max(A), inclusive
        self.K = random.randint(min(self.A), max(self.A))

        # Compute reference answer using CDQ divide-and-conquer counting
        self.reference_answer = self._count_subarrays_with_mean_geq_k(self.A, self.K)
        assert self.reference_answer > 0

        # Build problem prompt
        array_str = " ".join(map(str, self.A))
        self.current_problem = (
            f"Given an array A of length {self.N}:\n"
            f"{array_str}\n\n"
            f"How many nonempty contiguous subarrays have a mean greater than or equal to {self.K}?\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.N,
            "K": self.K,
            "A": self.A[:],  # copy for safety
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == user_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
            "A": self.A[:],
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} from the given text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _count_subarrays_with_mean_geq_k(self, a: List[int], k: int) -> int:
        """Count subarrays with mean >= k using transformed prefix sums and CDQ."""
        n = len(a)
        # Transform to B[i] = A[i] - K, count subarrays with sum >= 0
        v = [0] * (n + 1)
        for i in range(1, n + 1):
            v[i] = v[i - 1] + a[i - 1] - k

        tmp = [0] * (n + 1)
        res = 0

        def cdq(l: int, r: int) -> None:
            nonlocal res
            if l >= r:
                return
            mid = (l + r) // 2
            cdq(l, mid)
            cdq(mid + 1, r)

            i, j = l, mid + 1
            sum_left = 0
            for k_idx in range(l, r + 1):
                if j > r or (i <= mid and v[i] <= v[j]):
                    sum_left += 1
                    tmp[k_idx] = v[i]
                    i += 1
                else:
                    res += sum_left
                    tmp[k_idx] = v[j]
                    j += 1

            for k_idx in range(l, r + 1):
                v[k_idx] = tmp[k_idx]

        cdq(0, n)
        return res

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # Number of nonempty subarrays is at most N*(N+1)//2
        max_guess = (self.N if self.N is not None else self.max_n)
        max_val = max_guess * (max_guess + 1) // 2
        random_answer = random.randint(0, max_val)
        return f"\\boxed{{{random_answer}}}"