import random
from collections import deque
from typing import Any, Deque, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LongestMaxDiffBoundedIntervalEnv(Env):
    """Environment for finding the longest contiguous subarray with bounded max difference.

    Task:
      - Given an array A of length N and an integer K, find a contiguous subarray A[l : r]
        (0-indexed, r is exclusive) such that the maximum difference between any two elements
        in the subarray is at most K.
      - Output l and r separated by a single space, in \\boxed{l r} format.

    Reward:
      - 1.0 if the proposed interval is valid and has maximum possible length.
      - 0.0 if the proposed interval is invalid or not of maximum length.
      - -0.1 if the answer format is incorrect or cannot be parsed.
    """

    def __init__(self, N: int = 10, **kwargs):
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N

        # Problem state
        self.A: Optional[List[int]] = None
        self.K: Optional[int] = None
        self.gold_length: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an array subarray problem.\n"
            "Given an array A and an integer K, find the longest contiguous subarray A[l : r]\n"
            "(0-indexed, r is exclusive) such that the maximum difference between any two\n"
            "elements in the subarray is at most K.\n"
            "Provide your answer as two integers l and r (space-separated) in \\boxed{l r} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        A = [random.randint(0, N) for _ in range(N)]
        spread = max(A) - min(A)
        K = random.randint(0, max(spread - 1, 0))

        gold_length = self._compute_max_length(A, K)

        assert gold_length > 0, "The answer should be greater than 0"

        self.A = A
        self.K = K
        self.gold_length = gold_length

        array_desc = " ".join(f"A[{i}]={val}" for i, val in enumerate(A))
        self.current_problem = (
            f"You are given an array A of length {N}: {array_desc}\n\n"
            f"Please find the longest contiguous subarray A[l : r] (from index l to r - 1, inclusive)\n"
            f"such that the maximum difference between any two elements in the subarray is at most {K}.\n"
            f"Output l and r, separated by a space.\n\n"
            f"Output Format: \\boxed{{l r}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_max_length(self, A: List[int], K: int) -> int:
        """Compute the maximum length of a subarray where max(A[l:r]) - min(A[l:r]) <= K.

        This replicates the algorithmic logic used in the original RLVE environment.
        """
        max_deque: Deque[int] = deque()
        min_deque: Deque[int] = deque()
        left = 0
        answer = 0

        for right, value in enumerate(A):
            while max_deque and A[max_deque[-1]] <= value:
                max_deque.pop()
            max_deque.append(right)

            while min_deque and A[min_deque[-1]] >= value:
                min_deque.pop()
            min_deque.append(right)

            while A[max_deque[0]] - A[min_deque[0]] > K:
                if max_deque[0] < min_deque[0]:
                    left = max_deque[0] + 1
                    max_deque.popleft()
                else:
                    left = min_deque[0] + 1
                    min_deque.popleft()

            answer = max(answer, right - left + 1)

        return answer

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the proposed interval and return the result as a single-turn episode."""
        if self.A is None or self.K is None or self.gold_length is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        tokens = boxed_content.strip().split()
        if len(tokens) != 2:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            l = int(tokens[0])
            r = int(tokens[1])
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        N = self.N
        if not (0 <= l < r <= N):
            info = {
                "correct": False,
                "reason": "invalid_range",
                "N": N,
                "K": self.K,
                "A": self.A,
                "gold_length": self.gold_length,
                "user_interval": (l, r),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        subarray = self.A[l:r]
        if max(subarray) - min(subarray) > self.K:
            info = {
                "correct": False,
                "reason": "constraint_violated",
                "N": N,
                "K": self.K,
                "A": self.A,
                "gold_length": self.gold_length,
                "user_interval": (l, r),
                "user_length": r - l,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        user_length = r - l
        is_correct = (user_length == self.gold_length)

        reward: float = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "N": N,
            "K": self.K,
            "A": self.A,
            "gold_length": self.gold_length,
            "user_interval": (l, r),
            "user_length": user_length,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re

        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required boxed format."""
        if self.N < 2:
            return r"\boxed{0 1}"
        l = random.randint(0, self.N - 2)
        r = random.randint(l + 1, self.N)
        return f"\\boxed{{{l} {r}}}"