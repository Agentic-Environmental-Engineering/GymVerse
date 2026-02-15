import heapq
import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BoundedIntervalIntersectionEnv(Env):
    """Environment for counting subsets of intervals whose intersection length is at least K.

    Single-turn Q&A environment. The agent receives a list of intervals and an integer K, and must
    output the number of non-empty subsets of intervals whose intersection length is >= K.

    Answer must be provided in \\boxed{...} format.
    """

    def __init__(self, N: int = 10, **kwargs) -> None:
        """Initialize the environment.

        Args:
            N: Number of intervals to generate (must be >= 2).
        """
        super().__init__()
        assert N >= 2, "N should be greater than or equal to 2"
        self.N: int = N

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.intervals: List[Tuple[int, int]] = []
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an interval intersection counting problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Returns:
            observation: The full problem description as a string.
            info: Additional info dict (empty for this environment).
        """
        super().reset(seed)

        N = self.N
        # Generate intervals: l in [0, N], r in [l, N]
        self.intervals = []
        for _ in range(N):
            l = random.randint(0, N)
            r = random.randint(l, N)
            self.intervals.append((l, r))

        # Choose K: randint between 1 and max(min_length, 1)
        min_len = min((r - l) for l, r in self.intervals) if self.intervals else 0
        self.K = random.randint(1, max(min_len, 1))
        assert self.K > 0, "K should be greater than 0"

        # Compute reference answer
        self.reference_answer = self._solve_reference(self.intervals, self.K)

        # Build problem prompt
        intervals_str = "\n".join(f"[{l}, {r}]" for l, r in self.intervals)
        self.current_problem = (
            "An interval [l, r]'s length is defined as r - l. The length of an empty intersection "
            "is considered to be 0. The intersection of a set of intervals is the range covered by "
            "all of them simultaneously.\n\n"
            f"You are given {N} intervals:\n"
            f"{intervals_str}\n\n"
            f"Please count how many non-empty subsets (i.e., from the total of 2^{N} - 1 non-empty subsets) "
            f"have an intersection of length greater than or equal to {self.K}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _solve_reference(self, intervals: List[Tuple[int, int]], K: int) -> int:
        """Compute the number of non-empty subsets whose intersection length is at least K.

        This implements the same algorithm as the original RLVE environment:
        - Sort intervals by left endpoint.
        - Maintain a min-heap of right endpoints of intervals that can pair with the current interval
          to achieve intersection length at least K.
        - Count subsets uniquely by designating the interval with the maximum left endpoint in the subset.
        """
        intervals_sorted = sorted(intervals, key=lambda x: x[0])

        heap: List[int] = []
        ans = 0

        for l, r in intervals_sorted:
            if r - l >= K:
                # Remove intervals whose right endpoint is too small to intersect [l, l+K] with length >= K
                while heap and heap[0] < l + K:
                    heapq.heappop(heap)
                # Each subset includes current interval plus any subset of the eligible previous intervals
                ans += pow(2, len(heap))
                heapq.heappush(heap, r)

        return ans

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer.

        Args:
            action: The model's response containing \\boxed{...} with the final integer.

        Returns:
            observation: TERMINAL_STATE (single-turn environment).
            reward: 1.0 if correct, 0.0 if incorrect, -0.1 if format error.
            terminated: True (single-turn).
            truncated: False.
            info: Dict containing correctness and reference information or error type.
        """
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not initialized. Call reset() first."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
            "intervals": self.intervals,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # The answer cannot exceed 2^N - 1; sample within a plausible range.
        max_guess = max(0, (1 << self.N) - 1)
        random_answer = random.randint(0, max_guess)
        return f"\\boxed{{{random_answer}}}"