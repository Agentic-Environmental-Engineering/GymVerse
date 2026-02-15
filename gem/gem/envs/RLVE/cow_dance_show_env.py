import heapq
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CowDanceShowEnv(Env):
    """Cow Dance Show scheduling environment - single-turn Q&A.

    There are N cows labeled from 1 to N. The i-th cow takes d[i] time to dance.
    The stage can hold K cows simultaneously. Initially, cows 1 through K are on the stage.
    When a cow finishes, it immediately leaves and the next available cow by label enters.
    The task is to compute the time when all cows have finished dancing.

    The answer must be provided in \\boxed{...} format as a single integer.
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 100,
        fixed_n: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        assert min_n >= 3, "min_n should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"
        if fixed_n is not None:
            assert fixed_n >= 3, "fixed_n should be greater than or equal to 3"
        self.min_n = min_n
        self.max_n = max_n
        self.fixed_n = fixed_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.d: Optional[list[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving the Cow Dance Show scheduling problem.\n"
            "Please provide your answer in \\boxed{...} format as a single integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        N = self.fixed_n if self.fixed_n is not None else random.randint(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"

        d = [random.randint(1, N) for _ in range(N)]
        K = random.randint(2, N - 1)

        # Store state
        self.N = N
        self.K = K
        self.d = d

        # Build problem prompt
        d_str = ", ".join(f"d[{i}]={di}" for i, di in enumerate(d, start=1))
        self.current_problem = (
            f"There are {N} cows labeled from 1 to {N}, and the i-th cow takes d[i] time to dance. "
            f"The array d is given as: {d_str}\n\n"
            f"The cows dance on the stage as follows:\n"
            f"- Initially, the first {K} cows (cows 1 through {K}) are on the stage.\n"
            f"- Each cow dances for its own time d[i]. When a cow finishes dancing, it leaves the stage.\n"
            f"- As soon as a cow leaves, the next available cow in label order (if any) immediately takes its place. "
            f"For example, when the first cow leaves, cow {K} + 1 enters the stage.\n\n"
            f"Please output the time when all cows have finished dancing.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer using a min-heap
        self.reference_answer = self._compute_total_time(N, K, d)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_total_time(self, N: int, K: int, d: list[int]) -> int:
        """Compute total time for all cows to finish dancing using a min-heap."""
        heap = d[:K]
        heapq.heapify(heap)
        for i in range(K, N):
            t = heapq.heappop(heap)
            heapq.heappush(heap, t + d[i])
        return max(heap)

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the submitted answer."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)

        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and check correctness
        try:
            user_answer = int(answer_text)
            is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
            reward: float = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
            "d": self.d,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the given text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        upper = self.reference_answer if self.reference_answer is not None else (self.max_n * max(self.min_n, 3))
        upper = max(int(upper), 1)
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"