from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MonotonicStackEnv(Env):
    """Monotonic stack counting problem environment - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 2000,
        **kwargs
    ):
        """
        Initialize the MonotonicStackEnv.

        Args:
            N: If provided, use this fixed size for the array (must be >= 3).
            min_N: Minimum N when sampling (inclusive), must be >= 3.
            max_N: Maximum N when sampling (inclusive), must be >= min_N.
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N must be at least 3 when provided.")
        if min_N < 3:
            raise ValueError("min_N must be at least 3.")
        if max_N < min_N:
            raise ValueError("max_N must be >= min_N.")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_array: Optional[List[int]] = None
        self.current_N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a monotonic stack counting problem.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 3:
            raise ValueError("N must be at least 3.")

        # Generate array A with values in [1, N]
        A = [random.randint(1, N) for _ in range(N)]

        # Compute reference answer using a monotonic decreasing stack
        S: List[int] = []
        ans = 0
        for t in A:
            while S and S[-1] <= t:
                S.pop()
            ans += len(S)
            S.append(t)

        # Build problem text
        array_repr = ", ".join(f"A[{i}]={val}" for i, val in enumerate(A, start=1))
        problem_text = (
            f"You are given an array A indexed from 1 to {N}: {array_repr}\n\n"
            f"For each 1 ≤ i ≤ {N}, define C[i] as the number of indices j such that:\n"
            f"- i + 1 ≤ j ≤ {N}, and\n"
            f"- For every index k such that i + 1 ≤ k ≤ j, we have A[i] > A[k].\n\n"
            f"Tell me the value of C[1] + C[2] + ... + C[{N}].\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Save state
        self.current_problem = problem_text
        self.reference_answer = ans
        self.current_array = A
        self.current_N = N

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        answer_str = self._parse_answer(action)

        if answer_str is None:
            # Format error: no \\boxed{...} found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_str)
        except ValueError:
            # The boxed content is not a valid integer
            info = {
                "error": "invalid_answer",
                "reference_answer": self.reference_answer
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_N,
            "array": self.current_array
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by guessing an integer within a plausible range."""
        if self.current_N is not None:
            # The maximum possible answer is at most N*(N-1)/2
            upper = self.current_N * (self.current_N - 1) // 2
            guess = random.randint(0, max(0, upper))
        else:
            guess = random.randint(0, 1000)
        return f"\\boxed{{{guess}}}"