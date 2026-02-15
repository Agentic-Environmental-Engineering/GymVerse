from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaximumDivisorEnv(Env):
    """Maximum divisor floor-sum environment - single-turn Q&A.

    Task:
    Given an array A of length N and an integer K, find the maximum positive integer L
    such that floor(A[0] / L) + floor(A[1] / L) + ... + floor(A[N-1] / L) >= K.

    Answer format:
    The agent must output the final integer answer in \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 100,
        random_range_coefficient: int = 20,
        **kwargs,
    ):
        super().__init__()
        # Configuration for problem generation
        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N
        self.random_range_coefficient: int = random_range_coefficient

        # Validate static parameters
        assert self.random_range_coefficient > 0, "random_range_coefficient must be positive"
        assert self.min_N >= 2, "min_N must be at least 2"
        assert self.max_N >= self.min_N, "max_N must be >= min_N"
        if self.fixed_N is not None:
            assert self.fixed_N >= 2, "N must be at least 2"

        # Runtime state
        self.N: Optional[int] = None
        self.A: Optional[list[int]] = None
        self.K: Optional[int] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a maximum divisor floor-sum problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 2, "N must be at least 2"
        self.N = N

        # Generate K according to original logic
        K_upper = N * max(1, N // self.random_range_coefficient)
        K = random.randint(1, K_upper if K_upper >= 1 else 1)
        self.K = K

        # Generate array A
        A = [random.randint(1, N) for _ in range(N)]
        if sum(A) < K:
            # Adjust to ensure feasibility: sum(A) >= K
            A[0] += K - sum(A)
        assert sum(A) >= K, "sum(A) must be at least K"
        random.shuffle(A)
        self.A = A

        # Compute reference answer with binary search (original logic)
        def check(l: int) -> bool:
            return sum(ai // l for ai in A) >= K

        l, r = 1, max(A) + 1
        while l < r:
            m = (l + r) // 2
            if check(m):
                l = m + 1
            else:
                r = m
        self.reference_answer = l - 1

        # Build problem statement
        A_str = " ".join(map(str, A))
        self.current_problem = (
            f"You are given an array A of length {N}. The values are as follows (indexing starts at 0):\n"
            f"{A_str}\n\n"
            f"Please find the maximum positive integer L such that the following inequality holds: "
            f"[A[0] / L] + [A[1] / L] + ... + [A[{N - 1}] / L] >= {K}, where [x] denotes the floor function "
            f"(i.e., rounding down to the nearest integer).\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer and check correctness
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None) and (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
            "A": self.A,
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
        # Heuristic: sample within [1, max(A)] if available; otherwise a small positive integer
        if self.A:
            guess = random.randint(1, max(self.A))
        else:
            guess = random.randint(1, 10)
        return f"\\boxed{{{guess}}}"