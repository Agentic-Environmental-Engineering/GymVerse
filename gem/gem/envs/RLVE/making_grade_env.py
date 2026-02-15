import heapq
import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MakingGradeEnv(Env):
    """
    Making Grade problem environment - single-turn Q&A.

    Task:
    Given an array A of length N, find an array B of length N such that B is either
    monotonically non-decreasing or monotonically non-increasing, and the sum of
    |A[i] - B[i]| over all i is minimized.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 30,
        **kwargs: Any,
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, use this fixed length for the array. Must be >= 4.
        - min_N: Minimum value for N when sampled randomly (inclusive). Must be >= 4.
        - max_N: Maximum value for N when sampled randomly (inclusive). Must be >= min_N.
        """
        super().__init__()
        if N is not None and N < 4:
            raise ValueError("N should be greater than or equal to 4")
        if min_N < 4:
            raise ValueError("min_N should be greater than or equal to 4")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.N_fixed: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        self.current_problem: Optional[str] = None
        self.reference_best_cost: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a sequence adjustment problem.\n"
            "Given an array A, you must output an array B of the same length such that B is either\n"
            "monotonically non-decreasing or monotonically non-increasing, minimizing the sum of |A[i] - B[i]|.\n"
            "Answer Format: Provide your array as space-separated integers inside \\boxed{...}.\n"
            "Example: \\boxed{1 2 2 3}\n\n"
        )

    @staticmethod
    def _non_decreasing(arr: List[int]) -> bool:
        """Check if the array is non-decreasing."""
        return all(a <= b for a, b in zip(arr, arr[1:]))

    @staticmethod
    def _non_increasing(arr: List[int]) -> bool:
        """Check if the array is non-increasing."""
        return all(a >= b for a, b in zip(arr, arr[1:]))

    @staticmethod
    def _cost_nondecreasing(seq: List[int]) -> int:
        """
        Compute the minimum total decrease needed to make seq non-decreasing.
        This is equivalent to computing the minimal L1 adjustment to enforce non-decreasing order
        with unrestricted integers, using a max-heap technique.
        """
        heap: List[int] = []
        total = 0
        for a in seq:
            heapq.heappush(heap, -a)
            top = -heap[0]
            if a < top:
                total += top - a
                heapq.heapreplace(heap, -a)
        return total

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            N = self.N_fixed
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 4:
            raise AssertionError("N should be greater than or equal to 4")

        # Generate a non-monotonic A
        while True:
            A = [random.randint(0, N * N) for _ in range(N)]
            if not (self._non_decreasing(A) or self._non_increasing(A)):
                break

        # Compute reference best cost (minimal |A-B| with monotonic B)
        inc_cost = self._cost_nondecreasing(A)
        dec_cost = self._cost_nondecreasing([-x for x in A])
        gold_cost = min(inc_cost, dec_cost)
        if gold_cost <= 0:
            raise AssertionError("gold_answer should be greater than 0")

        # Store state
        self.N = N
        self.A = A
        self.reference_best_cost = gold_cost

        # Build problem text
        arr_str = ", ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A, start=1))
        self.current_problem = (
            f"There is an array A of length {N}: {arr_str}\n"
            f"Please find an array B of length {N} such that B is either monotonically non-decreasing "
            f"or monotonically non-increasing. Make the sum of |A[i] - B[i]| for all 1 ≤ i ≤ {N} as small as possible.\n"
            f"Output Format: Provide B as space-separated integers inside \\boxed{{...}}, e.g., \\boxed{{b1 b2 ... b{N}}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted B array."""
        # Parse boxed answer content
        content = self._parse_answer(action)
        if content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure environment is initialized
        if self.A is None or self.N is None or self.reference_best_cost is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Parse array B
        tokens = content.strip().split()
        try:
            B = [int(tok) for tok in tokens]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length
        if len(B) != self.N:
            info = {
                "error": "invalid_length",
                "expected_length": self.N,
                "received_length": len(B),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate monotonicity
        if not (self._non_decreasing(B) or self._non_increasing(B)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_monotonic"}

        # Compute cost
        user_cost = sum(abs(a - b) for a, b in zip(self.A, B))
        is_optimal = (user_cost == self.reference_best_cost)
        reward = 1.0 if is_optimal else 0.0

        info = {
            "correct": is_optimal,
            "reference_best_cost": self.reference_best_cost,
            "user_cost": user_cost,
            "N": self.N,
            "A": self.A,
            "B": B,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random valid action formatted as \\boxed{...}.
        This generates a random monotonic array B of the correct length.
        """
        if self.N is None or self.A is None:
            # Generic fallback
            return "\\boxed{0}"

        N = self.N
        max_val = N * N

        # Randomly choose monotonic direction
        if random.random() < 0.5:
            # Non-decreasing
            B = []
            current = random.randint(0, max_val // 2)
            for _ in range(N):
                inc = random.randint(0, max(0, max_val // N))
                current = min(max_val, current + inc)
                B.append(current)
        else:
            # Non-increasing
            B = []
            current = random.randint(max_val // 2, max_val)
            for _ in range(N):
                dec = random.randint(0, max(0, max_val // N))
                current = max(0, current - dec)
                B.append(current)

        b_str = " ".join(str(x) for x in B)
        return f"\\boxed{{{b_str}}}"