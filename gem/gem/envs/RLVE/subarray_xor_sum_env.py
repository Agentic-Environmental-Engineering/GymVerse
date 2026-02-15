from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SubarrayXorSumEnv(Env):
    """Single-turn environment for computing the sum of XORs of all contiguous subarrays."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 200,
        **kwargs
    ):
        """
        Initialize the SubarrayXorSum environment.

        Parameters:
        - N: If provided, the array length will be fixed to this value (must be >= 3).
        - min_N: Minimum value for N when sampling (inclusive, must be >= 3).
        - max_N: Maximum value for N when sampling (inclusive, must be >= min_N).
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.N_fixed: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_array: Optional[list[int]] = None
        self.current_N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an array problem about subarray XOR sums.\n"
            "Task: Given an integer array A of length N, compute the sum of XOR values of all contiguous subarrays of A.\n"
            "Answer Format: Provide a single integer enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        N = self.N_fixed if self.N_fixed is not None else random.randint(self.min_N, self.max_N)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Generate array A with elements in [0, N]
        A = [random.randint(0, N) for _ in range(N)]

        # Compute reference answer
        reference_answer = self._compute_subarray_xor_sum(A)

        # Build problem statement
        A_str = ", ".join(f"A[{i}]={val}" for i, val in enumerate(A, start=1))
        problem = (
            f"You are given an array A of {N} integers: {A_str}\n"
            f"This array has {N} × ({N} + 1) / 2 contiguous subarrays. "
            f"For each subarray, compute the bitwise XOR of its elements, then output the sum of all these subarray XOR values.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Store state
        self.current_N = N
        self.current_array = A
        self.reference_answer = reference_answer
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and terminate."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Compare with reference answer
        assert self.reference_answer is not None, "Reference answer is not set. Did you call reset()?"
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _compute_subarray_xor_sum(A: list[int]) -> int:
        """Compute the sum of XOR of all contiguous subarrays using prefix XOR bit counting."""
        # Determine number of bits needed based on OR of all elements
        or_all = 0
        for x in A:
            or_all |= x
        B = or_all.bit_length()

        # If all elements are zero, the result is zero
        if B == 0:
            return 0

        # Counts of previous prefixes with bit j == 0 or 1, including the initial prefix 0
        cnt_zero = [1] * B
        cnt_one = [0] * B
        prefix = 0
        ans = 0

        for x in A:
            prefix ^= x
            for j in range(B - 1, -1, -1):
                bit = (prefix >> j) & 1
                if bit:
                    ans += (1 << j) * cnt_zero[j]
                    cnt_one[j] += 1
                else:
                    ans += (1 << j) * cnt_one[j]
                    cnt_zero[j] += 1

        return ans

    def sample_random_action(self) -> str:
        """Sample a random boxed integer as an action."""
        # A rough range for random guess; using a moderate range
        N = self.current_N if self.current_N is not None else (self.N_fixed or self.min_N)
        random_answer = random.randint(0, max(1, N * N * N))
        return f"\\boxed{{{random_answer}}}"