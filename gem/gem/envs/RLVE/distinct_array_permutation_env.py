from typing import Any, List, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DistinctArrayPermutationEnv(Env):
    """Environment for the 'Distinct Array Permutation' single-turn Q&A problem."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 10,
        **kwargs
    ):
        """
        Initialize the DistinctArrayPermutationEnv instance.

        Parameters:
        - N: If provided, the environment will always use this fixed array size (must be >= 3).
        - min_N: Minimum array size (inclusive) when N is not fixed (must be >= 3).
        - max_N: Maximum array size (inclusive) when N is not fixed (must be >= min_N).
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N should be at least 3")
        if min_N < 3:
            raise ValueError("min_N should be at least 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        self.current_problem: Optional[str] = None
        self.current_array: Optional[List[int]] = None
        self.reference_answer_list: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.current_N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a permutation problem over arrays with distinct integers.\n"
            "Task: Given an array A of N distinct integers (1-indexed), construct an array B by permuting A such that\n"
            "for every non-empty proper subset of indices S = {x1, x2, ..., xk} (1 ≤ xi ≤ N, 0 < k < N),\n"
            "the sums of elements of A and B at those positions are different, i.e., sum_{i in S} A[i] != sum_{i in S} B[i].\n"
            "Output Format: Provide your permuted array B as space-separated integers enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 3:
            raise ValueError("N should be at least 3")
        self.current_N = N

        # Generate array with distinct integers using max_value = 2 * N
        self.current_array = random.sample(range(2 * N), N)

        # Compute a reference valid permutation (not necessarily unique)
        self.reference_answer_list = self._find_valid_permutation(self.current_array)
        self.reference_answer_str = " ".join(map(str, self.reference_answer_list))

        # Build problem description
        array_str = " ".join(map(str, self.current_array))
        self.current_problem = (
            f"You are given an array A with {N} distinct integers (1-indexing): {array_str}\n\n"
            "Construct an array B by permuting A such that for every non-empty proper subset of indices "
            f"S = {{x1, x2, ..., xk}} (1 ≤ xi ≤ {N}, 0 < k < {N}) the sums of elements at those positions "
            "in A and B are different.\n\n"
            "Your final answer should be a single line containing the permuted array B's elements in order, "
            "separated by spaces.\n"
            "Output Format: Enclose your space-separated array inside \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        # Extract boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse boxed content as a list of integers
        try:
            tokens = boxed.strip().split()
            user_list = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate against current problem
        assert self.current_array is not None, "Environment not properly reset."
        assert self.current_N is not None, "Environment not properly reset."

        is_perm = sorted(user_list) == sorted(self.current_array)
        subset_ok = False

        if is_perm:
            subset_ok = self._is_valid_permutation(self.current_array, user_list)

        is_correct = bool(is_perm and subset_ok)
        reward: float = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "is_permutation": is_perm,
            "subset_condition_satisfied": subset_ok,
            "array_A": self.current_array[:],
            "N": self.current_N,
            "reference_answer": self.reference_answer_str,
            "user_answer": " ".join(map(str, user_list))
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _find_valid_permutation(self, arr: List[int]) -> List[int]:
        """
        Find a valid permutation of arr such that all subset sums are different.
        Uses the elegant solution: sort indices by values, then cyclically assign next value.
        """
        n = len(arr)
        p = sorted([i for i in range(n)], key=lambda x: arr[x])
        b = [0] * n
        for i in range(n):
            b[p[i]] = arr[p[(i + 1) % n]]
        return b

    def _is_valid_permutation(self, arr_a: List[int], arr_b: List[int]) -> bool:
        """
        Check if arr_b is a valid permutation that satisfies the condition that
        for every non-empty proper subset S of indices, sum_A(S) != sum_B(S).
        """
        n = len(arr_a)

        # Check if it's actually a permutation
        if sorted(arr_a) != sorted(arr_b):
            return False

        # Check all non-empty proper subsets
        for mask in range(1, (1 << n) - 1):
            sum_a = 0
            sum_b = 0
            for i in range(n):
                if mask & (1 << i):
                    sum_a += arr_a[i]
                    sum_b += arr_b[i]
            if sum_a == sum_b:
                return False

        return True

    def sample_random_action(self) -> str:
        """Sample a random action: a random permutation boxed as required."""
        assert self.current_array is not None, "Environment not properly reset."
        perm = self.current_array[:]
        random.shuffle(perm)
        return f"\\boxed{{{' '.join(map(str, perm))}}}"