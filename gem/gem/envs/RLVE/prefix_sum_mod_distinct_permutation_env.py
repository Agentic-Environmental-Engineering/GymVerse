from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PrefixSumMODDistinctPermutationEnv(Env):
    """Environment for constructing a permutation with distinct prefix sums modulo N."""

    def __init__(
        self,
        max_n: int = 1000,
        min_n: int = 3,
        even_only: bool = True,
        fixed_n: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the environment.

        Args:
            max_n: Upper bound for N (inclusive). Must be >= 3.
            min_n: Lower bound for N (inclusive). Must be >= 3 and <= max_n.
            even_only: If True, generated N will be even (matching original logic).
            fixed_n: If provided, use this N instead of sampling.
        """
        super().__init__()
        assert max_n >= 3, "max_n should be greater than or equal to 3"
        assert min_n >= 3, "min_n should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"

        self.max_n = max_n
        self.min_n = min_n
        self.even_only = even_only
        self.fixed_n = fixed_n

        self.current_n: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer_list: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics construction problem.\n"
            "Task: Find a permutation of numbers 1..N such that all N prefix sums are distinct modulo N.\n"
            "Answer Format: Provide the permutation as N space-separated integers enclosed in \\boxed{...}.\n"
            "Do not include any extra text.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Choose N according to the original logic (even N only by default)
        if self.fixed_n is not None:
            N = self.fixed_n
            if self.even_only and (N % 2 == 1):
                raise ValueError("fixed_n must be even when even_only=True")
            if not (self.min_n <= N <= self.max_n):
                raise ValueError("fixed_n must be within [min_n, max_n]")
        else:
            while True:
                N = random.randint(self.min_n, self.max_n)
                if self.even_only and (N % 2 == 1):
                    continue
                break

        self.current_n = N

        # Construct the reference permutation using the original zig-zag construction for even N
        perm: List[int] = [N]
        for i in range(1, N):
            if i % 2 == 1:
                perm.append(i)
            else:
                perm.append(N - i)

        self.reference_answer_list = perm
        self.reference_answer_str = " ".join(map(str, perm))

        # Build the problem description
        problem = (
            f"Please find a permutation of the numbers from 1 to {N} such that all {N} prefix sums "
            f"(i.e., the sum of the first i numbers for all i from 1 to {N}) are distinct modulo {N}.\n\n"
            f"Output Format: Provide {N} integers separated by spaces inside \\boxed{{...}}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted permutation."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error (no boxed content)
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse into list of integers
        tokens = boxed_content.strip().split()
        try:
            user_perm = list(map(int, tokens))
        except ValueError:
            # Not all tokens are integers -> format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.current_n is None:
            # Environment not properly reset
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        N = self.current_n

        # Basic checks: length and permutation validity
        if len(user_perm) != N:
            info = {
                "correct": False,
                "reason": "length_mismatch",
                "expected_length": N,
                "received_length": len(user_perm),
                "reference_answer": self.reference_answer_str,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if set(user_perm) != set(range(1, N + 1)):
            info = {
                "correct": False,
                "reason": "not_a_permutation_of_1_to_N",
                "reference_answer": self.reference_answer_str,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check that all N prefix sums are distinct modulo N
        seen = [False] * N
        prefix_sum = 0
        valid = True
        for x in user_perm:
            prefix_sum = (prefix_sum + x) % N
            if seen[prefix_sum]:
                valid = False
                break
            seen[prefix_sum] = True

        is_correct = valid and all(seen)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "user_answer": " ".join(map(str, user_perm)),
            "n": N,
        }
        if not is_correct:
            info["reason"] = "prefix_sums_not_all_distinct_mod_n"

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} pair."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Return a sample action in the correct format (uses reference answer if available)."""
        if self.reference_answer_str is not None:
            return f"\\boxed{{{self.reference_answer_str}}}"
        else:
            # If no reference yet, return an empty boxed content
            return "\\boxed{}"