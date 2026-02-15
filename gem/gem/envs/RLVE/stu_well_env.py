from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class STUWellEnv(Env):
    """Single-turn environment for the terrain smoothing and well digging optimization problem."""

    def __init__(
        self,
        N: int,
        weight_multiple: int = 4,
        **kwargs
    ):
        """
        Initialize the STUWellEnv.

        Args:
            N: Length of the array X. Must be >= 3.
            weight_multiple: Multiplier to control the range of initial heights for X.
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N
        self.weight_multiple: int = weight_multiple

        # Problem state
        self.X: Optional[List[int]] = None
        self.M: Optional[int] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a terrain smoothing and well digging optimization problem.\n"
            "You will be given an array X and an operation budget M.\n"
            "You can decrease any X[i] by 1 per operation, at most M operations in total, and at the end at least one X[i] must be 0.\n"
            "Your goal is to minimize the maximum absolute difference between adjacent elements after operations.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new single-turn problem."""
        super().reset(seed)

        N = self.N
        # Generate X with values in [1, N * weight_multiple]
        X = [random.randint(1, N * self.weight_multiple) for _ in range(N)]
        # Generate M in [min(X), sum(X)]
        M = random.randint(min(X), sum(X))

        self.X = X
        self.M = M

        # Compute reference answer using binary search on z with feasibility check
        def check(z: int) -> bool:
            """
            Check if it's possible with maximum allowed adjacent slope z
            to dig down somewhere to water (height 0) using at most M operations.
            """
            assert self.X is not None and self.M is not None
            a = self.X[:]
            rem = self.M

            # Enforce |a[i] - a[i-1]| <= z by shaving excess from the right pass
            for i in range(1, N):
                excess = a[i] - (a[i - 1] + z)
                if excess > 0:
                    rem -= excess
                    a[i] = a[i - 1] + z

            # Enforce from the left pass
            for i in range(N - 2, -1, -1):
                excess = a[i] - (a[i + 1] + z)
                if excess > 0:
                    rem -= excess
                    a[i] = a[i + 1] + z

            if rem < 0:
                return False

            # Prefix sums of a
            prefix = [0] * N
            prefix[0] = a[0]
            for i in range(1, N):
                prefix[i] = prefix[i - 1] + a[i]

            # Compute left boundary L for each i
            L = [0] * N
            j = 0
            for i in range(N):
                while j < N and z * (i - j) > a[j]:
                    j += 1
                L[i] = j

            # Compute right boundary R for each i
            R = [0] * N
            j = N - 1
            for i in range(N - 1, -1, -1):
                while j >= 0 and z * (j - i) > a[j]:
                    j -= 1
                R[i] = j

            # Test each position i as the digging spot
            for i in range(N):
                li, ri = L[i], R[i]
                # sum of a[li..ri]
                segment_sum = prefix[ri] - (prefix[li - 1] if li > 0 else 0)
                # cost to carve the left half (from li up to i)
                left_len = i - li
                cost_left = z * left_len * (left_len + 1) // 2
                # cost to carve the right half (from i up to ri)
                right_len = ri - i
                cost_right = z * right_len * (right_len + 1) // 2
                # total additional digs needed to form the tent
                needed = segment_sum - cost_left - cost_right
                if needed <= rem:
                    return True

            return False

        lo, hi = 0, max(X)
        best_z = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if check(mid):
                best_z = mid
                hi = mid - 1
            else:
                lo = mid + 1

        self.reference_answer = best_z

        # Build problem prompt
        x_repr = " ".join(f"X[{i}]={Xi}" for i, Xi in enumerate(X))
        self.current_problem = (
            f"There is an array X of length {N}. Initially, X is: {x_repr}\n"
            f"You can perform the following operation at most {M} times: pick an arbitrary index i and decrease X[i] by 1 (i.e., X[i] -= 1); "
            f"at the end, you must ensure that there exists at least one index i such that X[i] = 0.\n"
            f"Try your best to minimize the value of max(|X[i] - X[i + 1]|) over all 0 <= i < {N} - 1 "
            f"(i.e., the maximum absolute difference between any two adjacent elements in X). "
            f"Output the minimum possible value of this maximum difference.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": N,
            "X": X,
            "M": M,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer and terminate the episode."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer and compare
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment must be reset before calling step()."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "X": self.X,
            "M": self.M,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required boxed format."""
        # Heuristic range for random guessing
        guess = random.randint(0, max(self.X) if self.X else 10 * self.weight_multiple)
        return f"\\boxed{{{guess}}}"