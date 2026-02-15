import random
import bisect
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BoundedSubarrayCountingEnv(Env):
    """Environment for counting subarrays with bounded sum in a repeated array (single-turn Q&A)."""

    def __init__(
        self,
        min_N: int = 1,
        max_N: int = 50,
        min_M: int = 2,
        max_M: int = 10,
        fixed_N: Optional[int] = None,
        fixed_M: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - min_N: minimum array length N (must be >= 1)
        - max_N: maximum array length N (must be >= min_N)
        - min_M: minimum repeat count M (must be >= 2)
        - max_M: maximum repeat count M (must be >= min_M)
        - fixed_N: if provided, use this fixed N instead of random (must be >= 1)
        - fixed_M: if provided, use this fixed M instead of random (must be >= 2)
        """
        super().__init__()
        assert min_N >= 1, "min_N should be greater than or equal to 1"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"
        assert min_M >= 2, "min_M should be greater than or equal to 2"
        assert max_M >= min_M, "max_M should be greater than or equal to min_M"
        if fixed_N is not None:
            assert fixed_N >= 1, "fixed_N should be greater than or equal to 1"
        if fixed_M is not None:
            assert fixed_M >= 2, "fixed_M should be greater than or equal to 2"

        self.min_N = min_N
        self.max_N = max_N
        self.min_M = min_M
        self.max_M = max_M
        self.fixed_N = fixed_N
        self.fixed_M = fixed_M

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.last_params: Dict[str, Any] = {}

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a bounded subarray counting problem on a repeated array.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Choose N and M either fixed or randomly within given ranges
        N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
        M = self.fixed_M if self.fixed_M is not None else random.randint(self.min_M, self.max_M)

        assert N >= 1, "N should be greater than or equal to 1"
        assert M >= 2, "M should be greater than or equal to 2"

        # Generate array A and threshold K
        A: List[int] = [random.randint(1, N) for _ in range(N)]
        total_A = sum(A)
        K = random.randint(max(A), total_A * M)

        # Build the problem prompt
        problem = (
            f"Given an array A of length {N}:\n"
            f"{' '.join(map(str, A))}\n\n"
            f"Repeat array A {M} times to form a new array B of length {N} * {M} = {N * M}. "
            f"In the new array B, how many (nonempty) contiguous subarrays have a total sum less than or equal to {K}?\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem

        # Compute the reference answer
        self.reference_answer = self._compute_answer(A, M, K)
        assert self.reference_answer is not None and self.reference_answer > 0

        # Store parameters for info/debugging
        self.last_params = {
            "N": N,
            "M": M,
            "A": A,
            "K": K,
            "NM": N * M,
        }

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_answer(self, A: List[int], M: int, K: int) -> int:
        """Compute the number of subarrays in repeated array B whose sum is <= K."""
        N = len(A)

        # Build prefix sums s[0..N]
        s = [0] * (N + 1)
        for i in range(1, N + 1):
            s[i] = s[i - 1] + A[i - 1]
        total = s[N]

        ans = 0
        # Precompute M*(M-1)/2 * N for the full-span case
        mmn = M * (M - 1) // 2 * N

        for i in range(1, N + 1):
            si = s[i]
            if si < K:
                # How many full repeats we can append after position i without exceeding K
                d = (K - si) // total
                if d < M - 1:
                    # Contributions from using 0,1,...,d full copies
                    ans += i * (d + 1) + d * (d + 1) // 2 * N

                    # Partial in the (d+1)-th copy
                    e = (K - si) % total
                    # Find smallest j with s[j] >= total - e
                    j = bisect.bisect_left(s, total - e)
                    # For each of the remaining (M-1-d) copies, we can take up to (N-j) more elements
                    ans += (i + d * N + (N - j)) * (M - 1 - d)
                else:
                    # We can take all M copies plus all possible full-span subarrays
                    ans += i * M + mmn
            else:
                # Even the prefix [1..i] exceeds K, so only shorter endings count
                # Find j so that s[i] - s[j] <= K  =>  s[j] >= s[i] - K
                j = bisect.bisect_left(s, si - K)
                ans += (i - j) * M

        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and validate the user's answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == user_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "params": self.last_params,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        guess = random.randint(0, max(1, (self.last_params.get("NM", 1))))
        return f"\\boxed{{{guess}}}"