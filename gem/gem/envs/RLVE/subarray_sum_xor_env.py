from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SubarraySumXorEnv(Env):
    """Subarray Sum XOR environment - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 100,
        **kwargs
    ):
        """
        Initialize the SubarraySumXorEnv instance.

        Parameters:
        - N: Optional fixed size of the array. If None, N will be randomly sampled in [min_n, max_n].
        - min_n: Minimum allowed N (default 3).
        - max_n: Maximum allowed N used for random sampling when N is None (default 100).
        """
        super().__init__()
        self.N = N
        self.min_n = min_n
        self.max_n = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_A: Optional[List[int]] = None
        self.current_N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving subarray sum XOR problems.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.N is not None:
            if self.N < self.min_n:
                raise ValueError(f"N should be greater than or equal to {self.min_n}")
            N = self.N
        else:
            N = random.randint(self.min_n, self.max_n)

        # Generate array A with elements in [0, N]
        A = [random.randint(0, N) for _ in range(N)]

        # Build problem prompt
        A_str = ", ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A, start=1))
        self.current_problem = (
            f"You are given an array A of {N} integers: {A_str}\n"
            f"This array has {N} × ({N} + 1) / 2 contiguous subarrays. "
            f"For each subarray, compute its sum; then, output the bitwise XOR of all these subarray sums.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(A)

        # Store current state
        self.current_A = A
        self.current_N = N

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_N,
            "A": self.current_A,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_reference_answer(self, A: List[int]) -> int:
        """Compute the bitwise XOR of all contiguous subarray sums."""
        N = len(A)
        if N < self.min_n:
            raise ValueError(f"N should be greater than or equal to {self.min_n}")

        # Build prefix sums S[0..N]
        S = [0] * (N + 1)
        for i in range(1, N + 1):
            S[i] = S[i - 1] + A[i - 1]
        mx = S[N]

        # Count how many times each prefix-sum value appears (excluding S[0])
        cnt = [0] * (mx + 1)
        for i in range(1, N + 1):
            cnt[S[i]] += 1

        # scnt[v] = sum of cnt[0..v]
        scnt = [0] * (mx + 1)
        scnt[0] = cnt[0]
        for v in range(1, mx + 1):
            scnt[v] = scnt[v - 1] + cnt[v]

        ans = 0
        # For each bit j, count how many subarray sums have that bit = 1
        for j in range(mx.bit_length()):
            K = 1 << j
            M = 1 << (j + 1)

            # f[v] = number of earlier prefix-sums s' with (v - s') in [K, M-1]
            f = [0] * (mx + 1)
            for v in range(mx + 1):
                prev = f[v - M] if v >= M else 0
                add1 = scnt[v - K] if v >= K else 0
                sub1 = scnt[v - M] if v >= M else 0
                f[v] = prev + add1 - sub1

            # g[v] = number of later prefix-sums s' with (s' - v) in [K, M-1]
            g = [0] * (mx + 1)
            for v in range(mx, -1, -1):
                prev = g[v + M] if v + M <= mx else 0
                hi = v + M - 1
                lo = v + K - 1
                add2 = scnt[hi] if hi <= mx else scnt[mx]
                sub2 = scnt[lo] if lo <= mx else scnt[mx]
                g[v] = prev + add2 - sub2

            # Sum up f[S[i]] + g[S[i]] for i=1..N, then divide by 2 to get the # of subarrays
            res = 0
            for i in range(1, N + 1):
                sv = S[i]
                res += f[sv] + g[sv]
            res //= 2

            # If that count is odd, set bit j in ans
            if res & 1:
                ans |= K

        # Finally, include the subarrays that start from index 1 (i.e., S[i] - S[0] = S[i])
        for i in range(1, N + 1):
            ans ^= S[i]

        return ans

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        random_answer = random.randint(0, 10**9)
        return f"\\boxed{{{random_answer}}}"