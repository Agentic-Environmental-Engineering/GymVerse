from typing import Any, Optional, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class EnergyStorageMeterEnv(Env):
    """Single-turn Q&A environment for computing a sum over XOR pairs."""

    def __init__(self, max_n_m: int = 64, **kwargs) -> None:
        """
        Initialize the environment.

        Args:
            max_n_m: Upper bound for N and M (inclusive). Must be >= 4.
        """
        super().__init__()
        if max_n_m < 4:
            raise ValueError("max_n_m should be greater than or equal to 4")
        self.max_n_m: int = max_n_m

        # State variables set in reset
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial sum problem over XOR values.\n"
            "Given integers N, M, and K, compute the value:\n"
            "S = sum_{i=0..N-1} sum_{j=0..M-1} max((i XOR j) - K, 0).\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem instance."""
        super().reset(seed)

        # Generate parameters
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)
        self.K = random.randint(0, self.max_n_m)

        # Build problem statement
        self.current_problem = (
            "I want to know the sum of max((i XOR j) - K, 0) over all pairs (i, j) such that "
            f"0 <= i < {self.N} and 0 <= j < {self.M}, where XOR denotes the bitwise XOR operation.\n"
            f"Given: N = {self.N}, M = {self.M}, K = {self.K}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference(self.N, self.M, self.K)

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "reference_answer": self.reference_answer,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer."""
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return (
                TERMINAL_STATE,
                -0.1,
                True,
                False,
                {"error": "format_error", "message": "Answer must be provided in \\boxed{...} format."},
            )

        try:
            user_answer = int(answer_str)
        except ValueError:
            return (
                TERMINAL_STATE,
                0.0,
                True,
                False,
                {"error": "invalid_answer", "message": "Boxed content is not an integer."},
            )

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "K": self.K,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if not matches:
            return None
        return matches[-1].strip()

    def _compute_reference(self, N: int, M: int, K: int) -> int:
        """Compute the reference answer using the original algorithm."""
        def S(l: int, r: int) -> int:
            # sum of integers from l to r inclusive
            if l > r:
                return 0
            cnt = r - l + 1
            return (l + r) * cnt // 2

        def calc(l: int, r: int, x: int) -> int:
            # corresponds to the original calc logic
            if l <= x <= r:
                return S(0, r - x)
            elif r < x:
                return 0
            else:  # x < l
                return S(l - x, r - x)

        # Collect set bit positions for N and M
        bitsN = [i for i in range(N.bit_length() + 1) if ((N >> i) & 1)]
        bitsM = [j for j in range(M.bit_length() + 1) if ((M >> j) & 1)]

        ans = 0
        for i in bitsN:
            for j in bitsM:
                u = i if i < j else j
                v = i ^ j ^ u  # equals max(i, j)

                # Clear lower (i+1) bits of N and (j+1) bits of M, then XOR
                ni = (N >> (i + 1)) << (i + 1)
                mj = (M >> (j + 1)) << (j + 1)
                x = ni ^ mj

                # Clear lower v bits of x
                if v > 0:
                    x = (x >> v) << v

                # r = x with its lower v bits set to 1
                r = (x | ((1 << v) - 1)) if v > 0 else x

                contrib = (1 << u) * calc(x, r, K)
                ans += contrib

        return ans

    def sample_random_action(self) -> str:
        """Sample a random action by guessing a non-negative integer in \\boxed{...} format."""
        # A conservative upper bound for sampling: XOR max times pair count
        xor_max = (1 << (self.max_n_m.bit_length())) - 1
        max_pairs = self.max_n_m * self.max_n_m
        upper_bound = max(1, xor_max * max_pairs)
        random_answer = random.randint(0, upper_bound)
        return f"\\boxed{{{random_answer}}}"