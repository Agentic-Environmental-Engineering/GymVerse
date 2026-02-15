from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class AndOr_Sequence_CountingEnv(Env):
    """Bitwise AND/OR sequence counting environment - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        M: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 8,
        min_M: int = 1,
        max_M: int = 8,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed length of the arrays (must be >= 2 if provided).
        - M: Optional fixed bit-width parameter (must be >= 1 if provided).
        - min_N, max_N: Range for sampling N when not provided.
        - min_M, max_M: Range for sampling M when not provided.
        """
        super().__init__()
        # Validation of ranges
        if min_N < 2:
            raise ValueError("min_N must be >= 2.")
        if max_N < min_N:
            raise ValueError("max_N must be >= min_N.")
        if min_M < 1:
            raise ValueError("min_M must be >= 1.")
        if max_M < min_M:
            raise ValueError("max_M must be >= min_M.")
        if N is not None and N < 2:
            raise ValueError("N should be greater than or equal to 2.")
        if M is not None and M < 1:
            raise ValueError("M should be greater than or equal to 1.")

        self.N_fixed = N
        self.M_fixed = M
        self.min_N = min_N
        self.max_N = max_N
        self.min_M = min_M
        self.max_M = max_M

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.A: Optional[list[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a bitwise AND/OR sequence counting problem.\n"
            "Please provide your answer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N and M
        self.N = self.N_fixed if self.N_fixed is not None else random.randint(self.min_N, self.max_N)
        self.M = self.M_fixed if self.M_fixed is not None else random.randint(self.min_M, self.max_M)

        # Validate N and M
        if self.N < 2:
            raise ValueError("N should be greater than or equal to 2.")
        if self.M < 1:
            raise ValueError("M should be greater than or equal to 1.")

        # Generate A
        power_2_M = 2 ** self.M
        self.A = [random.randint(0, power_2_M - 1) for _ in range(self.N)]

        # Build problem prompt
        A_str = " ".join(f"A[{i}]={v}" for i, v in enumerate(self.A))
        self.current_problem = (
            f"You are given an integer array `A` of length {self.N}:\n"
            f"{A_str}\n\n"
            f"Please count the number of valid integer arrays `B` of length {self.N} that satisfy the following conditions:\n"
            f"- For all indices 0 <= i <= {self.N - 1}, the value B[i] must be in the range: 0 <= B[i] < 2^{{M}} = {power_2_M}\n"
            f"- For all indices 0 <= i < {self.N - 1}, the following bitwise conditions hold:\n"
            f"  - (A[i] & B[i]) <= (A[i + 1] & B[i + 1])\n"
            f"  - (A[i] | B[i]) >= (A[i + 1] | B[i + 1])\n"
            f"  - (Here, `&` is the bitwise AND operator and `|` is the bitwise OR operator.)\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer using the original DP decomposition
        self.reference_answer = self._compute_reference_answer(self.N, self.M, self.A)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse answer from \boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _dp1(N: int, M_minus_1: int, A: list[int]) -> int:
        """
        DP for enforcing non-decreasing (A[i] & B[i]) across i, processed bit-by-bit.
        M_minus_1 corresponds to the maximum bit index to process (i.e., M-1).
        """
        F = [[[0] * N for _ in range(N)] for _ in range(2)]
        for l in range(N):
            for r in range(l, N):
                F[1][l][r] = 1

        # Iterate over bits 0..M_minus_1
        for b in range(M_minus_1 + 1):
            now = b % 2
            lst = now ^ 1

            # Reset current layer
            for i in range(N):
                for j in range(N):
                    F[now][i][j] = 0

            # Prefix sum of bit b in A
            Pre = [0] * (N + 1)
            for i in range(1, N + 1):
                Pre[i] = Pre[i - 1] + ((A[i - 1] >> b) & 1)

            # Combine segments
            for l in range(N):
                for r in range(l, N):
                    for x in range(l - 1, r + 1):
                        # Ensure all A[x+1..r] bit b are 1 (i.e., count equals r - x)
                        if Pre[r + 1] - Pre[x + 1] != (r - x):
                            continue
                        left_count = F[lst][l][x] if x >= l else 1
                        right_count = F[lst][x + 1][r] if x + 1 <= r else 1
                        F[now][l][r] += left_count * right_count

        return F[M_minus_1 % 2][0][N - 1]

    @staticmethod
    def _dp2(N: int, M_minus_1: int, A: list[int]) -> int:
        """
        DP for enforcing non-increasing (A[i] | B[i]) across i, processed bit-by-bit.
        M_minus_1 corresponds to the maximum bit index to process (i.e., M-1).
        """
        F = [[[0] * N for _ in range(N)] for _ in range(2)]
        for l in range(N):
            for r in range(l, N):
                F[1][l][r] = 1

        # Iterate over bits 0..M_minus_1
        for b in range(M_minus_1 + 1):
            now = b % 2
            lst = now ^ 1

            # Reset current layer
            for i in range(N):
                for j in range(N):
                    F[now][i][j] = 0

            # Prefix sum of bit b in A
            Pre = [0] * (N + 1)
            for i in range(1, N + 1):
                Pre[i] = Pre[i - 1] + ((A[i - 1] >> b) & 1)

            # Combine segments
            for l in range(N):
                for r in range(l, N):
                    for x in range(l - 1, r + 1):
                        # Ensure all A[x+1..r] bit b are 0 (i.e., count equals 0)
                        if Pre[r + 1] - Pre[x + 1] != 0:
                            continue
                        left_count = F[lst][l][x] if x >= l else 1
                        right_count = F[lst][x + 1][r] if x + 1 <= r else 1
                        F[now][l][r] += left_count * right_count

        return F[M_minus_1 % 2][0][N - 1]

    @classmethod
    def _compute_reference_answer(cls, N: int, M: int, A: list[int]) -> int:
        """
        Compute the reference answer by multiplying the results of the two DP procedures
        using bits 0..M-1.
        """
        return cls._dp1(N, M - 1, A) * cls._dp2(N, M - 1, A)

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        random_answer = random.randint(0, 10**6)
        return f"\\boxed{{{random_answer}}}"