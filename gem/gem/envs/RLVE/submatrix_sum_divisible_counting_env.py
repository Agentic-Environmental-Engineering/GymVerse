from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SubmatrixSumDivisibleCountingEnv(Env):
    """Environment for counting submatrices with sums divisible by K - single-turn Q&A."""

    def __init__(
        self,
        max_n_m: int = 10,
    ):
        """
        Initialize the environment.

        Args:
            max_n_m: Maximum size for N and M (both chosen in [2, max_n_m]).
        """
        super().__init__()
        if max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")
        self.max_n_m = max_n_m

        # Internal state for the current problem
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None
        self.matrix: Optional[List[List[int]]] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a matrix subproblem counting task.\n"
            "Task: Count the number of contiguous, non-empty submatrices whose sum is divisible by K.\n"
            "Please provide your answer in \\boxed{...} format with a single non-negative integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameters
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)
        self.K = random.randint(2, self.N * self.M)

        # Generate matrix with entries in [0, K-1]
        self.matrix = [[random.randint(0, self.K - 1) for _ in range(self.M)] for _ in range(self.N)]

        # Compute the reference answer using 2D technique with modulo K
        self.reference_answer = self._count_submatrices_divisible_by_k(self.matrix, self.K)

        # Build problem string
        matrix_str = self._format_matrix_string(self.matrix)
        self.current_problem = (
            f"You are given a matrix of size {self.N} × {self.M}, where each element is an integer. "
            f"Count the number of contiguous, non-empty submatrices whose sum is divisible by {self.K}. "
            f"The matrix is:\n{matrix_str}\n\n"
            "Notes:\n"
            "- Two submatrices are considered different if they differ in position, even if they contain identical elements.\n"
            "- The entire matrix itself is also considered a submatrix.\n"
            "- Output a single non-negative integer, which is the total number of submatrices whose sum is divisible by K.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the user's answer and return the terminal state."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            # Not an integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the value inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _format_matrix_string(self, matrix: List[List[int]]) -> str:
        """Format the matrix similarly to the original environment."""
        lines = [", ".join(map(str, row)) for row in matrix]
        return "[\n" + "\n".join(lines) + "\n]"

    def _count_submatrices_divisible_by_k(self, matrix: List[List[int]], K: int) -> int:
        """Count the number of contiguous submatrices with sum divisible by K."""
        N = len(matrix)
        M = len(matrix[0]) if N > 0 else 0

        # 2D prefix sums modulo K, 1-indexed
        a = [[0] * (M + 1) for _ in range(N + 1)]
        for i in range(1, N + 1):
            row = matrix[i - 1]
            ai = a[i]
            ai_1 = a[i - 1]
            for j in range(1, M + 1):
                v = row[j - 1]  # each a[i][j] <= K
                ai[j] = (v + ai_1[j] + ai[j - 1] + K - ai_1[j - 1]) % K

        ans = 0
        b = [0] * (M + 1)     # reuse across pairs of rows
        cnt = [0] * K         # frequency array modulo K

        # Enumerate pairs of rows (top=i+1 .. bottom=j)
        for i in range(0, N):
            ai = a[i]
            for j in range(i + 1, N + 1):
                aj = a[j]
                cnt[0] = 1  # empty prefix
                # Sweep columns, counting subarrays with sum % K == 0
                for k in range(1, M + 1):
                    v = aj[k] - ai[k]   # both already modulo K
                    if v < 0:
                        v += K
                    b[k] = v
                    ans += cnt[v]
                    cnt[v] += 1
                # reset only the touched buckets
                for k in range(1, M + 1):
                    cnt[b[k]] = 0

        return ans

    def sample_random_action(self) -> str:
        """Sample a random action (boxed integer)."""
        random_answer = random.randint(0, max(1, (self.N or 2) * (self.M or 2)))
        return f"\\boxed{{{random_answer}}}"