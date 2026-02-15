from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxThreeSquareSumEnv(Env):
    """Environment for maximizing the sum of three non-overlapping K×K squares in a grid."""

    def __init__(
        self,
        max_n_m: int = 10,
        # Optional configuration for cell values; by default, uses [0, max_n_m]
        max_cell_value: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n_m: Maximum value for N and M (inclusive lower bound is 4).
            max_cell_value: Maximum value (inclusive) for each cell in the grid. If None, defaults to max_n_m.
        """
        super().__init__()
        if max_n_m < 4:
            raise ValueError("max_n_m should be greater than or equal to 4")
        self.max_n_m: int = max_n_m
        self.max_cell_value: int = max_n_m if max_cell_value is None else max_cell_value

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None
        self.grid: Optional[list[list[int]]] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Given a grid of integers, find three non-overlapping K×K squares such that the sum of "
            "all values in the three squares is maximized.\n"
            "Answer Format: Provide a single integer wrapped in \\boxed{...}.\n"
            "Example: \\boxed{12345}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate N, M, K, and the grid
        N = random.randint(4, self.max_n_m)
        M = random.randint(4, self.max_n_m)
        K = random.randint(2, min(N, M) // 2)
        A = [[random.randint(0, self.max_cell_value) for _ in range(M)] for _ in range(N)]

        # Compute the reference answer using the algorithm from the original environment
        ans = self._compute_max_three_square_sum(A, N, M, K)

        # Persist state
        self.N = N
        self.M = M
        self.K = K
        self.grid = A
        self.reference_answer = ans

        # Build problem prompt
        grid_str = "\n".join(" ".join(map(str, row)) for row in A)
        self.current_problem = (
            f"You are given a grid of size {N} × {M}, where each cell contains an integer. "
            f"Please find three non-overlapping {K} × {K} squares in the grid such that the sum of all values "
            f"in the three squares is maximized. The grid is provided as follows:\n{grid_str}\n\n"
            "Output Format: Output a single integer in \\boxed{...} — the maximum possible sum of values "
            f"from the three non-overlapping {K} × {K} squares."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and return the result."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Verify numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer answer in boxed format)."""
        # Provide a random guess; the range is arbitrary since it's only for sampling purposes
        random_answer = random.randint(0, max(1, (self.max_cell_value * 3 * (self.K or 2) * (self.K or 2)) if self.K else 1000))
        return f"\\boxed{{{random_answer}}}"

    def _compute_max_three_square_sum(self, A: list[list[int]], N: int, M: int, K: int) -> int:
        """Compute the maximum sum of three non-overlapping K×K squares using prefix sums and DP."""
        # Prefix sum array S with padding
        S = [[0] * (M + 1) for _ in range(N + 1)]
        for i in range(N):
            for j in range(M):
                S[i + 1][j + 1] = A[i][j] + S[i][j + 1] + S[i + 1][j] - S[i][j]

        # Sum of K×K square ending at (i, j)
        def cal(i: int, j: int) -> int:
            if i < K - 1 or j < K - 1:
                return 0
            return (
                S[i + 1][j + 1]
                - S[i + 1 - K][j + 1]
                - S[i + 1][j + 1 - K]
                + S[i + 1 - K][j + 1 - K]
            )

        # mxx[i]: best K×K whose bottom row is i
        # mxy[j]: best K×K whose right column is j
        mxx = [0] * N
        mxy = [0] * M
        for i in range(K - 1, N):
            for j in range(K - 1, M):
                v = cal(i, j)
                if v > mxx[i]:
                    mxx[i] = v
                if v > mxy[j]:
                    mxy[j] = v

        # a[l][r] = max(mxx[t] for t in [l..r])
        a = [[0] * N for _ in range(N)]
        for l in range(N):
            a[l][l] = mxx[l]
            for r in range(l + 1, N):
                a[l][r] = max(a[l][r - 1], mxx[r])

        # b[l][r] = max(mxy[t] for t in [l..r])
        b = [[0] * M for _ in range(M)]
        for l in range(M):
            b[l][l] = mxy[l]
            for r in range(l + 1, M):
                b[l][r] = max(b[l][r - 1], mxy[r])

        # Four quadrant DP arrays
        lu = [[0] * M for _ in range(N)]
        for i in range(N):
            for j in range(M):
                best = cal(i, j)
                if i > 0:
                    best = max(best, lu[i - 1][j])
                if j > 0:
                    best = max(best, lu[i][j - 1])
                lu[i][j] = best

        ru = [[0] * M for _ in range(N)]
        for i in range(N):
            for j in range(M - 1, -1, -1):
                best = cal(i, j + K - 1) if j + K - 1 < M else 0
                if i > 0:
                    best = max(best, ru[i - 1][j])
                if j + 1 < M:
                    best = max(best, ru[i][j + 1])
                ru[i][j] = best

        ld = [[0] * M for _ in range(N)]
        for i in range(N - 1, -1, -1):
            for j in range(M):
                best = cal(i + K - 1, j) if i + K - 1 < N else 0
                if i + 1 < N:
                    best = max(best, ld[i + 1][j])
                if j > 0:
                    best = max(best, ld[i][j - 1])
                ld[i][j] = best

        rd = [[0] * M for _ in range(N)]
        for i in range(N - 1, -1, -1):
            for j in range(M - 1, -1, -1):
                best = cal(i + K - 1, j + K - 1) if i + K - 1 < N and j + K - 1 < M else 0
                if i + 1 < N:
                    best = max(best, rd[i + 1][j])
                if j + 1 < M:
                    best = max(best, rd[i][j + 1])
                rd[i][j] = best

        # Try all 3-square patterns
        ans = 0

        # 1) Three horizontal strips
        for i in range(N):
            for j in range(i + K, N - K):
                total = a[0][i] + a[i + K][j] + a[j + K][N - 1]
                if total > ans:
                    ans = total

        # 2) Three vertical strips
        for i in range(M):
            for j in range(i + K, M - K):
                total = b[0][i] + b[i + K][j] + b[j + K][M - 1]
                if total > ans:
                    ans = total

        # 3) L-shaped splits
        for i in range(N):
            for j in range(M):
                # Top split then horizontal split
                if i + K < N and j + 1 < M:
                    ans = max(ans, lu[i][j] + ru[i][j + 1] + a[i + K][N - 1])
                # Bottom split then horizontal split
                if i >= K and j + 1 < M:
                    ans = max(ans, ld[i][j] + rd[i][j + 1] + a[0][i - 1])
                # Left split then vertical split
                if j + K < M and i + 1 < N:
                    ans = max(ans, lu[i][j] + ld[i + 1][j] + b[j + K][M - 1])
                # Right split then vertical split
                if j >= K and i + 1 < N:
                    ans = max(ans, ru[i][j] + rd[i + 1][j] + b[0][j - 1])

        return ans