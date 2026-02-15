import random
from typing import Any, Optional, List, SupportsFloat, Tuple
from itertools import combinations
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumSumDifferenceSubmatrixEnv(Env):
    """Environment for selecting rows and columns of a matrix to minimize sum of absolute differences of adjacent elements."""

    def __init__(
        self,
        max_n_m: int = 10,
        **kwargs
    ):
        """
        Initialize environment with parameter constraints.

        Args:
            max_n_m: The maximum value for N and M (matrix dimensions). Must be >= 3.
            **kwargs: Additional unused keyword arguments for compatibility.
        """
        super().__init__()
        assert max_n_m >= 3, "max_n_m should be greater than or equal to 3"
        self.max_n_m = max_n_m

        # Internal state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.R: Optional[int] = None
        self.C: Optional[int] = None
        self.matrix: Optional[List[List[int]]] = None
        self.gold_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task description and output format instructions."""
        return (
            "You are given an integer matrix and must select R rows and C columns to form an R × C submatrix.\n"
            "Your goal is to minimize the sum of absolute differences between all horizontally and vertically adjacent pairs in the submatrix.\n"
            "Output Format:\n"
            "- Provide two lines inside \\boxed{...}\n"
            "  * First line: R strictly increasing row indices (0-indexed), separated by a single space\n"
            "  * Second line: C strictly increasing column indices (0-indexed), separated by a single space\n"
            "Example:\n"
            "\\boxed{\\n0 2\\n1 3\\n}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(3, self.max_n_m)
        M = random.randint(3, self.max_n_m)
        R = random.randint(2, N - 1)
        C = random.randint(2, M - 1)
        matrix = [[random.randint(1, N * M) for _ in range(M)] for _ in range(N)]

        # Store parameters
        self.N, self.M, self.R, self.C = N, M, R, C
        self.matrix = matrix

        # Compute gold answer using enumeration of rows and DP over columns
        max_val = max(max(row) for row in matrix)
        max_pairs = (R - 1) * C + R * (C - 1)
        INF = max_val * max_pairs + 1
        ans = INF

        for rows in combinations(range(N), R):
            # Precompute w[j][i]: contribution when picking column j then column i
            # w[i][i] is the vertical adjacency cost within column i
            w = [[0] * M for _ in range(M)]

            for i in range(M):
                # Vertical adjacencies in column i
                for idx in range(1, R):
                    r0 = rows[idx - 1]
                    r1 = rows[idx]
                    w[i][i] += abs(matrix[r1][i] - matrix[r0][i])

                # Cross-column differences between column j and column i
                for j in range(i):
                    s = 0
                    for r0 in rows:
                        s += abs(matrix[r0][i] - matrix[r0][j])
                    w[j][i] = s

            # DP over columns: dp[i][k] = min cost to pick k columns ending at column i
            dp = [[INF] * (C + 1) for _ in range(M)]
            for i in range(M):
                dp[i][1] = w[i][i]

            for k in range(2, C + 1):
                for i in range(M):
                    best = INF
                    for j in range(i):
                        cost = dp[j][k - 1] + w[j][i] + w[i][i]
                        if cost < best:
                            best = cost
                    dp[i][k] = best

            # Update global answer
            for i in range(M):
                if dp[i][C] < ans:
                    ans = dp[i][C]

        self.gold_answer = ans

        # Build problem statement
        matrix_str = "\n".join(" ".join(map(str, row)) for row in matrix)
        self.current_problem = (
            f"You are given a {N} × {M} matrix of integers (rows 0..{N-1}, columns 0..{M-1}).\n"
            f"Please select {R} rows and {C} columns, denoted as r[1..{R}] and c[1..{C}], such that:\n"
            f"- 0 ≤ r[1] < ... < r[{R}] ≤ {N-1}\n"
            f"- 0 ≤ c[1] < ... < c[{C}] ≤ {M-1}\n\n"
            f"The matrix is:\n{matrix_str}\n\n"
            f"From these, extract an {R} × {C} submatrix using the chosen rows and columns.\n"
            f"Minimize the sum of absolute differences between all adjacent (horizontally or vertically) elements in the submatrix.\n"
            f"Two elements are adjacent if their Manhattan distance is 1.\n\n"
            f"Output the two lines (rows then columns) inside \\boxed{{...}} as described above."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": N,
            "M": M,
            "R": R,
            "C": C,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and return the result."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the two lines of indices from boxed content
        try:
            lines = [line.strip() for line in boxed_content.splitlines() if line.strip()]
            if len(lines) != 2:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

            row_indices = list(map(int, lines[0].split()))
            col_indices = list(map(int, lines[1].split()))
        except ValueError:
            # Not all tokens are integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate lengths
        if self.R is None or self.C is None or self.N is None or self.M is None or self.matrix is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_error"}

        if len(row_indices) != self.R or len(col_indices) != self.C:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate ranges and ordering
        if not all(0 <= r < self.N for r in row_indices) or not all(0 <= c < self.M for c in col_indices):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}
        if not all(row_indices[i] < row_indices[i + 1] for i in range(len(row_indices) - 1)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}
        if not all(col_indices[i] < col_indices[i + 1] for i in range(len(col_indices) - 1)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        # Build submatrix and compute sum of absolute differences for adjacent pairs
        new_matrix = [[self.matrix[row][col] for col in col_indices] for row in row_indices]
        sum_diff = 0
        for i in range(self.R):
            for j in range(self.C):
                if i < self.R - 1:
                    sum_diff += abs(new_matrix[i + 1][j] - new_matrix[i][j])
                if j < self.C - 1:
                    sum_diff += abs(new_matrix[i][j + 1] - new_matrix[i][j])

        is_correct = (sum_diff == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.gold_answer,
            "user_sum": sum_diff,
            "selected_rows": row_indices,
            "selected_cols": col_indices,
            "N": self.N,
            "M": self.M,
            "R": self.R,
            "C": self.C,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns the last occurrence if multiple."""
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action: two lines of strictly increasing indices inside \\boxed{...}."""
        if self.N is None or self.M is None or self.R is None or self.C is None:
            # Default fallback
            rows = [0, 1]
            cols = [0, 1]
        else:
            rows = sorted(random.sample(range(self.N), self.R))
            cols = sorted(random.sample(range(self.M), self.C))
        rows_str = " ".join(map(str, rows))
        cols_str = " ".join(map(str, cols))
        return f"\\boxed{{\n{rows_str}\n{cols_str}\n}}"