from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MatrixPermutation_BothDiagonalOneEnv(Env):
    """Environment for matrix row/column permutation to make both diagonals all ones - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, use this fixed matrix size (must be at least 2).
        - min_N: Minimum matrix size when sampling N randomly (default 2).
        - max_N: Maximum matrix size when sampling N randomly (default 10).
        """
        super().__init__()
        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Internal state
        self.current_problem: Optional[str] = None
        self.N_current: Optional[int] = None
        self.A: Optional[List[List[int]]] = None
        self.reference_row_permutation: Optional[List[int]] = None
        self.reference_column_permutation: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a 0-indexed square binary matrix A of size N × N.\n"
            "Your task is to find a permutation of row indices and a permutation of column indices such that\n"
            "after reordering rows and columns (i.e., element (i, j) becomes A[a[i]][b[j]]), both the main diagonal\n"
            "(i = j) and the anti-diagonal (i + j = N - 1) of the resulting matrix contain only 1s.\n\n"
            "Answer Format:\n"
            "- Provide two lines inside a single \\boxed{...} block.\n"
            "- First line: the row permutation a[0] a[1] ... a[N-1].\n"
            "- Second line: the column permutation b[0] b[1] ... b[N-1].\n"
            "- Use spaces to separate adjacent integers. Do not include quotes.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            if self.fixed_N < 2:
                raise ValueError("N must be at least 2.")
            N = self.fixed_N
        else:
            if self.min_N < 2:
                raise ValueError("min_N must be at least 2.")
            if self.max_N < self.min_N:
                raise ValueError("max_N must be greater than or equal to min_N.")
            N = random.randint(self.min_N, self.max_N)

        self.N_current = N

        # Generate matrix A
        one_probability = random.random() / 4.0
        A = [[1 if random.random() < one_probability else 0 for _ in range(N)] for _ in range(N)]

        # Generate reference permutations ensuring solution exists
        row_permutation = list(range(N))
        random.shuffle(row_permutation)
        column_permutation = list(range(N))
        random.shuffle(column_permutation)

        for i in range(N):
            A[row_permutation[i]][column_permutation[i]] = 1
        for i in range(N):
            A[row_permutation[i]][column_permutation[N - 1 - i]] = 1

        self.A = A
        self.reference_row_permutation = row_permutation
        self.reference_column_permutation = column_permutation

        # Build problem statement
        matrix_str = "\n".join("".join(map(str, row)) for row in A)
        problem = (
            f"You are given a square matrix of size {N} × {N}, where each element is either 0 or 1. "
            "This matrix is 0-indexed.\n\n"
            "Please find:\n"
            f"- a permutation of the row indices: a[0], ..., a[{N-1}] (a reordering of 0 to {N-1}),\n"
            f"- a permutation of the column indices: b[0], ..., b[{N-1}] (a reordering of 0 to {N-1}),\n"
            "such that after applying these permutations to the rows and columns of matrix A "
            "(i.e., the element at position (i, j) becomes A[a[i]][b[j]]), both diagonals of the resulting matrix contain only 1s — "
            f"that is, all positions where i = j (main diagonal) and i + j = {N-1} (anti-diagonal).\n\n"
            f"Matrix A is given as follows:\n{matrix_str}\n\n"
            "Output Format:\n"
            "- Put your two-line answer inside a single \\boxed{...} block.\n"
            "- First line: a[0] a[1] ... a[N-1]\n"
            "- Second line: b[0] b[1] ... b[N-1]\n"
            "- Use spaces to separate adjacent integers."
        )

        self.current_problem = problem
        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": N,
            "A": A,
            "reference_row_permutation": row_permutation,
            "reference_column_permutation": column_permutation,
        }

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse permutations from boxed content
        parsed = self._parse_permutations(boxed_content)
        if parsed is None:
            # Format error within boxed content
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        row_perm, col_perm = parsed
        N = self.N_current
        A = self.A

        if N is None or A is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Validate permutations
        is_row_valid = (len(row_perm) == N and set(row_perm) == set(range(N)))
        is_col_valid = (len(col_perm) == N and set(col_perm) == set(range(N)))

        if not is_row_valid or not is_col_valid:
            info = {
                "error": "invalid_solution",
                "row_valid": is_row_valid,
                "col_valid": is_col_valid,
                "user_row_permutation": row_perm,
                "user_column_permutation": col_perm,
                "reference_row_permutation": self.reference_row_permutation,
                "reference_column_permutation": self.reference_column_permutation,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Apply permutations to form matrix B
        B = [[A[row_perm[i]][col_perm[j]] for j in range(N)] for i in range(N)]

        # Check both diagonals
        satisfied = 0
        total = 0
        correct = True
        for i in range(N):
            for j in range(N):
                if i == j or i + j == N - 1:
                    total += 1
                    if B[i][j] == 1:
                        satisfied += 1
                    else:
                        correct = False

        reward = 1.0 if correct else 0.0
        info = {
            "correct": correct,
            "satisfied": satisfied,
            "total": total,
            "N": N,
            "user_row_permutation": row_perm,
            "user_column_permutation": col_perm,
            "reference_row_permutation": self.reference_row_permutation,
            "reference_column_permutation": self.reference_column_permutation,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} block content."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text, flags=0)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_permutations(self, content: str) -> Optional[Tuple[List[int], List[int]]]:
        """Parse two lines of permutations from the boxed content."""
        try:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if len(lines) != 2:
                return None
            row_perm = list(map(int, lines[0].split()))
            col_perm = list(map(int, lines[1].split()))
            return row_perm, col_perm
        except Exception:
            return None

    def sample_random_action(self) -> str:
        """Sample a random action: random permutations boxed."""
        if self.N_current is None:
            # Provide a minimal valid format when called before reset
            return "\\boxed{0\n0}"
        N = self.N_current
        row = list(range(N))
        col = list(range(N))
        random.shuffle(row)
        random.shuffle(col)
        row_str = " ".join(map(str, row))
        col_str = " ".join(map(str, col))
        return f"\\boxed{{{row_str}\n{col_str}}}"