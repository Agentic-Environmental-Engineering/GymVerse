from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MatrixPermutation_MainDiagonalOneEnv(Env):
    """Environment for the Matrix Permutation Main Diagonal Ones problem (single-turn Q&A)."""

    def __init__(self, N: int = 5, **kwargs):
        """
        Initialize the environment.

        Args:
            N: Size of the square matrix (must be at least 2).
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer."
        assert N >= 2, "N must be at least 2."
        self.N: int = N

        # Problem state
        self.A: Optional[List[List[int]]] = None
        self.row_permutation_ref: Optional[List[int]] = None
        self.column_permutation_ref: Optional[List[int]] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the general task instructions."""
        return (
            "You are given a square matrix A with entries 0 or 1. The task is to find two permutations of row and column indices such that "
            "after permuting rows and columns accordingly, the main diagonal of the resulting matrix contains only 1s.\n"
            "Answer format:\n"
            "- Provide two lines inside a single \\boxed{...} block:\n"
            "  1) First line: the row permutation a[0] a[1] ... a[N-1]\n"
            "  2) Second line: the column permutation b[0] b[1] ... b[N-1]\n"
            "Use spaces to separate integers. Do not include commas or quotes.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N

        # Generate a random matrix with certain probability of 1's
        one_probability = random.random() / 2.0
        A = [[1 if random.random() < one_probability else 0 for _ in range(N)] for _ in range(N)]

        # Generate a random pair of permutations and enforce diagonal ones after applying them
        row_perm = list(range(N))
        random.shuffle(row_perm)
        col_perm = list(range(N))
        random.shuffle(col_perm)
        for i in range(N):
            A[row_perm[i]][col_perm[i]] = 1

        self.A = A
        self.row_permutation_ref = row_perm
        self.column_permutation_ref = col_perm

        # Build the problem description
        matrix_str = "\n".join("".join(map(str, row)) for row in A)
        self.current_problem = (
            f"You are given a square matrix of size {N} × {N}, where each element is either 0 or 1. This matrix is 0-indexed.\n\n"
            f"Please find:\n"
            f"- a permutation of the row indices: a[0], ..., a[{N - 1}] (a reordering of 0 to {N - 1}),\n"
            f"- a permutation of the column indices: b[0], ..., b[{N - 1}] (a reordering of 0 to {N - 1}),\n"
            f"- such that after applying these permutations to the rows and columns of the matrix A (i.e., the element at position (i, j) "
            f"becomes A[a[i]][b[j]]), the main diagonal of the resulting matrix contains only 1s (main diagonal refers to the elements at "
            f"position (i, i) for i from 0 to {N - 1}).\n\n"
            f"Matrix A is given as follows:\n{matrix_str}\n\n"
            f"Output Format:\n"
            f"Your final answer must be placed inside a single \\boxed{{...}} block and contain exactly two lines:\n"
            f"- First line: a[0] a[1] ... a[{N - 1}]\n"
            f"- Second line: b[0] b[1] ... b[{N - 1}]\n"
            f"Use spaces to separate adjacent integers."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted permutations and compute the reward."""
        # Parse the boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure problem is initialized
        if self.A is None or self.row_permutation_ref is None or self.column_permutation_ref is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        # Parse permutations from the boxed content
        lines = [line.strip() for line in boxed_content.splitlines() if line.strip()]
        if len(lines) != 2:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "must_provide_two_lines"}

        try:
            row_perm = list(map(int, lines[0].split()))
            col_perm = list(map(int, lines[1].split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "non_integer_values"}

        N = self.N
        # Validate permutations
        if not (len(row_perm) == N and set(row_perm) == set(range(N))):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "invalid_row_permutation"}
        if not (len(col_perm) == N and set(col_perm) == set(range(N))):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "invalid_column_permutation"}

        # Check if all diagonal entries become 1 after applying permutations:
        # B[i][i] = A[row_perm[i]][col_perm[i]]
        satisfied = sum(1 for i in range(N) if self.A[row_perm[i]][col_perm[i]] == 1)
        is_correct = (satisfied == N)

        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied": satisfied,
            "total": N,
            "user_row_permutation": row_perm,
            "user_column_permutation": col_perm,
            "reference_row_permutation": self.row_permutation_ref,
            "reference_column_permutation": self.column_permutation_ref,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text, allowing multiline content."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}. If available, returns a correct sample."""
        if self.row_permutation_ref is not None and self.column_permutation_ref is not None:
            row_line = " ".join(map(str, self.row_permutation_ref))
            col_line = " ".join(map(str, self.column_permutation_ref))
            return f"\\boxed{{{row_line}\n{col_line}}}"
        else:
            # Fallback: produce a random (possibly incorrect) action
            N = self.N
            row = list(range(N))
            col = list(range(N))
            random.shuffle(row)
            random.shuffle(col)
            row_line = " ".join(map(str, row))
            col_line = " ".join(map(str, col))
            return f"\\boxed{{{row_line}\n{col_line}}}"