import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Binario_NoAdjacencyRequirementEnv(Env):
    """Binario environment without adjacency requirement - single-turn Q&A."""

    def __init__(
        self,
        max_n_m: int = 5,
        sparsity: float = 0.2,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_n_m: Maximum value for N and M (N and M will be chosen uniformly from [2, max_n_m]).
        - sparsity: Fraction of cells to be replaced with '*' in the given matrix (0 < sparsity < 1).
        """
        super().__init__()

        # Validate parameters
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"

        self.max_n_m = max_n_m
        self.sparsity = sparsity

        # Runtime state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.solution_matrix: Optional[List[str]] = None
        self.given_matrix: Optional[List[str]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a (2 × N) × (2 × M) matrix filled with '0', '1', or '*' "
            "('*' means the cell is empty). Your task is to fill all '*' with '0' or '1' such that:\n"
            "1. Each row contains exactly M '0's and M '1's.\n"
            "2. Each column contains exactly N '0's and N '1's.\n\n"
            "Output Format: Your final answer must be the completed matrix enclosed in \\boxed{...}.\n"
            "Inside the box, output exactly (2 × N) lines, each of length (2 × M), with only '0' or '1'. "
            "Use one row per line, no separators.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Choose N and M randomly within [2, max_n_m]
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Generate a valid solution matrix using parity of row/col permutations
        row_perm = list(range(2 * self.N))
        col_perm = list(range(2 * self.M))
        random.shuffle(row_perm)
        random.shuffle(col_perm)

        solution = [
            [str((row_perm[i] + col_perm[j]) % 2) for j in range(2 * self.M)]
            for i in range(2 * self.N)
        ]
        self.solution_matrix = ["".join(row) for row in solution]
        self.reference_answer = "\n".join(self.solution_matrix)

        # Apply sparsity to create the given matrix with '*'
        total_cells = (2 * self.N) * (2 * self.M)
        num_empty = max(1, int(total_cells * self.sparsity))
        empty_indices = set(random.sample(range(total_cells), num_empty))

        given = []
        for idx in range(total_cells):
            r, c = divmod(idx, 2 * self.M)
            if idx in empty_indices:
                given.append('*')
            else:
                given.append(self.solution_matrix[r][c])

        # Convert to row strings
        self.given_matrix = [
            "".join(given[r * (2 * self.M):(r + 1) * (2 * self.M)])
            for r in range(2 * self.N)
        ]

        # Build problem prompt
        matrix_str = "\n".join(self.given_matrix)
        self.current_problem = (
            f"You are given a (2 × {self.N}) × (2 × {self.M}) matrix. Each cell contains either '0', '1', or '*' "
            f"('*' means the cell is empty). Please fill all '*' cells with either '0' or '1' such that:\n"
            f"1. Each row contains exactly {self.M} '0's and {self.M} '1's.\n"
            f"2. Each column contains exactly {self.N} '0's and {self.N} '1's.\n\n"
            f"The matrix is given in row-major order, with each row represented as a string of '0', '1', and '*':\n"
            f"{matrix_str}\n\n"
            f"Output Format: Your final answer should be (2 × {self.N}) lines, each containing (2 × {self.M}) "
            f"characters ('0' or '1'), enclosed in \\boxed{{...}}. Keep one row per line inside the box, no separators."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        boxed_content = self._parse_answer(action)

        # Format error: no boxed content found
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the matrix from boxed content
        lines = [ln.strip() for ln in boxed_content.splitlines() if ln.strip()]

        # Validate basic format (dimensions and characters)
        assert self.N is not None and self.M is not None
        expected_rows = 2 * self.N
        expected_cols = 2 * self.M

        if len(lines) != expected_rows or any(len(row) != expected_cols for row in lines):
            info = {
                "error": "wrong_format",
                "expected_rows": expected_rows,
                "expected_cols": expected_cols,
                "received_rows": len(lines),
                "received_cols": [len(row) for row in lines],
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if any(not all(ch in "01" for ch in row) for row in lines):
            return TERMINAL_STATE, 0.0, True, False, {"error": "wrong_format", "detail": "non_binary_characters"}

        # Validate that pre-filled cells are unchanged
        assert self.given_matrix is not None
        for i in range(expected_rows):
            for j in range(expected_cols):
                original = self.given_matrix[i][j]
                proposed = lines[i][j]
                if original != '*' and proposed != original:
                    return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "position": (i, j)}

        # Validate row constraints
        for i in range(expected_rows):
            ones = lines[i].count('1')
            zeros = lines[i].count('0')
            if ones != zeros or ones != self.M or zeros != self.M:
                return TERMINAL_STATE, 0.0, True, False, {
                    "error": "wrong_solution",
                    "detail": "row_balance_violation",
                    "row_index": i,
                    "ones": ones,
                    "zeros": zeros,
                    "expected_each": self.M,
                }

        # Validate column constraints
        for j in range(expected_cols):
            ones = sum(lines[i][j] == '1' for i in range(expected_rows))
            zeros = sum(lines[i][j] == '0' for i in range(expected_rows))
            if ones != zeros or ones != self.N or zeros != self.N:
                return TERMINAL_STATE, 0.0, True, False, {
                    "error": "wrong_solution",
                    "detail": "column_balance_violation",
                    "col_index": j,
                    "ones": ones,
                    "zeros": zeros,
                    "expected_each": self.N,
                }

        # If all checks pass, correct solution
        info = {
            "correct": True,
            "reference_answer": self.reference_answer,
            "user_answer": "\n".join(lines),
            "n": self.N,
            "m": self.M,
            "given_matrix": self.given_matrix,
        }
        return TERMINAL_STATE, 1.0, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} (supports multiline)."""
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by filling '*' randomly and keeping known cells."""
        if self.N is None or self.M is None or self.given_matrix is None:
            # Fallback if called before reset
            return "\\boxed{0}"

        rows = []
        for i in range(2 * self.N):
            row_chars = []
            for j in range(2 * self.M):
                c = self.given_matrix[i][j]
                if c == '*':
                    row_chars.append(random.choice(['0', '1']))
                else:
                    row_chars.append(c)
            rows.append("".join(row_chars))
        return "\\boxed{\n" + "\n".join(rows) + "\n}"