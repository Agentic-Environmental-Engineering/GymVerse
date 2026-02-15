from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SurvoPuzzleEnv(Env):
    """Survo Puzzle environment - single-turn Q&A.
    
    The agent is given an N x M matrix with some cells filled and others set to -1,
    along with the required row sums and column sums. The task is to fill the empty
    cells with numbers from 0 to N*M-1 such that:
      1. Each number from 0 to N*M-1 appears exactly once in the matrix.
      2. The sum of each row matches the given row sums.
      3. The sum of each column matches the given column sums.
      4. The pre-filled cells must remain unchanged.
      
    The final answer must be provided inside \\boxed{...}, containing N lines,
    each with M integers separated by spaces.
    """

    def __init__(
        self,
        max_n_m: int = 5,
        sparsity: float = 0.3,
        **kwargs
    ):
        super().__init__()
        # Parameters for puzzle generation
        self.max_n_m = max_n_m
        self.sparsity = sparsity

        # State variables
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_solution: Optional[List[List[int]]] = None
        self.puzzle_matrix: Optional[List[List[int]]] = None
        self.row_sums: Optional[List[int]] = None
        self.col_sums: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Survo Puzzle.\n"
            "Fill the empty cells (-1) in the given matrix with numbers from 0 to N*M-1 so that:\n"
            "1) Each number appears exactly once in the entire matrix.\n"
            "2) The sum of each row equals the provided row sums.\n"
            "3) The sum of each column equals the provided column sums.\n"
            "4) Any pre-filled cell must remain unchanged.\n\n"
            "Output Format: Provide the completed matrix inside \\boxed{...} with exactly N lines.\n"
            "Each line must contain M integers separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new puzzle."""
        super().reset(seed)

        # Parameter validation
        assert isinstance(self.max_n_m, int), "max_n_m must be an integer"
        assert self.max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        assert isinstance(self.sparsity, float), "sparsity must be a float"
        assert 0.0 < self.sparsity < 1.0, "sparsity should be between 0 and 1"

        # Generate dimensions
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Generate a random permutation solution
        permutation = list(range(self.N * self.M))
        random.shuffle(permutation)
        self.reference_solution = [
            [permutation[i * self.M + j] for j in range(self.M)]
            for i in range(self.N)
        ]
        self.reference_answer_str = "\n".join(" ".join(map(str, row)) for row in self.reference_solution)

        # Compute sums
        self.row_sums = [sum(row) for row in self.reference_solution]
        self.col_sums = [
            sum(self.reference_solution[i][j] for i in range(self.N))
            for j in range(self.M)
        ]

        # Create puzzle matrix by introducing -1 sparsity
        self.puzzle_matrix = [row[:] for row in self.reference_solution]
        empty_cells_count = max(1, int(self.N * self.M * self.sparsity))
        empty_cells = random.sample(range(self.N * self.M), empty_cells_count)
        for cell in empty_cells:
            r, c = divmod(cell, self.M)
            self.puzzle_matrix[r][c] = -1

        # Build problem prompt
        matrix_str = "\n".join(" ".join(map(str, row)) for row in self.puzzle_matrix)
        self.current_problem = (
            f"You are given a {self.N} × {self.M} matrix with some cells filled with numbers from 0 to {self.N * self.M - 1}, "
            f"and some cells empty (represented by -1). Please fill the empty cells with numbers from 0 to {self.N * self.M - 1} such that:\n"
            f"1. Each number from 0 to {self.N * self.M - 1} appears exactly once in the matrix.\n"
            f"2. The sum of each row (from top to bottom) is: {' '.join(map(str, self.row_sums))}\n"
            f"3. The sum of each column (from left to right) is: {' '.join(map(str, self.col_sums))}\n\n"
            f"The matrix is given as follows:\n{matrix_str}\n\n"
            f"Output Format: Your final answer should be provided inside \\boxed{{...}} containing {self.N} lines, "
            f"each with {self.M} numbers, separated by spaces. The numbers should represent the completed matrix in row-major order."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step and verify the answer."""
        # Parse boxed answer content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse the matrix
        user_matrix = self._parse_matrix(boxed_content)

        if user_matrix is None or self.N is None or self.M is None:
            # Parsing failed or dimensions incorrect
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate dimensions
        if len(user_matrix) != self.N or any(len(row) != self.M for row in user_matrix):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate uniqueness and range of numbers
        flat_values = [v for row in user_matrix for v in row]
        if set(flat_values) != set(range(self.N * self.M)):
            info = {"error": "invalid_solution", "reason": "values_not_permutation"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate pre-filled cells remain unchanged
        assert self.puzzle_matrix is not None
        for r in range(self.N):
            for c in range(self.M):
                original_value = self.puzzle_matrix[r][c]
                if original_value != -1 and original_value != user_matrix[r][c]:
                    info = {"error": "invalid_solution", "reason": "prefilled_mismatch"}
                    return TERMINAL_STATE, 0.0, True, False, info

        # Validate row sums and column sums
        user_row_sums = [sum(row) for row in user_matrix]
        user_col_sums = [sum(user_matrix[i][j] for i in range(self.N)) for j in range(self.M)]
        assert self.row_sums is not None and self.col_sums is not None

        rows_ok = all(a == b for a, b in zip(user_row_sums, self.row_sums))
        cols_ok = all(a == b for a, b in zip(user_col_sums, self.col_sums))
        is_correct = rows_ok and cols_ok

        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_solution": self.reference_solution,
            "user_solution": user_matrix,
            "row_sums_match": rows_ok,
            "col_sums_match": cols_ok,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_matrix(self, content: str) -> Optional[List[List[int]]]:
        """Parse a matrix from boxed content. Expects N lines with M integers per line."""
        try:
            lines = [line.strip() for line in content.splitlines()]
            # Filter out empty lines
            lines = [line for line in lines if line]
            matrix: List[List[int]] = []
            for line in lines:
                row = list(map(int, line.split()))
                matrix.append(row)
            return matrix
        except Exception:
            return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random permutation matrix formatted inside \\boxed{...}."""
        if self.N is None or self.M is None:
            # Default safe size if called before reset
            N, M = 2, 2
        else:
            N, M = self.N, self.M

        permutation = list(range(N * M))
        random.shuffle(permutation)
        rows = []
        for i in range(N):
            row = permutation[i * M:(i + 1) * M]
            rows.append(" ".join(map(str, row)))
        content = "\n".join(rows)
        return f"\\boxed{{\n{content}\n}}"