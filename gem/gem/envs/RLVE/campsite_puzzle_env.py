from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CampsitePuzzleEnv(Env):
    """Campsite puzzle environment - single turn Q&A in GEM format."""

    prompt_template = (
        "You are given a {N} × {M} matrix. Each cell contains either '0', '1', or '*' "
        "('*' means the cell is empty). Please fill all '*' cells with either '0' or '1' such that:\n"
        "1. No two (horizontally or vertically) adjacent cells can both contain 1.\n"
        "2. The number of 1s in each row (from top to bottom) is: {row_counts}.\n"
        "3. The number of 1s in each column (from left to right) is: {col_counts}.\n\n"
        "The matrix is given in row-major order, with each row represented as a string of '0', '1', and '*':\n"
        "{matrix}\n\n"
        "Output Format: Put your final matrix (exactly {N} lines, each with {M} characters of 0/1) "
        "inside a single \\boxed{{...}}. Lines should be separated by newline characters.\n"
    )

    def __init__(
        self,
        max_n_m: int = 6,
        sparsity: float = 0.5,
        # The following are preserved for compatibility with original environment defaults
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(satisfied/all)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        super().__init__()
        # Parameter validation and storage
        assert isinstance(max_n_m, int) and max_n_m >= 2, "max_n_m should be an integer >= 2"
        assert 0.0 < sparsity < 1.0, "sparsity should be between 0 and 1 (exclusive)"

        self.max_n_m = max_n_m
        self.sparsity = sparsity

        # Preserved but unused due to GEM reward scheme
        self.wrong_format = wrong_format
        self.invalid_solution = invalid_solution
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # State variables
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.original_matrix: Optional[List[str]] = None  # With '*' holes
        self.row_counts: Optional[List[int]] = None
        self.col_counts: Optional[List[int]] = None
        self.reference_answer: Optional[str] = None  # Full correct 0/1 matrix (N lines)
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a binary grid puzzle.\n"
            "Fill the empty cells ('*') with '0' or '1' to satisfy adjacency and count constraints.\n"
            "Submit your final matrix as N lines of 0/1 inside a single \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new puzzle instance."""
        super().reset(seed)

        # Choose dimensions
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Generate a valid full solution matrix (no adjacent '1's horizontally/vertically)
        full_matrix = self._generate_full_matrix(self.N, self.M)
        self.reference_answer = "\n".join("".join(row) for row in full_matrix)

        # Compute target row/column counts from the full solution
        self.row_counts = [sum(1 for c in row if c == "1") for row in full_matrix]
        self.col_counts = [
            sum(1 for i in range(self.N) if full_matrix[i][j] == "1") for j in range(self.M)
        ]

        # Create the puzzle by turning some cells into '*'
        puzzle_matrix = [row[:] for row in full_matrix]
        total_cells = self.N * self.M
        empty_count = max(1, int(total_cells * self.sparsity))
        empty_indices = random.sample(range(total_cells), empty_count)
        for idx in empty_indices:
            r, c = divmod(idx, self.M)
            puzzle_matrix[r][c] = "*"

        self.original_matrix = ["".join(row) for row in puzzle_matrix]

        problem_text = self.prompt_template.format(
            N=self.N,
            M=self.M,
            matrix="\n".join(self.original_matrix),
            row_counts=", ".join(map(str, self.row_counts)),
            col_counts=", ".join(map(str, self.col_counts)),
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the submitted solution and return the result."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse matrix from boxed content
        user_lines = [line.strip() for line in boxed_content.splitlines() if line.strip()]

        if self.N is None or self.M is None or self.original_matrix is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        if len(user_lines) != self.N:
            return TERMINAL_STATE, -0.1, True, False, {"error": "dimension_mismatch"}

        if any(len(row) != self.M for row in user_lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "dimension_mismatch"}

        if any(any(ch not in "01" for ch in row) for row in user_lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "invalid_characters"}

        # Check consistency with pre-filled cells
        for i in range(self.N):
            for j in range(self.M):
                orig_ch = self.original_matrix[i][j]
                if orig_ch != "*" and orig_ch != user_lines[i][j]:
                    return TERMINAL_STATE, 0.0, True, False, {"error": "prefilled_mismatch"}

        # Check adjacency constraint: no two adjacent '1's
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for i in range(self.N):
            for j in range(self.M):
                if user_lines[i][j] == "1":
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.N and 0 <= nj < self.M and user_lines[ni][nj] == "1":
                            return TERMINAL_STATE, 0.0, True, False, {"error": "adjacency_violation"}

        # Check row and column counts
        user_row_counts = [sum(1 for c in row if c == "1") for row in user_lines]
        user_col_counts = [
            sum(1 for i in range(self.N) if user_lines[i][j] == "1") for j in range(self.M)
        ]

        if user_row_counts != self.row_counts:
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "row_count_mismatch",
                "target": self.row_counts,
                "user": user_row_counts,
            }

        if user_col_counts != self.col_counts:
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "col_count_mismatch",
                "target": self.col_counts,
                "user": user_col_counts,
            }

        info = {
            "correct": True,
            "reference_answer": self.reference_answer,
            "user_answer": user_lines,
            "N": self.N,
            "M": self.M,
            "row_counts": self.row_counts,
            "col_counts": self.col_counts,
        }
        return TERMINAL_STATE, 1.0, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random matrix placed inside \\boxed{...}."""
        if self.N is None or self.M is None:
            # Default to a small random matrix if not initialized
            n, m = 2, 2
        else:
            n, m = self.N, self.M

        # Generate a random matrix (not necessarily valid)
        rows: List[str] = ["".join(random.choice("01") for _ in range(m)) for _ in range(n)]
        return "\\boxed{" + "\n".join(rows) + "}"

    def _generate_full_matrix(self, N: int, M: int) -> List[List[str]]:
        """Generate a full N x M matrix of '0'/'1' satisfying no-adjacent-ones constraint."""
        grid: List[List[Optional[str]]] = [[None] * M for _ in range(N)]
        all_cells = [(i, j) for i in range(N) for j in range(M)]
        random.shuffle(all_cells)

        def can_place_one(i: int, j: int) -> bool:
            # Check adjacent cells to ensure no adjacent '1's
            if i - 1 >= 0 and grid[i - 1][j] == "1":
                return False
            if i + 1 < N and grid[i + 1][j] == "1":
                return False
            if j - 1 >= 0 and grid[i][j - 1] == "1":
                return False
            if j + 1 < M and grid[i][j + 1] == "1":
                return False
            return True

        def fill(idx: int) -> bool:
            if idx == len(all_cells):
                return True
            i, j = all_cells[idx]

            # Try placing in random order to add variety
            for v in random.sample(["0", "1"], 2):
                if v == "1" and not can_place_one(i, j):
                    continue
                grid[i][j] = v
                # The original generation asserts the continuation is always possible
                assert fill(idx + 1)
                return True

            # Fallback should never be reached because '0' is always valid
            return False

        assert fill(0), "Failed to generate a valid matrix"
        return [[grid[i][j] if grid[i][j] is not None else "0" for j in range(M)] for i in range(N)]