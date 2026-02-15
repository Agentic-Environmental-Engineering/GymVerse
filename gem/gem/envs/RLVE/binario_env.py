import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BinarioEnv(Env):
    """Binairo-like puzzle environment - single-turn Q&A.

    The task: Fill a binary matrix (with '0', '1', and '*' for empty cells) so that:
    1) Each row has a specified number of 1s.
    2) Each column has a specified number of 1s.
    3) No more than two consecutive identical digits ('0' or '1') appear in any row or column.

    Output format: The completed matrix should be provided inside \\boxed{...}, with N lines (one line per row),
    each containing M characters ('0' or '1'). Line breaks inside the boxed braces are allowed and used to separate rows.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        sparsity: float = 0.5,
        # Preserve original reward-related parameters (not used in GEM reward mechanics)
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(satisfied/all)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        super().__init__()
        # Difficulty parameters
        self.max_n_m = max_n_m
        self.sparsity = sparsity

        # Preserved but unused parameters (kept for compatibility)
        self.rewards_config = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # Internal state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.row_counts: Optional[List[int]] = None
        self.col_counts: Optional[List[int]] = None
        self.puzzle_matrix: Optional[List[str]] = None  # with '*' for empty cells
        self.reference_answer: Optional[str] = None     # full solution lines joined by '\n'
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a binary matrix puzzle.\n"
            "Fill all '*' cells with '0' or '1' such that:\n"
            "1) Each row has the specified number of 1s.\n"
            "2) Each column has the specified number of 1s.\n"
            "3) No more than two consecutive identical digits appear in any row or column.\n\n"
            "Output Format: Provide the completed matrix inside \\boxed{...} with exactly N lines, "
            "one line per row, each with M characters ('0' or '1'). You may include line breaks inside the boxed braces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate parameters
        assert isinstance(self.max_n_m, int), "max_n_m must be an integer"
        assert self.max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        assert isinstance(self.sparsity, float), "sparsity must be a float"
        assert 0.0 < self.sparsity < 1.0, "sparsity should be between 0 and 1"

        # Generate N and M
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)

        def generate_matrix(n: int, m: int) -> Optional[List[List[str]]]:
            """Backtracking generation of a valid full solution matrix."""
            grid: List[List[Optional[str]]] = [[None] * m for _ in range(n)]
            all_cells = [(i, j) for i in range(n) for j in range(m)]
            random.shuffle(all_cells)

            backtrack_counting = 0

            def backtrack(idx: int) -> bool:
                # Finished filling all cells
                if idx == len(all_cells):
                    return True

                i, j = all_cells[idx]
                nonlocal backtrack_counting
                backtrack_counting += 1
                # Safety cap to avoid excessive recursion
                if backtrack_counting > 10_000_000:
                    return False

                # Try placing '0' or '1' in random order
                for v in random.sample(["0", "1"], 2):
                    # Row adjacency checks (prevent three consecutive same digits)
                    if j >= 2 and grid[i][j - 1] == v and grid[i][j - 2] == v:
                        continue
                    if j >= 1 and j + 1 < m and grid[i][j - 1] == v and grid[i][j + 1] == v:
                        continue
                    if j + 2 < m and grid[i][j + 1] == v and grid[i][j + 2] == v:
                        continue

                    # Column adjacency checks
                    if i >= 2 and grid[i - 1][j] == v and grid[i - 2][j] == v:
                        continue
                    if i >= 1 and i + 1 < n and grid[i - 1][j] == v and grid[i + 1][j] == v:
                        continue
                    if i + 2 < n and grid[i + 1][j] == v and grid[i + 2][j] == v:
                        continue

                    # Place v
                    grid[i][j] = v

                    # Recurse
                    if backtrack(idx + 1):
                        return True

                    # Backtrack
                    grid[i][j] = None

                # No valid assignment found at (i, j)
                return False

            success = backtrack(0)
            if not success:
                return None

            # Convert to concrete string grid
            return [[cell if cell is not None else "0" for cell in row] for row in grid]

        # Attempt to generate a valid full matrix
        matrix: Optional[List[List[str]]] = None
        for _ in range(50):
            matrix = generate_matrix(N, M)
            if matrix is not None:
                break
        if matrix is None:
            # Fallback: raise error if generation failed repeatedly
            raise RuntimeError("Failed to generate a valid matrix for the puzzle.")

        # Reference full solution
        self.reference_answer = "\n".join("".join(row) for row in matrix)

        # Row and column counts based on the reference solution
        row_counts = [sum(1 for cell in row if cell == "1") for row in matrix]
        col_counts = [sum(1 for i in range(N) if matrix[i][j] == "1") for j in range(M)]

        # Create puzzle by sparsifying the matrix with '*' cells
        puzzle_grid: List[List[str]] = [row[:] for row in matrix]
        total_cells = N * M
        # At least one cell is empty
        empty_count = max(1, int(total_cells * self.sparsity))
        empty_indices = random.sample(range(total_cells), empty_count)
        for cell in empty_indices:
            r, c = divmod(cell, M)
            puzzle_grid[r][c] = "*"

        puzzle_lines = ["".join(row) for row in puzzle_grid]

        # Store state
        self.N = N
        self.M = M
        self.row_counts = row_counts
        self.col_counts = col_counts
        self.puzzle_matrix = puzzle_lines

        # Build problem prompt
        problem = (
            f"You are given a {N} × {M} matrix. Each cell contains either '0', '1', or '*' ('*' means the cell is empty). "
            f"Please fill all '*' cells with either '0' or '1' such that:\n"
            f"1. The number of 1s in each row (from top to bottom) is: {', '.join(map(str, row_counts))}.\n"
            f"2. The number of 1s in each column (from left to right) is: {', '.join(map(str, col_counts))}.\n"
            "3. No more than two consecutive cells in a row or column can contain the same number.\n\n"
            "The matrix is given in row-major order, with each row represented as a string of '0', '1', and '*' (one row per line):\n"
            + "\n".join(puzzle_lines)
            + "\n\nOutput Format: Output exactly N lines, each containing M characters ('0' or '1'), "
              "inside \\boxed{...}. You may include line breaks inside the boxed braces to separate rows."
        )

        self.current_problem = problem
        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": N,
            "M": M,
            "row_counts": row_counts,
            "col_counts": col_counts,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process a single answer and return the terminal result."""
        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        assert self.N is not None and self.M is not None, "Environment not properly initialized. Call reset() first."
        assert self.puzzle_matrix is not None, "Puzzle matrix is not initialized."
        assert self.row_counts is not None and self.col_counts is not None, "Counts are not initialized."

        # Parse matrix from boxed content: split by lines, ignore empty lines
        lines: List[str] = []
        for line in boxed_content.splitlines():
            s = line.strip()
            if s:
                lines.append(s)

        # Validate shape and characters
        if len(lines) != self.N or any(len(row) != self.M for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "wrong_shape"}
        if any(not all(c in "01" for c in row) for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "invalid_characters"}

        # Validate fixed cells against puzzle (non '*' must match)
        for i in range(self.N):
            for j in range(self.M):
                original_cell = self.puzzle_matrix[i][j]
                if original_cell != "*" and lines[i][j] != original_cell:
                    return TERMINAL_STATE, 0.0, True, False, {
                        "error": "prefill_mismatch",
                        "position": (i, j),
                        "expected": original_cell,
                        "got": lines[i][j],
                    }

        # Check adjacency constraints: no three consecutive identical in any direction
        directions = [
            (+1, 0),
            (-1, 0),
            (0, +1),
            (0, -1),
        ]
        for i in range(self.N):
            for j in range(self.M):
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    nni, nnj = i + 2 * di, j + 2 * dj
                    if 0 <= ni < self.N and 0 <= nj < self.M and 0 <= nni < self.N and 0 <= nnj < self.M:
                        if lines[i][j] == lines[ni][nj] == lines[nni][nnj]:
                            return TERMINAL_STATE, 0.0, True, False, {
                                "error": "adjacency_violation",
                                "positions": [(i, j), (ni, nj), (nni, nnj)],
                                "value": lines[i][j],
                            }

        # Check row and column counts
        user_row_counts = [sum(1 for c in row if c == "1") for row in lines]
        user_col_counts = [sum(1 for i in range(self.N) if lines[i][j] == "1") for j in range(self.M)]

        rows_ok = all(a == b for a, b in zip(user_row_counts, self.row_counts))
        cols_ok = all(a == b for a, b in zip(user_col_counts, self.col_counts))

        if not (rows_ok and cols_ok):
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "count_mismatch",
                "user_row_counts": user_row_counts,
                "user_col_counts": user_col_counts,
                "target_row_counts": self.row_counts,
                "target_col_counts": self.col_counts,
            }

        # All constraints satisfied
        info = {
            "correct": True,
            "N": self.N,
            "M": self.M,
            "row_counts": self.row_counts,
            "col_counts": self.col_counts,
            "reference_answer": self.reference_answer,
            "user_answer": "\n".join(lines),
        }
        return TERMINAL_STATE, 1.0, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}, allowing multiline content."""
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random (likely incorrect) action by filling with random 0/1."""
        if self.N is None or self.M is None:
            # Provide a generic boxed random content if called before reset
            random_lines = ["0", "1"]
            return f"\\boxed{{{random_lines[0]}\n{random_lines[1]}}}"
        random_answer = "\n".join(
            "".join(random.choice("01") for _ in range(self.M)) for _ in range(self.N)
        )
        return f"\\boxed{{{random_answer}}}"