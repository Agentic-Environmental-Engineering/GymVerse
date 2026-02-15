from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class StarBattleEnv(Env):
    """
    Star Battle puzzle environment (single-turn Q&A).

    Task:
    - You are given an N x M grid with characters 'X' or '.'.
    - Place exactly one '*' in each row.
    - No column may contain more than one '*'.
    - No two '*' are adjacent, including diagonals (8-neighborhood).
    - Cells marked 'X' are blocked and cannot be changed.

    Observation:
    - A textual description of the puzzle and the grid.

    Action:
    - Provide the final grid with 'X', '.', and '*' in \\boxed{...} format.
    - The content inside the box should be N lines, each line having M characters.

    Reward:
    - Correct answer: 1.0
    - Wrong answer (constraint violation or invalid modification): 0.0
    - Format error (cannot parse or wrong shape/characters): -0.1
    """

    prompt_template = (
        "You are given a {N} × {M} grid. Each cell contains either 'X' or '.'. "
        "Please select some '.' cells to fill with '*' such that:\n"
        "1. Each row contains exactly one '*'.\n"
        "2. Each column contains no more than one '*'.\n"
        "3. No two '*' cells are adjacent (including diagonals — i.e., no two '*'s share an 8-neighbor relationship).\n\n"
        "The grid is given in row-major order, with each row represented as a string of 'X' and '.':\n"
        "{grid}\n\n"
        "Output Format: Output {N} lines, each containing {M} characters. Each character should be 'X', '.', or '*'. "
        "Place your final grid inside \\boxed{{...}} as a single block, preserving line breaks."
    )

    def __init__(
        self,
        max_n_m: int = 10,
        sparsity: float = 0.25,
        **kwargs: Any,
    ):
        """
        Initialize the StarBattle environment.

        Args:
            max_n_m: Upper bound for both N and M. Must be >= 3.
            sparsity: Fraction (0, 1) of empty cells ('.') to convert to 'X' after placing a valid solution, to create the puzzle.
        """
        super().__init__()
        # Validate parameters
        if max_n_m < 3:
            raise ValueError("max_n_m should be greater than or equal to 3.")
        if not (0 < sparsity < 1):
            raise ValueError("sparsity should be between 0 and 1.")

        self.max_n_m = max_n_m
        self.sparsity = sparsity

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.puzzle_grid: Optional[List[str]] = None  # Grid with 'X' and '.'
        self.reference_answer: Optional[str] = None   # Grid with 'X', '.', and '*' (solution)
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Star Battle placement puzzle.\n"
            "Make sure to output your final grid inside \\boxed{...} with the exact shape and characters.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new puzzle instance."""
        super().reset(seed)

        # Generate puzzle parameters and grids
        N, M, puzzle_grid, reference_answer = self._generate_puzzle()

        self.N = N
        self.M = M
        self.puzzle_grid = puzzle_grid
        self.reference_answer = reference_answer

        # Build problem description
        problem_text = self.prompt_template.format(
            N=N,
            M=M,
            grid="\n".join(puzzle_grid),
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + problem_text
        return obs, {}

    def _generate_puzzle(self) -> Tuple[int, int, List[str], str]:
        """
        Generate a Star Battle puzzle:
        - Create a valid solution (with '*') that satisfies all constraints.
        - Randomly convert a fraction of '.' cells to 'X' to create obstacles.
        - Clear '*' back to '.' to produce the puzzle, but keep the reference solution.
        """
        MAX_N_M = self.max_n_m

        while True:
            N = random.randint(2, MAX_N_M)
            M = random.randint(max(3, N), MAX_N_M)

            # Start with all '.'
            grid = [["."] * M for _ in range(N)]

            # Choose distinct columns for each row such that adjacent rows' columns differ by > 1
            permutation = random.sample(range(M), N)
            if any(abs(a - b) <= 1 for a, b in zip(permutation, permutation[1:])):
                continue

            # Place stars according to the permutation
            for row, col in enumerate(permutation):
                grid[row][col] = "*"

            # We have a valid solution so far
            break

        # Add sparsity: turn some '.' cells into 'X'
        empty_cells = [(i, j) for i in range(N) for j in range(M) if grid[i][j] == "."]
        num_to_block = max(1, int(len(empty_cells) * self.sparsity)) if empty_cells else 0
        for i, j in random.sample(empty_cells, num_to_block):
            grid[i][j] = "X"

        # Save reference answer (with '*')
        reference_answer = "\n".join("".join(row) for row in grid)

        # Produce the puzzle grid by removing '*' -> '.'
        puzzle_grid_matrix = []
        for i in range(N):
            row_out = []
            for j in range(M):
                if grid[i][j] == "*":
                    row_out.append(".")
                else:
                    # Should be either '.' or 'X'
                    if grid[i][j] not in ("X", "."):
                        raise ValueError("Grid should only contain 'X', '.', or '*' at this point.")
                    row_out.append(grid[i][j])
            puzzle_grid_matrix.append(row_out)

        # Convert to list of strings
        puzzle_grid = ["".join(row) for row in puzzle_grid_matrix]

        return N, M, puzzle_grid, reference_answer

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided solution and return the result."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Process lines to reconstruct the grid
        solution_lines = self._process_grid_text(boxed_content)

        # Validate format
        if self.N is None or self.M is None or self.puzzle_grid is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "internal_state_error"}

        N, M = self.N, self.M
        if len(solution_lines) != N or any(len(row) != M for row in solution_lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "shape_mismatch"}
        if not all(c in "X.*" for row in solution_lines for c in row):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "invalid_characters"}

        # Validate immutability of 'X' and allowed modifications on '.'
        for row_idx, (row, orig_row) in enumerate(zip(solution_lines, self.puzzle_grid)):
            for col_idx, (cell, orig_cell) in enumerate(zip(row, orig_row)):
                if orig_cell == "X" and cell != "X":
                    return TERMINAL_STATE, 0.0, True, False, {
                        "error": "invalid_solution",
                        "detail": f"modified_blocked_cell_at_{row_idx}_{col_idx}",
                    }
                if orig_cell == "." and cell not in ".*":
                    return TERMINAL_STATE, 0.0, True, False, {
                        "error": "invalid_solution",
                        "detail": f"invalid_edit_at_{row_idx}_{col_idx}",
                    }

        # Constraint checks
        # 1) Each row contains exactly one '*'
        if any(row.count("*") != 1 for row in solution_lines):
            return TERMINAL_STATE, 0.0, True, False, {"error": "wrong_solution", "detail": "row_star_count"}

        # 2) Each column contains no more than one '*'
        for col in zip(*solution_lines):
            if col.count("*") > 1:
                return TERMINAL_STATE, 0.0, True, False, {"error": "wrong_solution", "detail": "column_star_count"}

        # 3) No adjacency including diagonals
        for i in range(N):
            for j in range(M):
                if solution_lines[i][j] == "*":
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < N and 0 <= nj < M:
                                if solution_lines[ni][nj] == "*":
                                    return TERMINAL_STATE, 0.0, True, False, {
                                        "error": "wrong_solution",
                                        "detail": f"adjacent_stars_at_{i}_{j}_and_{ni}_{nj}",
                                    }

        # If all checks pass
        info = {
            "correct": True,
            "reference_answer": self.reference_answer,
            "puzzle_grid": self.puzzle_grid,
            "N": self.N,
            "M": self.M,
        }
        return TERMINAL_STATE, 1.0, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence, allowing multiline."""
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process_grid_text(self, text: str) -> List[str]:
        """Convert text to a list of non-empty, stripped lines."""
        grid: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                grid.append(line)
        return grid

    def sample_random_action(self) -> str:
        """
        Sample a random action: here we simply return the given puzzle grid inside \\boxed{...}.
        This is typically an invalid solution (no '*'), but serves as a random baseline.
        """
        if self.puzzle_grid is None:
            # If called before reset, produce an empty boxed answer
            return "\\boxed{}"
        content = "\n".join(self.puzzle_grid)
        return f"\\boxed{{\n{content}\n}}"