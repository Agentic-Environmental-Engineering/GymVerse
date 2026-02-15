from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SudokuEnv(Env):
    """Sudoku puzzle environment (single-turn Q&A) for GEM."""

    def __init__(
        self,
        max_n_m: int = 3,
        sparsity: float = 0.6,
        **kwargs
    ):
        """
        Initialize the Sudoku environment.

        Args:
            max_n_m: Maximum value for both N and M (each subgrid is N x M, and the full grid is (N*M) x (N*M)).
                     Must be >= 2. In each reset, N and M are chosen uniformly at random from [2, max_n_m].
            sparsity: Fraction of cells to blank out (set to 0) in the generated puzzle, must satisfy 0 < sparsity < 1.
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        assert 0 < sparsity < 1, "sparsity should be between 0 and 1"
        self.max_n_m = max_n_m
        self.sparsity = sparsity

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.NM: Optional[int] = None
        self.puzzle_grid: Optional[List[List[int]]] = None
        self.solution_grid: Optional[List[List[int]]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving Sudoku puzzles of size (N × M) × (M × N) = (N×M) × (N×M).\n"
            "Rules:\n"
            "1) Each row must contain all digits from 1 to N×M without repetition.\n"
            "2) Each column must contain all digits from 1 to N×M without repetition.\n"
            "3) The grid is divided into M × N subgrids; each subgrid is N rows by M columns, and must contain all digits from 1 to N×M without repetition.\n\n"
            "Output Format:\n"
            "- Provide the completed Sudoku grid in row-major order.\n"
            "- Use exactly N×M lines, each with N×M integers separated by spaces.\n"
            "- Place the entire grid inside \\boxed{...}. Newlines inside the box are allowed.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new Sudoku puzzle."""
        super().reset(seed)

        # Choose N and M
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        NM = N * M

        # Generate a valid completed Sudoku solution using a base pattern and shuffles
        base = [[(M * (row % N) + row // N + column) % NM + 1 for column in range(NM)] for row in range(NM)]

        perm = list(range(1, NM + 1))
        random.shuffle(perm)
        grid = [[perm[base[row][column] - 1] for column in range(NM)] for row in range(NM)]

        def shuffle_groups(data: List[List[int]], group_size: int) -> None:
            """Shuffle rows within each group and shuffle the groups themselves."""
            G = len(data) // group_size
            for g in range(G):
                start = g * group_size
                slice_ = data[start: start + group_size]
                random.shuffle(slice_)
                data[start: start + group_size] = slice_
            groups = [data[g * group_size:(g + 1) * group_size] for g in range(G)]
            random.shuffle(groups)
            data[:] = [row for group in groups for row in group]

        # Shuffle rows in groups of N, then columns in groups of M (via transpose trick)
        shuffle_groups(grid, N)
        grid_t = list(map(list, zip(*grid)))
        shuffle_groups(grid_t, M)
        grid = list(map(list, zip(*grid_t)))

        # Randomly transpose the grid and swap N and M
        if random.choice([True, False]):
            grid = list(map(list, zip(*grid)))
            N, M = M, N
            NM = N * M

        # Store the full solution
        solution_grid = [row[:] for row in grid]
        self.solution_grid = solution_grid

        # Reference answer as string (lines separated by newline, numbers by spaces)
        self.reference_answer = "\n".join(" ".join(map(str, row)) for row in solution_grid)

        # Create the puzzle by removing a fraction of cells
        empty_cells = random.sample(range(NM * NM), max(1, int(NM * NM * self.sparsity)))
        puzzle_grid = [row[:] for row in solution_grid]
        for cell in empty_cells:
            r, c = divmod(cell, NM)
            puzzle_grid[r][c] = 0

        # Save parameters
        self.N = N
        self.M = M
        self.NM = NM
        self.puzzle_grid = puzzle_grid

        # Build the problem prompt
        puzzle_text = "\n".join(" ".join(map(str, row)) for row in puzzle_grid)
        output_example = "\n".join(" ".join(map(str, range(1, NM + 1))) for _ in range(NM))
        self.current_problem = (
            f"Solve a Sudoku puzzle of size ({N} × {M}) × ({M} × {N}) = {NM} × {NM}.\n"
            f"Each number is in the range from 1 to {NM}, and empty cells are represented by 0.\n\n"
            f"Input grid:\n{puzzle_text}\n\n"
            f"Please return your completed grid in \\boxed{{...}}. The content inside the box must be {NM} lines, each with {NM} integers separated by spaces.\n"
            f"Example (NOT a valid Sudoku):\n"
            f"\\boxed{{\n{output_example}\n}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted Sudoku solution."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Missing or malformed \\boxed{...}."}

        # Attempt to parse the grid
        try:
            lines = [ln.strip() for ln in boxed_content.splitlines() if ln.strip()]
            if self.NM is None or self.puzzle_grid is None or self.N is None or self.M is None:
                return TERMINAL_STATE, -0.1, True, False, {"error": "internal_state_error"}

            NM = self.NM
            if len(lines) != NM:
                return TERMINAL_STATE, -0.1, True, False, {"error": "wrong_format", "message": f"Expected {NM} lines."}

            grid: List[List[int]] = []
            for ln in lines:
                nums = ln.split()
                if len(nums) != NM:
                    return TERMINAL_STATE, -0.1, True, False, {"error": "wrong_format", "message": f"Each line must contain {NM} integers."}
                grid.append([int(x) for x in nums])
        except Exception:
            # Parsing failed (non-integers or malformed content)
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Failed to parse integers from the boxed content."}

        # Validate cell ranges and consistency with given clues
        for r in range(NM):
            for c in range(NM):
                val = grid[r][c]
                if not (1 <= val <= NM):
                    return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_value", "position": (r, c), "value": val}
                given = self.puzzle_grid[r][c]
                if given != 0 and val != given:
                    return TERMINAL_STATE, 0.0, True, False, {"error": "mismatch_with_clue", "position": (r, c), "given": given, "value": val}

        # Validate rows
        for r in range(NM):
            if len(set(grid[r])) != NM:
                return TERMINAL_STATE, 0.0, True, False, {"error": "row_violation", "row": r}

        # Validate columns
        for c in range(NM):
            col_vals = [grid[r][c] for r in range(NM)]
            if len(set(col_vals)) != NM:
                return TERMINAL_STATE, 0.0, True, False, {"error": "column_violation", "column": c}

        # Validate subgrids: N rows by M columns, arranged as M by N subgrids
        N = self.N
        M = self.M
        for bi in range(M):
            for bj in range(N):
                sub = [grid[x][y] for x in range(bi * N, (bi + 1) * N) for y in range(bj * M, (bj + 1) * M)]
                if len(set(sub)) != NM:
                    return TERMINAL_STATE, 0.0, True, False, {
                        "error": "subgrid_violation",
                        "block_row_index": bi,
                        "block_col_index": bj
                    }

        # All checks passed
        info = {
            "correct": True,
            "reference_answer": self.reference_answer,
        }
        return TERMINAL_STATE, 1.0, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Supports multiline content."""
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random (likely incorrect) Sudoku grid action in boxed format."""
        if self.NM is None:
            # Provide a generic dummy action if called before reset
            return "\\boxed{1}"

        NM = self.NM
        # Random grid with values in range 1..NM
        lines = []
        for _ in range(NM):
            row = [str(random.randint(1, NM)) for _ in range(NM)]
            lines.append(" ".join(row))
        content = "\n".join(lines)
        return f"\\boxed{{\n{content}\n}}"