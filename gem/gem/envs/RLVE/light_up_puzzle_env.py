import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LightUpPuzzleEnv(Env):
    """Light Up puzzle environment - single-turn Q&A in GEM format."""

    prompt_template = r"""You are given a {N} × {M} grid. Each cell contains either a number from `0` to `4`, or a character `B` or `W`.
- All `W` cells are considered **white cells** (including those that may be replaced with `L` later).
- All other cells (`0`–`4` or `B`) are considered **black cells**.

You may replace some `W` cells with `L`, indicating the placement of a **light bulb**. A light bulb illuminates its own cell and extends light in all **four directions** (up, down, left, right), stopping when it hits a black cell or the edge of the grid. Please place light bulbs such that:
1. **Each white cell** is illuminated by **at least one** light bulb.
2. No light bulb is illuminated by another light bulb, i.e., no two light bulbs can be placed in the same row or column without a black cell in between.
3. **Each black cell** with a number from `0` to `4` must have **exactly that many** light bulbs in its 4 neighboring cells (up, down, left, right).

The grid is given in **row-major order**:
{grid}

Output Format: Output {N} lines, each containing {M} characters with no separators. Some `W` cells should be replaced with `L` to indicate light bulbs; all other cells remain unchanged."""

    def __init__(
        self,
        max_n_m: int = 10,
        density: float = 0.5,
        black_cell_density_range: tuple[float, float] = (0.6, 0.95),
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(satisfied/all)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 10.0,
        **kwargs,
    ):
        """
        Initialize the LightUpPuzzleEnv instance.

        Parameters:
        - max_n_m: maximum dimension for N and M (both chosen in [2, max_n_m])
        - density: proportion of black cells turned into numbered cells
        - black_cell_density_range: range for initial black cell density in the grid
        - wrong_format, invalid_solution, rewarding_strategy, rewarding_weight, rewarding_beta:
          retained for compatibility but not used in GEM rewards (see step()).
        """
        super().__init__()
        # Parameter validation (preserved logic)
        assert isinstance(max_n_m, int) and max_n_m >= 2, "max_n_m should be an integer >= 2"
        assert 0.0 < density < 1.0, "density should be between 0 and 1"
        assert (
            isinstance(black_cell_density_range, tuple)
            and len(black_cell_density_range) == 2
            and 0.0 < black_cell_density_range[0] < black_cell_density_range[1] < 1.0
        ), "black_cell_density_range should be a tuple of two floats in (0, 1)"

        # Store parameters
        self.max_n_m = max_n_m
        self.density = density
        self.black_cell_density_range = black_cell_density_range

        # Retained reward configuration (not used for final step reward in GEM)
        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # Runtime attributes
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None
        self.grid_puzzle: Optional[List[str]] = None  # grid with 'L' replaced by 'W'

    def _get_instructions(self) -> str:
        """Return general task instructions and output format."""
        return (
            "You are solving a Light Up puzzle. Place light bulbs (L) on white cells (W) to satisfy the rules.\n"
            "Please submit your final answer as N lines inside \\boxed{...}, with rows separated by newline characters.\n"
            "Each line must have exactly M characters, using only 'W' or 'L' for white cells, and keep all 'B' or numbered cells unchanged.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new puzzle instance."""
        super().reset(seed)

        # Generate grid size
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        self.N, self.M = N, M

        # Initialize grid with all white cells
        grid = [["W"] * M for _ in range(N)]

        # Place black cells based on density range
        black_cell_density = random.uniform(self.black_cell_density_range[0], self.black_cell_density_range[1])
        num_black = max(1, min(int(N * M * black_cell_density), N * M - 1))
        black_cells_flat = random.sample(range(N * M), num_black)
        for cell in black_cells_flat:
            r, c = divmod(cell, M)
            grid[r][c] = "B"

        # Ensure at least one white cell
        white_cells = [(i, j) for i in range(N) for j in range(M) if grid[i][j] == "W"]
        assert len(white_cells) >= 1, "There should be at least one white cell"

        # Place bulbs to illuminate all white cells, ensuring no conflicts
        random.shuffle(white_cells)
        illuminated = [[False] * M for _ in range(N)]
        for i, j in white_cells:
            if illuminated[i][j]:
                continue
            grid[i][j] = "L"
            illuminated[i][j] = True

            for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                ni, nj = i + di, j + dj
                while 0 <= ni < N and 0 <= nj < M:
                    if grid[ni][nj] == "B":
                        break
                    # No light bulb in same row/column without a black cell in between
                    assert grid[ni][nj] != "L", "There should be no light bulb in the same row or column without a black cell in between"
                    illuminated[ni][nj] = True
                    ni += di
                    nj += dj

        # Convert some black cells to numbered cells based on density
        black_cells = [(i, j) for i in range(N) for j in range(M) if grid[i][j] == "B"]
        num_numbered = max(1, int(len(black_cells) * self.density))
        numbered_black_cells = random.sample(black_cells, num_numbered)
        assert len(numbered_black_cells) > 0, "There should be at least one black cell with a number"

        for i, j in numbered_black_cells:
            count = 0
            for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < M and grid[ni][nj] == "L":
                    count += 1
            grid[i][j] = str(count)

        # Store reference answer (complete solved grid with 'L's)
        self.reference_answer = "\n".join("".join(row) for row in grid)

        # Puzzle grid provided to the user (replace 'L' with 'W')
        self.grid_puzzle = ["".join(cell if cell != "L" else "W" for cell in row) for row in grid]

        # Build problem prompt
        self.current_problem = self.prompt_template.format(
            N=N,
            M=M,
            grid="\n".join("".join(row) for row in self.grid_puzzle),
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {"N": N, "M": M}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted solution and return the terminal state."""
        # Parse boxed answer content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Process answer text into a list of lines
        solution_lines = self._process(boxed)
        if solution_lines is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        assert self.N is not None and self.M is not None and self.grid_puzzle is not None

        N, M = self.N, self.M

        # Check dimension format
        if len(solution_lines) != N or any(len(row) != M for row in solution_lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Check cells type consistency (W/L for white cells; exact match for black/numbers)
        for solution_row, original_row in zip(solution_lines, self.grid_puzzle):
            for solution_cell, original_cell in zip(solution_row, original_row):
                if original_cell == "W":
                    if solution_cell not in "WL":
                        return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}
                elif original_cell in "B01234":
                    if solution_cell != original_cell:
                        return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}
                else:
                    return TERMINAL_STATE, 0.0, True, False, {"error": f"invalid_cell:{original_cell}"}

        # Verify illumination rules and no bulb conflicts
        illuminated = [[False] * M for _ in range(N)]
        for i in range(N):
            for j in range(M):
                if solution_lines[i][j] == "L":
                    illuminated[i][j] = True
                    for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                        ni, nj = i + di, j + dj
                        while 0 <= ni < N and 0 <= nj < M:
                            if solution_lines[ni][nj] != "W":
                                if solution_lines[ni][nj] == "L":
                                    return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}  # bulb conflict
                                elif solution_lines[ni][nj] in "B01234":
                                    break
                                else:
                                    return TERMINAL_STATE, 0.0, True, False, {"error": f"unknown_cell:{solution_lines[ni][nj]}"}
                            illuminated[ni][nj] = True
                            ni += di
                            nj += dj

        # Check all white cells are illuminated
        if any(not illuminated[i][j] for i in range(N) for j in range(M) if self.grid_puzzle[i][j] == "W"):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        # Check numbered black cells constraints
        satisfied, total = 0, 0
        for i in range(N):
            for j in range(M):
                if self.grid_puzzle[i][j] in "01234":
                    total += 1
                    count = 0
                    for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M and solution_lines[ni][nj] == "L":
                            count += 1
                    if count == int(self.grid_puzzle[i][j]):
                        satisfied += 1

        correct = (total > 0 and satisfied == total)
        reward = 1.0 if correct else 0.0

        info = {
            "correct": correct,
            "reference_answer": self.reference_answer,
            "user_answer": "\n".join(solution_lines),
            "satisfied": satisfied,
            "total_constraints": total,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns the last boxed content if multiple."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _process(self, answer: Optional[str]) -> Optional[List[str]]:
        """Convert raw boxed content into a list of non-empty lines."""
        if answer is None:
            return None
        try:
            matrix: List[str] = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    matrix.append(line)
            return matrix
        except Exception:
            return None

    def sample_random_action(self) -> str:
        """Sample a random action: return the puzzle grid (unchanged) wrapped in \\boxed{...}."""
        if self.grid_puzzle is None:
            # If called before reset, provide a dummy 2x2 grid
            content = "WW\nWW"
        else:
            content = "\n".join(self.grid_puzzle)
        return f"\\boxed{{\n{content}\n}}"