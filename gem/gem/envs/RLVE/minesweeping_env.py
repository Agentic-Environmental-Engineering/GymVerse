from typing import Any, List, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinesweepingEnv(Env):
    """Minesweeping constraint satisfaction environment - single-turn Q&A.

    The agent is given a matrix where each element is either an integer in [0, 8] or -1.
    The task is to construct a same-sized grid of '*' and '.' such that:
      - Every non -1 cell in the original matrix must correspond to '.' in the output.
      - For any non -1 cell, its value equals the number of '*' present in its 8 neighboring cells.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        density: float = 0.5,
        mine_density_range: Tuple[float, float] = (0.4, 0.7),
        **kwargs
    ):
        super().__init__()
        # Parameter validations
        assert isinstance(max_n_m, int) and max_n_m >= 2, "max_n_m should be an integer >= 2"
        assert isinstance(density, float) and 0.0 < density < 1.0, "density should be a float in (0, 1)"
        assert (
            isinstance(mine_density_range, tuple)
            and len(mine_density_range) == 2
            and 0.0 < mine_density_range[0] < mine_density_range[1] < 1.0
        ), "mine_density_range should be a tuple of two floats in (0, 1) with lower < upper"

        self.max_n_m = max_n_m
        self.density = density
        self.mine_density_range = mine_density_range

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.clue_grid: Optional[List[List[int]]] = None  # -1 for unknown, else 0..8
        self.reference_answer: Optional[str] = None  # Solution grid ('.' and '*') as lines joined by '\n'
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given a matrix of hints for a Minesweeper-like puzzle.\n"
            "Each cell in the matrix is either -1 (unknown) or an integer in [0, 8].\n"
            "You must output a grid of the same size using only '*' and '.', such that:\n"
            "1) For any cell in the given matrix that is NOT -1, the output grid must have '.' in that cell.\n"
            "2) For any cell that is NOT -1, its given number equals the number of '*' in its 8 neighboring cells.\n"
            "Please include your entire grid (exactly N lines, each with M characters) inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Determine grid size
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        self.N, self.M = N, M

        # Generate the underlying solution grid
        solution_grid = [["."] * M for _ in range(N)]
        mine_density = random.uniform(self.mine_density_range[0], self.mine_density_range[1])
        num_mines = max(1, min(int(N * M * mine_density), N * M - 1))
        mine_cells = random.sample(range(N * M), num_mines)

        for cell in mine_cells:
            r, c = divmod(cell, M)
            solution_grid[r][c] = "*"

        # Store reference answer as string
        self.reference_answer = "\n".join("".join(row) for row in solution_grid)

        # Select some empty cells to expose numbers according to density
        empty_cells = [(i, j) for i in range(N) for j in range(M) if solution_grid[i][j] == "."]
        assert len(empty_cells) >= 1, "There should be at least one empty cell in the generated grid"

        sample_count = max(1, int(len(empty_cells) * self.density))
        sampled_empty_cells = random.sample(empty_cells, sample_count)

        # Build clue grid with numbers for sampled empty cells, -1 elsewhere
        clue_grid: List[List[int]] = [[-1] * M for _ in range(N)]
        for i, j in sampled_empty_cells:
            count = 0
            for di in (-1, 0, +1):
                for dj in (-1, 0, +1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < M and solution_grid[ni][nj] == "*":
                        if not (di == 0 and dj == 0):
                            count += 1
            assert 0 <= count <= 8
            clue_grid[i][j] = count

        self.clue_grid = clue_grid

        # Build problem prompt
        matrix_str = "\n".join(" ".join(map(str, row)) for row in self.clue_grid)
        problem_text = (
            f"You are given a {N} × {M} matrix. Each element is either a number in [0, 8] or -1. "
            f"Your task is to construct a grid of the same size, satisfying the following conditions:\n"
            f"1) Each cell is either '*' or '.'\n"
            f"2) For any cell in the original matrix that is NOT -1, the corresponding cell in the output grid must be '.'. "
            f"Also, its number must equal the number of '*' characters in its 8 neighboring cells.\n\n"
            f"The matrix is given in row-major order:\n{matrix_str}\n\n"
            f"Output Format: Output {N} lines, each containing {M} characters with no separators, "
            f"and put your entire grid inside \\boxed{{...}}."
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the submitted answer and return the result."""
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Missing or invalid \\boxed{...}."}

        # Convert boxed content to grid
        N, M = self.N, self.M
        clue_grid = self.clue_grid
        assert N is not None and M is not None and clue_grid is not None, "Environment not initialized properly."

        lines = [ln.strip() for ln in boxed.splitlines() if ln.strip() != ""]
        if len(lines) != N or any(len(row) != M for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {
                "error": "format_error",
                "message": f"Expected {N} non-empty lines each of length {M}."
            }

        if not all(all(ch in ("*", ".") for ch in row) for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {
                "error": "format_error",
                "message": "Grid must only contain '*' and '.'."
            }

        # Validate solution against clues
        satisfied = 0
        total = 0
        for i in range(N):
            for j in range(M):
                if clue_grid[i][j] != -1:
                    total += 1
                    # Cell must be '.'
                    if lines[i][j] != ".":
                        info = {
                            "correct": False,
                            "satisfied": satisfied,
                            "total": total,
                            "reason": "non_unknown_cell_must_be_dot",
                            "reference_answer": self.reference_answer,
                            "user_answer": boxed
                        }
                        return TERMINAL_STATE, 0.0, True, False, info

                    # Count neighboring '*'
                    count = 0
                    for di in (-1, 0, +1):
                        for dj in (-1, 0, +1):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < N and 0 <= nj < M and lines[ni][nj] == "*":
                                count += 1
                    if count == clue_grid[i][j]:
                        satisfied += 1

        is_correct = (satisfied == total)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied": satisfied,
            "total": total,
            "reference_answer": self.reference_answer,
            "user_answer": boxed
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns the last match if multiple boxes exist."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random grid action wrapped in \\boxed{...}."""
        if self.N is None or self.M is None:
            # Default small random grid if called before reset
            n, m = 2, 2
        else:
            n, m = self.N, self.M

        # Create a random guess grid
        guess = []
        for _ in range(n):
            row = []
            for _ in range(m):
                row.append("*" if random.random() < 0.5 else ".")
            guess.append("".join(row))
        content = "\n".join(guess)
        return f"\\boxed{{\n{content}\n}}"