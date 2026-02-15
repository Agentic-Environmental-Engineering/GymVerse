from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class QueenPlacementEnv(Env):
    """Environment for placing additional queens on a chessboard without conflicts."""

    def __init__(
        self,
        N: int = 8,
        **kwargs
    ):
        """
        Initialize the QueenPlacementEnv.

        Parameters:
            N: Size of the chessboard (N x N). Must be >= 3.
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.presented_grid: Optional[List[List[str]]] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a chess queen placement problem.\n"
            "Rules:\n"
            "- You must place the specified number of additional queens so that no two queens threaten each other.\n"
            "- A queen threatens another if they share the same row, column, or diagonal (both main and anti-diagonals).\n"
            "Output Format:\n"
            "- Provide exactly N lines, each of length N.\n"
            "- Use only 'Q' for a queen and '.' for an empty cell.\n"
            "- Wrap the entire N-line grid inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        N = self.N
        # Generate a maximal set of non-attacking queens by greedy placement over a random order of cells
        grid: List[List[str]] = [["." for _ in range(N)] for _ in range(N)]

        all_cells = [(i, j) for i in range(N) for j in range(N)]
        random.shuffle(all_cells)

        row_used, col_used, main_diag, anti_diag = set(), set(), set(), set()
        queens: List[Tuple[int, int]] = []
        for i, j in all_cells:
            if i in row_used or j in col_used or (i - j) in main_diag or (i + j) in anti_diag:
                continue
            grid[i][j] = "Q"
            queens.append((i, j))
            row_used.add(i)
            col_used.add(j)
            main_diag.add(i - j)
            anti_diag.add(i + j)

        # Store the full solution before removing K queens
        full_solution_grid: List[List[str]] = [row[:] for row in grid]
        self.reference_answer = "\n".join("".join(r) for r in full_solution_grid)

        # Remove K queens to create the presented grid
        K = random.randint(1, max(1, len(queens) // 2))
        removed = random.sample(queens, K) if queens else []
        for i, j in removed:
            grid[i][j] = "."

        self.presented_grid = grid
        self.K = K

        # Build problem text
        grid_str = "\n".join("".join(r) for r in grid)
        self.current_problem = (
            f"You are given an {N} × {N} chessboard grid. Some cells already contain queens (denoted by 'Q'), and the rest are empty ('.').\n"
            f"{grid_str}\n\n"
            f"Please place {K} additional queens such that no two queens threaten each other. "
            f"A queen threatens another if they share the same row, column, or diagonal (both main and anti-diagonals).\n\n"
            f"Output Format: Provide exactly {N} lines, each containing a string of length {N} using 'Q' and '.' characters, "
            f"and wrap your entire {N}-line grid inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": N,
            "K": K,
            "presented_grid": grid_str,
            "reference_answer": self.reference_answer,
        }
        return obs, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted grid solution."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {
                "error": "format_error",
                "message": "Answer must be provided inside \\boxed{...}."
            }

        # Process boxed content into lines
        lines = [line.strip() for line in boxed_content.splitlines() if line.strip()]

        N = self.N
        if len(lines) != N:
            return TERMINAL_STATE, -0.1, True, False, {
                "error": "format_error",
                "message": f"Expected exactly {N} non-empty lines inside \\boxed{{...}}."
            }
        if any(len(row) != N for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {
                "error": "format_error",
                "message": f"Each line must be exactly length {N}."
            }
        if any(cell not in "Q." for row in lines for cell in row):
            return TERMINAL_STATE, -0.1, True, False, {
                "error": "format_error",
                "message": "Only 'Q' and '.' characters are allowed."
            }

        if self.presented_grid is None or self.K is None:
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "state_error",
                "message": "Environment is not properly initialized."
            }

        # Verification logic: preserve existing queens, count added queens, ensure no conflicts
        counting = 0
        row_used, col_used, main_diag, anti_diag = set(), set(), set(), set()

        invalid_solution = False
        conflict_found = False

        for i, (original_row, current_row_str) in enumerate(zip(self.presented_grid, lines)):
            for j, (original_cell, current_cell) in enumerate(zip(original_row, current_row_str)):
                if original_cell == "Q":
                    if current_cell != "Q":
                        invalid_solution = True
                        break
                else:
                    if current_cell == "Q":
                        counting += 1

                if current_cell == "Q":
                    if i in row_used or j in col_used or (i - j) in main_diag or (i + j) in anti_diag:
                        conflict_found = True
                    row_used.add(i)
                    col_used.add(j)
                    main_diag.add(i - j)
                    anti_diag.add(i + j)
            if invalid_solution:
                break

        if invalid_solution:
            info = {
                "error": "invalid_solution",
                "message": "Pre-existing queens must remain unchanged.",
                "K": self.K,
                "reference_answer": self.reference_answer
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if conflict_found or counting != self.K:
            info = {
                "error": "wrong_solution",
                "message": "Either queens attack each other or incorrect number of added queens.",
                "added_queens": counting,
                "expected_added": self.K,
                "reference_answer": self.reference_answer
            }
            return TERMINAL_STATE, 0.0, True, False, info

        info = {
            "correct": True,
            "K": self.K,
            "reference_answer": self.reference_answer
        }
        return TERMINAL_STATE, 1.0, True, False, info

    def sample_random_action(self) -> str:
        """Sample a random action: return the full reference solution if available, else the presented grid."""
        if self.reference_answer is not None:
            return f"\\boxed{{\n{self.reference_answer}\n}}"
        if self.presented_grid is not None:
            grid_str = "\n".join("".join(r) for r in self.presented_grid)
            return f"\\boxed{{\n{grid_str}\n}}"
        # Fallback to an empty N x N grid
        empty_grid = "\n".join("." * self.N for _ in range(self.N))
        return f"\\boxed{{\n{empty_grid}\n}}"