from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import numpy as np
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


def magic_square(n: int) -> np.ndarray:
    """Generate a magic square of size n x n for odd and doubly-even n."""
    if n == 1:
        return np.array([[1]], dtype=int)

    if n % 2 == 1:
        return _magic_odd(n)
    elif n % 4 == 0:
        return _magic_doubly_even(n)
    else:
        raise NotImplementedError("Magic square for singly even n (e.g., 6, 10) is not implemented.")


def _magic_odd(n: int) -> np.ndarray:
    """Siamese method for odd-order magic squares."""
    magic = np.zeros((n, n), dtype=int)
    num = 1
    i, j = 0, n // 2
    while num <= n * n:
        magic[i, j] = num
        num += 1
        ni, nj = (i - 1) % n, (j + 1) % n
        if magic[ni, nj] != 0:
            i = (i + 1) % n
        else:
            i, j = ni, nj
    return magic


def _magic_doubly_even(n: int) -> np.ndarray:
    """Algorithm for doubly-even (divisible by 4) order magic squares."""
    magic = np.arange(1, n * n + 1, dtype=int).reshape(n, n)
    for i in range(n):
        for j in range(n):
            if (i % 4 == j % 4) or ((i % 4) + (j % 4) == 3):
                magic[i, j] = n * n + 1 - magic[i, j]
    return magic


def rotate(square: np.ndarray) -> np.ndarray:
    """Rotate the square by a random number of quarter turns (1-3)."""
    return np.rot90(square, random.randint(1, 3))


def mirror(square: np.ndarray) -> np.ndarray:
    """Mirror the square horizontally."""
    return np.fliplr(square)


def swap_rows(square: np.ndarray, i: int, j: int) -> np.ndarray:
    """Swap two rows and the corresponding symmetric columns (original logic preserved)."""
    n = square.shape[0]
    A = square.copy()
    A[[i, j], :] = A[[j, i], :]
    c1, c2 = n - 1 - i, n - 1 - j
    A[:, [c1, c2]] = A[:, [c2, c1]]
    # Preserve original return behavior
    return square


class MagicSquarePuzzleEnv(Env):
    """Magic Square puzzle environment - single turn Q&A."""

    prompt_template = (
        "Given a grid of size {N} × {N} filled with integers, some cells may be empty (represented by 0). "
        "Please complete the grid to form a magic square, such that:\n"
        "1. Each integer from 1 to {N}^2 appears exactly once.\n"
        "2. The sum of each row, each column, and both main diagonals is equal to {N} * ({N}^2 + 1) / 2 = {magic_constant}.\n\n"
        "The grid is given as follows:\n"
        "{grid}\n\n"
        "Output Format: Your final answer should contain {N} lines, each with {N} numbers, separated by spaces. "
        "Place your entire answer within \\boxed{{...}}. For example:\n"
        "\\boxed{{\n"
        "a11 a12 ... a1N\n"
        "a21 a22 ... a2N\n"
        "...\n"
        "aN1 aN2 ... aNN\n"
        "}}"
    )

    def __init__(
        self,
        N: int = 3,
        sparsity: float = 0.5,
        # The following parameters are preserved for compatibility, though reward is fixed in this GEM env
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(satisfied/all)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.N = N
        self.sparsity = sparsity

        # Store preserved parameters (not used for reward in GEM)
        self.wrong_format = wrong_format
        self.invalid_solution = invalid_solution
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # Internal state
        self.current_grid: Optional[List[List[int]]] = None
        self.full_solution_grid: Optional[List[List[int]]] = None
        self.reference_answer: Optional[str] = None
        self.magic_constant: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a Magic Square completion problem.\n"
            "Rules:\n"
            "- Use all integers from 1 to N^2 exactly once.\n"
            "- Every row, column, and both main diagonals must sum to the magic constant.\n"
            "- Do not change the given non-zero cells.\n"
            "Submission:\n"
            "- Provide your completed grid as N lines of N integers (space-separated), "
            "wrapped entirely within \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new magic square puzzle."""
        super().reset(seed)

        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 3, "N should be greater than or equal to 3"
        assert 0 < self.sparsity < 1, "sparsity should be between 0 and 1"

        N = self.N
        grid = magic_square(N)

        # Apply random transformations
        operation_distribution = [0.1, 0.1, 0.8]  # rotate, mirror, swap_rows
        for _ in range(N * N):
            operation = random.choices(["rotate", "mirror", "swap_rows"], weights=operation_distribution)[0]
            if operation == "rotate":
                grid = rotate(grid)
            elif operation == "mirror":
                grid = mirror(grid)
            elif operation == "swap_rows":
                while True:
                    row1, row2 = random.sample(range(N), 2)
                    if row1 != row2:
                        break
                grid = swap_rows(grid, row1, row2)
            else:
                raise RuntimeError("Unknown operation")

        # Store full solution as reference string (before making cells empty)
        self.reference_answer = "\n".join(" ".join(map(str, row)) for row in grid)

        # Convert to Python list for manipulation
        full_grid = [[cell.item() for cell in row] for row in grid]
        self.full_solution_grid = [row[:] for row in full_grid]

        # Sparsify the grid by setting some cells to 0
        masked_grid = [row[:] for row in full_grid]
        total_cells = N * N
        empty_count = max(1, int(total_cells * self.sparsity))
        empty_cells = random.sample(range(total_cells), empty_count)
        for cell in empty_cells:
            r, c = divmod(cell, N)
            masked_grid[r][c] = 0

        self.current_grid = masked_grid
        self.magic_constant = N * (N * N + 1) // 2

        # Build problem statement
        grid_text = "\n".join(" ".join(map(str, row)) for row in self.current_grid)
        self.current_problem = self.prompt_template.format(
            N=N,
            magic_constant=self.magic_constant,
            grid=grid_text,
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted answer and return terminal state."""
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert boxed content to grid
        parsed_grid = self._parse_grid_from_text(boxed)
        if parsed_grid is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        assert self.current_grid is not None, "Environment must be reset before calling step()"
        N = self.N
        M = self.magic_constant if self.magic_constant is not None else (N * (N * N + 1) // 2)

        # Validate size
        if len(parsed_grid) != N or any(len(row) != N for row in parsed_grid):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate values are permutation of 1..N^2
        values = [v for row in parsed_grid for v in row]
        if set(values) != set(range(1, N * N + 1)):
            info = {"error": "invalid_solution_value_set", "expected_set_size": N * N}
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate pre-filled cells are unchanged
        for r in range(N):
            for c in range(N):
                given = self.current_grid[r][c]
                if given != 0 and parsed_grid[r][c] != given:
                    info = {"error": "modified_given_cells", "row": r, "col": c, "given": given, "submitted": parsed_grid[r][c]}
                    return TERMINAL_STATE, 0.0, True, False, info

        # Validate sums for rows, columns, diagonals
        satisfied = 0
        # Rows
        for r in range(N):
            if sum(parsed_grid[r]) == M:
                satisfied += 1
        # Columns
        for c in range(N):
            if sum(parsed_grid[r][c] for r in range(N)) == M:
                satisfied += 1
        # Diagonals
        if sum(parsed_grid[i][i] for i in range(N)) == M:
            satisfied += 1
        if sum(parsed_grid[i][N - i - 1] for i in range(N)) == M:
            satisfied += 1

        is_correct = (satisfied == (2 * N + 2))
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied": satisfied,
            "total_checks": 2 * N + 2,
            "magic_constant": M,
            "reference_answer": self.reference_answer,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of content inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_grid_from_text(self, text: str) -> Optional[List[List[int]]]:
        """Parse a grid from the provided text: N lines with N integers each."""
        try:
            rows: List[List[int]] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                row = [int(tok) for tok in parts]
                rows.append(row)
            return rows if rows else None
        except Exception:
            return None

    def sample_random_action(self) -> str:
        """Sample a random action. If possible, return the correct solution wrapped in \\boxed{...}."""
        if self.reference_answer is not None:
            return f"\\boxed{{\n{self.reference_answer}\n}}"
        # Fallback: return a dummy boxed content
        return "\\boxed{1}"