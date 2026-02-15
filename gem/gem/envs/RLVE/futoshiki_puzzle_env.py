from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FutoshikiPuzzleEnv(Env):
    """
    Futoshiki puzzle environment - single-turn Q&A.

    Task:
    - Fill an N x N grid with integers in [0, N-1] so that each row and column contains all numbers 0..N-1 exactly once.
    - Respect given inequality constraints between cells.
    - Respect pre-filled cells (cells not equal to -1 must remain unchanged).

    Answer format:
    - The final answer must be enclosed in \\boxed{...}.
    - Inside the \\boxed{...}, provide N lines, each with N integers separated by spaces.
    """

    def __init__(
        self,
        N: int = 5,
        sparsity: float = 0.5,
        inequality_constraint_num_multiple: int = 2,
        **kwargs
    ):
        super().__init__()
        self.N: int = N
        self.sparsity: float = sparsity
        self.inequality_constraint_num_multiple: int = inequality_constraint_num_multiple

        # Generated instance state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.full_solution_matrix: Optional[List[List[int]]] = None
        self.given_matrix: Optional[List[List[int]]] = None
        self.inequalities: Optional[List[Tuple[int, int, int, int]]] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a Futoshiki-like Latin square puzzle.\n"
            "Fill the grid so that each row and each column contains all integers from 0 to N-1 exactly once,\n"
            "respect the provided inequality constraints, and do not change any pre-filled cells.\n"
            "Please provide your final answer inside \\boxed{...} with N lines, each line containing N integers separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new puzzle instance."""
        super().reset(seed)

        assert self.N >= 3, "N should be greater than or equal to 3"
        assert 0 < self.sparsity < 1, "sparsity should be between 0 and 1"

        N = self.N

        # Generate a Latin square via permutation of row/column
        permutation_row = list(range(N))
        permutation_col = list(range(N))
        random.shuffle(permutation_row)
        random.shuffle(permutation_col)

        full_matrix = [[(permutation_row[i] + permutation_col[j]) % N for j in range(N)] for i in range(N)]
        self.full_solution_matrix = [row[:] for row in full_matrix]
        self.reference_answer = "\n".join(" ".join(map(str, row)) for row in self.full_solution_matrix)

        # Build all possible inequality pairs where value(a) < value(b)
        all_inequalities: List[Tuple[int, int, int, int]] = []
        for x1 in range(N):
            for y1 in range(N):
                for x2 in range(N):
                    for y2 in range(N):
                        if full_matrix[x1][y1] < full_matrix[x2][y2]:
                            all_inequalities.append((x1, y1, x2, y2))

        k = random.randint(1, min(len(all_inequalities), self.inequality_constraint_num_multiple * N))
        inequalities = random.sample(all_inequalities, k=k)
        self.inequalities = inequalities

        # Create the puzzle by removing some entries (set to -1)
        puzzle_matrix = [row[:] for row in full_matrix]
        empty_count = max(1, int(N * N * self.sparsity))
        empty_cells = random.sample(range(N * N), empty_count)
        for cell in empty_cells:
            r, c = divmod(cell, N)
            puzzle_matrix[r][c] = -1
        self.given_matrix = puzzle_matrix

        # Build the problem text
        inequalities_str = "\n".join(f"c[{x1}][{y1}] < c[{x2}][{y2}]" for x1, y1, x2, y2 in inequalities)
        matrix_str = "\n".join(" ".join(map(str, row)) for row in puzzle_matrix)

        self.current_problem = (
            f"You are given a {N} × {N} matrix. Some cells are already filled with integers in the range [0, {N - 1}], "
            "and the rest are empty (denoted by -1). Please fill the empty cells with integers in the same range such that:\n"
            f"- Each row and each column contains all integers from 0 to {N - 1} exactly once.\n"
            "- The following inequality constraints between cells are satisfied (use c[i][j] to denote the cell at row i, column j, 0-indexed):\n"
            f"{inequalities_str}\n\n"
            f"The original matrix is as follows:\n{matrix_str}\n\n"
            f"Output Format: Your final answer should contain {N} lines, each with {N} integers separated by spaces. "
            "Each line represents a row of the completed matrix, and the entire matrix must be enclosed in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract content inside \\boxed{...}. Returns the last match if multiple are present."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_matrix_from_text(self, text: str) -> Optional[List[List[int]]]:
        """Parse a matrix from the given text. Each non-empty line contains space-separated integers."""
        lines = [ln.strip() for ln in text.splitlines()]
        # Filter out completely empty lines
        lines = [ln for ln in lines if ln]
        if not lines:
            return None
        matrix: List[List[int]] = []
        try:
            for line in lines:
                row = list(map(int, line.split()))
                matrix.append(row)
        except ValueError:
            return None
        return matrix

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted solution."""
        # Ensure a problem has been generated
        if self.given_matrix is None or self.inequalities is None or self.full_solution_matrix is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse matrix
        solution = self._parse_matrix_from_text(boxed_content)
        if solution is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.N
        info: dict[str, Any] = {}

        # Dimension check
        if len(solution) != N or any(len(row) != N for row in solution):
            info.update({"error": "dimension_mismatch"})
            return TERMINAL_STATE, 0.0, True, False, info

        # Row validity: each row must be a permutation of 0..N-1
        row_valid = all(set(row) == set(range(N)) for row in solution)
        # Column validity: each column must be a permutation of 0..N-1
        col_valid = all(set(solution[i][j] for i in range(N)) == set(range(N)) for j in range(N))

        # Check pre-filled cells
        filled_cells_respected = True
        for r in range(N):
            for c in range(N):
                original_val = self.given_matrix[r][c]
                if original_val != -1 and solution[r][c] != original_val:
                    filled_cells_respected = False
                    break
            if not filled_cells_respected:
                break

        # Check inequalities
        satisfied = 0
        total_ineq = len(self.inequalities)
        for x1, y1, x2, y2 in self.inequalities:
            if solution[x1][y1] < solution[x2][y2]:
                satisfied += 1

        all_inequalities_satisfied = (satisfied == total_ineq)

        is_correct = bool(row_valid and col_valid and filled_cells_respected and all_inequalities_satisfied)
        reward: float = 1.0 if is_correct else 0.0

        info.update({
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": solution,
            "row_valid": row_valid,
            "column_valid": col_valid,
            "filled_cells_respected": filled_cells_respected,
            "satisfied_inequalities": satisfied,
            "total_inequalities": total_ineq
        })

        return TERMINAL_STATE, reward, True, False, info

    def sample_random_action(self) -> str:
        """Sample a random action in the correct \\boxed{...} format. Returns the reference solution occasionally."""
        if self.full_solution_matrix is not None and random.random() < 0.5:
            # Return the correct solution
            content = "\n".join(" ".join(map(str, row)) for row in self.full_solution_matrix)
            return f"\\boxed{{\n{content}\n}}"

        # Otherwise, generate a random matrix (likely invalid but formatted correctly)
        N = self.N
        matrix = []
        for _ in range(N):
            row = [random.randint(0, N - 1) for _ in range(N)]
            matrix.append(row)
        content = "\n".join(" ".join(map(str, row)) for row in matrix)
        return f"\\boxed{{\n{content}\n}}"