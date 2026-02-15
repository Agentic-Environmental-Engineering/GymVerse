from typing import Any, List, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SkyscraperSumPuzzleEnv(Env):
    """Skyscraper Sum Puzzle environment - single-turn Q&A.

    The agent must fill an N x N grid with integers 0..N-1 so that each row and column
    is a permutation of 0..N-1 exactly once. The "sum height" of visible buildings
    from each side is provided. A building is visible from a direction if it is not
    preceded by any taller building when looking from that direction. The sum for a
    side is the sum of the heights of all visible buildings from that side.

    The agent should output the completed grid inside a \\boxed{...} block, where the
    content consists of N lines, each containing N integers separated by spaces.
    """

    prompt_template = (
        "You are given a {N} × {N} grid. Your task is to place a building of height in the range [0, {N_minus_1}] in each cell such that:\n"
        "- Each row and each column contains all integer heights from 0 to {N_minus_1} exactly once.\n"
        "- A building is visible from a direction if there are no taller buildings before it in that direction.\n\n"
        "The sum height of visible buildings is specified as follows:\n"
        "- From the left of each row: {left}\n"
        "- From the right of each row: {right}\n"
        "- From the top of each column: {top}\n"
        "- From the bottom of each column: {bottom}\n\n"
        "Output Format: Your final answer should be the entire {N}×{N} grid placed inside a single \\boxed{{...}} block.\n"
        "Inside the \\boxed, provide exactly {N} lines; each line should contain {N} integers separated by single spaces. Each line corresponds to a grid row."
    )

    def __init__(
        self,
        N: Optional[int] = None,
        N_min: int = 3,
        N_max: int = 10,
        **kwargs: Any,
    ):
        """Initialize the SkyscraperSumPuzzleEnv.

        Args:
            N: If provided, fixes the puzzle size to N (must be >= 3).
            N_min: Minimum N if sampling (inclusive, must be >= 3).
            N_max: Maximum N if sampling (inclusive, must be >= N_min).

        Note: Rewards are fixed as:
            - Correct answer: 1.0
            - Wrong answer: 0.0
            - Format error: -0.1
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if N is None:
            if N_min < 3:
                raise ValueError("N_min should be greater than or equal to 3")
            if N_max < N_min:
                raise ValueError("N_max should be greater than or or equal to N_min")
        self.fixed_N: Optional[int] = N
        self.N_min: int = N_min
        self.N_max: int = N_max

        # Problem state
        self.N: Optional[int] = None
        self.left: Optional[List[int]] = None
        self.right: Optional[List[int]] = None
        self.top: Optional[List[int]] = None
        self.bottom: Optional[List[int]] = None
        self.reference_solution_grid: Optional[List[List[int]]] = None
        self.reference_solution_text: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Skyscraper Sum Puzzle.\n"
            "Please submit your entire grid answer inside a single \\boxed{...} block.\n"
            "Inside the \\boxed, provide exactly N lines; each line should have N integers separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new puzzle instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.N_min, self.N_max)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        self.N = N

        # Generate a Latin-square-like solution grid using two permutations
        permutation_row = list(range(N))
        permutation_col = list(range(N))
        random.shuffle(permutation_row)
        random.shuffle(permutation_col)
        grid = [[(permutation_row[i] + permutation_col[j]) % N for j in range(N)] for i in range(N)]

        # Compute visibility sums
        left = [sum(int(grid[i][j] == max(grid[i][: j + 1])) * grid[i][j] for j in range(N)) for i in range(N)]
        right = [sum(int(grid[i][j] == max(grid[i][j:])) * grid[i][j] for j in range(N)) for i in range(N)]
        transposed_grid = [[grid[j][i] for j in range(N)] for i in range(N)]
        top = [sum(int(transposed_grid[i][j] == max(transposed_grid[i][: j + 1])) * transposed_grid[i][j] for j in range(N)) for i in range(N)]
        bottom = [sum(int(transposed_grid[i][j] == max(transposed_grid[i][j:])) * transposed_grid[i][j] for j in range(N)) for i in range(N)]

        # Store problem state
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.reference_solution_grid = grid
        self.reference_solution_text = "\n".join(" ".join(map(str, row)) for row in grid)

        # Build problem text
        problem = self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            left=" ".join(map(str, left)),
            right=" ".join(map(str, right)),
            top=" ".join(map(str, top)),
            bottom=" ".join(map(str, bottom)),
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted solution."""
        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Answer must be inside \\boxed{...}."}

        # Parse grid from boxed content
        grid = self._parse_grid(boxed_content)
        N = self.N
        if N is None:
            # Should not happen if reset was called
            return TERMINAL_STATE, -0.1, True, False, {"error": "internal_error", "message": "Environment not initialized."}
        if grid is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Failed to parse grid from boxed content."}

        # Validate shape
        if len(grid) != N or any(len(row) != N for row in grid):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": f"Grid must be {N}x{N}."}

        # Validate row and column permutations
        required_set = set(range(N))
        if not all(set(row) == required_set for row in grid):
            info = {"correct": False, "reason": "row_constraint_violation"}
            info.update(self._final_info(grid))
            return TERMINAL_STATE, 0.0, True, False, info
        if not all(set(grid[i][j] for i in range(N)) == required_set for j in range(N)):
            info = {"correct": False, "reason": "column_constraint_violation"}
            info.update(self._final_info(grid))
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute visibility sums for submitted solution
        left = [sum(int(grid[i][j] == max(grid[i][: j + 1])) * grid[i][j] for j in range(N)) for i in range(N)]
        right = [sum(int(grid[i][j] == max(grid[i][j:])) * grid[i][j] for j in range(N)) for i in range(N)]
        transposed_solution = [[grid[j][i] for j in range(N)] for i in range(N)]
        top = [sum(int(transposed_solution[i][j] == max(transposed_solution[i][: j + 1])) * transposed_solution[i][j] for j in range(N)) for i in range(N)]
        bottom = [sum(int(transposed_solution[i][j] == max(transposed_solution[i][j:])) * transposed_solution[i][j] for j in range(N)) for i in range(N)]

        # Check constraints
        assert self.left is not None and self.right is not None and self.top is not None and self.bottom is not None
        is_correct = (left == self.left) and (right == self.right) and (top == self.top) and (bottom == self.bottom)

        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_solution": self.reference_solution_text,
            "user_solution": "\n".join(" ".join(map(str, row)) for row in grid),
            "N": N,
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
            "user_left": left,
            "user_right": right,
            "user_top": top,
            "user_bottom": bottom,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text, flags=0)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_grid(self, content: str) -> Optional[List[List[int]]]:
        """Parse a grid from the boxed content."""
        try:
            rows: List[List[int]] = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                row = list(map(int, parts))
                rows.append(row)
            return rows
        except Exception:
            return None

    def _final_info(self, user_grid: List[List[int]]) -> dict[str, Any]:
        """Assemble final info dictionary snippets."""
        return {
            "reference_solution": self.reference_solution_text,
            "user_solution": "\n".join(" ".join(map(str, row)) for row in user_grid),
            "N": self.N,
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
        }

    def sample_random_action(self) -> str:
        """Sample a random action by producing a random grid inside \\boxed{...}."""
        if self.N is None:
            N = self.fixed_N if self.fixed_N is not None else max(3, self.N_min)
        else:
            N = self.N
        # Generate a random grid (not necessarily valid)
        grid = [[random.randint(0, N - 1) for _ in range(N)] for _ in range(N)]
        text = "\n".join(" ".join(map(str, row)) for row in grid)
        return f"\\boxed{{{text}}}"