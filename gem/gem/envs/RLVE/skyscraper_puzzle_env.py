from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SkyscraperPuzzleEnv(Env):
    """Skyscraper puzzle environment - single-turn Q&A.

    The task is to fill an N x N grid with integers from 0 to N-1 such that each row and column
    contains each integer exactly once. Visibility counts from each direction are provided as clues.
    """

    def __init__(self, N: int = 5, **kwargs):
        super().__init__()
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        self.N: int = N

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_grid: Optional[List[List[int]]] = None
        self.reference_answer: Optional[str] = None

        # Clues
        self.left: Optional[List[int]] = None
        self.right: Optional[List[int]] = None
        self.top: Optional[List[int]] = None
        self.bottom: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are given a Skyscraper puzzle.\n"
            "Please return your final grid inside \\boxed{...} with exactly N lines, each containing N integers separated by spaces.\n"
            "Do not include any extra text outside the box.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        N = self.N

        # Generate a Latin square using two random permutations
        permutation_row = list(range(N))
        permutation_col = list(range(N))
        random.shuffle(permutation_row)
        random.shuffle(permutation_col)
        grid = [[(permutation_row[i] + permutation_col[j]) % N for j in range(N)] for i in range(N)]

        # Compute visibility clues
        left = [self._count_visible_from_left(grid[i]) for i in range(N)]
        right = [self._count_visible_from_right(grid[i]) for i in range(N)]

        transposed = list(map(list, zip(*grid)))
        top = [self._count_visible_from_left(transposed[i]) for i in range(N)]
        bottom = [self._count_visible_from_right(transposed[i]) for i in range(N)]

        # Store references
        self.reference_grid = grid
        self.reference_answer = "\n".join(" ".join(map(str, row)) for row in grid)
        self.left, self.right, self.top, self.bottom = left, right, top, bottom

        # Build problem statement
        problem = (
            f"You are given a {N} × {N} grid. Your task is to place a building of height in the range [0, {N - 1}] in each cell such that:\n"
            f"- Each row and each column contains all integer heights from 0 to {N - 1} exactly once.\n"
            f"- A building is visible from a direction if there are no taller buildings before it in that direction.\n\n"
            f"The number of visible buildings is specified as follows:\n"
            f"- From the left of each row: {' '.join(map(str, left))}\n"
            f"- From the right of each row: {' '.join(map(str, right))}\n"
            f"- From the top of each column: {' '.join(map(str, top))}\n"
            f"- From the bottom of each column: {' '.join(map(str, bottom))}\n\n"
            f"Output Format: Your final answer must be placed inside \\boxed{{...}} and contain {N} lines, each with {N} integers (heights), separated by spaces. "
            f"Each line represents a row of the grid."
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted solution."""
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse grid from boxed content
        grid = self._parse_grid(boxed)
        if grid is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        N = self.N

        # Validate grid shape
        if len(grid) != N or not all(isinstance(row, list) and len(row) == N for row in grid):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate row and column sets
        all_vals = set(range(N))
        rows_ok = all(set(row) == all_vals for row in grid)
        cols_ok = all(set(grid[i][j] for i in range(N)) == all_vals for j in range(N))

        # Compute clues for submitted solution
        left_ans = [self._count_visible_from_left(grid[i]) for i in range(N)]
        right_ans = [self._count_visible_from_right(grid[i]) for i in range(N)]
        transposed = list(map(list, zip(*grid)))
        top_ans = [self._count_visible_from_left(transposed[i]) for i in range(N)]
        bottom_ans = [self._count_visible_from_right(transposed[i]) for i in range(N)]

        clues_match = (
            left_ans == self.left
            and right_ans == self.right
            and top_ans == self.top
            and bottom_ans == self.bottom
        )

        is_correct = rows_ok and cols_ok and clues_match
        reward = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "user_left": left_ans,
            "user_right": right_ans,
            "user_top": top_ans,
            "user_bottom": bottom_ans,
            "gold_left": self.left,
            "gold_right": self.right,
            "gold_top": self.top,
            "gold_bottom": self.bottom,
            "reference_answer": self.reference_answer,
        }

        if not rows_ok or not cols_ok:
            info["error"] = "invalid_solution"

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_grid(self, content: str) -> Optional[List[List[int]]]:
        """Parse a grid from text content: lines of space-separated integers."""
        try:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            grid: List[List[int]] = []
            for line in lines:
                tokens = line.split()
                row = list(map(int, tokens))
                grid.append(row)
            return grid
        except Exception:
            return None

    @staticmethod
    def _count_visible_from_left(row: List[int]) -> int:
        """Count visible buildings from the left (prefix maxima count)."""
        count = 0
        current_max = None
        for x in row:
            if current_max is None or x >= current_max:
                count += 1
                current_max = x
        return count

    @staticmethod
    def _count_visible_from_right(row: List[int]) -> int:
        """Count visible buildings from the right (suffix maxima count)."""
        count = 0
        current_max = None
        for x in reversed(row):
            if current_max is None or x >= current_max:
                count += 1
                current_max = x
        return count

    def sample_random_action(self) -> str:
        """Sample a random action. Here we return the reference answer if available, otherwise a random grid."""
        if self.reference_answer is not None:
            return f"\\boxed{{\n{self.reference_answer}\n}}"
        # Fallback: generate a random grid of correct shape (not guaranteed to satisfy clues)
        grid = [[random.randint(0, self.N - 1) for _ in range(self.N)] for _ in range(self.N)]
        content = "\n".join(" ".join(map(str, row)) for row in grid)
        return f"\\boxed{{\n{content}\n}}"