import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GridComponentEnv(Env):
    """Grid largest connected component environment - single turn Q&A."""

    def __init__(
        self,
        max_n_m: int = 10,
        reward_correct: float = 1.0,
        reward_wrong: float = 0.0,
        reward_format_error: float = -0.1,
        **kwargs
    ):
        super().__init__()
        # Parameter validation
        if max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")
        self.max_n_m = max_n_m

        # Rewards
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.reward_format_error = reward_format_error

        # Internal state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.grid: Optional[List[str]] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return instructions for the task."""
        return (
            "You are given a binary grid (each cell is '0' or '1'). "
            "Your task is to compute the size of the largest connected component of '1's. "
            "Cells are connected if they share an edge (up, down, left, right).\n"
            "Please provide your final answer in \\boxed{...} format containing a single integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate grid dimensions
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Generate grid with a random probability of '1'
        one_probability = random.uniform(0.1, 0.9)
        self.grid = [
            "".join("01"[random.random() < one_probability] for _ in range(self.M))
            for _ in range(self.N)
        ]

        # Compute reference answer using DFS-based labeling
        self.reference_answer = self._compute_largest_component(self.grid)

        # Build the problem statement
        grid_str = "\n".join(self.grid)
        self.current_problem = (
            f"You are given a {self.N} × {self.M} grid. Each cell contains either 0 or 1. "
            f"Please compute the largest connected component of 1's in the grid, where a connected component "
            f"is defined as a group of 1 cells that are reachable from each other by moving up, down, left, or right.\n\n"
            f"The grid is given as follows:\n{grid_str}\n\n"
            f"Output Format: Output a single integer in \\boxed{{...}} — the size of the largest connected component. "
            f"If there are no 1's in the grid, output 0."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and end the episode."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, self.reward_format_error, True, False, {"error": "format_error"}

        # Validate integer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, self.reward_wrong, True, False, {"error": "invalid_answer"}

        # Compare with reference
        is_correct = (user_answer == self.reference_answer)
        reward = self.reward_correct if is_correct else self.reward_wrong

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_largest_component(self, grid: List[str]) -> int:
        """Compute the size of the largest connected component of '1's in the grid."""
        N = len(grid)
        M = len(grid[0]) if N > 0 else 0
        labels = [[0] * M for _ in range(N)]

        def dfs_label(sx: int, sy: int) -> None:
            """Iterative DFS to label all cells connected to (sx, sy)."""
            stack = [(sx, sy)]
            while stack:
                x, y = stack.pop()
                for dx, dy in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < M and grid[nx][ny] == "1":
                        if labels[nx][ny] == 0:
                            labels[nx][ny] = labels[x][y]
                            stack.append((nx, ny))
                        else:
                            # This assertion mirrors the original environment's consistency check
                            assert labels[nx][ny] == labels[x][y], "Labels should match for connected components"

        total = 0
        counting = [0]
        for x in range(N):
            for y in range(M):
                if grid[x][y] == "1":
                    if labels[x][y] == 0:
                        total += 1
                        counting.append(0)
                        labels[x][y] = total
                        dfs_label(x, y)
                    counting[labels[x][y]] += 1

        return max(counting) if counting else 0

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # A reasonable range for the component size is [0, N*M]
        max_guess = (self.N or self.max_n_m) * (self.M or self.max_n_m)
        random_answer = random.randint(0, max_guess)
        return f"\\boxed{{{random_answer}}}"