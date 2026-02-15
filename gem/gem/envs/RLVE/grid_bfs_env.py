from typing import Any, List, Optional, SupportsFloat, Tuple
import random
from collections import deque
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GridBFSEnv(Env):
    """Grid BFS distance problem environment - single-turn Q&A.

    The task: Given an N x M grid with cells containing '0', '1', or 'X',
    compute for each cell the shortest distance to any '1' cell using
    four-directional moves (up, down, left, right) and not moving through 'X'.
    If a cell cannot reach any '1', its distance is -1. A '1' cell has distance 0,
    and an 'X' cell has distance -1.

    The answer must be provided as N lines, each containing M integers separated by spaces,
    enclosed inside \\boxed{...}. Newlines are allowed inside the box.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        # The following parameters are preserved for compatibility with the original RLVE environment,
        # but they are not used in the GEM reward scheme. Rewards are fixed as specified.
        wrong_format: float = -1.0,
        rewarding_strategy: str = "mean([gold=answer])^beta",
        rewarding_beta: float = 10.0,
        rewarding_weight: float = +1.0,
        **kwargs: Any
    ):
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        self.max_n_m = max_n_m

        # Compatibility placeholders (not used in GEM reward computation)
        self._compat_rewards = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_beta": rewarding_beta,
            "rewarding_weight": rewarding_weight,
        }

        # State variables
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.grid: Optional[List[List[str]]] = None
        self.gold_answer: Optional[List[List[int]]] = None
        self.reference_answer_text: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions for the environment."""
        return (
            "You are solving a shortest-distance-on-grid problem.\n"
            "For an N × M grid with cells '0', '1', or 'X':\n"
            "1) You may move up, down, left, or right to adjacent cells.\n"
            "2) You cannot move through 'X' cells.\n"
            "3) For each cell, compute the shortest distance to any '1' cell.\n"
            "4) If a cell cannot reach any '1', its distance is -1.\n"
            "5) The distance for a '1' cell is 0; the distance for an 'X' cell is also -1.\n\n"
            "Output Format:\n"
            "- Output N lines, each containing M integers separated by spaces.\n"
            "- Enclose the entire output inside \\boxed{...}. Newlines are allowed inside the box.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new grid instance."""
        super().reset(seed)

        # Generate dimensions
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Random cell distribution for '0', '1', 'X'
        weights = [random.randint(1, self.N * self.M) for _ in range(3)]
        total = sum(weights)
        probs = [w / total for w in weights]
        self.grid = [
            [random.choices(["0", "1", "X"], weights=probs, k=1)[0] for _ in range(self.M)]
            for _ in range(self.N)
        ]

        # Compute BFS distances to nearest '1'
        self.gold_answer = [[-1] * self.M for _ in range(self.N)]
        queue: deque[Tuple[int, int]] = deque()

        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j] == "1":
                    self.gold_answer[i][j] = 0
                    queue.append((i, j))

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.N
                    and 0 <= ny < self.M
                    and self.grid[nx][ny] != "X"
                    and self.gold_answer[nx][ny] == -1
                ):
                    self.gold_answer[nx][ny] = self.gold_answer[x][y] + 1
                    queue.append((nx, ny))

        # Build reference answer text (N lines, each with M integers separated by spaces)
        self.reference_answer_text = "\n".join(
            " ".join(map(str, row)) for row in self.gold_answer
        )

        # Build the problem prompt
        grid_str = "\n".join("".join(row) for row in self.grid)
        problem = (
            f"You are given a {self.N} × {self.M} grid. Each cell contains '0', '1', or 'X'. "
            f"For each cell, compute its shortest distance to any cell containing '1'.\n"
            f"The grid is given as follows:\n{grid_str}\n\n"
            f"Output Format: Output {self.N} lines, each containing {self.M} integers separated by spaces, "
            f"enclosed inside \\boxed{{...}}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return terminal state."""
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse matrix from boxed content
        matrix: List[List[int]] = []
        try:
            for line in boxed.splitlines():
                stripped = line.strip()
                if stripped:
                    row = list(map(int, stripped.split()))
                    matrix.append(row)
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate shape
        if self.N is None or self.M is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "state_error"}

        if len(matrix) != self.N or any(len(r) != self.M for r in matrix):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Check correctness
        is_correct = matrix == self.gold_answer
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_text,
            "user_answer": matrix,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re

        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random N x M integer matrix boxed."""
        if self.N is None or self.M is None:
            # If called before reset, provide a generic boxed value
            return "\\boxed{0}"

        random_matrix = [
            [random.randint(-1, max(self.N, self.M)) for _ in range(self.M)]
            for _ in range(self.N)
        ]
        content = "\n".join(" ".join(map(str, row)) for row in random_matrix)
        return f"\\boxed{{{content}}}"