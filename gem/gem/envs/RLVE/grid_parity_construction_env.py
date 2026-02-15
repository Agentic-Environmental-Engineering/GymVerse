from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GridParityConstructionEnv(Env):
    """Grid parity construction environment - single-turn Q&A."""

    def __init__(
        self,
        max_n_m: int = 10,
        wrong_format_reward: float = -0.1,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
            max_n_m: Maximum size for N and M (inclusive). Must be >= 2.
            wrong_format_reward: Reward to return when the answer format is invalid.
        """
        super().__init__()
        self.max_n_m = max_n_m
        self.wrong_format_reward = wrong_format_reward

        # Generated problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.parity: Optional[List[List[int]]] = None
        self.reference_grid: Optional[List[str]] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a grid parity matrix construction task.\n"
            "Definition (Parity Matrix): For each cell (i, j), its parity is the XOR of the cell’s value and the values of its four neighbors (up, down, left, right). "
            "A neighbor outside the grid is treated as 0.\n\n"
            "Answer Format:\n"
            "- Output exactly N lines, each with M characters (each '0' or '1'), without separators.\n"
            "- Wrap your entire multiline answer inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        assert self.max_n_m >= 2, "max_n_m should be greater than or equal to 2"

        # Generate N, M
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Generate a random grid with a bias controlled by one_probability
        one_probability = random.random()
        self.reference_grid = [
            "".join("01"[random.random() < one_probability] for _ in range(self.M))
            for _ in range(self.N)
        ]

        # Compute parity from the reference grid
        self.parity = self._compute_parity(self.reference_grid)

        # Build problem statement
        parity_str = "\n".join("".join(map(str, row)) for row in self.parity)
        self.current_problem = (
            f"Please construct a {self.N} × {self.M} binary matrix (each cell is either 0 or 1) "
            f"such that its parity matrix is:\n{parity_str}\n\n"
            f"Output Format: Output {self.N} lines, each with {self.M} characters (each '0' or '1'), without separators. "
            f"Wrap your entire multiline answer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the submitted answer."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Process content into matrix lines
        matrix_lines = []
        for line in boxed_content.splitlines():
            line = line.strip()
            if line:
                matrix_lines.append(line)

        # Validate dimensions and characters
        if self.N is None or self.M is None or self.parity is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error"}

        if len(matrix_lines) != self.N or any(len(row) != self.M for row in matrix_lines):
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        if any(not all(c in "01" for c in row) for row in matrix_lines):
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Compute parity of user's grid
        user_parity = self._compute_parity(matrix_lines)

        # Compare parities
        satisfied = sum(
            int(user_parity[i][j] == self.parity[i][j])
            for i in range(self.N)
            for j in range(self.M)
        )
        total = self.N * self.M
        is_correct = (satisfied == total)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied": satisfied,
            "total": total,
            "reference_answer": "\n".join(self.reference_grid) if self.reference_grid else None,
            "user_answer": "\n".join(matrix_lines),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Supports multiline content."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_parity(self, grid: List[str]) -> List[List[int]]:
        """Compute the parity matrix for the given grid."""
        N = len(grid)
        M = len(grid[0]) if N > 0 else 0
        parity = [[0] * M for _ in range(N)]
        for i in range(N):
            for j in range(M):
                parity[i][j] ^= int(grid[i][j])
                for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < M:
                        parity[i][j] ^= int(grid[ni][nj])
        return parity

    def sample_random_action(self) -> str:
        """Sample a random valid action: a random binary matrix of the correct size."""
        if self.N is None or self.M is None:
            # If called before reset, create a small default matrix
            n, m = 2, 2
        else:
            n, m = self.N, self.M
        random_grid = [
            "".join(random.choice("01") for _ in range(m))
            for _ in range(n)
        ]
        content = "\n".join(random_grid)
        return f"\\boxed{{\n{content}\n}}"