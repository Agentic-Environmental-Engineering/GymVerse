from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TwiddlePuzzleEnv(Env):
    """Twiddle Puzzle environment - single-turn Q&A.

    Task:
    - You are given an N x M grid with digits 0 to N*M-1.
    - You may select a top-left corner (i, j) with 0 <= i <= N-K and 0 <= j <= M-K.
    - Perform a 90-degree counterclockwise rotation on the K x K subgrid starting at (i, j).
    - Starting from the given start grid, transform it into the destination grid by applying a sequence of such rotations.

    Response format:
    - Provide each action on its own line as: i j
    - Wrap the entire action sequence inside a single \\boxed{...} block.
    """

    def __init__(
        self,
        max_n_m: int = 5,
        steps: int = 3,
        **kwargs
    ):
        """
        Initialize the TwiddlePuzzleEnv.

        Args:
            max_n_m: Maximum value for N and M (both sampled in [2, max_n_m]). Must be >= 2.
            steps: Number of random rotations applied to generate the destination grid. Must be >= 1.
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        assert steps >= 1, "steps should be greater than or equal to 1"
        self.max_n_m = max_n_m
        self.steps = steps

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None
        self.start_grid: Optional[List[List[int]]] = None
        self.destination_grid: Optional[List[List[int]]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a grid-twiddling puzzle.\n"
            "At any time, you may select a cell (i, j) with 0 <= i <= N - K and 0 <= j <= M - K, "
            "and perform a 90-degree counterclockwise rotation on the K x K subgrid starting at (i, j).\n"
            "Your goal is to transform the start grid into the destination grid.\n\n"
            "Output Format: Provide your sequence of actions, one per line, each as 'i j'. "
            "Wrap the entire sequence inside a single \\boxed{...} block. For example:\n"
            "\\boxed{0 1\\n2 0}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new puzzle instance."""
        super().reset(seed)

        # Sample dimensions
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        K = random.randint(2, min(N, M))

        # Generate start grid as a random permutation of 0..N*M-1
        start_perm = list(range(N * M))
        random.shuffle(start_perm)
        start_grid = [[start_perm[i * M + j] for j in range(M)] for i in range(N)]

        # Apply a sequence of 'steps' random rotations to produce the destination grid
        destination_grid = [row[:] for row in start_grid]
        reference_actions: List[Tuple[int, int]] = []
        for _ in range(self.steps):
            i = random.randint(0, N - K)
            j = random.randint(0, M - K)
            reference_actions.append((i, j))
            destination_grid = self._apply_rotation(destination_grid, i, j, K)

        # Store state
        self.N, self.M, self.K = N, M, K
        self.start_grid = start_grid
        self.destination_grid = destination_grid
        self.reference_answer = "\n".join(f"{i} {j}" for i, j in reference_actions)

        # Build problem text
        problem_text = (
            f"You are given a {N} x {M} grid, where each cell contains a digit from 0 to {N * M - 1}. "
            f"At any time, you may select a cell (i, j) such that 0 <= i <= {N} - {K} and 0 <= j <= {M} - {K}. "
            f"Then, you perform a 90-degree counterclockwise rotation on the {K} x {K} subgrid starting at position (i, j).\n\n"
            f"You start with the following grid:\n"
            f"{self._grid_to_string(start_grid)}\n\n"
            f"Your goal is to transform it into the following grid:\n"
            f"{self._grid_to_string(destination_grid)}\n\n"
            f"Output Format: Each action should be written on its own line as 'i j'. "
            f"Wrap the entire sequence inside a single \\boxed{{...}}. Example: \\boxed{{0 1\\n2 0}}"
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted sequence of actions."""
        # Parse content inside \boxed{...}
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse actions
        actions: List[Tuple[int, int]] = []
        for line in boxed.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            try:
                i_val = int(parts[0])
                j_val = int(parts[1])
            except ValueError:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            actions.append((i_val, j_val))

        # Validate and apply actions
        if self.start_grid is None or self.destination_grid is None or self.N is None or self.M is None or self.K is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "state_error"}

        current_grid = [row[:] for row in self.start_grid]
        for (i, j) in actions:
            if not (0 <= i <= self.N - self.K and 0 <= j <= self.M - self.K):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_action"}
            current_grid = self._apply_rotation(current_grid, i, j, self.K)

        # Check correctness
        is_correct = self._grids_equal(current_grid, self.destination_grid)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_actions": actions
        }

        return TERMINAL_STATE, reward, True, False, info

    def _apply_rotation(self, grid: List[List[int]], i: int, j: int, K: int) -> List[List[int]]:
        """Apply a 90-degree counterclockwise rotation to the K x K subgrid at (i, j)."""
        N = len(grid)
        M = len(grid[0]) if N > 0 else 0
        new_grid = [grid[row][:] for row in range(N)]
        for x in range(K):
            for y in range(K):
                new_grid[i + K - 1 - y][j + x] = grid[i + x][j + y]
        return new_grid

    def _grid_to_string(self, grid: List[List[int]]) -> str:
        """Convert a grid to a string with rows separated by newlines and columns by spaces."""
        return "\n".join(" ".join(map(str, row)) for row in grid)

    def _grids_equal(self, A: List[List[int]], B: List[List[int]]) -> bool:
        """Check if two grids are equal."""
        if len(A) != len(B):
            return False
        for ra, rb in zip(A, B):
            if len(ra) != len(rb):
                return False
            for a, b in zip(ra, rb):
                if a != b:
                    return False
        return True

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a plausible action: use the reference solution when available; otherwise random valid action."""
        if self.reference_answer is not None:
            # Return the known solution to achieve a correct answer
            return f"\\boxed{{{self.reference_answer}}}"
        # Fallback: a single random valid action
        if self.N is None or self.M is None or self.K is None:
            return "\\boxed{}"
        i = random.randint(0, self.N - self.K)
        j = random.randint(0, self.M - self.K)
        return f"\\boxed{{{i} {j}}}"