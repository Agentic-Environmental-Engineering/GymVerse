import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class WhackAMoleEnv(Env):
    """Whack-a-Mole grid problem environment - single-turn Q&A.

    The task:
    - Given an N x M grid of non-negative integers representing moles in each cell.
    - You can define a fixed hammer size r x c (1 ≤ r ≤ N, 1 ≤ c ≤ M) before starting.
    - Each swing chooses a subrectangle of size r x c fully within the grid, where each cell has at least 1 mole.
    - Each swing removes exactly 1 mole from each cell in the chosen subrectangle.
    - Goal: remove all moles with the minimum number of swings.

    Output format: a single integer in \\boxed{...}.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        max_beat: int = 3,
        **kwargs
    ):
        """Initialize the environment parameters.

        Args:
            max_n_m: Maximum value for N and M (both sampled in [2, max_n_m]).
            max_beat: Maximum number of moles added per generated rectangle swing in the internal grid construction.
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        assert max_beat >= 1, "max_beat should be greater than or equal to 1"
        self.max_n_m = max_n_m
        self.max_beat = max_beat

        # Internal state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.grid: Optional[List[List[int]]] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.total_moles: int = 0

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Whack-a-Mole grid hammer problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: The problem statement string.
            info: Additional information (empty dict by default).
        """
        super().reset(seed)

        # Generate grid dimensions
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        self.N, self.M = N, M

        # Randomly choose a generation hammer size R x C (used only to construct a valid grid via difference arrays)
        R = random.randint(1, N)
        C = random.randint(1, M)

        # Construct grid using 2D difference method to ensure feasibility with some hammer swings
        grid = [[0] * M for _ in range(N)]
        # Apply random rectangles
        for i in range(N - R + 1):
            for j in range(M - C + 1):
                num_moles = random.randint(0, self.max_beat)
                grid[i][j] += num_moles
                if i + R < N:
                    grid[i + R][j] -= num_moles
                if j + C < M:
                    grid[i][j + C] -= num_moles
                if i + R < N and j + C < M:
                    grid[i + R][j + C] += num_moles

        # Convert difference array to actual grid using prefix sums
        for i in range(N):
            for j in range(M):
                if i > 0:
                    grid[i][j] += grid[i - 1][j]
                if j > 0:
                    grid[i][j] += grid[i][j - 1]
                if i > 0 and j > 0:
                    grid[i][j] -= grid[i - 1][j - 1]

        self.grid = grid

        total = sum(sum(row) for row in grid)
        self.total_moles = total

        # Compute reference answer: minimal number of swings with an optimal fixed hammer size
        if total == 0:
            best_area = R * C  # Any valid area works; minimal swings is 0
            reference_answer = 0
        else:
            best_area = 0

            # Try every possible hammer area, largest first
            for area in range(N * M + 1, 0, -1):
                if total % area != 0:
                    continue
                if area <= best_area:
                    continue
                for r in range(1, area + 1):
                    if area % r != 0:
                        continue
                    c = area // r
                    if not (1 <= r <= N and 1 <= c <= M):
                        continue
                    if area <= best_area:
                        continue

                    # 2D difference array, size (N+1)x(M+1)
                    diff = [[0] * (M + 1) for _ in range(N + 1)]
                    ok = True

                    # Sweep through grid and simulate scheduling hammer swings
                    for i in range(N):
                        for j in range(M):
                            # accumulate 2D prefix sum at (i, j)
                            if i > 0:
                                diff[i][j] += diff[i - 1][j]
                            if j > 0:
                                diff[i][j] += diff[i][j - 1]
                            if i > 0 and j > 0:
                                diff[i][j] -= diff[i - 1][j - 1]

                            # If overshoot, fail
                            if diff[i][j] > grid[i][j]:
                                ok = False
                                break

                            # If we need more hits, schedule them if possible
                            if diff[i][j] < grid[i][j]:
                                if i + r > N or j + c > M:
                                    ok = False
                                    break
                                t = grid[i][j] - diff[i][j]
                                diff[i][j] += t
                                diff[i + r][j] -= t
                                diff[i][j + c] -= t
                                diff[i + r][j + c] += t
                        if not ok:
                            break

                    if ok:
                        best_area = area

            # The minimum number of swings is total moles divided by the largest valid hammer area
            assert best_area >= R * C, "best_area should be at least R * C"
            reference_answer = total // best_area

        self.reference_answer = reference_answer

        # Build problem description
        grid_str = "\n".join(" ".join(map(str, row)) for row in grid)
        problem_text = (
            f"You are given an {N} × {M} grid, where each cell contains a non-negative integer representing the number of moles in that hole:\n"
            f"{grid_str}\n\n"
            f"You are allowed to define a fixed hammer size of r × c (1 ≤ r ≤ {N}, 1 ≤ c ≤ {M}) before starting. Each time you swing the hammer:\n"
            f"- You choose an r × c subrectangle in the grid (without rotation).\n"
            f"- This subrectangle must be fully within the grid.\n"
            f"- Each cell in the subrectangle must contain at least 1 mole.\n"
            f"- Each cell in the subrectangle has exactly 1 mole removed (so r × c moles are removed per swing).\n\n"
            f"You may swing the hammer multiple times, but you cannot change its size after choosing r and c. "
            f"Your goal is to remove all the moles from the grid with the minimum number of swings.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}} — the minimum number of hammer swings required to remove all moles from the grid."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step by verifying the provided answer.

        Args:
            action: The agent's answer text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE as this is a single-turn environment.
            reward: 1.0 if correct, 0.0 if wrong, -0.1 if format error.
            terminated: True always (single-turn).
            truncated: False always.
            info: Additional info such as correctness and reference answer.
        """
        answer_text = self._parse_answer(action)

        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        upper = self.total_moles if self.total_moles > 0 else (self.max_n_m * self.max_n_m * self.max_beat)
        upper = max(upper, 1)
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"