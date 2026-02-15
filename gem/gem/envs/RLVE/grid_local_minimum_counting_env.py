import random
from typing import Any, Optional, SupportsFloat, Tuple, Dict, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GridLocalMinimumCountingEnv(Env):
    """Grid Local Minimum Counting Environment - Single-turn Q&A.

    The environment generates a grid with marks 'X' and '.' based on local minima obtained
    from a random permutation of numbers 1..N*M placed in the grid. The task is to count
    the number of valid labelings (permutations) such that the set of local minima is
    exactly the marked 'X' cells.
    """

    def __init__(self, max_n_m: int = 5, **kwargs):
        """Initialize the environment.

        Args:
            max_n_m: Maximum size for N and M (both selected uniformly in [2, max_n_m]).
                     Must be >= 2.
            **kwargs: Extra arguments ignored for compatibility.
        """
        super().__init__()
        self.max_n_m: int = max_n_m
        if self.max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")

        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.grid: Optional[List[List[str]]] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial counting problem on a grid.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Args:
            seed: Optional random seed.

        Returns:
            observation: The problem description as a string.
            info: Additional information (empty dict).
        """
        super().reset(seed)

        # Generate grid size
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        self.N, self.M = N, M

        # Create a random permutation and derive the local minima grid
        permutation = list(range(1, N * M + 1))
        random.shuffle(permutation)

        def get_num(i: int, j: int) -> int:
            return permutation[i * M + j]

        grid = [['.'] * M for _ in range(N)]
        for i in range(N):
            for j in range(M):
                local_minimum = True
                for dx, dy in [
                    (-1, -1), (-1, 0), (-1, +1),
                    (0, -1),           (0, +1),
                    (+1, -1), (+1, 0), (+1, +1)
                ]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < N and 0 <= nj < M and get_num(ni, nj) <= get_num(i, j):
                        local_minimum = False
                        break
                if local_minimum:
                    grid[i][j] = 'X'

        self.grid = grid

        # Compute reference answer
        self.reference_answer = self._compute_reference(grid, N, M)

        # Build problem statement
        grid_str = "\n".join("".join(row) for row in grid)
        self.current_problem = (
            f"Consider a grid of size {N} × {M}, where the numbers from 1 to {N*M} are placed "
            f"in the cells such that each number appears exactly once.\n"
            f"A cell is considered a local minimum if its value is strictly less than all of its 8 neighbors "
            f"(adjacent vertically, horizontally, or diagonally); if a neighbor does not exist, it is considered to be infinitely large.\n"
            f"You are given a grid of size {N} × {M} where some cells are marked with 'X' and others with '.'. "
            f"Please count how many valid numberings exist such that the local minima are exactly those marked with 'X'. "
            f"The grid is given as follows:\n{grid}\n\n"
            f"Output Format: Output a single integer in \\boxed{{...}}."
        ).replace("{grid}", grid_str)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference(self, raw_grid: List[List[str]], N: int, M: int) -> int:
        """Compute the number of valid labelings that produce exactly the given set of local minima.

        This implementation preserves the original algorithm and logic from the RLVE environment.
        """
        # Build boolean map of required local minima
        grid_bool = [[(raw_grid[i][j] == 'X') for j in range(M)] for i in range(N)]

        # Quick invalid check: no two required 'X's may be adjacent (including diagonals)
        for i in range(N):
            for j in range(M):
                if grid_bool[i][j]:
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < N and 0 <= nj < M and grid_bool[ni][nj]:
                                raise AssertionError("Invalid grid: two local minima are adjacent")

        ans = 0

        def inrange(x: int, y: int) -> bool:
            return 0 <= x < N and 0 <= y < M

        def calc() -> int:
            """Inclusion-exclusion DP counting for exact local minima configuration."""
            pos = [(i, j) for i in range(N) for j in range(M) if grid_bool[i][j]]
            cntX = len(pos)
            total = N * M

            # dp[used_cells][subset_mask]
            dp = [[0] * (1 << cntX) for _ in range(total + 2)]
            dp[0][0] = 1

            for s in range(1 << cntX):
                # mark all cells "blocked" by the minima NOT in subset s
                blocked = [[False] * M for _ in range(N)]
                free_cells = total
                for k in range(cntX):
                    if not (s & (1 << k)):
                        x, y = pos[k]
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                ni, nj = x + di, y + dj
                                if inrange(ni, nj) and not blocked[ni][nj]:
                                    blocked[ni][nj] = True
                                    free_cells -= 1

                for used in range(free_cells + 1):
                    v = dp[used][s]
                    if not v:
                        continue
                    # place a non-min in one of the remaining free cells
                    dp[used + 1][s] += v * (free_cells - used)
                    # or turn one of the excluded minima into an actual minima
                    for k in range(cntX):
                        if not (s & (1 << k)):
                            dp[used + 1][s | (1 << k)] += v

            # We want all total cells assigned, and all minima chosen
            return dp[total][(1 << cntX) - 1]

        def dfs(i: int, j: int, sign: int) -> None:
            nonlocal ans
            if i == N:
                ans += sign * calc()
                return

            ni, nj = (i, j + 1) if j + 1 < M else (i + 1, 0)

            # option 1: don't add a minima here
            dfs(ni, nj, sign)

            # option 2: if this cell is not already a minima, and none of its neighbors is one, we can add it
            if not grid_bool[i][j]:
                ok = True
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ai, aj = i + di, j + dj
                        if inrange(ai, aj) and grid_bool[ai][aj]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    grid_bool[i][j] = True
                    dfs(ni, nj, -sign)
                    grid_bool[i][j] = False

        dfs(0, 0, 1)
        assert ans > 0
        return ans

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step by verifying the provided answer.

        Args:
            action: The agent's answer text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE since this is a single-turn environment.
            reward: 1.0 if correct, 0.0 if wrong, -0.1 if format error.
            terminated: Always True after one step.
            truncated: Always False.
            info: Additional information including correctness and answers.
        """
        # Parse the boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer and compare
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == user_answer)
        reward = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "grid": ["".join(row) for row in (self.grid or [])],
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (a random integer) in boxed format."""
        # The true answer may be large; we just sample a small non-negative integer as a placeholder.
        random_answer = random.randint(0, 1000)
        return f"\\boxed{{{random_answer}}}"