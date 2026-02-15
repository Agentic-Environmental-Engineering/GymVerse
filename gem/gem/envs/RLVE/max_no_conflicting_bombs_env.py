import sys
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxNoConflictingBombsEnv(Env):
    """
    Max No-Conflicting Bombs environment - single-turn Q&A.

    Task:
    - You are given an N x M grid with characters '#', 'x', or '*'.
    - You may replace some '*' cells with 'B', subject to the constraint that
      no two 'B' cells appear in the same row or column unless there is at least one '#'
      between them in that row or column.
    - Your goal is to maximize the number of 'B' cells.

    Action/Answer:
    - Output only the maximum possible number of 'B' cells, as a single integer
      enclosed in \\boxed{...}.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n_m: The maximum value for both N and M (N, M are sampled uniformly from [2, max_n_m]).
        """
        super().__init__()
        assert max_n_m >= 2, "MAX_N_M should be greater than or equal to 2"
        self.max_n_m: int = max_n_m

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.grid: Optional[List[List[str]]] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an N × M grid. Each cell contains one of '#', 'x', or '*'. "
            "You may replace some '*' cells with 'B', under the condition that no two 'B' cells "
            "may appear in the same row or column unless there is at least one '#' between them. "
            "Your goal is to maximize the number of 'B' cells.\n\n"
            "Answer Format: Provide only the maximum possible number of 'B' cells, as a single "
            "integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample N and M
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)

        # Random distribution over ['#', 'x', '*']
        distribution = [random.randint(1, N * M) for _ in range(3)]
        s = sum(distribution)
        weights = [x / s for x in distribution]

        # Generate grid
        A = [random.choices(["#", "x", "*"], weights=weights, k=M) for _ in range(N)]

        # Store for later use
        self.N = N
        self.M = M
        self.grid = A

        # Compute the reference answer (maximum number of bombs) using maximum bipartite matching
        gold = self._compute_max_bombs(A, N, M)
        self.reference_answer = gold

        # Build problem description
        grid_str = "\n".join("".join(row) for row in A)
        self.current_problem = (
            f"You are given a {N} × {M} grid. Each cell contains one of the following characters: '#', 'x', or '*'. "
            f"You may replace some '*' cells with 'B', under the following condition: no two 'B' cells may appear "
            f"in the same row or column unless there is at least one '#' between them. Try to maximize the number of 'B' cells.\n"
            f"The grid is given in row-major order:\n{grid_str}\n\n"
            f"Output Format: Provide only the maximum possible number of 'B' cells as a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_max_bombs(self, A: List[List[str]], N: int, M: int) -> int:
        """Compute the maximum number of bombs using bipartite matching on row/column segments."""
        # Assign row-segment IDs to each non-# cell
        ROW = [[-1] * M for _ in range(N)]
        tot = 0
        for i in range(N):
            j = 0
            while j < M:
                if A[i][j] == '#':
                    j += 1
                else:
                    k = j
                    while k < M and A[i][k] != '#':
                        ROW[i][k] = tot
                        k += 1
                    tot += 1
                    j = k
        row_cnt = tot

        # Assign column-segment IDs to each non-# cell
        COL = [[-1] * M for _ in range(N)]
        tot = 0
        for j in range(M):
            i = 0
            while i < N:
                if A[i][j] == '#':
                    i += 1
                else:
                    k = i
                    while k < N and A[k][j] != '#':
                        COL[k][j] = tot
                        k += 1
                    tot += 1
                    i = k
        col_cnt = tot

        # Build bipartite graph: row segments -> col segments for '*' cells
        G: List[List[int]] = [[] for _ in range(row_cnt)]
        for i in range(N):
            for j in range(M):
                if A[i][j] == '*':
                    u = ROW[i][j]
                    v = COL[i][j]
                    if u != -1 and v != -1:
                        G[u].append(v)

        # Maximum bipartite matching via DFS (Kuhn's algorithm)
        MATCH = [-1] * col_cnt
        sys.setrecursionlimit(10000)

        def dfs(u: int, seen: List[bool]) -> bool:
            for v in G[u]:
                if not seen[v]:
                    seen[v] = True
                    if MATCH[v] == -1 or dfs(MATCH[v], seen):
                        MATCH[v] = u
                        return True
            return False

        result = 0
        for u in range(row_cnt):
            seen = [False] * col_cnt
            if dfs(u, seen):
                result += 1

        return result

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Must be an integer
        try:
            user_answer = int(parsed)
        except ValueError:
            # Not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Reference answer must be available
        assert self.reference_answer is not None, "Environment not properly reset or reference answer missing."

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random integer boxed."""
        # Bound the random answer by a plausible range: at most number of '*' cells.
        if self.grid is not None:
            max_possible = sum(row.count('*') for row in self.grid)
        else:
            # Fallback if not reset yet
            max_possible = self.max_n_m * self.max_n_m
        random_answer = random.randint(0, max_possible)
        return f"\\boxed{{{random_answer}}}"