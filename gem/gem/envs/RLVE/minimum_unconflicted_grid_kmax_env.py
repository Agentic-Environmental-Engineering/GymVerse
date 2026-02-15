from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumUnconflictedGridKMaxEnv(Env):
    """
    Minimum Unconflicted Grid K-th Maximum Problem Environment - Single-turn Q&A

    Task:
      - You are given an N × M grid A of non-negative integers (1-indexed).
      - Choose N distinct column indices p[1], p[2], ..., p[N] in [1, M], one for each row.
      - For each row i, take the value A[i][p[i]]; among these N values, consider the K-th largest value.
      - The goal is to minimize this K-th largest value.
      - Output your column choices as 'p1 p2 ... pN' inside \\boxed{...}.

    Reward:
      - Correct (achieves minimal possible K-th largest value): 1.0
      - Wrong: 0.0
      - Format error: -0.1
    """

    def __init__(self, max_n_m: int = 20, **kwargs) -> None:
        super().__init__()
        assert max_n_m >= 3, "max_n_m must be >= 3"
        self.max_n_m: int = max_n_m

        # Problem parameters
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None
        self.A: Optional[List[List[int]]] = None

        # Cached problem statement and reference answer
        self.current_problem: Optional[str] = None
        self.gold_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a grid selection optimization problem.\n"
            "Choose N distinct column indices (1-indexed), one per row, to minimize the K-th largest selected value.\n"
            "Please provide your selected indices in \\boxed{p1 p2 ... pN} format, separated by single spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(3, self.max_n_m)
        M = random.randint(N, self.max_n_m)
        K = random.randint(1, N)
        A = [[random.randint(1, N * M) for _ in range(M)] for _ in range(N)]

        self.N, self.M, self.K, self.A = N, M, K, A

        # Compute minimal achievable K-th largest value (reference answer)
        self.gold_answer = self._compute_gold(N, M, K, A)

        # Build problem description
        grid_str_lines = []
        for i, row in enumerate(A, start=1):
            row_repr = ", ".join(f"A[{i}][{j}]={val}" for j, val in enumerate(row, start=1))
            grid_str_lines.append(row_repr)
        grid_str = "\n".join(grid_str_lines)

        self.current_problem = (
            f"You are given an {N} × {M} grid of non-negative integers A[i][j] (1-indexed). "
            f"The matrix A is:\n{grid_str}\n\n"
            f"Choose {N} distinct column indices p[1], p[2], ..., p[{N}] in the range [1, {M}]. "
            f"For each row i, take the value A[i][p[i]]; among these {N} values, "
            f"consider the {K}-th largest value; your goal is to minimize this {K}-th largest value.\n\n"
            f"Output Format: Provide your selected indices as a single line in \\boxed{{p1 p2 ... p{N}}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the proposed selection."""
        # Parse boxed content
        content = self._parse_answer(action)
        if content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure problem is initialized
        if self.N is None or self.M is None or self.K is None or self.A is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "uninitialized_problem"}

        # Parse indices
        tokens = content.replace(",", " ").split()
        try:
            user_cols = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer_parse"}

        info: dict[str, Any] = {
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "grid": self.A,
            "reference_answer": self.gold_answer,
            "user_columns": user_cols,
        }

        # Validate selection
        if len(user_cols) != self.N:
            info["error"] = "invalid_solution_length"
            return TERMINAL_STATE, 0.0, True, False, info
        if len(set(user_cols)) != self.N:
            info["error"] = "non_distinct_columns"
            return TERMINAL_STATE, 0.0, True, False, info
        if not all(1 <= x <= self.M for x in user_cols):
            info["error"] = "columns_out_of_range"
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute user's K-th largest value
        selected_values = [self.A[i][user_cols[i] - 1] for i in range(self.N)]
        kth_largest = sorted(selected_values, reverse=True)[self.K - 1]
        info["user_kth_largest"] = kth_largest

        # Correct if user's K-th largest equals minimal possible
        is_correct = (kth_largest == self.gold_answer)
        info["correct"] = is_correct

        reward: float = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_gold(self, N: int, M: int, K: int, A: List[List[int]]) -> int:
        """
        Compute the minimal achievable K-th largest value using a binary search + bipartite matching.
        Transform K to K' = N - K + 1 (as in the original logic) and find the smallest lim such that
        a matching of size >= K' exists using only entries <= lim.
        """
        # Transform K as in the original code
        K_required = N - K + 1

        # Determine search bounds
        LIM = max(max(row) for row in A)
        l, r = 1, LIM

        def check(x: int) -> int:
            # Maximum matching size when only entries <= x are allowed
            vis = [0] * (M + 1)  # visitation stamps for columns
            lin = [0] * (M + 1)  # matched row for each column
            tot = 1
            ans = 0

            def dfs(u: int, lim: int) -> bool:
                for j in range(1, M + 1):
                    if A[u - 1][j - 1] <= lim and vis[j] != tot:
                        vis[j] = tot
                        if lin[j] == 0 or dfs(lin[j], lim):
                            lin[j] = u
                            return True
                return False

            for i in range(1, N + 1):
                if dfs(i, x):
                    ans += 1
                tot += 1
            return ans

        while l < r:
            mid = (l + r) // 2
            if check(mid) >= K_required:
                r = mid
            else:
                l = mid + 1

        return l

    def sample_random_action(self) -> str:
        """Sample a random valid-looking action in \\boxed{...} format."""
        if self.N is None or self.M is None:
            # No problem initialized, return a safe placeholder
            return "\\boxed{1}"
        cols = random.sample(range(1, self.M + 1), self.N)
        action_str = " ".join(map(int.__str__, cols))
        return f"\\boxed{{{action_str}}}"