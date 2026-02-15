import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Axis_KCenterEnv(Env):
    """Axis K-Center problem environment - single-turn Q&A.

    The task is: given N points on a line (sorted by position), select K distinct indices
    such that the sum of distances from all points to their nearest selected point is minimized.
    The answer must be provided in \\boxed{...} containing space-separated indices.
    """

    def __init__(
        self,
        position_multiple: int = 5,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 100,
        **kwargs
    ):
        """
        Initialize the Axis K-Center environment.

        Parameters:
        - position_multiple: multiplier to determine the range of positions.
        - N: fixed number of points if provided; otherwise N will be randomly chosen.
        - min_n: minimum N when N is generated randomly.
        - max_n: maximum N when N is generated randomly.
        """
        super().__init__()
        self.position_multiple = position_multiple
        self.N_config = N
        self.min_n = min_n
        self.max_n = max_n

        # Runtime variables
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.X: Optional[List[int]] = None
        self.gold_cost: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an axis K-center problem on a line.\n"
            "Please provide your answer as K distinct indices inside \\boxed{...}, separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N_config is not None:
            assert self.N_config >= 3, "N should be greater than or equal to 3"
            N = self.N_config
        else:
            N = random.randint(self.min_n, max(self.min_n, self.max_n))
        assert N >= 3, "N should be greater than or equal to 3"

        # Determine K
        K = random.randint(1, N - 1)

        # Generate positions
        X = random.sample(range(N * self.position_multiple + 1), N)
        X.sort()

        # Compute optimal cost using dynamic programming with Knuth optimization
        INF = N * (X[-1] - X[0] + 1)

        # Precompute w[l][r]: cost of one center serving villages l..r (inclusive)
        w = [[0] * N for _ in range(N)]
        for l in range(N):
            for r in range(l + 1, N):
                m = (l + r) // 2
                w[l][r] = w[l][r - 1] + (X[r] - X[m])

        # dp[i][j]: minimum total distance covering the first i villages with j centers
        dp = [[INF] * (K + 1) for _ in range(N + 1)]
        # d[i][j]: the k giving the optimum for dp[i][j] (Knuth optimization)
        d = [[0] * (K + 1) for _ in range(N + 2)]

        dp[0][0] = 0

        for j in range(1, K + 1):
            d[N + 1][j] = N
            for i in range(N, 0, -1):
                best = INF
                argk = 0
                start = d[i][j - 1]
                end = d[i + 1][j]
                if end > i - 1:
                    end = i - 1
                for k in range(start, end + 1):
                    cost = dp[k][j - 1] + w[k][i - 1]
                    if cost < best:
                        best = cost
                        argk = k
                dp[i][j] = best
                d[i][j] = argk

        gold_answer = dp[N][K]

        # Save problem state
        self.N = N
        self.K = K
        self.X = X
        self.gold_cost = gold_answer

        # Build problem prompt
        positions_str = " ".join(map(str, X))
        self.current_problem = (
            f"You are given {N} points on a line, labeled from 0 to {N - 1}. "
            f"Their positions (from left to right) are: {positions_str}\n\n"
            f"Please select a set of {K} distinct points. Try your best to minimize the total distance "
            f"from all points to their nearest selected point (the distance is the absolute difference between positions).\n\n"
            f"Output Format: Your final answer should be a single line containing the indices of the selected {K} points "
            f"in any order, separated by spaces, and wrapped in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": N,
            "K": K,
            "positions": X,
            "reference_cost": gold_answer,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step: parse and verify the user's selected indices."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure environment was reset
        if self.N is None or self.K is None or self.X is None or self.gold_cost is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_ready"}

        # Parse indices
        try:
            tokens = boxed_content.strip().split()
            selected_indices = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate selection
        if len(selected_indices) != len(set(selected_indices)):
            info = {"error": "invalid_solution", "reason": "duplicate_indices"}
            return TERMINAL_STATE, 0.0, True, False, info
        if len(selected_indices) != self.K:
            info = {"error": "invalid_solution", "reason": "wrong_number_of_indices"}
            return TERMINAL_STATE, 0.0, True, False, info
        if not all(0 <= u < self.N for u in selected_indices):
            info = {"error": "invalid_solution", "reason": "index_out_of_bounds"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute user's cost
        X = self.X
        assert X is not None
        user_cost = sum(
            min(abs(X[u] - X[v]) for v in selected_indices)
            for u in range(self.N)
        )
        reference_cost = self.gold_cost
        assert reference_cost is not None

        is_optimal = (user_cost == reference_cost)
        reward = 1.0 if is_optimal else 0.0

        info = {
            "correct": is_optimal,
            "reference_cost": reference_cost,
            "user_cost": user_cost,
            "N": self.N,
            "K": self.K,
            "positions": self.X,
            "selected_indices": selected_indices,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action: random K distinct indices formatted in \\boxed{...}."""
        if self.N is None or self.K is None:
            # Default random sample if environment is not ready
            n = self.N_config if self.N_config is not None else max(self.min_n, 3)
            k = min(1, max(1, n - 1))
            indices = sorted(random.sample(range(n), k))
        else:
            indices = sorted(random.sample(range(self.N), self.K))
        return "\\boxed{" + " ".join(map(str, indices)) + "}"