import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Minimum_DominatingIntervalEnv(Env):
    """Minimum Dominating Interval environment - single-turn Q&A.
    
    Task: Select K distinct points such that each selected point is covered by at least one interval.
    The cost is the sum of costs of all intervals that cover at least one selected point.
    The goal is to minimize the total cost.
    """

    def __init__(
        self,
        N: int = 10,
        M: int = 5,
        K_density: float = 0.5,
        cost_range: int = 10,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 5.0,
        **kwargs,
    ):
        super().__init__()
        # Parameters controlling problem generation
        self.N = N
        self.M = M
        self.K_density = K_density
        self.cost_range = cost_range

        # Original reward configuration retained for compatibility, not used in GEM scoring
        self.rewards_cfg = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[int] = None

        # Problem data
        self.intervals: Optional[List[Tuple[int, int, int]]] = None
        self.N_actual: Optional[int] = None
        self.M_actual: Optional[int] = None
        self.K_actual: Optional[int] = None
        self.dominated_points: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a combinatorial optimization problem on intervals.\n"
            "You must select K distinct points covered by the given intervals to minimize total cost.\n"
            "Answer format: put the K selected points separated by spaces inside \\boxed{...}.\n"
            "Example: \\boxed{1 3 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Validate parameters
        assert self.N >= 3, "N should be greater than or equal to 3"
        assert self.M >= 2, "M should be greater than or equal to 2"
        assert 0.0 <= self.K_density <= 1.0, "K_density should be between 0.0 and 1.0"

        N = self.N
        M = self.M

        # Generate all possible intervals and sample M of them with random positive costs
        all_intervals = [(l, r, random.randint(1, self.cost_range)) for l in range(1, N + 1) for r in range(l, N + 1)]
        assert len(all_intervals) == (N * (N + 1) // 2)
        intervals = random.sample(all_intervals, min(len(all_intervals), M))

        # Compute K based on density and the number of dominated points
        def full_point_set_size(intervals_list: List[Tuple[int, int, int]]) -> int:
            dominated = set()
            for interval in intervals_list:
                Li, Ri = interval[0], interval[1]
                dominated.update(range(Li, Ri + 1))
            return len(dominated)

        dominated_size = full_point_set_size(intervals)
        K = max(1, int(self.K_density * dominated_size))

        # Prepare for DP computation
        L, R, C = zip(*intervals) if intervals else ([], [], [])
        Sum_Ci = [[0] * (N + 1) for _ in range(N + 1)]
        for i in range(M):
            Li, Ri, Ci = L[i], R[i], C[i]
            Sum_Ci[Li][Ri] = Sum_Ci[Li][Ri] + Ci
        for l in range(1, N + 1):
            for r in range(N - 1, 0, -1):
                Sum_Ci[l][r] += Sum_Ci[l][r + 1]

        dpF = [[None] * (N + 1) for _ in range(0, K + 1)]
        dpG = [[None] * (N + 1) for _ in range(0, K + 1)]
        for i in range(1, N + 1):
            if not any(Li <= i <= Ri for Li, Ri in zip(L, R)):
                continue
            dpF[1][i] = 0
            for l in range(1, i + 1):
                dpF[1][i] += Sum_Ci[l][i]
        for k in range(2, K + 1):
            for i in range(1, N + 1):
                if not any(Li <= i <= Ri for Li, Ri in zip(L, R)):
                    continue
                Sum = 0
                for j in range(i, 0, -1):
                    Sum += Sum_Ci[j][i]
                    if dpF[k - 1][j - 1] is not None:
                        val = dpF[k - 1][j - 1] + Sum
                        if dpF[k][i] is None or val < dpF[k][i]:
                            dpF[k][i] = val
                            dpG[k][i] = j - 1

        last = None
        for i in range(1, N + 1):
            if dpF[K][i] is None:
                continue
            if dpF[K][i] is not None and (last is None or dpF[K][i] < dpF[K][last]):
                last = i
        pickeds: List[int] = []
        for k in range(K, 0, -1):
            assert last is not None
            pickeds.append(last)
            last = dpG[k][last]
        assert last is None
        pickeds.reverse()

        # Store answers
        reference_answer = " ".join(map(str, pickeds))
        gold_answer = sum(C[i] for i in range(M) if any(L[i] <= picked <= R[i] for picked in pickeds))
        assert gold_answer > 0

        # Save state
        self.intervals = intervals
        self.N_actual = N
        self.M_actual = M
        self.K_actual = K
        self.reference_answer = reference_answer
        self.gold_answer = gold_answer

        dominated_set = set()
        for Li, Ri, _Ci in intervals:
            dominated_set.update(range(Li, Ri + 1))
        self.dominated_points = sorted(dominated_set)

        # Build the problem prompt
        interval_lines = "\n".join(
            f"L[{i}]={L[i]}, R[{i}]={R[i]}, C[{i}]={C[i]}"
            for i in range(M)
        )
        first_K_points = " ".join(map(str, range(1, K + 1)))
        self.current_problem = (
            f"There are {N} points labeled 1 through {N} on a line. You are given {M} intervals [L[i], R[i]] (1 <= L[i] <= R[i] <= {N}), each with a cost C[i]:\n"
            f"{interval_lines}\n\n"
            f"Please select {K} distinct points such that each selected point is covered by at least one of the intervals.\n"
            f"The cost of a selection is the sum of the costs (C[i]) of all intervals that cover at least one of the selected points.\n"
            f"Try your best to minimize the total cost of the selection.\n\n"
            f"Output Format: Your final answer should be a single line containing the {K} selected points, separated by spaces, wrapped in \\boxed{{...}}.\n"
            f"Example: \\boxed{{{first_K_points}}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and return the result."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure problem is available
        assert self.intervals is not None and self.K_actual is not None and self.gold_answer is not None

        # Parse points from boxed content
        parts = boxed_content.strip().split()
        try:
            user_points = list(map(int, parts))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate selection
        if len(user_points) != self.K_actual:
            info = {
                "error": "invalid_solution",
                "reason": "wrong_number_of_points",
                "expected_K": self.K_actual,
                "got_K": len(user_points),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if len(set(user_points)) != self.K_actual:
            info = {
                "error": "invalid_solution",
                "reason": "points_not_distinct",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        L, R, C = zip(*self.intervals)
        if not all(any(Li <= picked <= Ri for Li, Ri in zip(L, R)) for picked in user_points):
            info = {
                "error": "invalid_solution",
                "reason": "points_not_covered",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute user's cost
        user_cost = sum(C[i] for i in range(self.M_actual) if any(L[i] <= picked <= R[i] for picked in user_points))
        is_correct = (user_cost == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "gold_cost": self.gold_answer,
            "user_cost": user_cost,
            "user_points": user_points,
            "reference_answer": self.reference_answer,
            "intervals": self.intervals,
            "N": self.N_actual,
            "M": self.M_actual,
            "K": self.K_actual,
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
        """Sample a random valid action in \\boxed{...} format."""
        assert self.dominated_points is not None and self.K_actual is not None
        # Ensure we can sample K distinct dominated points
        pool = self.dominated_points if len(self.dominated_points) >= self.K_actual else list(range(1, self.N_actual + 1))
        sampled = random.sample(pool, self.K_actual)
        sampled_str = " ".join(map(str, sampled))
        return f"\\boxed{{{sampled_str}}}"