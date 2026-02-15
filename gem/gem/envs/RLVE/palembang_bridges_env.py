import heapq
import random
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PalembangBridgesEnv(Env):
    """Palembang Bridges optimization environment - single-turn Q&A.

    Task:
      - Given two arrays S and T of length N, choose K integers P[j] (1 <= j <= K) to minimize:
        sum over i=1..N of min_j (|P[j] - S[i]| + |P[j] - T[i]|).
      - Output the K integers separated by spaces inside \\boxed{...}.

    This environment generates a random instance and validates the submitted P values
    by comparing the achieved total cost with the optimal (reference) cost.
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 50,
        allow_k_min: int = 1,
        allow_k_max: int = 2,
        **kwargs
    ):
        """Initialize the environment with parameter ranges.

        Parameters:
            min_n: Minimum N (must be >= 3).
            max_n: Maximum N (must be >= min_n).
            allow_k_min: Minimum K value (default 1).
            allow_k_max: Maximum K value (default 2).
        """
        super().__init__()
        assert min_n >= 3, "min_n should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"
        assert allow_k_min >= 1, "allow_k_min should be at least 1"
        assert allow_k_max >= allow_k_min, "allow_k_max should be >= allow_k_min"

        self.min_n = min_n
        self.max_n = max_n
        self.allow_k_min = allow_k_min
        self.allow_k_max = allow_k_max

        # Current instance storage
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.S: Optional[List[int]] = None
        self.T: Optional[List[int]] = None
        self.reference_cost: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving the Palembang Bridges optimization problem.\n"
            "Given arrays S and T (each of length N), choose K integers P[j] to minimize\n"
            "the total cost: sum over i of min_j (|P[j] - S[i]| + |P[j] - T[i]|).\n"
            "Output Format: Provide exactly K integers separated by single spaces, enclosed in \\boxed{...}.\n"
            "Example: \\boxed{3 7}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and generate a new problem instance.

        Returns:
            observation: The problem description as a string.
            info: Additional metadata dictionary.
        """
        super().reset(seed)

        # Generate parameters
        self.N = random.randint(self.min_n, self.max_n)
        self.K = random.randint(self.allow_k_min, self.allow_k_max)
        # As in the original environment, S[i], T[i] are integers in [0, N]
        self.S = [random.randint(0, self.N) for _ in range(self.N)]
        self.T = [random.randint(0, self.N) for _ in range(self.N)]

        # Compute the reference optimal cost
        cross_pairs = list(zip(self.S, self.T))
        self.reference_cost = self._compute_optimal_cost(self.K, cross_pairs)

        # Build problem statement
        pairs_str = "; ".join(
            f"S[{i}]={s}, T[{i}]={t}"
            for i, (s, t) in enumerate(cross_pairs, start=1)
        )

        self.current_problem = (
            f"You are given two arrays S and T, each of length {self.N}, provided as: {pairs_str}\n\n"
            f"Your task is to choose {self.K} integers P[j] (1 <= j <= {self.K}) such that the following total cost is minimized: "
            f"for each i from 1 to {self.N}, compute min(|P[j] - S[i]| + |P[j] - T[i]|) over all 1 ≤ j ≤ {self.K}, and take the sum over all i.\n"
            f"Output exactly {self.K} integers P[j] in a single line, separated by spaces, enclosed in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem

        info: Dict[str, Any] = {
            "N": self.N,
            "K": self.K,
            "S": self.S,
            "T": self.T,
            "reference_cost": self.reference_cost,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step by verifying the submitted answer.

        Parameters:
            action: The model's response text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE as this is a single-turn environment.
            reward: 1.0 if optimal cost achieved, 0.0 otherwise; -0.1 for format error.
            terminated: Always True (single turn).
            truncated: Always False.
            info: Dictionary with evaluation details.
        """
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers separated by spaces
        tokens = boxed.strip().split()
        try:
            user_values = list(map(int, tokens))
        except ValueError:
            # Content inside \\boxed{...} is not all integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate number of values equals K
        if self.K is None or len(user_values) != self.K:
            return TERMINAL_STATE, 0.0, True, False, {"error": "wrong_number_of_values", "expected_K": self.K, "got": len(user_values)}

        # Compute user cost
        assert self.S is not None and self.T is not None
        user_cost = sum(
            min(abs(p - s) + abs(p - t) for p in user_values)
            for s, t in zip(self.S, self.T)
        )

        # Compare with reference cost
        assert self.reference_cost is not None
        is_correct = (user_cost == self.reference_cost)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_cost": self.reference_cost,
            "user_cost": user_cost,
            "N": self.N,
            "K": self.K,
            "S": self.S,
            "T": self.T,
            "user_values": user_values,
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
        """Sample a random action: random P values inside \\boxed{...}."""
        if self.N is None or self.K is None:
            # Fallback random values if reset was not called
            k = random.randint(self.allow_k_min, self.allow_k_max)
            n = random.randint(self.min_n, self.max_n)
            vals = [random.randint(0, n) for _ in range(k)]
        else:
            vals = [random.randint(0, self.N) for _ in range(self.K)]
        return "\\boxed{" + " ".join(map(str, vals)) + "}"

    @staticmethod
    def _compute_optimal_cost(K: int, cross_pairs: List[Tuple[int, int]]) -> int:
        """Compute the optimal minimal cost for the given pairs and K bridges.

        This preserves the original algorithm:
          - For K == 1: place at the median of all endpoints (S and T) to minimize sum of absolute deviations.
          - For K == 2: sort by (S[i] + T[i]) and split into two contiguous groups; sum minimal costs of each group.
        """

        class MedianCostSolver:
            """Two-heap median maintenance with sum tracking to compute sum of absolute deviations."""

            def __init__(self) -> None:
                # Max-heap for lower half (store negatives), min-heap for upper half
                self.left: List[int] = []
                self.right: List[int] = []
                self.left_sum: int = 0
                self.right_sum: int = 0

            def insert(self, a: int) -> None:
                if not self.left:
                    heapq.heappush(self.left, -a)
                    self.left_sum += a
                else:
                    median = -self.left[0]
                    if a <= median:
                        heapq.heappush(self.left, -a)
                        self.left_sum += a
                    else:
                        heapq.heappush(self.right, a)
                        self.right_sum += a

                # Rebalance so that left has (total+1)//2 elements
                total = len(self.left) + len(self.right)
                target = (total + 1) // 2

                while len(self.left) > target:
                    v = -heapq.heappop(self.left)
                    self.left_sum -= v
                    heapq.heappush(self.right, v)
                    self.right_sum += v

                while len(self.left) < target:
                    v = heapq.heappop(self.right)
                    self.right_sum -= v
                    heapq.heappush(self.left, -v)
                    self.left_sum += v

            def query(self) -> int:
                """Return the sum of absolute deviations from the median."""
                if not self.left:
                    return 0
                total = len(self.left) + len(self.right)
                cnt = (total + 1) // 2
                median = -self.left[0]
                # cost = sum_{i in left} (median - x_i) + sum_{j in right} (x_j - median)
                return cnt * median - self.left_sum + self.right_sum - (total - cnt) * median

        m = len(cross_pairs)

        if K == 1:
            solver = MedianCostSolver()
            for a, b in cross_pairs:
                solver.insert(a)
                solver.insert(b)
            return solver.query()

        # K == 2 case
        cross_pairs_sorted = sorted(cross_pairs, key=lambda x: x[0] + x[1])

        # pre[i]: best cost for first i pairs with one bridge
        pre: List[int] = [0] * (m + 1)
        solver1 = MedianCostSolver()
        for i in range(m):
            a, b = cross_pairs_sorted[i]
            solver1.insert(a)
            solver1.insert(b)
            pre[i + 1] = solver1.query()

        # suf[i]: best cost for pairs i..m-1 with one bridge
        suf: List[int] = [0] * (m + 2)
        solver2 = MedianCostSolver()
        for i in range(m - 1, -1, -1):
            a, b = cross_pairs_sorted[i]
            solver2.insert(a)
            solver2.insert(b)
            suf[i + 1] = solver2.query()

        # Try all splits
        best = pre[0] + suf[1]
        for i in range(m + 1):
            cost = pre[i] + suf[i + 1]
            if cost < best:
                best = cost

        return best