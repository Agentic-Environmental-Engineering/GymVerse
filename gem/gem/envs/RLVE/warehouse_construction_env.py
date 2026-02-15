import random
from collections import deque
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class WarehouseConstructionEnv(Env):
    """
    Warehouse Construction environment (single-turn Q&A).

    Problem:
    - There are N factories along a mountain, indexed 0..N-1 from top to bottom.
    - Distances D[0..N-1] are strictly increasing with D[0] = 0.
    - Each factory i has P[i] products and warehouse construction cost C[i].
    - You may build warehouses at any subset of factories. A factory without a warehouse
      must send all its products downhill (to a higher index) to the nearest factory
      that has a warehouse. Transporting one product over one unit of distance costs 1.
    - Total cost = sum of chosen C[i] + sum of transportation costs.
    Goal: minimize total cost.

    Task:
    - Output the indices where warehouses should be built (any order).
    - Answer format must be \\boxed{i1 i2 ... ik}.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        min_n: int = 2,
        max_n: int = 20,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            n: If provided, fixes the number of factories N for each reset.
            min_n: Minimum N to sample when n is None.
            max_n: Maximum N to sample when n is None.
        """
        super().__init__()
        assert min_n >= 2, "min_n should be >= 2"
        assert max_n >= min_n, "max_n should be >= min_n"
        if n is not None:
            assert n >= 2, "n should be >= 2"

        self.fixed_n = n
        self.min_n = min_n
        self.max_n = max_n

        # Problem state
        self.N: Optional[int] = None
        self.D: Optional[List[int]] = None
        self.P: Optional[List[int]] = None
        self.C: Optional[List[int]] = None

        # Reference optimal total cost
        self.reference_min_cost: Optional[int] = None

        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a warehouse construction optimization problem.\n"
            "Provide the indices of factories where warehouses should be built (any order).\n"
            "Answer format: Put your final indices separated by spaces inside \\boxed{...}.\n"
            "Example: \\boxed{0 3 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # Choose N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 2, "N should be greater than or equal to 2"

        # Generate strictly increasing distances with D[0] = 0
        D_sample = random.sample(range(1, 2 * N + 1), N - 1)
        D_sample.sort()
        D = [0] + D_sample
        assert len(D) == N, "D should have length N"
        assert all(di < di1 for di, di1 in zip(D, D[1:])), "D should be strictly increasing"

        # Generate products and costs
        P = [random.randint(0, N) for _ in range(N)]
        C = [random.randint(1, N * 2) for _ in range(N)]

        # Store problem parameters
        self.N = N
        self.D = D
        self.P = P
        self.C = C

        # Compute reference minimal total cost using DP with convex hull trick
        self.reference_min_cost = self._compute_optimal_cost(N, D, P, C)

        # Build problem prompt
        self.current_problem = self._build_problem_text(N, D, P, C)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_optimal_cost(self, N: int, D: List[int], P: List[int], C: List[int]) -> int:
        """Compute the minimal total cost using the original DP + convex hull trick."""
        Q = [0] * (N + 1)  # Prefix sums of P
        R = [0] * (N + 1)  # Prefix sums of D[i-1] * P[i-1]
        for i in range(1, N + 1):
            Q[i] = Q[i - 1] + P[i - 1]
            R[i] = R[i - 1] + D[i - 1] * P[i - 1]

        # f[i]: minimal cost up to considering factory i-1, with a warehouse at i-1
        f = [0] * (N + 1)

        def decx(idx: int) -> int:
            return Q[idx]

        def decy(idx: int) -> int:
            return f[idx] + R[idx]

        def maked(i: int, u: int) -> int:
            # f[u] + D[i-1]*(Q[i]-Q[u]) - (R[i]-R[u]) + C[i-1]
            return f[u] + D[i - 1] * (Q[i] - Q[u]) - (R[i] - R[u]) + C[i - 1]

        dq = deque([0])

        for i in range(1, N + 1):
            # Pop from the left while next is better at x = D[i-1]
            while len(dq) >= 2:
                u1, u2 = dq[0], dq[1]
                if decy(u2) - decy(u1) <= D[i - 1] * (decx(u2) - decx(u1)):
                    dq.popleft()
                else:
                    break

            u = dq[0]
            f[i] = maked(i, u)

            # Maintain lower hull
            while len(dq) >= 2:
                u1, u2 = dq[-1], dq[-2]
                if (decy(u1) - decy(u2)) * (decx(i) - decx(u1)) >= (decy(i) - decy(u1)) * (decx(u1) - decx(u2)):
                    dq.pop()
                else:
                    break

            dq.append(i)

        ans = f[N]
        x = N
        # If trailing factories have zero products, we can consider skipping the last warehouse
        while x > 0 and P[x - 1] == 0:
            x -= 1
            ans = min(ans, f[x])

        return ans

    def _build_problem_text(self, N: int, D: List[int], P: List[int], C: List[int]) -> str:
        """Construct the problem description string."""
        D_str = " ".join(f"D[{i}]={Di}" for i, Di in enumerate(D))
        P_str = " ".join(f"P[{i}]={Pi}" for i, Pi in enumerate(P))
        C_str = " ".join(f"C[{i}]={Ci}" for i, Ci in enumerate(C))

        return (
            f"You are given {N} factories arranged from top to bottom along a mountain, "
            f"indexed from 0 to {N - 1}. Factory 0 is at the top and factory {N - 1} is at the bottom.\n\n"
            f"Each factory has\n"
            f"- Distance from factory 0: {D_str}\n"
            f"- Number of products: {P_str}\n"
            f"- Cost to build a warehouse at that factory: {C_str}\n\n"
            f"You can choose to build warehouses at any subset of factories.\n"
            f"- A warehouse can store any number of products.\n"
            f"- If a factory does not build a warehouse, all its products must be sent downhill "
            f"(i.e., to a factory with a higher index) to a factory with a warehouse. "
            f"Transporting one product over one unit of distance costs 1.\n"
            f"- The total cost is the sum of warehouse construction costs and product transportation costs. "
            f"Try your best to minimize the total cost.\n\n"
            f"Output Format: Output a single line containing the indices of the factories where warehouses "
            f"should be built, separated by spaces (in any order), and place them in \\boxed{{...}}.\n"
            f"Example: \\boxed{{0 2 5}}\n"
        )

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's proposed warehouse indices."""
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices list
        try:
            indices = self._parse_indices_list(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate and compute user's total cost
        assert self.N is not None and self.D is not None and self.P is not None and self.C is not None
        user_cost, valid = self._compute_user_cost(indices, self.N, self.D, self.P, self.C)

        info: dict[str, Any] = {
            "indices": indices,
            "reference_min_cost": self.reference_min_cost,
            "user_cost": user_cost,
            "valid": valid,
        }

        if not valid:
            info["error"] = "invalid_solution"
            return TERMINAL_STATE, 0.0, True, False, info

        assert self.reference_min_cost is not None
        is_correct = (user_cost == self.reference_min_cost)
        info["correct"] = is_correct

        reward = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _compute_user_cost(
        self,
        indices: List[int],
        N: int,
        D: List[int],
        P: List[int],
        C: List[int],
    ) -> Tuple[int, bool]:
        """
        Compute the total cost of the user's proposed set of warehouse indices.
        Returns (cost, valid), where valid indicates feasibility.
        """
        # Check indices range
        for idx in indices:
            if idx < 0 or idx >= N:
                return 0, False

        built = [False] * N
        cost = 0
        for idx in indices:
            if not built[idx]:
                built[idx] = True
                cost += C[idx]

        nearest_warehouse: Optional[int] = None
        for i in range(N - 1, -1, -1):
            if built[i]:
                nearest_warehouse = i
            if P[i] > 0:
                if nearest_warehouse is None:
                    # No downhill warehouse to receive products from i
                    return 0, False
                cost += P[i] * (D[nearest_warehouse] - D[i])

        return cost, True

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns None if not found."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_indices_list(self, text: str) -> List[int]:
        """
        Parse a space- or comma-separated list of integers from text.
        Returns the list of indices. Raises ValueError if parsing fails.
        """
        cleaned = text.replace(",", " ").strip()
        if cleaned == "":
            return []
        parts = [p for p in cleaned.split() if p]
        indices = [int(p) for p in parts]
        return indices

    def sample_random_action(self) -> str:
        """Sample a random subset of factories as a candidate action in boxed format."""
        if self.N is None:
            # If called before reset, default to an empty action
            return "\\boxed{}"
        # Random subset
        indices = [i for i in range(self.N) if random.random() < 0.3]
        content = " ".join(map(str, indices))
        return f"\\boxed{{{content}}}"