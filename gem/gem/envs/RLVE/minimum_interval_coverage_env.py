import random
import networkx as nx
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumIntervalCoverageEnv(Env):
    """Minimum Interval Coverage problem environment - single-turn Q&A.
    
    Task:
    - You are given M intervals within [1, N]. Each interval [L[i], R[i]] has cost C[i].
    - You can select each interval any number of times (including 0).
    - For each point i in [1, N], it must be covered by at least NEED[i] selected intervals.
    - The goal is to minimize the total cost of selected intervals.
    
    Output:
    - Provide M non-negative integers indicating the selection counts of each interval, in order.
    - The answer must be in \\boxed{t1 t2 ... tM} format.
    
    Scoring:
    - Reward = 1.0 if the solution is feasible and achieves the minimum total cost.
    - Reward = 0.0 otherwise.
    - Reward = -0.1 for format errors.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        M: Optional[int] = None,
        min_N: int = 1,
        max_N: int = 20,
        min_M: int = 1,
        max_M: int = 20,
        cost_multiple: int = 3,
        **kwargs
    ):
        super().__init__()
        # Problem size configuration
        self.N_fixed = N
        self.M_fixed = M
        self.min_N = min_N
        self.max_N = max_N
        self.min_M = min_M
        self.max_M = max_M

        # Cost configuration
        self.cost_multiple = cost_multiple

        # Runtime state
        self.N: int = 0
        self.M: int = 0
        self.intervals: List[Tuple[int, int, int]] = []
        self.needs: List[int] = []
        self.gold_cost: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a Minimum Interval Coverage optimization problem.\n"
            "Provide your answer as M non-negative integers inside \\boxed{...}, separated by spaces.\n"
            "Example: \\boxed{0 1 2 0 3}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N and M
        if self.N_fixed is not None:
            assert self.N_fixed >= 1, "N must be at least 1"
            self.N = self.N_fixed
        else:
            assert self.min_N >= 1 and self.max_N >= self.min_N, "Invalid N range"
            self.N = random.randint(self.min_N, self.max_N)

        if self.M_fixed is not None:
            assert self.M_fixed >= 1, "M must be at least 1"
            self.M = self.M_fixed
        else:
            assert self.min_M >= 1 and self.max_M >= self.min_M, "Invalid M range"
            self.M = random.randint(self.min_M, self.max_M)

        # Generate intervals
        self.intervals = []
        for _ in range(self.M):
            L = random.randint(1, self.N)
            R = random.randint(1, self.N)
            if L > R:
                L, R = R, L
            C = random.randint(1, self.cost_multiple * (R - L + 1))
            self.intervals.append((L, R, C))

        # Generate needs
        self.needs = []
        for i in range(1, self.N + 1):
            covered = any(L <= i <= R for L, R, _ in self.intervals)
            self.needs.append(random.randint(0, self.N) if covered else 0)

        # Compute gold minimum total cost using min-cost flow transformation
        self.gold_cost = self._compute_min_cost(self.N, self.intervals, self.needs)

        # Build problem prompt
        intervals_str = "\n".join(
            f"L[{i+1}]={L} R[{i+1}]={R} C[{i+1}]={C}" for i, (L, R, C) in enumerate(self.intervals)
        )
        needs_str = " ".join(f"NEED[{i+1}]={need}" for i, need in enumerate(self.needs))

        self.current_problem = (
            f"You are given {self.M} intervals within [1, {self.N}]. Each interval is defined as [L[i], R[i]] "
            f"with an associated cost C[i]. The intervals are:\n{intervals_str}\n\n"
            f"You can select each interval any number of times (including 0). For each point i in [1, {self.N}], "
            f"you must ensure it is covered by at least NEED[i] selected intervals, where the array NEED is:\n{needs_str}\n\n"
            f"Your goal is to minimize the total cost of the selected intervals while satisfying the above condition.\n\n"
            f"Output Format: Provide {self.M} integers — the number of times you select each interval, in order, "
            f"separated by spaces, inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": self.N,
            "M": self.M,
            "intervals": self.intervals,
            "needs": self.needs,
        }

    def _compute_min_cost(self, N: int, intervals: List[Tuple[int, int, int]], needs: List[int]) -> int:
        """Compute the minimum total cost using a min-cost flow formulation."""
        # Pad NEED with zeros at both ends for difference calculation
        need_padded = [0] + needs + [0]  # length N+2

        # Build node demands: DEMANDS[k] = NEED[k+1] - NEED[k] for k in 0..N
        demands = [need_padded[i] - need_padded[i - 1] for i in range(1, N + 2)]

        # Build the directed multigraph
        G = nx.MultiDiGraph()
        INF = sum(needs)

        # Add nodes with demand attribute
        for node, d in enumerate(demands):
            G.add_node(node, demand=d)

        # Add chain edges i -> i+1 with large capacity and zero cost
        for i in range(N):
            G.add_edge(i, i + 1, capacity=INF, weight=0)

        # Add an edge for each interval (s, t, c) as t -> (s-1) with cost c
        for s, t, c in intervals:
            u = t      # node t
            v = s - 1  # node s-1
            G.add_edge(u, v, capacity=INF, weight=c)

        # Compute the minimum-cost flow
        cost, _ = nx.network_simplex(G)
        return cost

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted solution and return the result."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers
        try:
            times = list(map(int, boxed_content.split()))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate length
        if len(times) != self.M:
            info = {"error": "wrong_length", "expected_M": self.M, "received_M": len(times)}
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate non-negativity
        if any(t < 0 for t in times):
            return TERMINAL_STATE, 0.0, True, False, {"error": "negative_times"}

        # Validate coverage constraints
        for i in range(1, self.N + 1):
            coverage = sum((1 if L <= i <= R else 0) * times[j] for j, (L, R, _) in enumerate(self.intervals))
            if coverage < self.needs[i - 1]:
                return TERMINAL_STATE, 0.0, True, False, {
                    "error": "constraint_unsatisfied",
                    "position": i,
                    "required": self.needs[i - 1],
                    "coverage": coverage
                }

        # Compute user's total cost
        user_cost = sum(times[j] * C for j, (_, _, C) in enumerate(self.intervals))
        gold_cost = self.gold_cost if self.gold_cost is not None else 0

        # Validate optimality
        is_correct = (user_cost == gold_cost)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "gold_cost": gold_cost,
            "user_cost": user_cost,
            "N": self.N,
            "M": self.M,
            "intervals": self.intervals,
            "needs": self.needs,
            "times": times
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the correct format."""
        # Random small non-negative integers for each interval
        times = [random.randint(0, 2) for _ in range(self.M if self.M else 1)]
        return f"\\boxed{{{' '.join(map(str, times))}}}"