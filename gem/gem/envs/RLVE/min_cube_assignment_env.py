from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from collections import deque
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinCubeAssignmentEnv(Env):
    """Minimum Cube Assignment environment - single turn Q&A.

    Task:
        Given a P × Q grid, assign each cell (i, j) an integer value f(i, j) in [0, R).
        Each assignment contributes a cost c(i, j, f(i, j)), and adjacent cells must satisfy
        |f(i, j) - f(i', j')| ≤ D for any adjacent pair.
        Compute the minimal possible total cost under these constraints.

    Output:
        The final minimal total cost should be provided in \\boxed{...} format.
    """

    def __init__(
        self,
        max_p_q_r: int = 10,
        fixed_P: Optional[int] = None,
        fixed_Q: Optional[int] = None,
        fixed_R: Optional[int] = None,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
            max_p_q_r: Upper bound for randomly sampling P, Q, R if fixed values are not provided.
            fixed_P: Optional fixed number of rows P.
            fixed_Q: Optional fixed number of columns Q.
            fixed_R: Optional fixed number of levels R.
        """
        super().__init__()
        assert max_p_q_r >= 2, "max_p_q_r should be greater than or equal to 2"
        self.max_p_q_r = max_p_q_r
        self.fixed_P = fixed_P
        self.fixed_Q = fixed_Q
        self.fixed_R = fixed_R

        self.P: Optional[int] = None
        self.Q: Optional[int] = None
        self.R: Optional[int] = None
        self.D: Optional[int] = None
        self.costs: Optional[List[List[List[int]]]] = None

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a minimum assignment problem on a grid with smoothness constraints.\n"
            "Your task is to compute the minimal total cost under the given constraints.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters
        P = self.fixed_P if self.fixed_P is not None else random.randint(2, self.max_p_q_r)
        Q = self.fixed_Q if self.fixed_Q is not None else random.randint(2, self.max_p_q_r)
        R = self.fixed_R if self.fixed_R is not None else random.randint(2, self.max_p_q_r)
        assert P >= 2 and Q >= 2 and R >= 2, "P, Q, R should be each >= 2"

        # Costs: costs[i][j][f] where 0 <= f < R
        costs = [[[random.randint(1, P * Q) for _ in range(R)] for _ in range(Q)] for _ in range(P)]
        D = random.randint(0, R - 1)

        # Store parameters
        self.P, self.Q, self.R, self.D, self.costs = P, Q, R, D, costs

        # Compute minimal total cost using max-flow/min-cut construction (Dinic)
        self.reference_answer = self._compute_min_cost(P, Q, R, D, costs)

        # Build problem statement
        costs_desc = "\n".join(
            " ".join(f"c({i},{j},{f})={c}" for f, c in enumerate(costs[i][j]))
            for i in range(P) for j in range(Q)
        )
        problem = (
            f"You are given a {P} × {Q} grid. You need to assign each cell (i, j) an integer value f(i, j) in the range [0, {R}). "
            "Each cell (i, j) contributes a cost of c(i, j, f(i, j)) to the total cost, where the cost function c is defined as:\n"
            f"{costs_desc}\n\n"
            f"In addition, for every pair of adjacent cells (i, j) and (i', j') (i.e., cells such that |i - i'| + |j - j'| = 1), "
            f"the assigned values must satisfy |f(i, j) - f(i', j')| ≤ {D}. "
            "Please compute the minimal total cost.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "P": P,
            "Q": Q,
            "R": R,
            "D": D,
        }
        return obs, info

    def _compute_min_cost(self, P: int, Q: int, R: int, D: int, val: List[List[List[int]]]) -> int:
        """Compute minimal total cost using a standard min-cut construction and Dinic's algorithm."""
        # Compute INF based on input total cost
        total = 0
        for k in range(R):
            for i in range(P):
                for j in range(Q):
                    total += val[i][j][k]
        INF = total + 1

        # Node indexing: S=0, for (i,j,k): id = 1 + k*(P*Q) + i*Q + j, T = 1 + (R+1)*P*Q
        node_count = 1 + (R + 1) * P * Q + 1
        S = 0
        T = node_count - 1

        class Edge:
            """Edge structure for Dinic's algorithm."""
            __slots__ = ("to", "cap", "rev")

            def __init__(self, to: int, cap: int, rev: int) -> None:
                self.to = to
                self.cap = cap
                self.rev = rev

        adj: List[List[Edge]] = [[] for _ in range(node_count)]

        def add_edge(u: int, v: int, c: int) -> None:
            adj[u].append(Edge(v, c, len(adj[v])))
            adj[v].append(Edge(u, 0, len(adj[u]) - 1))

        def node_id(i: int, j: int, k: int) -> int:
            return 1 + k * (P * Q) + i * Q + j

        # Source to layer 0 and vertical edges through layers
        for i in range(P):
            for j in range(Q):
                add_edge(S, node_id(i, j, 0), INF)
                for k in range(R):
                    add_edge(node_id(i, j, k), node_id(i, j, k + 1), val[i][j][k])
                add_edge(node_id(i, j, R), T, INF)

        # Smoothness constraints: infinite edges for height differences greater than D
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(P):
            for j in range(Q):
                for dx, dy in dirs:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < P and 0 <= nj < Q:
                        for k in range(D + 1, R + 2):
                            u = node_id(i, j, k - 1)
                            v = node_id(ni, nj, k - D - 1)
                            add_edge(u, v, INF)

        level = [0] * node_count
        it = [0] * node_count

        def bfs() -> bool:
            for idx in range(node_count):
                level[idx] = -1
            queue = deque([S])
            level[S] = 0
            while queue:
                u = queue.popleft()
                for e in adj[u]:
                    if e.cap > 0 and level[e.to] < 0:
                        level[e.to] = level[u] + 1
                        if e.to == T:
                            return True
                        queue.append(e.to)
            return level[T] >= 0

        def dfs(u: int, flow: int) -> int:
            if u == T:
                return flow
            for idx in range(it[u], len(adj[u])):
                e = adj[u][idx]
                if e.cap > 0 and level[u] < level[e.to]:
                    d = dfs(e.to, min(flow, e.cap))
                    if d > 0:
                        e.cap -= d
                        adj[e.to][e.rev].cap += d
                        return d
                it[u] += 1
            return 0

        flow = 0
        while bfs():
            it = [0] * node_count
            while True:
                pushed = dfs(S, INF)
                if pushed == 0:
                    break
                flow += pushed

        assert flow > 0, "Flow should be greater than 0, indicating a valid assignment exists"
        return flow

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the boxed minimal cost answer."""
        # Parse answer in boxed format
        raw = self._parse_answer(action)
        if raw is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(raw)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Reference answer must be computed in reset()."
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "P": self.P,
            "Q": self.Q,
            "R": self.R,
            "D": self.D,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...} from the input text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random boxed integer answer."""
        # Use a heuristic range based on P, Q, R to generate a random guess
        # If not initialized, default to a moderate range
        if self.P is not None and self.Q is not None and self.R is not None:
            guess_upper = max(10, self.P * self.Q * max(1, self.R))
        else:
            guess_upper = 100
        random_answer = random.randint(0, guess_upper)
        return f"\\boxed{{{random_answer}}}"