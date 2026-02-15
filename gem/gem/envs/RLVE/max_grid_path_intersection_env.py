import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from collections import deque
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxGridPathIntersectionEnv(Env):
    """Environment for maximizing total sum over K right/down paths on an N x N grid."""

    def __init__(
        self,
        N: int = 3,
        **kwargs
    ):
        """
        Initialize the MaxGridPathIntersectionEnv.

        Parameters:
        - N: Size of the grid (must be >= 3). The grid is N x N.
        """
        super().__init__()
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        self.N = N

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.grid: Optional[List[List[int]]] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a grid path maximization problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        N = self.N
        K = random.randint(1, N // 2)
        self.K = K
        self.grid = [[random.randint(0, N) for _ in range(N)] for _ in range(N)]

        grid_str = "\n".join(" ".join(map(str, row)) for row in self.grid)

        self.current_problem = (
            f"You are given an {N} × {N} grid (0-indexed) of non-negative integers (given in row-major order):\n"
            f"{grid_str}\n\n"
            f"You will start at cell (0, 0) and move to cell ({N - 1}, {N - 1}) exactly {K} times. "
            f"Each time, you can only move right or down at each step. When you step on a cell during a path, "
            f"you collect its value and set it to 0 (so future paths will see it as 0). Your goal is to maximize "
            f"the total sum collected across all {K} paths.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        self.reference_answer = self._max_cost_flow(N, K, self.grid)

        obs = self._get_instructions() + self.current_problem
        return obs, {"N": N, "K": K}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step by validating the provided answer."""
        answer_str = self._parse_answer(action)

        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (answer) in boxed format."""
        # Heuristic random guess based on grid size
        random_answer = random.randint(0, self.N * self.N * self.N)
        return f"\\boxed{{{random_answer}}}"

    def _max_cost_flow(self, N: int, K: int, A: List[List[int]]) -> int:
        """
        Compute the maximum total sum collected across K paths using a maximum-cost flow approach.

        Each cell has an in-node and an out-node. Moving right or down is represented as edges
        between out-nodes of a cell to in-nodes of adjacent cells. Picking up a cell value is modeled
        by one unit of capacity with the cell's value as cost, and the remaining K-1 passes with zero cost.
        """
        # Number of nodes: each cell has in-node and out-node
        total_nodes = 2 * N * N
        # Adjacency list: each entry is [to, capacity, cost, rev]
        ADJ: List[List[List[Any]]] = [[] for _ in range(total_nodes)]

        def add_edge(u: int, v: int, cap: int, cost: int) -> None:
            # forward edge
            forward = [v, cap, cost, None]  # [to, capacity, cost, reverse_edge]
            # reverse edge
            backward = [u, 0, -cost, None]
            # link edges for capacity updates
            forward[3] = backward
            backward[3] = forward
            ADJ[u].append(forward)
            ADJ[v].append(backward)

        def node_id(i: int, j: int, is_out: bool) -> int:
            # 0-indexed: cells at (i, j) share indices 0..N*N-1 for in-nodes,
            # N*N..2*N*N-1 for out-nodes
            base = N * N if is_out else 0
            return base + i * N + j

        # Build the flow network
        for i in range(N):
            for j in range(N):
                in_id = node_id(i, j, False)
                out_id = node_id(i, j, True)
                # Pick the cell's value on one of the K visits
                add_edge(in_id, out_id, 1, A[i][j])    # one with reward
                add_edge(in_id, out_id, K - 1, 0)      # others free
                # Move right or down (up to K walkers)
                if j + 1 < N:
                    add_edge(out_id, node_id(i, j + 1, False), K, 0)
                if i + 1 < N:
                    add_edge(out_id, node_id(i + 1, j, False), K, 0)

        s = node_id(0, 0, False)
        t = node_id(N - 1, N - 1, True)
        total_cost = 0

        # If K is zero, there is no flow and cost is zero
        if K == 0:
            return 0

        # Successive SPFA for maximum-cost flow
        while True:
            DIST = [float('-inf')] * total_nodes
            FLOW = [0] * total_nodes
            INQUEUE = [False] * total_nodes
            PREV_NODE: List[Optional[int]] = [None] * total_nodes
            PREV_EDGE: List[Optional[List[Any]]] = [None] * total_nodes

            queue = deque([s])
            DIST[s] = 0
            FLOW[s] = K   # maximum possible augment per iteration
            INQUEUE[s] = True

            # Find longest path from s to t in residual graph
            while queue:
                u = queue.popleft()
                INQUEUE[u] = False
                for edge in ADJ[u]:
                    v, cap, cost, rev = edge
                    if cap > 0 and DIST[v] < DIST[u] + cost:
                        DIST[v] = DIST[u] + cost
                        FLOW[v] = min(FLOW[u], cap)
                        PREV_NODE[v] = u
                        PREV_EDGE[v] = edge
                        if not INQUEUE[v]:
                            queue.append(v)
                            INQUEUE[v] = True

            # If there's no augmenting path, we are done
            if DIST[t] == float('-inf'):
                break

            # Augment along the path
            f = FLOW[t]
            total_cost += f * DIST[t]
            v = t
            while v != s:
                edge = PREV_EDGE[v]
                # reduce forward capacity
                edge[1] -= f
                # increase reverse capacity
                edge[3][1] += f
                v = PREV_NODE[v]  # type: ignore

        return int(total_cost)