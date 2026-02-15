import random
from collections import deque
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinCostReducingLNDSEnv(Env):
    """
    Environment for the problem:
    Given arrays A and B of length N, remove a set of indices from A (paying costs in B)
    to reduce the length of the longest non-decreasing subsequence (LNDS) by at least 1,
    with minimum total cost.

    Single-turn Q&A environment:
    - reset() generates a new instance
    - step(action) validates the proposed indices
    """

    def __init__(
        self,
        N: Optional[int] = None,
        n_min: int = 3,
        n_max: int = 50,
        **kwargs: Any,
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, use this fixed N for all generated instances.
        - n_min: Minimum N (used when N is None).
        - n_max: Maximum N (used when N is None).
        """
        super().__init__()
        self.fixed_N: Optional[int] = N
        self.n_min: int = n_min
        self.n_max: int = n_max

        # Problem state
        self.N: int = 0
        self.A: List[int] = []
        self.B: List[int] = []
        self.original_lnds_length: int = 0
        self.gold_cost: int = 0
        self.reference_indices: List[int] = []  # one optimal set (0-based indices)
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task description and output format instructions."""
        return (
            "You are given two arrays A and B of the same length N.\n"
            "You may erase any (distinct) elements from A. When you erase element A[i], you must pay a cost of B[i].\n"
            "Your goal is to reduce the length of the longest non-decreasing subsequence (LNDS) of A by at least 1,\n"
            "while minimizing the total cost. Provide the indices you choose to erase, separated by spaces.\n\n"
            "Output Format: Put your final answer inside \\boxed{...}.\n"
            "Example: \\boxed{0 3 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            if self.fixed_N < 3:
                raise ValueError("N should be greater than or equal to 3")
            self.N = self.fixed_N
        else:
            self.N = random.randint(self.n_min, self.n_max)
            if self.N < 3:
                self.N = 3  # ensure minimum

        # Generate arrays A and B
        self.A = [random.randint(1, max(1, self.N * 2)) for _ in range(self.N)]
        self.B = [random.randint(1, self.N) for _ in range(self.N)]

        # Compute gold minimal cost and one optimal set via max-flow min-cut on LNDS DAG
        self._compute_gold_and_reference()

        # Build problem prompt
        A_str = " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(self.A))
        B_str = " ".join(f"B[{i}]={Bi}" for i, Bi in enumerate(self.B))
        self.current_problem = (
            f"You are given two arrays A and B, both of length {self.N}:\n"
            f"A: {A_str}\n"
            f"B: {B_str}\n"
            "You may erase any (distinct) elements from A. When you erase element A[i], you must pay a cost of B[i]. "
            "Please reduce the length of the longest non-decreasing subsequence (not necessarily contiguous) of A by at least 1, "
            "while minimizing the total cost of the erased elements.\n"
            "Output Format: Output a single line containing the indices of the elements you choose to erase, separated by spaces, "
            "inside \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_gold_and_reference(self) -> None:
        """Compute the original LNDS length, minimal deletion cost (gold), and one optimal set of indices."""
        N = self.N
        # 1-based copies for graph construction
        A = [0] + self.A.copy()
        B = [0] + self.B.copy()
        C = [0] + list(range(1, N + 1))  # indices 1..N

        # Graph: Node split for each i: i (in), N+i (out)
        V = 2 * N + 2
        SRC, SINK = 0, V - 1

        class Edge:
            __slots__ = ("to", "rev", "cap", "orig")

            def __init__(self, to: int, rev: int, cap: int):
                self.to = to
                self.rev = rev
                self.cap = cap
                self.orig = cap

        adj: List[List[Edge]] = [[] for _ in range(V)]

        def add_edge(u: int, v: int, c: int) -> None:
            adj[u].append(Edge(v, len(adj[v]), c))
            adj[v].append(Edge(u, len(adj[u]) - 1, 0))

        # 1) Node-split edges and record them
        id_info: List[Optional[tuple[int, int, int, int]]] = [None] * (N + 1)
        for i in range(1, N + 1):
            u, v = i, N + i
            idx_u = len(adj[u])
            idx_v = len(adj[v])
            adj[u].append(Edge(v, idx_v, B[i]))
            adj[v].append(Edge(u, idx_u, 0))
            id_info[i] = (u, idx_u, v, idx_v)

        # 2) Compute dp[i] = LNDS ending at i
        dp = [0] * (N + 1)
        for i in range(1, N + 1):
            best = 1
            for j in range(1, i):
                if A[j] <= A[i] and dp[j] + 1 > best:
                    best = dp[j] + 1
            dp[i] = best

        K = max(dp[1:]) if N > 0 else 0
        self.original_lnds_length = K

        # 3) Add DAG edges with infinite capacity
        S = sum(B[1:]) + 1
        INF = S

        for i in range(1, N + 1):
            if dp[i] == 1:
                add_edge(SRC, i, INF)
            if dp[i] == K:
                add_edge(N + i, SINK, INF)
            for j in range(1, i):
                if A[j] <= A[i] and dp[j] + 1 == dp[i]:
                    add_edge(N + j, i, INF)

        level = [-1] * V
        ptr = [0] * V

        def bfs_level() -> bool:
            for i in range(V):
                level[i] = -1
            dq = deque([SRC])
            level[SRC] = 0
            while dq:
                u = dq.popleft()
                for e in adj[u]:
                    if e.cap > 0 and level[e.to] < 0:
                        level[e.to] = level[u] + 1
                        dq.append(e.to)
            return level[SINK] >= 0

        def dfs_flow(u: int, f: int) -> int:
            if u == SINK:
                return f
            i = ptr[u]
            while i < len(adj[u]):
                e = adj[u][i]
                if e.cap > 0 and level[e.to] == level[u] + 1:
                    pushed = dfs_flow(e.to, min(f, e.cap))
                    if pushed:
                        e.cap -= pushed
                        adj[e.to][e.rev].cap += pushed
                        return pushed
                i += 1
                ptr[u] = i
            return 0

        def dinic() -> int:
            total_flow = 0
            while bfs_level():
                ptr[:] = [0] * V
                while True:
                    pushed = dfs_flow(SRC, INF)
                    if not pushed:
                        break
                    total_flow += pushed
            return total_flow

        def reachable(u: int, t: int) -> bool:
            vis = [False] * V
            dq = deque([u])
            vis[u] = True
            while dq:
                x = dq.popleft()
                if x == t:
                    return True
                for e in adj[x]:
                    if e.cap > 0 and not vis[e.to]:
                        vis[e.to] = True
                        dq.append(e.to)
            return False

        flow = dinic()
        if flow <= 0:
            # With positive costs B[i], this should not happen.
            # Still, ensure a valid gold_cost.
            flow = 0
        self.gold_cost = flow

        # 5) Greedy extract lexicographically smallest C-sorted cut
        vc = sorted((C[i], i) for i in range(1, N + 1))
        ans = []
        remaining_flow = flow

        for _, idx in vc:
            # If idx.in cannot reach idx.out in residual graph, it is essential
            if not reachable(idx, N + idx):
                ans.append(idx)
                # Permanently remove its split edge
                u, iu, v, iv = id_info[idx]  # type: ignore
                e1 = adj[u][iu]
                e2 = adj[v][iv]
                e1.orig = 0
                e2.orig = 0
                # Reset all capacities to their original values
                for u0 in range(V):
                    for e in adj[u0]:
                        e.cap = e.orig
                # Recompute remaining flow
                level = [-1] * V
                ptr = [0] * V
                remaining_flow = dinic()
                if remaining_flow == 0:
                    break

        # Convert to 0-based indices
        ans_zero_based = [i - 1 for i in ans]
        # Sanity check: sum of B over selected indices equals gold_cost
        assert self.gold_cost == sum(self.B[i] for i in ans_zero_based), (
            f"Gold cost {self.gold_cost} does not match computed cost "
            f"{sum(self.B[i] for i in ans_zero_based)}"
        )
        self.reference_indices = ans_zero_based

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Validate the submitted indices.
        Reward scheme:
        - Format error: -0.1
        - Correct minimal-cost solution that reduces LNDS by at least 1: 1.0
        - Otherwise: 0.0
        """
        content = self._parse_answer(action)
        if content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices (space-separated integers)
        tokens = content.strip().split()
        if len(tokens) == 0:
            # No indices selected; cannot reduce LNDS
            info = {
                "correct": False,
                "reason": "no_indices",
                "gold_cost": self.gold_cost,
                "original_lnds_length": self.original_lnds_length,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        try:
            indices = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate indices: distinct and in range
        seen = set()
        for i in indices:
            if not (0 <= i < self.N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "index_out_of_range"}
            if i in seen:
                return TERMINAL_STATE, 0.0, True, False, {"error": "duplicate_indices"}
            seen.add(i)

        # Check LNDS reduction
        erased = [False] * self.N
        for i in indices:
            erased[i] = True
        newA = [Ai for i, Ai in enumerate(self.A) if not erased[i]]

        # Compute LNDS of newA (O(n^2) DP)
        def lnds_length(arr: List[int]) -> int:
            if not arr:
                return 0
            F = [1] * len(arr)
            for i in range(len(arr)):
                for j in range(i):
                    if arr[j] <= arr[i]:
                        F[i] = max(F[i], F[j] + 1)
            return max(F) if F else 0

        new_lnds = lnds_length(newA)
        if new_lnds >= self.original_lnds_length:
            info = {
                "correct": False,
                "reason": "no_reduction",
                "new_lnds": new_lnds,
                "original_lnds_length": self.original_lnds_length,
                "gold_cost": self.gold_cost,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check cost minimality
        user_cost = sum(self.B[i] for i in indices)
        is_correct = (user_cost == self.gold_cost)

        info = {
            "correct": is_correct,
            "user_cost": user_cost,
            "gold_cost": self.gold_cost,
            "original_lnds_length": self.original_lnds_length,
            "new_lnds": new_lnds,
            "reference_indices": self.reference_indices,
        }
        reward: float = 1.0 if is_correct else 0.0

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re

        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random subset of indices inside \\boxed{...}."""
        k = random.randint(0, self.N) if self.N > 0 else 0
        subset = sorted(random.sample(range(self.N), k)) if k > 0 else []
        content = " ".join(map(str, subset))
        return f"\\boxed{{{content}}}"