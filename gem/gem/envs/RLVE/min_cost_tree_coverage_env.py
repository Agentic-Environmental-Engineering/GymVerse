from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinCostTreeCoverageEnv(Env):
    """Minimum Cost Tree Coverage environment - single-turn Q&A.

    Task:
    - Given a tree with N vertices (0-indexed), a set of required vertices to cover,
      a distance limit D, and a cost array W for selecting vertices,
      select a subset of vertices to cover all required vertices where a selected
      vertex covers all vertices within distance <= D. The goal is to minimize the total cost.
    - Submit the selected vertex indices (space-separated) inside \\boxed{...}.
    - Any optimal subset (achieving the minimal total cost) is accepted.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 30,
        **kwargs
    ):
        super().__init__()
        if min_N < 3:
            min_N = 3
        if max_N < min_N:
            max_N = min_N
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Problem state
        self.N: Optional[int] = None
        self.edges: List[tuple[int, int]] = []
        self.covered_vertices: List[int] = []
        self.D: int = 1
        self.W: List[int] = []
        self.current_problem: Optional[str] = None

        # Reference for checking
        self.gold_cost: Optional[int] = None

        # Cached adjacency for validation
        self.adj: List[List[int]] = []

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a minimum cost coverage problem on a tree.\n"
            "- You will be given a tree with N vertices labeled 0..N-1 and N-1 undirected edges.\n"
            "- Selecting a vertex u covers all vertices within distance <= D from u.\n"
            "- You must ensure that all required vertices are covered.\n"
            "- Each selected vertex u incurs cost W[u]. Minimize the total cost.\n"
            "Output Format: Provide the selected vertex indices (space-separated) inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 3:
            N = 3

        # Generate a random tree using a randomized parent linkage from a permutation
        permutations = list(range(N))
        random.shuffle(permutations)

        depths: List[Optional[int]] = [None] * N
        edges: List[tuple[int, int]] = []
        for index, vertex in enumerate(permutations):
            if index == 0:
                depths[vertex] = 0
                continue
            u = vertex
            v = random.choice(permutations[:index])
            # depth assignment based on parent's depth
            assert depths[v] is not None
            depths[u] = depths[v] + 1
            u2, v2 = (u, v) if u < v else (v, u)
            edges.append((u2, v2))
        random.shuffle(edges)

        # Sanity checks
        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1

        # Sample covered vertices (at least one)
        covered_vertices = random.sample(range(N), k=random.randint(1, N))

        # Choose D based on depths distribution
        max_depth_for_covered = max(depths[c] for c in covered_vertices if depths[c] is not None)
        D = random.randint(1, max(1, max_depth_for_covered // 2))

        # Costs
        W = [random.randint(1, N) for _ in range(N)]

        # Build adjacency
        adj = [[] for _ in range(N)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Compute minimal cost (gold) using DP as in the original environment
        important = [False] * N
        for x in covered_vertices:
            important[x] = True

        K = D
        INF = sum(W) + 1

        dp = [[INF] * (K + 1) for _ in range(N)]
        fdp = [[INF] * (K + 1) for _ in range(N)]

        for i in range(N):
            dp[i][K] = W[i]  # place a guard on i
            if important[i]:
                fdp[i][0] = 0  # covered by ancestor is fine
            else:
                dp[i][0] = 0  # no guard needed (not important)

        parent = [-1] * N
        children = [[] for _ in range(N)]
        order: List[int] = []

        stack = [0]
        parent[0] = 0
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if parent[v] == -1:
                    parent[v] = u
                    children[u].append(v)
                    stack.append(v)

        for u in reversed(order):
            for v in children[u]:
                tru = [0] * (K + 1)
                trv = [0] * (K + 1)

                tru[0] = min(dp[u])
                for i in range(1, K + 1):
                    tru[i] = min(tru[i - 1], fdp[u][i - 1])

                trv[0] = min(dp[v])
                for i in range(1, K + 1):
                    trv[i] = min(trv[i - 1], fdp[v][i - 1])

                new_dp = [0] * (K + 1)
                new_fdp = [0] * (K + 1)

                for i in range(K):
                    new_dp[i] = min(dp[u][i] + trv[i], dp[v][i + 1] + tru[i + 1])
                    if new_dp[i] > INF:
                        new_dp[i] = INF
                new_dp[K] = dp[u][K] + trv[K]
                if new_dp[K] > INF:
                    new_dp[K] = INF

                new_fdp[0] = fdp[u][0] + trv[0]
                if new_fdp[0] > INF:
                    new_fdp[0] = INF
                for i in range(1, K + 1):
                    new_fdp[i] = min(fdp[u][i] + trv[i], fdp[v][i - 1] + tru[i])
                    if new_fdp[i] > INF:
                        new_fdp[i] = INF

                dp[u] = new_dp
                fdp[u] = new_fdp

        gold_cost = min(dp[0])

        # Store state
        self.N = N
        self.edges = edges
        self.covered_vertices = covered_vertices
        self.D = D
        self.W = W
        self.adj = adj
        self.gold_cost = gold_cost

        # Build problem prompt
        edges_str = "\n".join(f"{u} {v}" for u, v in edges)
        W_str = " ".join(f"W[{i}]={w}" for i, w in enumerate(W))
        covered_str = " ".join(map(str, covered_vertices))
        problem_text = (
            f"You are given a tree with {N} vertices labeled from 0 to {N - 1}. "
            f"The tree contains {N - 1} undirected edges. Each edge is represented as a tuple (u, v), "
            f"meaning there is an undirected edge connecting vertex u to vertex v:\n"
            f"{edges_str}\n\n"
            f"You may select any subset of vertices. When a vertex u is selected, it covers all vertices that are reachable "
            f"from u by a path containing at most {D} edges (i.e., within distance ≤ {D} in terms of edge count). "
            f"You are required to cover the following vertices: {covered_str}\n"
            f"Each selected vertex u incurs a cost of W[u]. The cost array is: {W_str}\n"
            f"Try your best to minimize the total cost of the selected vertices while ensuring all required vertices are covered.\n\n"
            f"Output Format: Provide the selected vertex indices (space-separated) inside \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step: parse and verify the submitted solution."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices from boxed content (space-separated integers)
        tokens = boxed_content.strip().split()
        selected: List[int] = []
        if tokens:
            try:
                selected = [int(tok) for tok in tokens]
            except ValueError:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate current state exists
        if self.N is None or self.gold_cost is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Check for duplicates
        if len(selected) != len(set(selected)):
            info = {
                "correct": False,
                "error": "duplicates_in_selection",
                "user_selection": selected,
                "gold_cost": self.gold_cost,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check range and compute coverage with cost
        for v in selected:
            if not (0 <= v < self.N):
                info = {
                    "correct": False,
                    "error": "index_out_of_range",
                    "invalid_vertex": v,
                    "user_selection": selected,
                    "gold_cost": self.gold_cost,
                }
                return TERMINAL_STATE, 0.0, True, False, info

        user_cost = 0
        covered = [False] * self.N
        for vertex in selected:
            user_cost += self.W[vertex]
            visited = [False] * self.N
            visited[vertex] = True
            stack = [(vertex, 0)]
            while stack:
                u, d = stack.pop()
                covered[u] = True
                if d == self.D:
                    continue
                for w in self.adj[u]:
                    if not visited[w]:
                        visited[w] = True
                        stack.append((w, d + 1))

        covered_ok = all(covered[c] for c in self.covered_vertices)

        is_correct = (covered_ok and (user_cost == self.gold_cost))
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "covered_ok": covered_ok,
            "user_cost": user_cost,
            "gold_cost": self.gold_cost,
            "user_selection": selected,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns None if not found."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, str(text))
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random subset of vertices wrapped in \\boxed{...}."""
        if self.N is None:
            # Fallback: random small example
            sample = []
        else:
            k = random.randint(0, self.N)  # allow empty selection
            sample = sorted(random.sample(range(self.N), k=k))
        content = " ".join(map(str, sample))
        return f"\\boxed{{{content}}}"