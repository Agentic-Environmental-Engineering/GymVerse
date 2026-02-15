from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GraphContainTreeCountingEnv(Env):
    """Environment for counting the number of bijections embedding a tree T into an undirected graph G.

    The task:
    - Given an undirected graph G and a tree T on the same number of vertices N (labeled 0..N-1),
      count the number of bijections p from vertices of T to vertices of G such that for every
      edge (u, v) in T, (p(u), p(v)) is an edge in G.

    Interaction:
    - Single-turn QA. The agent must return the answer in \\boxed{...} format.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        edge_density: Optional[float] = None,
        min_N: int = 3,
        max_N: int = 10,
        min_edge_density: float = 0.0,
        max_edge_density: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        # Parameter configuration
        if N is not None:
            assert isinstance(N, int), "N must be an integer"
            assert N >= 3, "N should be greater than or equal to 3"
        assert isinstance(min_N, int) and isinstance(max_N, int), "min_N and max_N must be integers"
        assert min_N >= 3, "min_N should be greater than or equal to 3"
        assert min_N <= max_N, "min_N should be less than or equal to max_N"

        if edge_density is not None:
            assert 0.0 <= edge_density <= 1.0, "edge_density must be in [0, 1]"
        assert 0.0 <= min_edge_density <= 1.0, "min_edge_density must be in [0, 1]"
        assert 0.0 <= max_edge_density <= 1.0, "max_edge_density must be in [0, 1]"
        assert min_edge_density <= max_edge_density, "min_edge_density should be <= max_edge_density"

        self.N_fixed = N
        self.edge_density_fixed = edge_density
        self.min_N = min_N
        self.max_N = max_N
        self.min_edge_density = min_edge_density
        self.max_edge_density = max_edge_density

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.edge_density_used: Optional[float] = None
        self.T_edges: List[Tuple[int, int]] = []
        self.G_edges: List[Tuple[int, int]] = []

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a graph counting problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Choose parameters
        self.N = self.N_fixed if self.N_fixed is not None else random.randint(self.min_N, self.max_N)
        assert self.N is not None and self.N >= 3
        if self.edge_density_fixed is not None:
            self.edge_density_used = self.edge_density_fixed
        else:
            self.edge_density_used = random.uniform(self.min_edge_density, self.max_edge_density)

        # Generate a random tree T with N vertices using a randomized parent selection
        N = self.N
        permutation = list(range(N))
        random.shuffle(permutation)

        T_edges: List[Tuple[int, int]] = []
        for index, vertex in enumerate(permutation):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutation[:index])
            if u > v:
                u, v = v, u
            T_edges.append((u, v))
        random.shuffle(T_edges)

        # Validations for T
        for u, v in T_edges:
            assert 0 <= u < v < N
        assert len(T_edges) == len(set(T_edges)) == N - 1

        # Build graph G
        G_edges: List[Tuple[int, int]] = []
        random.shuffle(permutation)
        for u, v in T_edges:
            pu, pv = permutation[u], permutation[v]
            if pu > pv:
                pu, pv = pv, pu
            G_edges.append((pu, pv))

        # Add additional edges to reach target density
        edge_density = float(self.edge_density_used)
        total_possible = N * (N - 1) // 2
        num_edges_target = int(edge_density * total_possible)
        if len(G_edges) < num_edges_target:
            all_pairs = {(i, j) for i in range(N) for j in range(i + 1, N)}
            existing = set(G_edges)
            remaining_edges = list(all_pairs - existing)
            extra_needed = min(len(remaining_edges), num_edges_target - len(G_edges))
            G_edges += random.sample(remaining_edges, extra_needed)

        random.shuffle(G_edges)

        # Validations for G
        for u, v in G_edges:
            assert 0 <= u < v < N
        assert len(G_edges) == len(set(G_edges)), "G edges should be unique"

        # Compute reference answer using inclusion–exclusion with tree DP
        G_adj = [[False] * N for _ in range(N)]
        for u, v in G_edges:
            G_adj[u][v] = True
            G_adj[v][u] = True

        ADJ: List[List[int]] = [[] for _ in range(N)]
        for u, v in T_edges:
            ADJ[u].append(v)
            ADJ[v].append(u)

        vis = [False] * N
        f = [[0] * N for _ in range(N)]
        ans = 0

        def dfs(u: int, parent: int, whi: List[int]) -> None:
            for w in ADJ[u]:
                if w == parent:
                    continue
                dfs(w, u, whi)
            for x in whi:
                total_prod = 1
                for w in ADJ[u]:
                    if w == parent:
                        continue
                    total = 0
                    for y in whi:
                        if G_adj[x][y]:
                            total += f[w][y]
                    total_prod *= total
                f[u][x] = total_prod

        def solve_subset() -> None:
            nonlocal ans
            whi = [i for i in range(N) if vis[i]]
            # Run DP rooted at 0
            if len(whi) > 0:
                dfs(0, -1, whi)
            # Inclusion–exclusion
            if (N - len(whi)) & 1:
                for x in whi:
                    ans -= f[0][x]
            else:
                for x in whi:
                    ans += f[0][x]

        def enumerate_subsets(dep: int = 0) -> None:
            if dep == N:
                solve_subset()
                return
            vis[dep] = False
            enumerate_subsets(dep + 1)
            vis[dep] = True
            enumerate_subsets(dep + 1)

        enumerate_subsets()
        assert ans > 0, "There should be at least one valid bijection"

        self.reference_answer = ans
        self.T_edges = T_edges
        self.G_edges = G_edges

        # Build the problem prompt
        G_edges_str = "\n".join(f"({u}, {v})" for (u, v) in self.G_edges)
        T_edges_str = "\n".join(f"({u}, {v})" for (u, v) in self.T_edges)
        self.current_problem = (
            f"You are given an undirected graph G and a tree T, each with {N} vertices labeled from 0 to {N-1}.\n\n"
            f"- Graph G has the following undirected edge set E1:\n{G_edges_str}\n\n"
            f"- Tree T has the following undirected edge set E2:\n{T_edges_str}\n\n"
            f"Please compute the number of bijections p (i.e., permutations) from the vertices of T to the vertices of G such that: "
            f"for every edge (u, v) in E2, the edge (p(u), p(v)) exists in E1.\n\n"
            f"Output Format: A single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.N,
            "edge_density": self.edge_density_used,
            "T_edges": self.T_edges,
            "G_edges": self.G_edges,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer."""
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not initialized. Call reset() first."
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
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
        """Sample a random action (random integer answer) in \\boxed{...} format."""
        if self.reference_answer is not None and self.reference_answer >= 0:
            hi = max(1, self.reference_answer * 2 + 10)
            guess = random.randint(0, hi)
        else:
            guess = random.randint(0, 1000)
        return f"\\boxed{{{guess}}}"