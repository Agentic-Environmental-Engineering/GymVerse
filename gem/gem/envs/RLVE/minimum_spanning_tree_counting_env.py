from typing import Any, Optional, SupportsFloat, Tuple, List, Dict, Set
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumSpanningTreeCountingEnv(Env):
    """Minimum Spanning Tree Counting environment - single-turn Q&A.

    Task:
      - Given an undirected weighted graph with N vertices and a list of edges (u, v, w),
        compute the number of minimum spanning trees modulo MOD.

    Answer format:
      - The agent must output the final integer answer in \\boxed{...} format.
    """

    def __init__(
        self,
        N_min: int = 3,
        N_max: int = 20,
        edge_ratio: float = 2.5,
        weight_range_divisor: int = 10,
        MAX_MOD: int = 10000,
        **kwargs,
    ):
        super().__init__()
        # Validate parameters
        assert N_min >= 3, "N_min should be greater than or equal to 3"
        assert N_max >= N_min, "N_max should be greater than or equal to N_min"
        assert weight_range_divisor > 0, "weight_range_divisor should be greater than 0"
        assert MAX_MOD > 1, "MAX_MOD should be greater than 1"
        assert edge_ratio > 0.0, "edge_ratio should be greater than 0"

        self.N_min = N_min
        self.N_max = N_max
        self.edge_ratio = edge_ratio
        self.weight_range_divisor = weight_range_divisor
        self.MAX_MOD = MAX_MOD

        # Episode-specific state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_edges: List[Tuple[int, int, int]] = []
        self.current_N: Optional[int] = None
        self.current_MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions for the agent."""
        return (
            "You are solving a graph theory problem: counting the number of minimum spanning trees (MSTs).\n"
            "Given an undirected graph with weighted edges, compute how many MSTs exist modulo the given MOD.\n"
            "Output Format: Your final answer must be a single integer placed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample problem parameters
        N = random.randint(self.N_min, self.N_max)
        edge_ratio = self.edge_ratio
        weight_range = max(1, int(edge_ratio * N / self.weight_range_divisor)) + 1

        # Generate edges ensuring initial connectivity
        edges: List[Tuple[int, int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            edges.append((u, v, random.randint(1, weight_range)))

        # Add extra edges up to num_edges = int(edge_ratio * N), without duplicates
        num_edges = int(edge_ratio * N)
        if len(edges) < num_edges:
            existing_pairs = set((u, v) for u, v, _ in edges)
            all_pairs = set((u, v) for u in range(N) for v in range(u + 1, N))
            remaining_edges = list(all_pairs - existing_pairs)
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges:
                edges.append((u, v, random.randint(1, weight_range)))
        random.shuffle(edges)

        # Basic sanity checks
        for u, v, w in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, _ in edges)), "edges should be unique"

        # Choose modulo P
        P = random.randint(2, self.MAX_MOD)

        # Store current episode data
        self.current_N = N
        self.current_edges = edges
        self.current_MOD = P

        # Build problem statement
        edges_text = "\n".join(f"({u}, {v}, {w})" for u, v, w in edges)
        problem_text = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}.\n"
            f"The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w), "
            f"meaning an undirected edge connecting vertex u to vertex v with weight w:\n"
            f"{edges_text}\n\n"
            f"Consider a subset of edges T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)] such that:\n"
            f"- k = {N - 1} (i.e., you select exactly {N - 1} edges),\n"
            f"- The selected edges form a spanning tree — they connect all {N} vertices without forming any cycles,\n"
            f"- The total weight w_1 + w_2 + ... + w_k is minimized among all such spanning trees (i.e., T is a minimum spanning tree).\n\n"
            f"Please compute the number of such minimum spanning trees modulo {P}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        # Compute reference answer
        self.reference_answer = self._count_mst(N, edges, P)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and terminate."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and score
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        mod = self.current_MOD if self.current_MOD is not None else 1
        # Treat out-of-range as incorrect (reward 0.0)
        in_range = 0 <= user_answer < mod

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "in_range": in_range,
            "modulo": mod,
        }
        if not in_range:
            info["error"] = "out_of_range"

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        mod = self.current_MOD if self.current_MOD is not None else max(2, self.MAX_MOD)
        random_answer = random.randint(0, mod - 1)
        return f"\\boxed{{{random_answer}}}"

    def _count_mst(self, N: int, edges: List[Tuple[int, int, int]], P: int) -> int:
        """Count the number of MSTs modulo P using Kruskal-style grouping and Matrix-Tree theorem."""

        def find(parent: List[int], x: int) -> int:
            # Path compression
            if parent[x] != x:
                parent[x] = find(parent, parent[x])
            return parent[x]

        def union(parent: List[int], a: int, b: int) -> None:
            # Simple union
            ra = find(parent, a)
            rb = find(parent, b)
            if ra != rb:
                parent[rb] = ra

        def det_mod(mat: List[List[int]], mod: int) -> int:
            """
            Compute determinant of a matrix modulo mod using a gcd-based elimination
            that avoids division by non-invertible elements.
            """
            n = len(mat)
            f = 1
            tp = 1
            # Normalize entries into [0, mod)
            for i in range(n):
                for j in range(n):
                    mat[i][j] %= mod

            for i in range(n):
                # Eliminate entries below mat[i][i]
                for j in range(i + 1, n):
                    a = mat[i][i]
                    b = mat[j][i]
                    while b:
                        t = a // b
                        a, b = b, a - t * b
                        # row_i = row_i - t * row_j (from column i onward)
                        for k in range(i, n):
                            mat[i][k] = (mat[i][k] - t * mat[j][k]) % mod
                        # swap row_i and row_j (from column i onward)
                        for k in range(i, n):
                            mat[i][k], mat[j][k] = mat[j][k], mat[i][k]
                        f = -f
                if mat[i][i] % mod == 0:
                    return 0
                tp = (tp * (mat[i][i] % mod)) % mod

            res = f * tp % mod
            return res if res >= 0 else res + mod

        # Sort edges by weight
        sorted_edges = edges.copy()
        sorted_edges.sort(key=lambda x: x[2])

        parent = list(range(N))
        ans = 1
        M = len(sorted_edges)
        i = 0

        # Process edges by weight groups
        while i < M:
            w = sorted_edges[i][2]
            j = i
            while j < M and sorted_edges[j][2] == w:
                j += 1
            group = sorted_edges[i:j]

            # Build multigraph on current DSU components
            adj_count: Dict[Tuple[int, int], int] = {}
            nodes: Set[int] = set()
            for u, v, _ in group:
                ru = find(parent, u)
                rv = find(parent, v)
                if ru != rv:
                    nodes.add(ru)
                    nodes.add(rv)
                    adj_count[(ru, rv)] = adj_count.get((ru, rv), 0) + 1
                    adj_count[(rv, ru)] = adj_count.get((rv, ru), 0) + 1

            # Find connected components in this subgraph and apply Matrix-Tree Theorem
            visited: Set[int] = set()
            for u in nodes:
                if u in visited:
                    continue
                # DFS to collect one connected component
                stack = [u]
                comp: List[int] = []
                visited.add(u)
                while stack:
                    x = stack.pop()
                    comp.append(x)
                    # Look at neighbors of x
                    for (a, b), cnt in adj_count.items():
                        if a == x and b not in visited:
                            visited.add(b)
                            stack.append(b)

                t = len(comp)
                if t > 1:
                    m = t - 1
                    # Construct Laplacian minor (remove last node)
                    mat = [[0] * m for _ in range(m)]
                    for xi in range(m):
                        ni = comp[xi]
                        # Degree within component (consider multiplicity)
                        deg = 0
                        for nj in comp:
                            deg += adj_count.get((ni, nj), 0)
                        deg %= P
                        mat[xi][xi] = deg
                        # Off-diagonals
                        for yj in range(m):
                            if xi != yj:
                                nj = comp[yj]
                                mat[xi][yj] = (-adj_count.get((ni, nj), 0)) % P

                    # Multiply by number of spanning trees of this component
                    ans = (ans * det_mod(mat, P)) % P

            # Unite DSU by all useful edges in this group
            for u, v, _ in group:
                ru = find(parent, u)
                rv = find(parent, v)
                if ru != rv:
                    union(parent, ru, rv)

            i = j

        # Check if the graph is connected
        roots = {self._find_root(parent, x) for x in range(N)}
        if len(roots) != 1:
            return 0
        return ans % P

    @staticmethod
    def _find_root(parent: List[int], x: int) -> int:
        """Helper to find final root after all unions."""
        while parent[x] != x:
            x = parent[x]
        return x