import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Clique_IndependentSet_Partitioning_CountingEnv(Env):
    """Environment for counting partitions of a graph into a clique and an independent set."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 20,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed number of vertices. If None, N will be sampled in reset().
        - min_N: Minimum number of vertices to sample when N is not fixed. Must be >= 3.
        - max_N: Maximum number of vertices to sample when N is not fixed. Must be >= min_N.
        """
        super().__init__()
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int]] = []

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a graph partitioning problem.\n"
            "Provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)

        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N

        # Generate a graph with a random clique and independent set
        clique = random.sample(range(N), random.randint(2, N - 1))
        independent_set = list(set(range(N)) - set(clique))
        edges: List[Tuple[int, int]] = []

        # Add all edges within the clique
        for u in clique:
            for v in clique:
                if u < v:
                    edges.append((u, v))

        # Add a random subset of edges between clique and independent set
        cross_edges_pool = [(min(u, v), max(u, v)) for u in clique for v in independent_set]
        k = random.randint(0, len(cross_edges_pool))
        edges += random.sample(cross_edges_pool, k)

        # Shuffle edges for presentation
        random.shuffle(edges)

        # Validate edges
        for u, v in edges:
            assert 0 <= u < v < N, "Edge endpoints must satisfy 0 <= u < v < N"
        assert len(edges) == len(set(edges)), "Edges should be unique"

        self.edges = edges

        # Build adjacency matrix
        flg = [[False] * N for _ in range(N)]
        for u, v in edges:
            flg[u][v] = True
            flg[v][u] = True

        # 2-SAT implication graph on 2*N nodes:
        # indices 0..N-1   == X_i (i is in S1)
        # indices N..2N-1  == ¬X_i (i is in S2)
        dfn = [0] * (2 * N)
        low = [0] * (2 * N)
        in_stack = [False] * (2 * N)
        col = [0] * (2 * N)
        stack: List[int] = []
        tot, colid = 0, 0

        def tarjan(u: int) -> None:
            nonlocal tot, colid
            tot += 1
            dfn[u] = low[u] = tot
            in_stack[u] = True
            stack.append(u)

            pos = u % N
            for i in range(N):
                if i == pos:
                    continue
                v = -1
                # If u represents ¬X_pos (i.e., u>=N) and pos is connected to i,
                # then add implication ¬X_pos → X_i
                if u >= N and flg[pos][i]:
                    v = i
                # If u represents X_pos (u<N) and pos is not connected to i,
                # then X_pos → ¬X_i
                if u < N and not flg[pos][i]:
                    v = i + N

                if v != -1:
                    if dfn[v] == 0:
                        tarjan(v)
                        low[u] = min(low[u], low[v])
                    elif in_stack[v]:
                        low[u] = min(low[u], dfn[v])

            if low[u] == dfn[u]:
                colid += 1
                while True:
                    x = stack.pop()
                    in_stack[x] = False
                    col[x] = colid
                    if x == u:
                        break

        # Run Tarjan on all nodes
        for u in range(2 * N):
            if dfn[u] == 0:
                tarjan(u)

        # Check unsatisfiable: X_i and ¬X_i in same SCC
        for i in range(N):
            if col[i] == col[i + N]:
                assert False, "The problem is unsatisfiable: X_i and ¬X_i are in the same strongly connected component."
                return TERMINAL_STATE, {}

        # Build one satisfying assignment:
        # if col[X_i] < col[¬X_i], put i in S1, else in S2
        S1: List[int] = []
        S2: List[int] = []
        for i in range(N):
            if col[i] < col[i + N]:
                S1.append(i)
            else:
                S2.append(i)

        # Precompute for each vertex how many "cross-edges" they have
        deg = [0] * N
        # For any i in S1, count how many j in S2 that i is connected to
        for i in S1:
            for j in S2:
                if flg[i][j]:
                    deg[i] += 1
        # For any j in S2, count how many i in S1 that j is NOT connected to
        for j in S2:
            for i in S1:
                if not flg[i][j]:
                    deg[j] += 1

        # Count all valid partitions reachable by swapping at most one
        # member between S1 and S2 (including the “no swap” case).
        ans = 0
        cnt1 = len(S1)
        cnt2 = len(S2)

        # Use None as the “dummy” to represent “no element swapped”
        S1d: List[Optional[int]] = [None] + S1
        S2d: List[Optional[int]] = [None] + S2

        for x in S1d:
            for y in S2d:
                # New sizes after removing x (if any) from S1 and adding y (if any)
                C1 = cnt1 - (1 if x is not None else 0) + (1 if y is not None else 0)
                C2 = cnt2 - (1 if y is not None else 0) + (1 if x is not None else 0)
                if C1 == 0 or C2 == 0:
                    continue

                v1 = deg[x] if x is not None else 0
                v2 = deg[y] if y is not None else 0

                # If we swapped two real vertices, adjust the double-counted edge
                if x is not None and y is not None:
                    if flg[x][y]:
                        v1 -= 1
                    else:
                        v2 -= 1

                if v1 == 0 and v2 == 0:
                    ans += 1

        assert ans > 0, "The number of counted valid partitions should be positive."
        self.reference_answer = ans

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        self.current_problem = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}. "
            f"The graph contains the following undirected edges:\n{edges_str}\n\n"
            "Please output the number of ways to partition the vertices into two non-empty sets S and T such that:\n"
            "- S is a clique (i.e., every pair of distinct vertices in S is connected by an edge),\n"
            "- T is an independent set (i.e., no pair of distinct vertices in T is connected by an edge),\n"
            "- S and T are disjoint (i.e., S ∩ T = ∅).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {"N": N, "edges": self.edges, "reference_answer": self.reference_answer}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Verify answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "edges": self.edges,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        # Heuristic range for random answer
        max_guess = (self.N or self.max_N) * (self.N or self.max_N)
        random_answer = random.randint(0, max_guess)
        return f"\\boxed{{{random_answer}}}"