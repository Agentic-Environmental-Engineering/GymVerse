import random
import networkx
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TreeCenterEnv(Env):
    """Tree center problem environment - single-turn Q&A.

    The task is to select a root vertex r in a weighted tree to minimize
    sum_i dist(i, r) * C[i], where dist(i, r) is the sum of edge weights along
    the unique path between i and r.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 2,
        max_n: int = 50,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed number of vertices (must be >= 2 if provided).
             If None, a random N in [min_n, max_n] will be sampled on reset.
        - min_n: Minimum N when sampling randomly. Must be >= 2.
        - max_n: Maximum N when sampling randomly. Must be >= min_n.
        """
        super().__init__()
        if min_n < 2:
            raise ValueError("min_n should be greater than or equal to 2")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        if N is not None and N < 2:
            raise ValueError("N should be greater than or equal to 2")

        self.fixed_N = N
        self.min_n = min_n
        self.max_n = max_n

        # State placeholders
        self.N: Optional[int] = None
        self.C: Optional[List[int]] = None
        self.edges: Optional[List[Tuple[int, int, int]]] = None
        self.adjacent: Optional[List[List[Tuple[int, int]]]] = None

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.gold_cost: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a weighted tree and a cost for each vertex.\n"
            "Your task is to select a single vertex r (0-indexed) that minimizes the sum of\n"
            "dist(i, r) * C[i] over all vertices i, where dist(i, r) is the path length between i and r.\n"
            "Please provide your final answer in \\boxed{...} format, containing only the selected integer r.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)

        # Validate N
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")

        self.N = N

        # Generate costs C[i] in [1, N]
        self.C = [random.randint(1, N) for _ in range(N)]

        # Generate a random tree
        permutations = list(range(N))
        random.shuffle(permutations)
        edges: List[Tuple[int, int, int]] = []
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            w = random.randint(1, N)
            edges.append((u, v, w))
        random.shuffle(edges)

        # Validate edges
        for u, v, w in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)) == N - 1

        # Build graph and validate tree property
        tree = networkx.Graph()
        tree.add_weighted_edges_from(edges)
        assert networkx.is_tree(tree)

        self.edges = edges

        # Build adjacency list
        adjacent: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
        for u, v, w in edges:
            adjacent[u].append((v, w))
            adjacent[v].append((u, w))
        self.adjacent = adjacent

        # Compute initial cost using root 0 and subtree sums
        reference_answer = 0
        gold_cost = 0
        subtree_sumC = [0] * N

        def dfs_initial(u: int, parent: int, depth: int) -> None:
            nonlocal gold_cost
            subtree_sumC[u] = self.C[u]
            gold_cost += depth * self.C[u]
            for v, w in adjacent[u]:
                if v == parent:
                    continue
                dfs_initial(v, u, depth + w)
                subtree_sumC[u] += subtree_sumC[v]

        dfs_initial(0, -1, 0)

        # Re-rooting to find minimal cost and corresponding root
        def find_solution(u: int, parent: int, now_answer: int) -> None:
            nonlocal reference_answer, gold_cost
            if now_answer < gold_cost:
                reference_answer = u
                gold_cost = now_answer
            for v, w in adjacent[u]:
                if v == parent:
                    continue
                # Move root from u to v
                # New answer = now + (sumC_total - sumC[v]) * w - sumC[v] * w
                new_answer = now_answer + (subtree_sumC[0] - subtree_sumC[v]) * w - subtree_sumC[v] * w
                find_solution(v, u, new_answer)

        find_solution(0, -1, gold_cost)
        assert gold_cost > 0

        self.reference_answer = reference_answer
        self.gold_cost = gold_cost

        # Build problem prompt
        C_lines = "\n".join(f"C[{i}]={Ci}" for i, Ci in enumerate(self.C))
        edges_lines = "\n".join(f"({u}, {v}, {w})" for u, v, w in self.edges)
        self.current_problem = (
            f"You are given a tree (connected undirected graph with no cycles) with {N} vertices, "
            f"labeled from 0 to {N - 1}.\n\n"
            f"Each vertex has a cost, given as a list C of length {N}, where C[i] is the cost of vertex i:\n"
            f"{C_lines}\n\n"
            f"The tree contains the following {N} - 1 = {N - 1} undirected edges. "
            f"Each edge is represented as a tuple (u, v, w), meaning there is an undirected edge connecting "
            f"vertex u to vertex v with weight w:\n"
            f"{edges_lines}\n\n"
            f"Your task is to select a single vertex r (where r is in the range 0 to {N - 1}).\n"
            f"Try your best to minimize dist(0, r) * C[0] + dist(1, r) * C[1] + ... + dist({N - 1}, r) * C[{N - 1}],\n"
            f"where dist(i, j) is the distance between vertices i and j in the tree. "
            f"The distance between two vertices is defined as the sum of the weights of the edges on the unique path connecting them.\n\n"
            f"Output Format: Your final answer should be a single integer r in \\boxed{{...}} format. Example: \\boxed{{0}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the user's answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_root = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate range
        if self.N is None or self.C is None or self.edges is None or self.adjacent is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        if not (0 <= user_root < self.N):
            return TERMINAL_STATE, 0.0, True, False, {"error": "out_of_range"}

        # Compute user's cost by DFS rooted at user_root
        visited = [False] * self.N
        user_cost = 0

        def dfs_user(u: int, parent: int, depth: int) -> None:
            nonlocal user_cost
            visited[u] = True
            user_cost += depth * self.C[u]
            for v, w in self.adjacent[u]:
                if v == parent:
                    continue
                dfs_user(v, u, depth + w)

        dfs_user(user_root, -1, 0)

        # Verify correctness (minimal cost)
        assert self.gold_cost is not None
        is_correct = (user_cost == self.gold_cost)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_root,
            "gold_cost": self.gold_cost,
            "user_cost": user_cost,
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
        """Sample a random action (random vertex index in boxed format)."""
        if self.N is None:
            # Fallback in case reset has not been called
            # Sample from a reasonable default range
            random_answer = 0
        else:
            random_answer = random.randint(0, self.N - 1)
        return f"\\boxed{{{random_answer}}}"