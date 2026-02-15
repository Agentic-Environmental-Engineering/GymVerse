import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from collections import deque
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxTree_KPathCoveraheEnv(Env):
    """Single-turn environment for the 'Maximize K-Path Coverage on a Tree' problem."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 4,
        max_n: int = 100,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed number of vertices. If None, N will be sampled in [min_n, max_n].
        - min_n: Minimum possible number of vertices (must be >= 4).
        - max_n: Maximum possible number of vertices (must be >= min_n).
        """
        super().__init__()
        assert min_n >= 4, "min_n should be greater than or equal to 4"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"
        if N is not None:
            assert N >= 4, "N should be greater than or equal to 4"

        self.fixed_N = N
        self.min_n = min_n
        self.max_n = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.edges: List[tuple[int, int]] = []

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a tree K-path coverage maximization problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 4, "N should be greater than or equal to 4"
        self.N = N

        # Generate a random tree
        edges: List[tuple[int, int]] = []
        degrees = [0] * N
        permutations = list(range(N))
        random.shuffle(permutations)

        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            edges.append((u, v))
            degrees[u] += 1
            degrees[v] += 1

        random.shuffle(edges)

        # Validate edges
        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1

        # Verify it is a tree
        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree), "Generated graph must be a tree"

        # Choose K based on number of leaves
        leaf_count = sum(1 for d in degrees if d == 1)
        upper = max(1, leaf_count // 2 - 1)
        K = random.randint(1, upper)

        self.K = K
        self.edges = edges

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(N, edges, K)

        # Build problem statement
        self.current_problem = self._build_problem_text(N, K, edges)
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _build_problem_text(self, N: int, K: int, edges: List[tuple[int, int]]) -> str:
        """Construct the problem description string."""
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        return (
            f"You are given a tree (i.e., a connected undirected graph with no cycles) with {N} vertices "
            f"labeled from 0 to {N - 1}. The tree contains the following {N - 1} undirected edges. "
            f"Each edge is represented as a tuple (u, v), meaning there is an undirected edge connecting vertex u and vertex v:\n"
            f"{edges_str}\n\n"
            f"You need to choose exactly {K} unordered pairs of distinct vertices (u, v). For each selected pair, define the set of all vertices on the unique path between u and v (inclusive) as covered. "
            f"Please maximize the total number of unique vertices that are covered by at least one of the {K} paths. "
            f"Output a single integer — the maximum number of vertices that can be covered.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

    def _compute_reference_answer(self, N: int, edges: List[tuple[int, int]], K: int) -> int:
        """Compute the maximum number of vertices covered by at least one of K paths."""
        M = K * 2

        # Build adjacency list
        adjacency = [[] for _ in range(N)]
        for A, B in edges:
            adjacency[A].append(B)
            adjacency[B].append(A)

        # Peeling process (distance layers from original leaves)
        d = [len(adjacency[i]) - 1 for i in range(N)]
        dep = [0] * N
        q = deque()

        # Initialize queue with all original leaves (d[i] == 0)
        for i in range(N):
            if d[i] == 0:
                q.append(i)
                dep[i] = 1

        # Count nodes per peeling round
        cnt = [0] * (N + 1)
        maxd = 0

        while q:
            x = q.popleft()
            depth = dep[x]
            cnt[depth] += 1
            if depth > maxd:
                maxd = depth
            for y in adjacency[x]:
                d[y] -= 1
                if d[y] == 0:
                    dep[y] = depth + 1
                    q.append(y)

        # Sum min(cnt[k], 2*K) over layers
        ans = 0
        for k in range(1, maxd + 1):
            ans += min(cnt[k], M)
        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer."""
        answer_text = self._parse_answer(action)

        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
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
            "edges": self.edges,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} format."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer answer in boxed format)."""
        max_val = self.N if (self.N is not None) else 10
        random_answer = random.randint(0, max_val)
        return f"\\boxed{{{random_answer}}}"