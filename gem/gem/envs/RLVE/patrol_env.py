import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from collections import deque
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PatrolEnv(Env):
    """Patrol problem environment - single-turn Q&A in GEM format.

    Problem:
    You are given a tree with N vertices labeled from 1 to N. It contains N-1 undirected edges.
    You are allowed to add K arbitrary edges to the tree (K in {1, 2}), where each added edge can
    connect any two existing vertices (including possibly the same vertex); it is allowed to be a
    duplicate of an existing edge. After adding these K edges, you must start at vertex 1 (and also
    end at vertex 1) and traverse a path that:
      - Visits each original edge at least once, and
      - Visits each added edge exactly once.
    Output the minimum total number of edges traversed (edges traversed multiple times are counted that many times).
    """

    def __init__(self, N: int, **kwargs):
        """Initialize the PatrolEnv.

        Parameters:
        - N: number of vertices in the tree (must be >= 4).

        Notes:
        - K is sampled uniformly from {1, 2} at each reset.
        """
        super().__init__()
        self.N: int = N
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.edges: Optional[List[Tuple[int, int]]] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a tree patrol optimization problem.\n"
            "Task: Given a tree and an integer K (1 or 2), you may add K arbitrary edges to minimize\n"
            "the length of a closed walk starting and ending at node 1 that traverses every original\n"
            "edge at least once and each added edge exactly once. Return the minimal total number of\n"
            "edge traversals.\n\n"
            "Answer format: Provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)
        assert isinstance(self.N, int), "N must be provided as an integer"
        assert self.N >= 4, "N should be greater than or equal to 4"
        N = self.N

        # Generate a random tree with N nodes, vertices labeled 1..N
        permutations = list(range(N))
        random.shuffle(permutations)
        edges: List[Tuple[int, int]] = []
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            edges.append((u + 1, v + 1))  # Convert to 1-based indexing
        random.shuffle(edges)

        # Validate tree properties
        for u, v in edges:
            assert 1 <= u < v <= N
        assert len(edges) == len(set(edges)) == N - 1

        # Sample K from {1, 2}
        K = random.randint(1, 2)

        # Build adjacency list for the tree
        adj: List[List[int]] = [[] for _ in range(N + 1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS helper
        def bfs(start: int, record_parent: bool = False):
            dist = [-1] * (N + 1)
            parent = [0] * (N + 1)
            q = deque([start])
            dist[start] = 0
            far_node = start
            maxd = 0
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        q.append(y)
                        if dist[y] > maxd:
                            maxd = dist[y]
                            far_node = y
            if record_parent:
                return far_node, maxd, parent, dist
            return far_node, maxd

        # Find tree diameter
        u, _ = bfs(1)
        v, L1, parent, _ = bfs(u, record_parent=True)

        # Compute minimal traversal result based on K
        if K == 1:
            result = 2 * (N - 1) - L1 + 1
        else:
            # Mark nodes on one diameter path (from v back to u using parent from BFS rooted at u)
            on_path = [False] * (N + 1)
            node = v
            while node != 0:
                on_path[node] = True
                node = parent[node]

            # DP over tree to compute L2 (weighted diameter where edges on the diameter have weight -1, others +1)
            d = [0] * (N + 1)
            L2 = [0]

            def dfs(x: int, p: int) -> None:
                for y in adj[x]:
                    if y == p:
                        continue
                    dfs(y, x)
                    w = -1 if (on_path[x] and on_path[y]) else 1
                    L2[0] = max(L2[0], d[x] + d[y] + w)
                    d[x] = max(d[x], d[y] + w)

            dfs(1, 0)
            result = 2 * N - L1 - L2[0]

        # Store state
        self.K = K
        self.edges = edges
        self.reference_answer = result

        # Build problem prompt
        edges_str = "\n".join(f"{u} {v}" for u, v in edges)
        problem = (
            f"You are given a tree (a connected undirected graph with no cycles) with {N} vertices "
            f"labeled from 1 to {N}. It contains the following {N - 1} undirected edges:\n"
            f"{edges_str}\n\n"
            f"You are allowed to add {K} arbitrary edges to the tree. Each added edge can connect any two existing "
            f"vertices (including possibly the same vertex); it is allowed to be a duplicate of an existing edge. "
            f"After adding these {K} edges, you must start at vertex 1 (and also end at vertex 1) and traverse a path that:\n"
            f"- Visits each original edge at least once, and\n"
            f"- Visits each added edge exactly once.\n\n"
            f"Please output the minimum total number of edges traversed. "
            f"Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and terminate."""
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
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
        """Extract the numeric answer from \\boxed{...} in the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the expected \\boxed{...} format."""
        # A loose guess for the range: result is around 2*N typically
        random_answer = random.randint(0, 3 * max(1, self.N))
        return f"\\boxed{{{random_answer}}}"