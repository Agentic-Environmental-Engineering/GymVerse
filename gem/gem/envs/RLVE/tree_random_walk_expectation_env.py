from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import sys
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TreeRandomWalkExpectationEnv(Env):
    """Environment for the Tree Random Walk Expectation problem - single-turn Q&A."""

    MOD = 998244353

    def __init__(
        self,
        fixed_n: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 1000,
        modulo: int = 998244353,
        **kwargs
    ):
        """
        Initialize the TreeRandomWalkExpectationEnv.

        Parameters:
        - fixed_n: If provided, use this fixed number of vertices N for all episodes.
        - min_n: Minimum N to sample when fixed_n is not provided.
        - max_n: Maximum N to sample when fixed_n is not provided.
        - modulo: The modulus to use for the output.
        """
        super().__init__()
        self.fixed_n = fixed_n
        self.min_n = min_n
        self.max_n = max_n
        self.modulo = modulo

        if self.fixed_n is not None:
            assert self.fixed_n >= 3, "N should be greater than or equal to 3"

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_n: Optional[int] = None
        self.current_edges: Optional[List[Tuple[int, int]]] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a tree random walk expectation problem.\n"
            "Please provide your answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Select N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"
        self.current_n = N

        # Generate a random tree
        edges: List[Tuple[int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            edges.append((u, v))
        random.shuffle(edges)

        # Validate edges
        for u, v in edges:
            assert 0 <= u < v < N, "Edge endpoints must satisfy 0 <= u < v < N"
        assert len(edges) == len(set(edges)) == N - 1, "Edges must form a valid tree"

        self.current_edges = edges

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(N, edges, self.modulo)

        # Build problem description
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        self.current_problem = (
            f"You are given a tree with {N} vertices labeled from 0 to {N - 1}. "
            f"The tree has the following {N - 1} undirected edges:\n{edges_str}\n\n"
            "A random walk on the tree is defined as follows: from the current vertex, "
            "you move to one of its neighbors uniformly at random at each step. "
            "Define E(S, T) as the expected number of steps to reach vertex T starting "
            "from vertex S (the walk stops immediately upon reaching T).\n\n"
            f"Please compute the sum of all E(S, T) over all ordered pairs (S, T), divided by {N}^2. "
            f"Output this value modulo {self.modulo}.\n\n"
            "Output Format: A single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the submitted answer and return the evaluation result."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)

        if answer_str is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare answer
        try:
            user_answer = int(answer_str)
        except ValueError:
            # Not a valid integer
            info = {
                "error": "invalid_answer"
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if not (0 <= user_answer < self.modulo):
            # Out of range according to original environment verification
            info = {
                "error": "out_of_range",
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "N": self.current_n,
                "edges": self.current_edges,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_n,
            "edges": self.current_edges,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the expected \\boxed{...} format."""
        random_answer = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, N: int, edges: List[Tuple[int, int]], modulo: int) -> int:
        """Compute the reference answer using the original algorithm."""
        # Increase recursion limit for deep trees
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10**6))

        adj = [[] for _ in range(N)]
        d = [0] * N

        # Build adjacency and initial degrees
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            d[u] += 1
            d[v] += 1

        totd = sum(d)

        sz = [0] * N
        parent = [-1] * N

        # DFS to compute subtree sizes and accumulate degree sums
        def dfs(u: int, p: int) -> None:
            parent[u] = p
            sz[u] = 1
            for v in adj[u]:
                if v == p:
                    continue
                dfs(v, u)
                sz[u] += sz[v]
                d[u] += d[v]

        dfs(0, -1)

        # Modular inverse of N^2
        rev = pow((N * N) % modulo, modulo - 2, modulo)

        ans = 0
        for u in range(N):
            for v in adj[u]:
                if v == parent[u]:
                    # Edge from u up to its parent
                    ans = (ans + d[u] * sz[u] * (N - sz[u])) % modulo
                else:
                    # Edge from u down to child v
                    ans = (ans + (totd - d[v]) * sz[v] * (N - sz[v])) % modulo

        return ans * rev % modulo