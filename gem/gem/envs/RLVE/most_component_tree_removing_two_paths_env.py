from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MostComponentTreeRemovingTwoPathsEnv(Env):
    """Environment for the problem: Maximize the number of connected components in a tree
    after removing two edge-disjoint paths (paths may share vertices). Single-turn Q&A.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 4,
        max_n: int = 100,
        **kwargs
    ):
        super().__init__()
        assert min_n >= 4, "min_n should be greater than or equal to 4"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"
        if N is not None:
            assert N >= 4, "N should be greater than or equal to 4"

        self.N_fixed: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.edges: Optional[List[tuple[int, int]]] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are given a tree problem. You should output your final answer in \\boxed{...} format.\n"
            "Only a single integer should be placed inside the box.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            N = self.N_fixed
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 4, "N should be greater than or equal to 4"
        self.N = N

        # Generate a random tree with N nodes labeled 1..N
        edges: List[tuple[int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v) + 1, max(u, v) + 1
            edges.append((u, v))
        random.shuffle(edges)

        for u, v in edges:
            assert 1 <= u < v <= N
        assert len(edges) == len(set(edges)) == N - 1

        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree)

        self.edges = edges

        # Compute reference answer using the DP algorithm
        self.reference_answer = self._compute_reference_answer(N, edges)

        # Build problem statement
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        problem = (
            f"You are given a tree with {N} vertices labeled from 1 to {N}, where vertex 1 is the root. "
            f"The tree contains the following {N - 1} undirected edges:\n"
            f"{edges_str}\n\n"
            "Your task is to choose two paths (each from any vertex to any vertex; a path could be just one single vertex) such that:\n"
            "- The two paths do NOT share any edge (but they can share vertices).\n"
            "- You remove all vertices on both paths, along with all their adjacent edges.\n"
            "- After this removal, the remaining structure is a forest. Try your best to maximize the number of connected components in the resulting forest.\n\n"
            "Output Format: A single integer in \\boxed{...} — the maximum number of connected components you can achieve."
        )

        self.current_problem = problem
        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse the boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem_generated"}

        try:
            user_answer = int(answer_str)
            is_correct = (user_answer == self.reference_answer)
            reward: float = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer
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

    def _compute_reference_answer(self, N: int, edges: List[tuple[int, int]]) -> int:
        """Compute the reference answer using the original DP logic."""
        adj: List[List[int]] = [[] for _ in range(N + 1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Build a child-only adjacency by rooting at 1
        visited = [False] * (N + 1)
        visited[1] = True
        stack = [1]
        children = [[] for _ in range(N + 1)]
        order: List[int] = []
        while stack:
            u = stack.pop()
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    children[u].append(v)
                    stack.append(v)

        # DP in post-order
        ans = 0
        f0 = [0] * (N + 1)
        f1 = [0] * (N + 1)
        f2 = [0] * (N + 1)
        f3 = [0] * (N + 1)
        for u in reversed(order):
            deg_u = len(children[u])
            dp0 = deg_u
            dp1 = 1
            dp2 = deg_u
            dp3 = deg_u
            ret = 0
            off = 1 if u == 1 else 0
            for q in children[u]:
                c0, c1, c2, c3 = f0[q], f1[q], f2[q], f3[q]
                # Update global answer
                val = dp3 + c0 - off
                if val > ans:
                    ans = val
                val = dp0 + c3 - off
                if val > ans:
                    ans = val
                val = dp1 + c2
                if val > ans:
                    ans = val
                val = dp1 + c1 - 1
                if val > ans:
                    ans = val
                val = dp2 + c1 - off
                if val > ans:
                    ans = val
                val = dp2 + c2 - off
                if val > ans:
                    ans = val
                # Transitions for f1
                if c1 > dp1:
                    dp1 = c1
                if c2 + 1 > dp1:
                    dp1 = c2 + 1
                # Transitions for f3
                val = dp0 + c2 - 1
                if val > dp3:
                    dp3 = val
                val = dp0 + c1 - 1
                if val > dp3:
                    dp3 = val
                val = dp2 + c0 - 1
                if val > dp3:
                    dp3 = val
                val = c3 + deg_u - 1
                if val > dp3:
                    dp3 = val
                val = c0 + deg_u + ret - 2
                if val > dp3:
                    dp3 = val
                # Transitions for f2
                val = dp0 + c0 - 1
                if val > dp2:
                    dp2 = val
                # Transitions for f0
                val = c0 + deg_u - 1
                if val > dp0:
                    dp0 = val
                # Update ret for next child
                if c1 > ret:
                    ret = c1
                if c2 > ret:
                    ret = c2
            f0[u], f1[u], f2[u], f3[u] = dp0, dp1, dp2, dp3
        return ans

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        guess_upper = self.N if self.N is not None else 100
        random_answer = random.randint(0, max(0, guess_upper))
        return f"\\boxed{{{random_answer}}}"