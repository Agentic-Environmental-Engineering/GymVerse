from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from collections import deque
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Tree_DistanceEqualTriad_CountingEnv(Env):
    """Single-turn environment for counting triads with equal pairwise distances in a tree."""

    def __init__(
        self,
        min_n: int = 4,
        max_n: int = 100,
        n: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.min_n = min_n
        self.max_n = max_n
        self.n = n

        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int]] = []
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Count the number of vertex triads in a tree such that all pairwise distances are equal.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine number of vertices
        if self.n is not None:
            N = self.n
        else:
            N = random.randint(self.min_n, self.max_n)

        if N < 4:
            raise ValueError("N should be greater than or equal to 4")

        self.N = N

        # Generate a random tree with N vertices using the given algorithm
        edges: List[Tuple[int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            edges.append((u + 1, v + 1))  # Convert to 1-based indexing
        random.shuffle(edges)

        # Validate edge set
        for u, v in edges:
            if not (1 <= u < v <= N):
                raise ValueError("Invalid edge generated")
        if not (len(edges) == len(set(edges)) == N - 1):
            raise ValueError("Generated edges do not form a tree")

        self.edges = edges

        # Compute the reference answer
        self.reference_answer = self._compute_reference_answer(N, edges)

        # Build problem prompt
        edge_lines = "\n".join(f"{u} {v}" for u, v in edges)
        self.current_problem = (
            f"You are given a tree (i.e., a connected undirected graph with no cycles) with {N} vertices, "
            f"labeled from 1 to {N}. It contains the following {N - 1} undirected edges:\n"
            f"{edge_lines}\n\n"
            "Please compute the number of three-vertex sets (a triad of vertices A, B, and C such that "
            f"1 ≤ A < B < C ≤ {N}) for which the pairwise distances are all equal — that is, the distance between "
            "A and B, between A and C, and between B and C are all the same. The distance between two vertices is "
            "the number of edges on the shortest path connecting them.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        # Parse the boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare
        try:
            user_answer = int(answer_text)
            is_correct = (user_answer == self.reference_answer)
            reward = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "edges": self.edges,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in boxed format."""
        # A naive bound for the answer is at most C(N, 3)
        N = self.N if self.N is not None else max(self.min_n, 4)
        upper = max(0, N * (N - 1) * (N - 2) // 6)
        random_answer = random.randint(0, upper if upper > 0 else 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, N: int, edges: List[Tuple[int, int]]) -> int:
        """Compute the number of triads with equal pairwise distances using branch BFS and symmetric sums."""
        adjacency: List[List[int]] = [[] for _ in range(N + 1)]
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)

        ans = 0

        # For each candidate center c, we look at its branches (one per neighbor).
        # In each branch we BFS to record how many nodes lie at each distance d from c.
        # Then for each distance d we have counts [c1, c2, ..., ck] across branches,
        # and the number of ways to pick one node in three distinct branches all at that
        # same distance is the 3rd elementary symmetric sum:
        #    e3 = sum_{i<j<k} ci*cj*ck = (S1^3 - 3 S1 S2 + 2 S3) / 6,
        # where S1 = sum ci, S2 = sum ci^2, S3 = sum ci^3.

        for c in range(1, N + 1):
            if len(adjacency[c]) < 3:
                continue  # need at least 3 branches

            visited = [False] * (N + 1)
            visited[c] = True

            branch_counts: List[List[int]] = []
            max_depth = 0

            # BFS each branch separately, marking visited to avoid overlap
            for nbr in adjacency[c]:
                if visited[nbr]:
                    continue
                visited[nbr] = True
                q = deque([(nbr, 1)])
                local: List[int] = []  # local[d] = number of nodes at distance d in this branch
                while q:
                    u, d = q.popleft()
                    # ensure local is long enough
                    if d >= len(local):
                        local.extend([0] * (d - len(local) + 1))
                    local[d] += 1
                    if d > max_depth:
                        max_depth = d
                    for w in adjacency[u]:
                        if not visited[w]:
                            visited[w] = True
                            q.append((w, d + 1))
                branch_counts.append(local)

            b = len(branch_counts)
            if b < 3:
                continue

            # for each possible distance t, compute the 3-way product sum
            for t in range(1, max_depth + 1):
                S1 = 0
                S2 = 0
                S3 = 0
                for f in branch_counts:
                    cnt = f[t] if t < len(f) else 0
                    S1 += cnt
                    S2 += cnt * cnt
                    S3 += cnt * cnt * cnt
                # elementary symmetric sum of order 3
                e3 = (S1 * S1 * S1 - 3 * S1 * S2 + 2 * S3) // 6
                ans += e3

        return ans