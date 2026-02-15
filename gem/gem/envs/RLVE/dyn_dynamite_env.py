import queue
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DynDynamiteEnv(Env):
    """Dynamic Dynamite tree center selection environment - single-turn Q&A.

    Task:
    You are given a tree with N vertices (0-indexed), a list of edges, and a set of key vertices.
    You must select exactly M centers (from all N vertices) to minimize the maximum distance
    (in number of edges) from any key vertex to its nearest selected center.

    Answer format:
    Provide exactly M space-separated vertex indices inside \\boxed{...}.
    Example: \\boxed{0 3 7}
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 30,
        # Original RLVE reward configuration parameters preserved (not used in GEM step logic)
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 5.0,
        **kwargs
    ):
        super().__init__()
        # Parameterization and validation
        assert min_N >= 3, "min_N should be greater than or equal to 3"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"
        if N is not None:
            assert N >= 3, "N should be greater than or equal to 3"

        self.N_fixed = N
        self.min_N = min_N
        self.max_N = max_N

        # Preserve original reward-related fields (not used directly in GEM reward scheme)
        self.rewards_config = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # Runtime state
        self.current_problem: Optional[str] = None
        self.gold_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int]] = []
        self.key_vertices: List[int] = []
        self.M: Optional[int] = None
        self.adj: List[List[int]] = []

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a tree center selection optimization problem.\n"
            "Task: Given a tree, a list of key vertices, and an integer M, "
            "select exactly M centers to minimize the maximum distance "
            "from any key vertex to its nearest selected center.\n"
            "Answer Format: Provide exactly M space-separated vertex indices inside \\boxed{...}.\n"
            "Example: \\boxed{0 3 7}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            self.N = self.N_fixed
        else:
            self.N = random.randint(self.min_N, self.max_N)

        # Generate a random tree with N vertices
        N = self.N
        edges: List[Tuple[int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
        random.shuffle(edges)

        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1

        # Sample key vertices and M
        key_vertices = random.sample(range(N), random.randint(2, N))
        M = random.randint(1, len(key_vertices) - 1)

        # Build adjacency list
        adj: List[List[int]] = [[] for _ in range(N)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

        # Prepare arrays for the DP computation
        d = [0] * N
        for kv in key_vertices:
            d[kv] = 1

        parent = [-1] * N
        order: List[int] = []
        stack: List[int] = [0]
        parent[0] = -1
        while stack:
            x = stack.pop()
            order.append(x)
            for v in adj[x]:
                if v == parent[x]:
                    continue
                parent[v] = x
                stack.append(v)

        NEG_INF = -(N + 1)
        INF = N + 1

        def needed(t: int) -> int:
            """Compute the minimum number of ignitions (centers) needed to cover key vertices within distance t."""
            f = [NEG_INF] * N
            g = [INF] * N
            cnt = 0

            # Process in reverse preorder (children before parent)
            for x in reversed(order):
                # Discard ignition if it covers nearest uncovered bomb within t
                if f[x] + g[x] <= t:
                    f[x] = NEG_INF

                # If there's an uncovered bomb here and this room has a bomb, place an ignition here
                if g[x] > t and d[x] == 1:
                    if f[x] < 0:
                        f[x] = 0

                # If an ignition at distance exactly t reaches here, count it
                if f[x] == t:
                    f[x] = NEG_INF
                    g[x] = 0
                    cnt += 1

                # Propagate distances up to the parent
                p = parent[x]
                if p != -1:
                    val_f = f[x] + 1
                    if val_f > f[p]:
                        f[p] = val_f
                    val_g = g[x] + 1
                    if val_g < g[p]:
                        g[p] = val_g

            # If there's still an ignition reaching the root, count it
            if f[0] >= 0:
                cnt += 1
            return cnt

        # Binary search minimal t in [0, N] such that needed(t) <= M
        l, r = 0, N
        while l < r:
            mid = (l + r) // 2
            if needed(mid) <= M:
                r = mid
            else:
                l = mid + 1
        gold_answer = l

        # Save state
        self.edges = edges
        self.key_vertices = key_vertices
        self.M = M
        self.adj = adj
        self.gold_answer = gold_answer

        # Build problem statement
        edges_str = "\n".join(f"{u} {v}" for u, v in edges)
        key_vertices_str = " ".join(map(str, key_vertices))
        self.current_problem = (
            f"You are given a tree with {N} vertices, labeled from 0 to {N - 1}.\n"
            f"It contains the following {N - 1} undirected edges (each line is 'u v'):\n{edges_str}\n\n"
            f"Key vertices: {key_vertices_str}\n"
            f"Please select exactly {M} vertices (from all {N} vertices) to serve as centers.\n"
            f"Your goal is to minimize the maximum distance (number of edges) from any key vertex to its nearest selected center.\n\n"
            f"Output Format: Provide exactly {M} centers as space-separated integers inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": N,
            "M": M,
            "edges": edges,
            "key_vertices": key_vertices,
            "gold_answer": gold_answer,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the centers."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse centers as integers
        try:
            tokens = boxed_content.strip().split()
            selected_centers = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate selection
        if self.M is None or self.N is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        if len(selected_centers) != self.M:
            info = {
                "error": "invalid_solution",
                "reason": "wrong_number_of_centers",
                "expected_M": self.M,
                "provided_count": len(selected_centers),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(0 <= v < self.N for v in selected_centers):
            info = {
                "error": "invalid_solution",
                "reason": "vertex_out_of_range",
                "N": self.N,
                "provided_centers": selected_centers,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute distances via BFS from selected centers
        dist: List[Optional[int]] = [None] * self.N
        Q: queue.Queue[int] = queue.Queue()
        for s in selected_centers:
            if dist[s] is None:
                dist[s] = 0
                Q.put(s)
            else:
                # Duplicate center, still fine
                pass

        while not Q.empty():
            u = Q.get()
            for v in self.adj[u]:
                if dist[v] is None:
                    dist[v] = dist[u] + 1
                    Q.put(v)

        # Compute max distance among key vertices
        try:
            user_max_distance = max(dist[u] for u in self.key_vertices if dist[u] is not None)
        except ValueError:
            # Just in case, though tree is connected so this should not happen
            return TERMINAL_STATE, 0.0, True, False, {"error": "distance_computation_failed"}

        is_correct = (user_max_distance == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "gold_answer": self.gold_answer,
            "user_max_distance": user_max_distance,
            "selected_centers": selected_centers,
            "N": self.N,
            "M": self.M,
            "edges": self.edges,
            "key_vertices": self.key_vertices,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: choose M random centers."""
        if self.N is None or self.M is None:
            # Fallback: provide an empty box if not initialized
            return "\\boxed{}"
        centers = random.sample(range(self.N), self.M)
        centers_str = " ".join(map(str, centers))
        return f"\\boxed{{{centers_str}}}"