import random
from typing import Any, Optional, SupportsFloat, Tuple, List
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TreeAddOneEdgeDiameterEnv(Env):
    """Environment for the 'Add One Edge to a Tree to Minimize Diameter' problem.

    Single-turn QA:
    - reset() generates a random weighted tree and a weight L for the new edge.
    - The agent must output two vertices x y (1-based) in \\boxed{x y} to connect with an edge of weight L.
    - The environment verifies whether the resulting graph's diameter equals the minimal achievable diameter.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 30,
        max_weight: Optional[int] = None,
        max_L: Optional[int] = None,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            N: If provided, use this fixed number of vertices; must be >= 4.
            min_N: Minimum number of vertices when sampling N randomly.
            max_N: Maximum number of vertices when sampling N randomly.
            max_weight: If provided, edge weights are sampled from [0, max_weight]. Otherwise use [0, N].
            max_L: If provided, L is sampled from [0, max_L]. Otherwise use [0, N].
        """
        super().__init__()
        if N is not None and N < 4:
            raise ValueError("N should be greater than or equal to 4")
        if min_N < 4:
            raise ValueError("min_N should be greater than or equal to 4")
        if min_N > max_N:
            raise ValueError("min_N should be less than or equal to max_N")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N
        self.max_weight: Optional[int] = max_weight
        self.max_L: Optional[int] = max_L

        # Problem instance state
        self.N: Optional[int] = None
        self.edges: Optional[List[Tuple[int, int, int]]] = None
        self.L: Optional[int] = None
        self.gold_min_diameter: Optional[int] = None

        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general instructions for the task."""
        return (
            "You are solving a graph optimization problem on trees.\n"
            "Given a weighted tree, you must add exactly one undirected edge of a given weight L\n"
            "between two vertices (1-based labels) so that the diameter (the maximum shortest-path distance\n"
            "between any two vertices) of the resulting graph is minimized.\n"
            "Output Format: Provide your chosen vertices as two integers in \\boxed{x y}.\n"
            "Note: The checker verifies whether your edge choice achieves the minimal possible diameter.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem instance."""
        super().reset(seed)

        # Sample N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 4:
            raise ValueError("N should be greater than or equal to 4")

        # Generate a random tree with N vertices and random weights
        edges: List[Tuple[int, int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        weight_upper = self.max_weight if self.max_weight is not None else N

        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            # Convert to 1-based indexing for the stored edges
            w = random.randint(0, weight_upper)
            edges.append((u + 1, v + 1, w))
        random.shuffle(edges)

        for u, v, w in edges:
            assert 1 <= u < v <= N
        assert len(edges) == len(set((u, v) for u, v, _ in edges)) == N - 1

        # Sample L
        L_upper = self.max_L if self.max_L is not None else N
        L = random.randint(0, L_upper)

        # Compute the minimal achievable diameter after adding the new edge of weight L
        gold = self._compute_min_diameter(N, edges, L)

        # Store state
        self.N = N
        self.edges = edges
        self.L = L
        self.gold_min_diameter = gold

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v}, {w})" for u, v, w in edges)
        self.current_problem = (
            f"You are given a tree with {N} vertices labeled from 1 to {N}. "
            f"The tree contains the following {N-1} undirected weighted edges (u, v, w):\n"
            f"{edges_str}\n\n"
            f"Add exactly one undirected edge with weight {L}. Your goal is to minimize the longest distance "
            f"(i.e., the diameter) between any two vertices in the resulting graph.\n\n"
            f"Output two integers x y (do NOT include quotes), separated by a space, indicating the two vertices "
            f"to which the new edge of weight {L} is added.\n\n"
            f"Output Format: \\boxed{{x y}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Parse the agent's action, verify the solution, and return reward."""
        # Parse answer from \boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Expect two integers x and y
        try:
            parts = parsed.replace(",", " ").split()
            if len(parts) != 2:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            x, y = int(parts[0]), int(parts[1])
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate bounds
        assert self.N is not None and self.edges is not None and self.L is not None and self.gold_min_diameter is not None
        if not (1 <= x <= self.N and 1 <= y <= self.N):
            # Out of range, considered invalid but format is correct
            # In this framework, treat as wrong answer
            user_diameter = None
            is_optimal = False
            reward = 0.0
            info = {
                "correct": is_optimal,
                "reference_answer": self.gold_min_diameter,
                "user_edge": (x, y),
                "user_diameter": user_diameter,
                "error": "invalid_vertices"
            }
            return TERMINAL_STATE, reward, True, False, info

        # Build graph and compute resulting diameter
        G = networkx.MultiGraph()
        G.add_weighted_edges_from(self.edges)
        G.add_edge(x, y, weight=self.L)

        try:
            user_diameter = max(
                max(networkx.single_source_dijkstra_path_length(G, u, weight="weight").values())
                for u in G.nodes()
            )
        except Exception:
            # If computation fails, treat as wrong answer
            return TERMINAL_STATE, 0.0, True, False, {"error": "compute_error"}

        gold = self.gold_min_diameter
        is_optimal = (user_diameter == gold)
        reward = 1.0 if is_optimal else 0.0

        info = {
            "correct": is_optimal,
            "reference_answer": gold,
            "user_edge": (x, y),
            "user_diameter": user_diameter
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format \\boxed{x y}."""
        if self.N is None:
            # Reasonable default if called before reset
            x = random.randint(1, 5)
            y = random.randint(1, 5)
        else:
            x = random.randint(1, self.N)
            y = random.randint(1, self.N)
        return f"\\boxed{{{x} {y}}}"

    def _compute_min_diameter(self, N: int, edges: List[Tuple[int, int, int]], L: int) -> int:
        """Compute the minimal achievable diameter after adding one edge of weight L.

        This function implements the same core algorithm as in the original RLVE environment.
        """
        # Build adjacency list
        e: List[List[Tuple[int, int]]] = [[] for _ in range(N + 1)]
        NEG_INF = 0
        for u, v, w in edges:
            e[u].append((v, w))
            e[v].append((u, w))
            NEG_INF -= w + 1

        # 1) Find S: the farthest node from node 1
        dis1 = [0] * (N + 1)
        stack = [(1, 0)]
        while stack:
            u, p = stack.pop()
            for v, w in e[u]:
                if v == p:
                    continue
                dis1[v] = dis1[u] + w
                stack.append((v, u))
        S = max(range(1, N + 1), key=lambda i: dis1[i])

        # 2) DFS from S to compute distances (dis) and subtree max-distance (mx), plus parent pointers
        dis = [0] * (N + 1)
        mx = [0] * (N + 1)
        parent = [0] * (N + 1)
        stack2 = [(S, 0, 0)]  # (node, parent, state) state=0: pre, state=1: post
        while stack2:
            u, p, st = stack2.pop()
            if st == 0:
                parent[u] = p
                stack2.append((u, p, 1))
                for v, w in e[u]:
                    if v == p:
                        continue
                    dis[v] = dis[u] + w
                    stack2.append((v, u, 0))
            else:
                mxd = dis[u]
                for v, _ in e[u]:
                    if v == p:
                        continue
                    if mx[v] > mxd:
                        mxd = mx[v]
                mx[u] = mxd

        # 3) Find T: the farthest node from S, and record the original diameter
        T = max(range(1, N + 1), key=lambda i: dis[i])
        diam = dis[T]

        # 4) Extract the diameter path from S to T
        p_nodes: List[int] = []
        u = T
        while True:
            p_nodes.append(u)
            if u == S:
                break
            u = parent[u]
        p_nodes.reverse()
        cnt = len(p_nodes)

        # 5) Compute prefix distances along the path (pre) and branch depths (val)
        pre = [0] * (cnt + 2)
        val = [0] * (cnt + 2)
        for i in range(1, cnt + 1):
            pre[i] = dis[p_nodes[i - 1]]
        for i in range(1, cnt + 1):
            node = p_nodes[i - 1]
            prev_node = p_nodes[i - 2] if i > 1 else None
            next_node = p_nodes[i] if i < cnt else None
            best = 0
            for v, _ in e[node]:
                if v == prev_node or v == next_node:
                    continue
                depth = mx[v] - dis[node]
                if depth > best:
                    best = depth
            val[i] = best

        # 6) Prepare sorted index lists for the two-pointer checks
        p1 = [0] + sorted(range(1, cnt + 1), key=lambda i: val[i] + pre[i])
        p2 = [0] + sorted(range(1, cnt + 1), key=lambda i: val[i] - pre[i], reverse=True)

        # 7) Feasibility check: can we achieve diameter <= x after adding the new edge?
        def check(x: int) -> bool:
            A = B = C = D = NEG_INF
            mx1 = mx2 = NEG_INF
            j = 0

            # First pass: accumulate constraints from violating pairs
            for idx in range(1, cnt + 1):
                i_idx = p1[idx]
                while j + 1 <= cnt and (val[i_idx] + pre[i_idx] + val[p2[j + 1]] - pre[p2[j + 1]] > x):
                    j += 1
                    k = p2[j]
                    c1 = val[k] + pre[k]
                    if c1 > mx1:
                        mx1 = c1
                    c2 = val[k] - pre[k]
                    if c2 > mx2:
                        mx2 = c2

                # Update A, B, C, D
                t = val[i_idx] + pre[i_idx] + mx1
                if t > A:
                    A = t
                t = val[i_idx] - pre[i_idx] + mx1
                if t > B:
                    B = t
                t = val[i_idx] + pre[i_idx] + mx2
                if t > C:
                    C = t
                t = val[i_idx] - pre[i_idx] + mx2
                if t > D:
                    D = t

                # If no pairs violated for all i, it's already feasible
                if idx == cnt and j == 0:
                    return True

            # Adjust constraints by (L - x)
            delta = L - x
            A += delta
            B += delta
            C += delta
            D += delta

            # Second pass: sliding-window ranges
            a, b, c, d = cnt + 1, 1, 0, cnt
            for i_idx in range(1, cnt + 1):
                while a > 1 and pre[i_idx] + pre[a - 1] >= A:
                    a -= 1
                while b <= cnt and -pre[i_idx] + pre[b] < B:
                    b += 1
                while c < cnt and pre[i_idx] - pre[c + 1] >= C:
                    c += 1
                while d >= 1 and -pre[i_idx] - pre[d] < D:
                    d -= 1

                left = a if a > b else b
                r1 = c if c < d else d
                right = i_idx - 1 if i_idx - 1 < r1 else r1
                if left <= right:
                    return True

            return False

        # 8) Binary search for the minimal achievable diameter
        left, right, ans = 0, diam, diam
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans