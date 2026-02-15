import random
import networkx
from collections import deque
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TreeChangeOneEdgeDiameterEnv(Env):
    """
    Single-turn environment: Given a tree, remove one edge and add one edge to minimize or maximize the diameter.
    The agent must output four integers (u1 v1 u2 v2) inside \\boxed{...} specifying the edge to remove and the edge to add.
    Reward:
    - 1.0 if the resulting tree is valid and achieves the optimal diameter (min or max as specified)
    - 0.0 otherwise
    - -0.1 if the answer format is invalid
    """

    def __init__(
        self,
        n: Optional[int] = None,
        min_n: int = 4,
        max_n: int = 50,
        mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the environment.

        Args:
            n: If provided, fixes the number of vertices. Must be >= 4.
            min_n: Minimum number of vertices if n is not provided (>= 4).
            max_n: Maximum number of vertices if n is not provided and must be >= min_n.
            mode: Either "minimize" or "maximize". If None, a mode will be chosen randomly at reset.
        """
        super().__init__()
        if n is not None and n < 4:
            raise ValueError("n should be greater than or equal to 4")
        if min_n < 4:
            raise ValueError("min_n should be greater than or equal to 4")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        if mode is not None and mode not in ("minimize", "maximize"):
            raise ValueError("mode should be either 'minimize' or 'maximize'")

        self.fixed_n: Optional[int] = n
        self.min_n: int = min_n
        self.max_n: int = max_n
        self.fixed_mode: Optional[str] = mode

        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int]] = []
        self.mode: str = "minimize"
        self.current_problem: Optional[str] = None
        self.reference_action: Optional[str] = None
        self.optimal_diameter: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a tree transformation problem.\n"
            "You may remove exactly one edge and add exactly one edge so that the graph remains a tree.\n"
            "Your goal is to either minimize or maximize the diameter (longest path length in edges).\n"
            "Answer format: Provide four integers 'u1 v1 u2 v2' inside \\boxed{...}, where:\n"
            "- (u1, v1) is the edge to remove\n"
            "- (u2, v2) is the edge to add\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new tree and objective."""
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # Choose N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 4
        self.N = N

        # Generate a random tree with vertices 1..N
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

        for u, v in edges:
            assert 1 <= u < v <= N
        assert len(edges) == len(set(edges)) == N - 1
        self.edges = edges

        # Choose objective (minimize or maximize)
        if self.fixed_mode is not None:
            self.mode = self.fixed_mode
        else:
            self.mode = random.choice(["minimize", "maximize"])

        # Compute optimal diameter and reference actions
        optimal_diameter, reference_action = self._compute_optimal_action_and_diameter(N, edges, self.mode)
        self.optimal_diameter = optimal_diameter
        self.reference_action = reference_action

        # Build the user-facing problem statement
        problem_text = self._build_problem_text(N, edges, self.mode)
        self.current_problem = problem_text

        obs = self._get_instructions() + problem_text
        return obs, {}

    def _build_problem_text(self, N: int, edges: List[Tuple[int, int]], mode: str) -> str:
        """Build the problem prompt string."""
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        maximize_or_minimize = "maximize" if mode == "maximize" else "minimize"
        return (
            f"You are given a tree (a connected undirected acyclic graph) with {N} vertices labeled 1 through {N}.\n"
            f"The tree contains the following edges:\n{edges_str}\n\n"
            f"You may remove one edge from the tree and add a new edge (possibly the same edge) while keeping the graph a tree.\n"
            f"Your goal is to {maximize_or_minimize} the diameter of the resulting tree. The diameter is the number of edges on the longest path.\n\n"
            f"Output Format: Provide four integers u1 v1 u2 v2 inside \\boxed{{...}}, separated by spaces."
        )

    def _compute_optimal_action_and_diameter(
        self, N: int, edges: List[Tuple[int, int]], mode: str
    ) -> Tuple[int, str]:
        """Compute the optimal diameter after a single edge change and provide a reference action."""

        # Build adjacency
        A = [[] for _ in range(N + 1)]
        for u, v in edges:
            A[u].append(v)
            A[v].append(u)

        def get_diameter(start: int, skip_u: Optional[int] = None, skip_v: Optional[int] = None) -> List[int]:
            # First BFS to find one end of the diameter
            dist = [-1] * (N + 1)
            dist[start] = 0
            q = deque([start])
            far = start
            while q:
                u = q.popleft()
                for v in A[u]:
                    if skip_u is not None and ((u == skip_u and v == skip_v) or (u == skip_v and v == skip_u)):
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        if dist[v] > dist[far]:
                            far = v
            # Second BFS from that end to find the other end and record parents
            P = [-1] * (N + 1)
            dist2 = [-1] * (N + 1)
            dist2[far] = 0
            q = deque([far])
            far2 = far
            while q:
                u = q.popleft()
                for v in A[u]:
                    if skip_u is not None and ((u == skip_u and v == skip_v) or (u == skip_v and v == skip_u)):
                        continue
                    if dist2[v] == -1:
                        dist2[v] = dist2[u] + 1
                        P[v] = u
                        q.append(v)
                        if dist2[v] > dist2[far2]:
                            far2 = v
            # Reconstruct the diameter path
            D = []
            u = far2
            while u != -1:
                D.append(u)
                u = P[u]
            return D

        def get_farthest(start: int, skip_u: Optional[int] = None, skip_v: Optional[int] = None) -> int:
            dist = [-1] * (N + 1)
            dist[start] = 0
            q = deque([start])
            far = start
            while q:
                u = q.popleft()
                for v in A[u]:
                    if skip_u is not None and ((u == skip_u and v == skip_v) or (u == skip_v and v == skip_u)):
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        if dist[v] > dist[far]:
                            far = v
            return far

        # Original diameter path
        D = get_diameter(1)
        InDiameter = [False] * (N + 1)
        for u in D:
            InDiameter[u] = True

        # f[u]: longest chain from u into a subtree off the diameter
        # g[u]: diameter within u's off-diameter subtree
        f = [0] * (N + 1)
        g = [0] * (N + 1)

        def tree_dp(u: int, p: int) -> None:
            for v in A[u]:
                if v == p:
                    continue
                tree_dp(v, u)
                if InDiameter[v]:
                    continue
                old_f = f[u]
                g[u] = max(g[u], g[v], f[v] + 1 + old_f)
                f[u] = max(old_f, f[v] + 1)

        tree_dp(D[0], 0)

        L = len(D)
        # prefix DP
        pref = [0] * L
        cur = 0
        for i in range(L):
            u = D[i]
            if i == 0:
                pref[i] = max(0, g[u], cur + f[u])
            else:
                pref[i] = max(pref[i - 1], g[u], cur + f[u])
            cur = max(cur + 1, f[u] + 1)

        INF = N + 5
        kmin = INF
        kmax = -INF
        x1min = y1min = x2min = y2min = None
        x1max = y1max = x2max = y2max = None

        # suffix DP + find best removal for min/max
        R = 0
        cur = 0
        for i in range(L - 1, 0, -1):
            u = D[i]
            R = max(R, g[u], cur + f[u])
            cur = max(cur + 1, f[u] + 1)
            left = pref[i - 1]
            # candidate for minimal new diameter
            cand_min = max(left, R, (R + 1) // 2 + (left + 1) // 2 + 1)
            if cand_min < kmin:
                kmin = cand_min
                x1min, y1min = u, D[i - 1]
            # candidate for maximal new diameter
            if R + 1 + left > kmax:
                kmax = R + 1 + left
                x1max, y1max = u, D[i - 1]

        # also consider removing a single off-diameter branch edge for max
        for u in D:
            for v in A[u]:
                if not InDiameter[v]:
                    if L + g[v] > kmax:
                        kmax = L + g[v]
                        x1max, y1max = u, v

        # find the new-edge endpoints for the minimal case
        if x1min is None or y1min is None:
            # Safety fallback; should not happen in a valid tree
            x1min, y1min = D[0], D[1] if L > 1 else D[0]
        D1 = get_diameter(x1min, x1min, y1min)
        x2min = D1[(len(D1) - 1) // 2]
        D2 = get_diameter(y1min, x1min, y1min)
        y2min = D2[(len(D2) - 1) // 2]

        # and for the maximal case
        if x1max is None or y1max is None:
            # Safety fallback; should not happen in a valid tree
            x1max, y1max = D[0], D[1] if L > 1 else D[0]
        x2max = get_farthest(x1max, x1max, y1max)
        y2max = get_farthest(y1max, x1max, y1max)

        if mode == "minimize":
            gold = kmin
            reference = f"{x1min} {y1min} {x2min} {y2min}"
        else:
            gold = kmax
            reference = f"{x1max} {y1max} {x2max} {y2max}"

        return gold, reference

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step: parse and evaluate the user's action."""
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.N is None or self.optimal_diameter is None or self.edges is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        N = self.N
        edges = list(self.edges)

        # Try to parse four integers
        try:
            import re

            tokens = re.findall(r"-?\d+", parsed)
            if len(tokens) != 4:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            u1, v1, u2, v2 = map(int, tokens)
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate vertices
        if not (1 <= u1 <= N and 1 <= v1 <= N and u1 != v1 and 1 <= u2 <= N and 1 <= v2 <= N and u2 != v2):
            info = {
                "error": "invalid_solution",
                "mode": self.mode,
                "reference_action": self.reference_action,
                "optimal_diameter": self.optimal_diameter,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Remove one edge (undirected, normalized)
        e_remove = (min(u1, v1), max(u1, v1))
        new_edges = [e for e in edges if e != e_remove]
        if len(new_edges) != N - 2:
            # Edge to remove not in the tree
            info = {
                "error": "invalid_solution",
                "mode": self.mode,
                "reference_action": self.reference_action,
                "optimal_diameter": self.optimal_diameter,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Add new edge
        e_add = (min(u2, v2), max(u2, v2))
        if e_add in new_edges:
            info = {
                "error": "invalid_solution",
                "mode": self.mode,
                "reference_action": self.reference_action,
                "optimal_diameter": self.optimal_diameter,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        new_edges.append(e_add)

        # Check if resulting graph is a valid tree
        G = networkx.Graph()
        G.add_nodes_from(range(1, N + 1))
        G.add_edges_from(new_edges)
        if not networkx.is_tree(G):
            info = {
                "error": "invalid_solution",
                "mode": self.mode,
                "reference_action": self.reference_action,
                "optimal_diameter": self.optimal_diameter,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute resulting diameter
        user_diameter = networkx.diameter(G)

        # Check optimality
        is_optimal = (user_diameter == self.optimal_diameter)

        info = {
            "correct": is_optimal,
            "mode": self.mode,
            "user_diameter": user_diameter,
            "optimal_diameter": self.optimal_diameter,
            "reference_action": self.reference_action,
            "user_action": f"{u1} {v1} {u2} {v2}",
        }
        reward = 1.0 if is_optimal else 0.0

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...}."""
        import re

        pattern = r"\\boxed\{([^}]*)\}"
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid-looking action (may not be optimal)."""
        if self.N is None or not self.edges:
            # Return a dummy action
            return "\\boxed{1 2 1 3}"
        N = self.N
        edges = self.edges
        # Randomly pick an existing edge to remove
        u1, v1 = random.choice(edges)
        # Randomly pick a non-existing edge to add
        existing = set(edges)
        while True:
            u2 = random.randint(1, N)
            v2 = random.randint(1, N)
            if u2 != v2:
                e = (min(u2, v2), max(u2, v2))
                if e not in existing or e == (min(u1, v1), max(u1, v1)):
                    break
        return f"\\boxed{{{u1} {v1} {u2} {v2}}}"