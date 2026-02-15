import random
import networkx
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumWeightedSpanningTreeEnv(Env):
    """Environment for the Minimum Weighted Spanning Tree problem with depth-weighted edge costs.

    Single-turn Q&A environment. The agent must choose a root and a spanning tree that minimizes
    the total cost, where each edge contributes its weight multiplied by the depth (distance in edges)
    of its child node from the chosen root.
    """

    def __init__(
        self,
        N: int = 5,
        edge_density: float = 0.5,
        **kwargs
    ):
        """
        Initialize the environment with parameters.

        Args:
            N: Number of vertices in the graph (>= 3).
            edge_density: Density of the undirected edges, in [0.0, 1.0].
        """
        super().__init__()
        self.N = N
        self.edge_density = edge_density

        # Internal state
        self.edges: List[Tuple[int, int, int]] = []
        self.reference_cost: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an undirected weighted graph and must select a root and a spanning tree (N-1 edges).\n"
            "Each non-root vertex t contributes w × K to the cost, where w is the weight of the edge to its parent,\n"
            "and K is the number of edges on the path from the root to t. Your goal is to minimize the total cost.\n\n"
            "Answer format: Provide your output inside \\boxed{...} as a single line of space-separated integers:\n"
            "root u_1 v_1 u_2 v_2 ... u_k v_k, where k = N - 1.\n"
            "Example: \\boxed{0 0 1 1 2 1 3}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Validate parameters
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 3, "N should be greater than or equal to 3"
        assert isinstance(self.edge_density, float), "edge_density must be a float"
        assert 0.0 <= self.edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        # Generate a connected undirected graph with weights
        N = self.N
        edges: List[Tuple[int, int, int]] = []

        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v, random.randint(0, N)))

        num_edges = int(self.edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges:
            remaining_edges = list(
                set((u, v) for u in range(N) for v in range(u + 1, N))
                - set((u, v) for u, v, w in edges)
            )
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges:
                edges.append((u, v, random.randint(0, N)))
        random.shuffle(edges)

        # Validation of generated edges
        for u, v, w in edges:
            assert 0 <= u < v < N, "Edge endpoints must satisfy 0 <= u < v < N"
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "Edges should be unique"

        self.edges = edges

        # Compute the minimal total cost (gold reference) using DP
        total_length = sum(w for _, _, w in edges)
        INF = total_length * N + 1

        # Build adjacency matrix
        A = [[INF] * N for _ in range(N)]
        for x, y, v in edges:
            if v < A[x][y]:
                A[x][y] = A[y][x] = v

        S = (1 << N) - 1

        # Precompute low-bit index
        lg = [0] * (S + 1)
        for i in range(N):
            lg[1 << i] = i

        # f[i][j] = min cost to attach subset j (disjoint from i) to i by exactly |j| edges
        f: List[Dict[int, int]] = [dict() for _ in range(S + 1)]
        # Make f[0][j] = 0 for all j
        f[0] = {j: 0 for j in range(S + 1)}
        # Base case: attaching an empty set costs 0
        for i in range(1, S + 1):
            f[i][0] = 0

        ne = [0] * (S + 1)
        for i in range(1, S + 1):
            s = S ^ i
            prev = 0
            j = s
            # Build reverse linked list of submasks of s
            while j:
                ne[j] = prev
                prev = j
                j = (j - 1) & s

            # Traverse that linked list
            j = prev
            while j:
                x = lg[j & -j]
                # Find cheapest edge from x into i
                best = INF
                tmp = i
                while tmp:
                    yb = tmp & -tmp
                    y = lg[yb]
                    if A[x][y] < best:
                        best = A[x][y]
                    tmp ^= yb

                without_low = j ^ (j & -j)
                f[i][j] = f[i][without_low] + best
                j = ne[j]

        # g[l][i] = min cost to excavate exactly the set i using l roads
        g = [[INF] * (S + 1) for _ in range(N + 1)]
        # With 0 roads, only singletons are free
        for i in range(N):
            g[0][1 << i] = 0

        # Build g
        for l in range(1, N + 1):
            for i in range(1, S + 1):
                j = i
                while j:
                    prev_set = i ^ j
                    cost = g[l - 1][prev_set] + f[prev_set][j] * l
                    if cost < g[l][i]:
                        g[l][i] = cost
                    j = (j - 1) & i

        # Answer is min over all l
        ans = min(g[l][S] for l in range(N + 1))
        self.reference_cost = ans

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v}, {w})" for u, v, w in self.edges)
        problem_text = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}.\n"
            f"The graph contains the following undirected edges (u, v, w), meaning an edge between u and v with weight w:\n"
            f"{edges_str}\n\n"
            "Your task is to select a subset of edges T = [(u_1, v_1, w_1), ..., (u_k, v_k, w_k)] such that:\n"
            f"- k = {N} - 1 (i.e., you select exactly {N - 1} edges).\n"
            f"- The selected edges form a spanning tree — they connect all {N} vertices without forming any cycles.\n"
            "- Choose one vertex as the root. Then, every non-root vertex has exactly one incoming edge in the tree.\n\n"
            "Cost definition:\n"
            "- For each vertex t ≠ root, suppose (s, t, w) is the single incoming edge on the path from the root to t,\n"
            "  and the number of edges from the root to t is K. The cost of this edge is w × K.\n"
            "- Total cost is the sum of these edge costs for all t ≠ root.\n\n"
            "Goal: Minimize the total cost as defined above.\n\n"
            "Output Format:\n"
            "Your final answer must be inside \\boxed{...} and contain a single line of space-separated integers:\n"
            "root u_1 v_1 u_2 v_2 ... u_k v_k. Example: \\boxed{0 0 1 1 2 1 3}\n"
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {"N": self.N, "edges": self.edges}
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step by validating the user's answer."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Extract integers from boxed content
        try:
            answer_array = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if not answer_array:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        N = self.N
        root = answer_array[0]
        mst_flat = answer_array[1:]

        # Validate format: pairs of endpoints
        if len(mst_flat) % 2 != 0:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        mst_edges: List[Tuple[int, int]] = [(mst_flat[i], mst_flat[i + 1]) for i in range(0, len(mst_flat), 2)]

        # Structural validations
        if not (0 <= root < N):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "invalid_root"}

        if len(mst_edges) != N - 1:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "wrong_number_of_edges"}

        # Check coverage of all vertices
        covered_vertices = set(u for u, v in mst_edges) | set(v for u, v in mst_edges)
        if covered_vertices != set(range(N)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "not_spanning"}

        # Build subgraph and ensure edges exist in the given graph
        subgraph = networkx.Graph()
        edge2weight: Dict[Tuple[int, int], int] = {(min(u, v), max(u, v)): w for u, v, w in self.edges}

        for u, v in mst_edges:
            a, b = min(u, v), max(u, v)
            if (a, b) not in edge2weight:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "edge_not_in_graph"}
            subgraph.add_edge(a, b)

        # Connectivity and tree check
        if not networkx.is_connected(subgraph):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "not_connected"}

        if not networkx.is_tree(subgraph):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "not_a_tree"}

        # Compute user's cost using DFS with depth tracking
        adjacent_list: List[List[int]] = [[] for _ in range(N)]
        for u, v in mst_edges:
            adjacent_list[u].append(v)
            adjacent_list[v].append(u)

        visited = [False] * N
        user_cost = 0

        def dfs(vertex: int, parent: int, depth: int) -> None:
            nonlocal user_cost
            visited[vertex] = True
            for neighbor in adjacent_list[vertex]:
                if neighbor == parent:
                    continue
                edge_weight = edge2weight[(min(vertex, neighbor), max(vertex, neighbor))]
                user_cost += edge_weight * (depth + 1)
                dfs(neighbor, vertex, depth + 1)

        dfs(root, -1, 0)

        # If some vertices are not visited due to invalid root connectivity
        if not all(visited):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "root_not_reaching_all_vertices"}

        # Compare with reference minimal cost
        assert self.reference_cost is not None, "Reference cost not computed"
        is_correct = (user_cost == self.reference_cost)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_cost": self.reference_cost,
            "user_cost": user_cost,
            "root": root,
            "selected_edges": [(min(u, v), max(u, v)) for u, v in mst_edges]
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract answer inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid-looking action (not necessarily optimal)."""
        N = self.N
        root = random.randint(0, N - 1)

        # Build full graph
        G = networkx.Graph()
        for u, v, _ in self.edges:
            G.add_edge(u, v)

        # Generate a BFS spanning tree from the root
        # If for some reason BFS fails (should not, graph is connected), fallback to chain
        try:
            bfs_tree = networkx.bfs_tree(G, root)
            # bfs_tree is a directed graph; convert to undirected edge list
            tree_edges = []
            for u, v in bfs_tree.edges():
                tree_edges.append((min(u, v), max(u, v)))
        except Exception:
            # Fallback: simple chain
            order = list(range(N))
            random.shuffle(order)
            tree_edges = [(min(order[i], order[i + 1]), max(order[i], order[i + 1])) for i in range(N - 1)]

        # Ensure exactly N - 1 edges and they are unique
        if len(tree_edges) > N - 1:
            tree_edges = tree_edges[:N - 1]
        elif len(tree_edges) < N - 1:
            # Add random edges to complete the tree if needed (unlikely)
            remaining = [
                (min(u, v), max(u, v))
                for u in range(N)
                for v in range(u + 1, N)
                if (min(u, v), max(u, v)) in {(min(a, b), max(a, b)) for a, b, _ in self.edges}
                and (min(u, v), max(u, v)) not in set(tree_edges)
            ]
            random.shuffle(remaining)
            tree_edges += remaining[: (N - 1 - len(tree_edges))]

        flat = []
        for u, v in tree_edges:
            flat.extend([u, v])

        return f"\\boxed{{{root} {' '.join(map(str, flat))}}}"