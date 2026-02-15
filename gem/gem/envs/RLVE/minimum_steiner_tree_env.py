import random
from collections import deque
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict

import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumSteinerTreeEnv(Env):
    """
    Minimum Steiner Tree environment (single-turn Q&A).

    Task:
    - Given an undirected weighted graph and a subset of K required vertices,
      select a set of edges that forms a connected subgraph covering all required
      vertices with minimum total weight.

    Answer format:
    - The agent must output the endpoints of the selected edges as space-separated integers
      inside \\boxed{...}, e.g., \\boxed{0 1 1 2 2 3}.
    """

    def __init__(
        self,
        N: int = 10,
        edge_density: float = 0.5,
        **kwargs,
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Number of vertices (must be >= 4)
        - edge_density: Desired density of edges in the undirected graph (0.0 to 1.0)

        Notes:
        - This is a single-turn environment.
        """
        super().__init__()
        self.N = N
        self.edge_density = edge_density

        # Generated per episode
        self.edges: List[Tuple[int, int, int]] = []
        self.K: int = 0
        self.to_be_connected: List[int] = []
        self.gold_answer: Optional[int] = None  # minimal total weight as defined by original logic
        self.edge2weight: Dict[Tuple[int, int], int] = {}

        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Minimum Steiner Tree problem on an undirected weighted graph.\n"
            "Please provide your final answer in \\boxed{...} format, containing the endpoints of the selected edges as space-separated integers.\n"
            "Example: \\boxed{0 1 1 2 2 3}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem.

        Returns:
        - observation: str
        - info: dict
        """
        super().reset(seed)
        assert self.N >= 4, "N should be greater than or equal to 4"
        assert 0.0 <= self.edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        N = self.N
        edges: List[Tuple[int, int, int]] = []

        # Ensure connectivity by creating a random spanning structure
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v, random.randint(1, N)))

        # Add extra edges to reach target edge count based on density
        target_edges = int(self.edge_density * N * (N - 1) / 2)
        if len(edges) < target_edges:
            existing = set((u, v) for u, v, _ in edges)
            all_pairs = set((u, v) for u in range(N) for v in range(u + 1, N))
            remaining_edges = list(all_pairs - existing)
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), target_edges - len(edges)))
            for u, v in remaining_edges:
                edges.append((u, v, random.randint(1, N)))
        random.shuffle(edges)

        # Validate edges
        for u, v, w in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, _ in edges)), "edges should be unique"

        # Sample required vertices (terminals)
        K = random.randint(3, min(20, N - 1))
        to_be_connected = random.sample(range(N), K)

        # Build adjacency list
        adj: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        # Dynamic programming for Steiner Tree (preserving original algorithm and logic)
        full_mask = (1 << K) - 1
        dp: List[List[Optional[int]]] = [[None] * (full_mask + 1) for _ in range(N)]
        for i in range(K):
            dp[to_be_connected[i]][1 << i] = 0

        for s1 in range(1, full_mask + 1):
            # Merge subsets
            for i in range(N):
                s2 = (s1 - 1) & s1
                while s2:
                    a = dp[i][s2]
                    b = dp[i][s1 ^ s2]
                    if a is not None and b is not None:
                        v = a + b
                        cur = dp[i][s1]
                        if cur is None or v < cur:
                            dp[i][s1] = v
                    s2 = (s2 - 1) & s1

            # SPFA-like relaxation over graph for subset s1
            vis = [False] * N
            q = deque()
            for i in range(N):
                if dp[i][s1] is not None:
                    q.append(i)
                    vis[i] = True
            while q:
                u = q.popleft()
                vis[u] = False
                du = dp[u][s1]
                assert du is not None
                for v, w in adj[u]:
                    nd = du + w
                    cur = dp[v][s1]
                    if cur is None or nd < cur:
                        dp[v][s1] = nd
                        if not vis[v]:
                            q.append(v)
                            vis[v] = True

        # Preserve original logic for gold answer
        gold_answer: Optional[int] = dp[to_be_connected[0]][full_mask]

        # Store for later verification
        self.edges = edges
        self.K = K
        self.to_be_connected = to_be_connected
        self.gold_answer = gold_answer
        self.edge2weight = {(min(u, v), max(u, v)): w for u, v, w in self.edges}

        # Build problem text
        problem_text = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}.\n\n"
            "The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w), "
            "meaning an undirected edge connecting vertex u to vertex v with weight w:\n"
            + "\n".join(f"({u}, {v}, {w})" for u, v, w in self.edges)
            + "\n\n"
            f"Your task is to select a subset of edges T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)] such that:\n"
            f"- The selected edges form a connected graph that contains these {K} vertices: "
            + " ".join(map(str, self.to_be_connected))
            + "\n"
            "- Your goal is to minimize the total weight of the selected edges: w_1 + w_2 + ... + w_k.\n\n"
            "Output Format:\n"
            "Your final answer should be a single line containing the endpoints of the selected edges in order: "
            "u_1 v_1 u_2 v_2 ... u_k v_k, separated by spaces, and wrapped in \\boxed{...}.\n"
            "Example: \\boxed{0 1 1 2 2 3}\n"
        )

        self.current_problem = self._get_instructions() + problem_text
        return self.current_problem, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step and verify the user's answer.

        Returns:
        - observation (TERMINAL_STATE)
        - reward: 1.0 if correct, 0.0 if wrong, -0.1 if format error
        - terminated: True
        - truncated: False
        - info: dict with diagnostic information
        """
        # Extract the boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers
        try:
            tokens = boxed.strip().split()
            answer_array = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Verify answer format: must be even-length list representing edge endpoints
        if len(answer_array) % 2 != 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer_format"}

        # Build edge list from pairs
        mst_pairs = [(answer_array[i], answer_array[i + 1]) for i in range(0, len(answer_array), 2)]

        # Basic vertex range check and required vertices inclusion
        N = self.N
        used_vertices = set()
        for u, v in mst_pairs:
            if not (0 <= u < N and 0 <= v < N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "vertex_out_of_range"}
            used_vertices.add(u)
            used_vertices.add(v)

        if not set(self.to_be_connected).issubset(used_vertices):
            return TERMINAL_STATE, 0.0, True, False, {"error": "missing_required_vertices"}

        # Validate edges exist in the original graph and compute total weight
        total_weight = 0
        subgraph = networkx.Graph()
        for u, v in mst_pairs:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in self.edge2weight:
                return TERMINAL_STATE, 0.0, True, False, {"error": "edge_not_in_graph"}
            w = self.edge2weight[(a, b)]
            total_weight += w
            subgraph.add_edge(a, b)

        # Check connectivity of the selected subgraph
        if subgraph.number_of_nodes() == 0 or not networkx.is_connected(subgraph):
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_connected"}

        # Compare against reference minimal cost (preserving original logic)
        reference = self.gold_answer
        if reference is None:
            # In an unlikely case where DP did not compute a value
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_reference"}

        is_correct = (total_weight == reference)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_min_cost": reference,
            "user_total_cost": total_weight,
            "num_selected_edges": len(mst_pairs),
            "required_vertices": self.to_be_connected,
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
        """
        Sample a random action by selecting a small random subset of edges and
        returning their endpoints in \\boxed{...} format. This does not aim to be correct.
        """
        if not self.edges:
            # Fallback: produce an empty selection
            return r"\boxed{}"
        m = random.randint(1, min(3, len(self.edges)))
        chosen = random.sample(self.edges, m)
        seq: List[int] = []
        for u, v, _ in chosen:
            seq.extend([u, v])
        return "\\boxed{" + " ".join(map(str, seq)) + "}"