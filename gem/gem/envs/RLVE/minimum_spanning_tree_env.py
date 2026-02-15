from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumSpanningTreeEnv(Env):
    """Minimum Spanning Tree problem environment - single-turn Q&A."""

    def __init__(
        self,
        N_min: int = 3,
        N_max: int = 10,
        edge_density: float = 0.5,
        **kwargs
    ):
        """
        Initialize the MinimumSpanningTreeEnv.

        Args:
            N_min: Minimum number of vertices (must be >= 3).
            N_max: Maximum number of vertices (must be >= N_min).
            edge_density: Edge density in [0.0, 1.0], determines the number of edges.
        """
        super().__init__()
        assert N_min >= 3, "N_min should be greater than or equal to 3"
        assert N_max >= N_min, "N_max should be greater than or equal to N_min"
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        self.N_min = N_min
        self.N_max = N_max
        self.edge_density = edge_density

        # State for the current problem instance
        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.edges: Optional[List[Tuple[int, int, int]]] = None
        self.reference_answer: Optional[str] = None
        self.gold_weight: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Minimum Spanning Tree (MST) problem on an undirected, weighted graph.\n"
            "Your task is to select exactly N-1 edges that connect all vertices (0..N-1) without cycles,\n"
            "and minimize the total weight. The selected edges must be chosen from the provided edge list.\n\n"
            "Output Format:\n"
            "- Provide your final answer inside \\boxed{...}.\n"
            "- Inside the box, list the endpoints of the selected edges in order: u1 v1 u2 v2 ... uk vk (space-separated).\n"
            "- Example: \\boxed{0 1 1 2 2 3} (for a graph with 4 vertices).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new MST problem instance."""
        super().reset(seed)

        # Sample problem size
        N = random.randint(self.N_min, self.N_max)
        assert N >= 3, "N should be greater than or equal to 3"

        # Generate edges ensuring connectivity and desired density
        edges: List[Tuple[int, int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)

        weight_upper_bound = max(1, int(self.edge_density * N * (N - 1) / 2))

        # First, build a random spanning tree to ensure connectivity
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            w = random.randint(1, weight_upper_bound)
            edges.append((u, v, w))

        # Determine target number of edges given density
        num_edges = int(self.edge_density * N * (N - 1) / 2)

        # Add additional random edges if needed (do not remove if len(edges) > num_edges)
        if len(edges) < num_edges:
            existing = set((u, v) for u, v, _ in edges)
            all_possible = set((u, v) for u in range(N) for v in range(u + 1, N))
            remaining_edges = list(all_possible - existing)
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges:
                w = random.randint(1, weight_upper_bound)
                edges.append((u, v, w))

        random.shuffle(edges)

        # Validate edges
        for u, v, _ in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, _ in edges)), "edges should be unique"

        # Compute MST using NetworkX
        G = networkx.Graph()
        G.add_weighted_edges_from(edges)
        mst = networkx.minimum_spanning_tree(G)
        reference_answer = " ".join("{} {}".format(u, v) for u, v in mst.edges())
        gold_weight = sum(mst[u][v]["weight"] for u, v in mst.edges())
        assert gold_weight > 0, "The gold answer should be greater than 0"

        # Store problem state
        self.N = N
        self.edges = edges
        self.reference_answer = reference_answer
        self.gold_weight = gold_weight

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v}, {w})" for u, v, w in edges)
        problem_text = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}. "
            f"The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w), "
            f"meaning an undirected edge connecting vertex u to vertex v with weight w:\n"
            f"{edges_str}\n\n"
            f"Your task is to select a subset of edges T = [(u1, v1, w1), (u2, v2, w2), ..., (uk, vk, wk)] such that:\n"
            f"- k = {N} - 1 = {N - 1} (i.e., you select exactly {N - 1} edges).\n"
            f"- The selected edges form a spanning tree — they connect all {N} vertices without forming any cycles.\n"
            f"- Your goal is to minimize the total weight of the selected edges.\n\n"
            f"Output Format: Your final answer should be the endpoints of the selected edges in order: "
            f"u1 v1 u2 v2 ... uk vk (space-separated), and should be wrapped in \\boxed{{...}}.\n"
            f"Example: \\boxed{{0 1 1 2 2 3}}"
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Validate the submitted MST answer."""
        # Extract content inside \boxed{...}
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure problem is initialized
        if self.N is None or self.edges is None or self.gold_weight is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Parse integers from boxed content
        tokens = boxed.strip().split()
        try:
            nums = [int(t) for t in tokens]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Must have even number of integers to form pairs
        if len(nums) % 2 != 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "uneven_endpoint_count"}

        # Build edge pairs
        pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]

        # Validate count equals N - 1
        if len(pairs) != self.N - 1:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution_edge_count"}

        # Validate vertex coverage
        vertices_used = set()
        for u, v in pairs:
            vertices_used.add(u)
            vertices_used.add(v)
        if vertices_used != set(range(self.N)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution_vertex_coverage"}

        # Validate edges exist and compute weight; also check connectivity and acyclicity
        subgraph = networkx.Graph()
        edge2weight: Dict[Tuple[int, int], int] = {(min(u, v), max(u, v)): w for u, v, w in self.edges}
        answer_weight = 0
        for u, v in pairs:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in edge2weight:
                return TERMINAL_STATE, 0.0, True, False, {"error": "edge_not_in_graph"}
            answer_weight += edge2weight[(a, b)]
            subgraph.add_edge(a, b)

        # Connectivity and tree structure
        if not networkx.is_connected(subgraph):
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_connected"}
        if not networkx.is_tree(subgraph):
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_a_tree"}

        # Correctness: minimal total weight (equal to gold_weight)
        is_correct = (answer_weight == self.gold_weight)
        reward = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "gold_weight": self.gold_weight,
            "user_weight": answer_weight,
            "N": self.N,
            "num_edges": len(self.edges),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a valid action; returns the reference MST as a correct solution if available."""
        if self.reference_answer is not None:
            return f"\\boxed{{{self.reference_answer}}}"
        # Fallback: random integers (likely incorrect)
        random_answer = "0 1"
        return f"\\boxed{{{random_answer}}}"