import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Set
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BridgeEnv(Env):
    """Bridge (cut-edge) detection environment - single-turn Q&A.

    The task: Given an undirected graph with N vertices and a list of edges,
    find all bridge edges (u, v) such that removing (u, v) disconnects u and v.

    The answer must be provided in \\boxed{...} format, containing a space-separated
    list of vertex indices representing pairs: u1 v1 u2 v2 ... uk vk.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        component_num: Optional[int] = None,
        edge_density: Optional[float] = None,
        N_min: int = 2,
        N_max: int = 50,
        edge_density_min: float = 0.0,
        edge_density_max: float = 1.0,
        **kwargs
    ):
        """Initialize the BridgeEnv.

        Parameters:
        - N: Number of vertices. If None, a random N in [N_min, N_max] will be used.
        - component_num: Number of components used to generate the graph structure.
                         If None, a random value in [2, N] will be used.
        - edge_density: Target density in [0.0, 1.0]. If None, sampled uniformly within [edge_density_min, edge_density_max].
        - N_min, N_max: Range for random N if N is not provided.
        - edge_density_min, edge_density_max: Range for random edge density if edge_density is not provided.
        """
        super().__init__()
        # Parameters controlling problem generation
        assert N_min >= 2, "N_min should be at least 2"
        assert N_max >= N_min, "N_max should be greater than or equal to N_min"
        assert 0.0 <= edge_density_min <= 1.0, "edge_density_min should be in [0.0, 1.0]"
        assert 0.0 <= edge_density_max <= 1.0, "edge_density_max should be in [0.0, 1.0]"
        assert edge_density_max >= edge_density_min, "edge_density_max should be >= edge_density_min"

        self.N_fixed = N
        self.component_num_fixed = component_num
        self.edge_density_fixed = edge_density

        self.N_min = N_min
        self.N_max = N_max
        self.edge_density_min = edge_density_min
        self.edge_density_max = edge_density_max

        # State for current problem
        self.current_problem: Optional[str] = None
        self.reference_answer_str: Optional[str] = None
        self.reference_bridges: List[Tuple[int, int]] = []
        self.edges: List[Tuple[int, int]] = []
        self.N: int = 0

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given an undirected graph and must find all bridge edges.\n"
            "A bridge edge (u, v) is one whose removal disconnects u and v.\n"
            "Output Format: Provide your final answer as a space-separated sequence "
            "of vertex pairs u1 v1 u2 v2 ... uk vk inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            self.N = self.N_fixed
        else:
            self.N = random.randint(self.N_min, self.N_max)
        assert self.N >= 2, "N should be greater than or equal to 2"

        # Determine component_num
        if self.component_num_fixed is not None:
            component_num = self.component_num_fixed
            assert 2 <= component_num <= self.N, "component_num should be between 2 and N"
        else:
            component_num = random.randint(2, self.N)

        # Determine edge_density
        if self.edge_density_fixed is not None:
            edge_density = self.edge_density_fixed
            assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"
        else:
            edge_density = random.uniform(self.edge_density_min, self.edge_density_max)

        # Generate a graph with at least one bridge (retry if necessary)
        self.edges = []
        self.reference_bridges = []
        while True:
            # Assign vertices to components ensuring at least two components used
            while True:
                components = [random.randint(0, component_num - 1) for _ in range(self.N)]
                if len(set(components)) >= 2:
                    break

            component2vertices: List[List[int]] = [[] for _ in range(component_num)]
            for vertex, comp in enumerate(components):
                component2vertices[comp].append(vertex)

            edges: List[Tuple[int, int]] = []
            remaining_edges: List[Tuple[int, int]] = []

            previous_vertices: List[int] = []
            for comp in range(component_num):
                vertices = component2vertices[comp]
                if len(vertices) == 0:
                    continue
                if previous_vertices:
                    u = random.choice(previous_vertices)
                    v = random.choice(vertices)
                    edges.append((min(u, v), max(u, v)))
                # add potential intra-component edges to remaining pool
                for u in vertices:
                    for v in vertices:
                        if u < v:
                            remaining_edges.append((u, v))
                previous_vertices += vertices

            # Add additional edges based on density
            num_edges_target = int(edge_density * self.N * (self.N - 1) / 2)
            if len(edges) < num_edges_target:
                k = min(len(remaining_edges), num_edges_target - len(edges))
                if k > 0:
                    edges += random.sample(remaining_edges, k)
            random.shuffle(edges)

            # Validate edge uniqueness and ranges
            for u, v in edges:
                assert 0 <= u < v < self.N, "Edge vertices must satisfy 0 <= u < v < N"
            assert len(edges) == len(set(edges)), "Edges should be unique"

            # Compute bridges using Tarjan's algorithm
            bridges_set = self._compute_bridges(self.N, edges)

            if len(bridges_set) > 0:
                self.edges = edges
                self.reference_bridges = sorted(list(bridges_set))
                break
            # Otherwise, regenerate

        # Build reference answer string
        self.reference_answer_str = " ".join(f"{u} {v}" for u, v in self.reference_bridges)

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v})" for u, v in self.edges)
        example_two = " ".join(f"{u} {v}" for u, v in self.edges[:2])
        self.current_problem = (
            f"You are given an undirected graph with {self.N} vertices labeled from 0 to {self.N - 1}. "
            f"The graph contains the following undirected edges:\n{edges_str}\n\n"
            "Your task is to find all edges (u, v) such that removing the edge (u, v) from the graph "
            "would disconnect vertices u and v (which are initially connected).\n\n"
            f"Output Format: Assuming the edges are (u1, v1), (u2, v2), ..., (uk, vk), your final answer should be "
            f"a single line containing `u1 v1 u2 v2 ... uk vk`, where the vertices are separated by spaces. "
            f"Example: {example_two} (do NOT include quotes or backticks). "
            "Please provide your answer in \\boxed{...} format."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to parse integers
        try:
            tokens = boxed_content.strip().split()
            ints = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate pairing
        if len(ints) % 2 != 0:
            # Odd number of integers cannot form complete pairs
            reward = 0.0
            info = {"error": "odd_length", "user_raw": boxed_content}
            return TERMINAL_STATE, reward, True, False, info

        user_pairs: List[Tuple[int, int]] = []
        for i in range(0, len(ints), 2):
            u, v = ints[i], ints[i + 1]
            pair = (min(u, v), max(u, v))
            user_pairs.append(pair)

        # Duplicates are considered incorrect
        if len(user_pairs) != len(set(user_pairs)):
            reward = 0.0
            info = {"error": "duplicate_pairs", "user_pairs": user_pairs}
            return TERMINAL_STATE, reward, True, False, info

        user_set: Set[Tuple[int, int]] = set(user_pairs)
        gold_set: Set[Tuple[int, int]] = set(self.reference_bridges)

        is_correct = (user_set == gold_set)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_pairs": self.reference_bridges,
            "user_pairs": user_pairs,
            "num_vertices": self.N,
            "edges": self.edges,
            "reference_answer": self.reference_answer_str,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (answer). By default, returns the correct answer if available."""
        if self.reference_answer_str is not None:
            return f"\\boxed{{{self.reference_answer_str}}}"
        # Fallback: random empty or random pair
        return "\\boxed{}"

    @staticmethod
    def _compute_bridges(N: int, edges: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Compute all bridge edges using Tarjan's algorithm."""
        adj: List[List[int]] = [[] for _ in range(N)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        disc = [-1] * N
        low = [0] * N
        timer = 0
        bridges: Set[Tuple[int, int]] = set()

        def dfs(u: int, parent: int) -> None:
            nonlocal timer
            disc[u] = low[u] = timer
            timer += 1
            for v in adj[u]:
                if v == parent:
                    continue
                if disc[v] == -1:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.add((min(u, v), max(u, v)))
                else:
                    low[u] = min(low[u], disc[v])

        for u in range(N):
            if disc[u] == -1:
                dfs(u, -1)

        return bridges