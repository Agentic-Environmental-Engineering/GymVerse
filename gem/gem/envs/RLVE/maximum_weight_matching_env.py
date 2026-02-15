from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import re
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaximumWeightMatchingEnv(Env):
    """Maximum Weight Matching environment - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        edge_density: Optional[float] = None,
        min_N: int = 2,
        max_N: int = 20,
        min_edge_density: float = 0.1,
        max_edge_density: float = 1.0,
        **kwargs
    ):
        """
        Initialize the MaximumWeightMatchingEnv.

        Parameters:
        - N: If provided, the number of vertices in the graph (must be >= 2).
             If None, N will be sampled uniformly from [min_N, max_N].
        - edge_density: If provided, the density of edges (between 0.0 and 1.0).
                        If None, density will be sampled uniformly from [min_edge_density, max_edge_density].
        - min_N, max_N: Range for random N when N is not provided.
        - min_edge_density, max_edge_density: Range for random edge_density when it is not provided.
        """
        super().__init__()
        self.fixed_N = N
        self.fixed_edge_density = edge_density
        self.min_N = min_N
        self.max_N = max_N
        self.min_edge_density = min_edge_density
        self.max_edge_density = max_edge_density

        # Runtime attributes populated during reset
        self.N: Optional[int] = None
        self.edge_density: Optional[float] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_weight: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an undirected weighted graph. Your task is to select a subset of edges forming a matching "
            "(no two selected edges share a vertex) to maximize the total weight.\n"
            "Please provide your answer in \\boxed{...} format as a sequence of vertex endpoints separated by spaces.\n"
            "For example: \\boxed{0 1 3 4} corresponds to selecting edges (0,1) and (3,4).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            assert self.fixed_N >= 2, "N should be greater than or equal to 2"
            self.N = self.fixed_N
        else:
            self.N = random.randint(self.min_N, self.max_N)
            assert self.N >= 2, "N should be greater than or equal to 2"

        # Determine edge density
        if self.fixed_edge_density is not None:
            assert 0.0 <= self.fixed_edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"
            self.edge_density = self.fixed_edge_density
        else:
            self.edge_density = random.uniform(self.min_edge_density, self.max_edge_density)
            assert 0.0 <= self.edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        # Generate edges
        all_possible = [(u, v, random.randint(1, self.N)) for u in range(self.N) for v in range(u + 1, self.N)]
        k = int(self.edge_density * self.N * (self.N - 1) / 2)
        k = min(k, len(all_possible))
        self.edges = random.sample(all_possible, k)
        random.shuffle(self.edges)

        # Validate uniqueness and bounds
        assert len(self.edges) == len(set((u, v) for u, v, w in self.edges)), "edges should be unique"
        for u, v, w in self.edges:
            assert 0 <= u < v < self.N, "edge endpoints must satisfy 0 <= u < v < N"

        # Compute optimal matching using NetworkX
        G = networkx.Graph()
        G.add_weighted_edges_from(self.edges)
        matching = networkx.max_weight_matching(G, maxcardinality=False)

        # Prepare reference answer and gold weight
        self.reference_answer = " ".join(f"{u} {v}" for u, v in matching)
        edge2weight = {(min(u, v), max(u, v)): w for u, v, w in self.edges}
        self.gold_weight = sum(edge2weight[(min(u, v), max(u, v))] for u, v in matching)

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v}, {w})" for u, v, w in self.edges)
        self.current_problem = (
            f"You are given an undirected graph with {self.N} vertices, labeled from 0 to {self.N - 1}.\n\n"
            f"The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w), "
            f"meaning an undirected edge connecting vertex u to vertex v with weight w:\n{edges_str}\n\n"
            "Your task is to select a subset of edges S = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)] such that:\n"
            "- Each selected edge must exist in the graph.\n"
            "- Each vertex appears in at most one edge in the set S — in other words, no two edges in S share a vertex.\n"
            "- Your goal is to maximize the total weight of the selected edges w_1 + w_2 + ... + w_k.\n\n"
            "Output Format:\n"
            "Your final answer should be the endpoints of the selected edges in order: u_1 v_1 u_2 v_2 ... u_k v_k, "
            "inside \\boxed{...} and separated by spaces.\n"
            "Example: \\boxed{0 1 3 4}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Verify the provided answer."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error (no boxed content)
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to parse a sequence of integers
        tokens = boxed_content.strip().split()
        try:
            numbers = list(map(int, tokens))
        except ValueError:
            # Contents are not all integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Must be even length to form pairs
        if len(numbers) % 2 != 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer_length"}

        # Form pairs
        matches = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]

        # Check vertex uniqueness (no two selected edges share a vertex)
        used_vertices = []
        for u, v in matches:
            used_vertices.append(u)
            used_vertices.append(v)
        if len(set(used_vertices)) != len(used_vertices):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution_vertices_reused"}

        # Validate edges and compute total weight
        edge2weight = {(min(u, v), max(u, v)): w for u, v, w in self.edges}
        answer_weight = 0
        for u, v in matches:
            a, b = min(u, v), max(u, v)
            if (a, b) not in edge2weight:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution_edge_not_exist"}
            answer_weight += edge2weight[(a, b)]

        # Compare to optimal (gold) weight
        is_correct = (self.gold_weight is not None and answer_weight == self.gold_weight)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_pairs": matches,
            "reference_weight": self.gold_weight,
            "user_weight": answer_weight,
            "N": self.N,
            "edge_density": self.edge_density,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random matching and format it in boxed form."""
        # Greedy random matching construction
        used = set()
        random_edges = self.edges[:]
        random.shuffle(random_edges)
        selected_pairs: List[Tuple[int, int]] = []
        for u, v, w in random_edges:
            if u not in used and v not in used:
                selected_pairs.append((u, v))
                used.add(u)
                used.add(v)

        flat = []
        for u, v in selected_pairs:
            flat.extend([u, v])

        return f"\\boxed{{{' '.join(map(str, flat))}}}"