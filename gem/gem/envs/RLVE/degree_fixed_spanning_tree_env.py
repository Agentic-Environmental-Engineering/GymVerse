import random
from typing import Any, Optional, SupportsFloat, Tuple, List
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DegreeFixed_SpanningTreeEnv(Env):
    """Single-turn environment for finding a spanning tree with fixed vertex degrees."""

    prompt_template = (
        "You are given an undirected graph with {N} vertices, labeled from 0 to {N_minus_1}.\n\n"
        "The graph contains the following undirected edges. Each edge is represented as a tuple (u, v), "
        "meaning an undirected edge connecting vertex u to vertex v:\n"
        "{edges}\n\n"
        "Your task is to select a subset of edges T = [(u_1, v_1), (u_2, v_2), ..., (u_k, v_k)] such that:\n"
        "- The selected edges form a spanning tree — that is, they connect all {N} vertices without forming any cycles.\n"
        "- Each vertex i has a fixed degree of d_i, meaning it must be connected to exactly d_i edges in the selected subset: {degrees}\n\n"
        "Output Format:\n"
        "Your final answer should be a single line containing the endpoints of the selected edges in order, "
        "inside \\boxed{{...}}: u_1 v_1 u_2 v_2 ... u_k v_k (separated by spaces).\n"
        "Example: \\boxed{{0 1 1 2 2 3}}; this means the spanning tree includes the edges (0, 1), (1, 2), and (2, 3)."
    )

    def __init__(self, N: int = 5, edge_density: float = 0.5, **kwargs):
        """
        Initialize the environment.

        Parameters:
        - N: Number of vertices in the graph (must be >= 3).
        - edge_density: Desired density of edges in the undirected graph (must be between 0.0 and 1.0).
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"
        assert isinstance(edge_density, (int, float)), "edge_density must be a number"
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        self.N: int = N
        self.edge_density: float = float(edge_density)

        # Problem state variables
        self.edges: List[Tuple[int, int]] = []
        self.degrees: List[int] = []
        self.reference_answer: str = ""
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a graph spanning tree problem with fixed vertex degrees.\n"
            "Please provide your answer enclosed in \\boxed{...}.\n"
            "Inside the box, list the endpoints of the selected edges in order, separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        edge_density = self.edge_density

        # Generate a random spanning tree as the reference solution, then add extra edges to reach target density.
        degrees: List[int] = [0] * N
        edges: List[Tuple[int, int]] = []
        reference_answer_parts: List[str] = []

        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            reference_answer_parts.append(f"{u} {v}")
            uu, vv = (u, v) if u < v else (v, u)
            edges.append((uu, vv))
            degrees[uu] += 1
            degrees[vv] += 1

        self.reference_answer = " ".join(reference_answer_parts)

        # Determine target number of edges by density
        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges:
            existing = set(edges)
            all_possible = {(u, v) for u in range(N) for v in range(u + 1, N)}
            remaining_edges = list(all_possible - existing)
            k = min(len(remaining_edges), num_edges - len(edges))
            if k > 0:
                edges += random.sample(remaining_edges, k)

        random.shuffle(edges)
        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"

        # Save generated data
        self.edges = edges
        self.degrees = degrees

        # Build the problem prompt
        edges_str = "\n".join(f"({u}, {v})" for u, v in self.edges)
        degrees_str = ", ".join(f"d_{i}={d}" for i, d in enumerate(self.degrees))
        self.current_problem = self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            edges=edges_str,
            degrees=degrees_str,
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided answer and return the result."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from the boxed content
        try:
            tokens = boxed_content.strip().split()
            st_list = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # The list length must be even to form pairs
        if len(st_list) % 2 != 0:
            info = {
                "correct": False,
                "reason": "odd_number_of_tokens",
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Form edge pairs
        st_pairs: List[Tuple[int, int]] = [(st_list[i], st_list[i + 1]) for i in range(0, len(st_list), 2)]

        N = self.N

        # Must contain exactly N-1 edges
        if len(st_pairs) != N - 1:
            info = {
                "correct": False,
                "reason": "wrong_number_of_edges",
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # All vertices must appear
        vertices_used = set(u for u, _ in st_pairs) | set(v for _, v in st_pairs)
        if vertices_used != set(range(N)):
            info = {
                "correct": False,
                "reason": "not_all_vertices_covered",
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate edges exist in the given graph
        edge_set = set(self.edges)
        subgraph = networkx.Graph()
        for u, v in st_pairs:
            uu, vv = (u, v) if u < v else (v, u)
            if (uu, vv) not in edge_set:
                info = {
                    "correct": False,
                    "reason": "edge_not_in_graph",
                    "invalid_edge": (uu, vv),
                    "reference_answer": self.reference_answer,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            subgraph.add_edge(uu, vv)

        # Check connectivity and tree property
        if not networkx.is_connected(subgraph):
            info = {
                "correct": False,
                "reason": "not_connected",
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info
        if not networkx.is_tree(subgraph):
            info = {
                "correct": False,
                "reason": "not_a_tree",
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Degree check
        degrees_answer = [0] * N
        for u, v in subgraph.edges():
            degrees_answer[u] += 1
            degrees_answer[v] += 1

        is_correct = all(da == dg for da, dg in zip(degrees_answer, self.degrees))
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": boxed_content,
            "target_degrees": self.degrees,
            "user_degrees": degrees_answer,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns the last match if multiple exist."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a plausible action (the reference spanning tree) in boxed format."""
        if not self.reference_answer:
            # In case called before reset
            return "\\boxed{}"
        return f"\\boxed{{{self.reference_answer}}}"