from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SubgraphIsomorphismEnv(Env):
    """Subgraph Isomorphism environment - single-turn Q&A.

    The task: Given two undirected graphs G1 and G2, find an injection p from vertices of G1 to vertices of G2
    such that for every pair (u, v), edge (u, v) exists in G1 if and only if edge (p(u), p(v)) exists in G2.

    Answer format: The agent must output the mapping p(0), p(1), ..., p(N1-1) as space-separated integers,
    wrapped in \\boxed{...}.
    """

    def __init__(
        self,
        N2: int = 6,
        edge_density: float = 0.5,
        **kwargs
    ):
        super().__init__()
        # Parameters controlling problem generation
        self.N2_default = N2
        self.edge_density = edge_density

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.reference_mapping: Optional[List[int]] = None
        self.N1: Optional[int] = None
        self.N2: Optional[int] = None
        self.G1_edges: Optional[List[Tuple[int, int]]] = None
        self.G2_edges: Optional[List[Tuple[int, int]]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving subgraph isomorphism problems for undirected graphs.\n"
            "You must find an injection p from vertices of G1 to vertices of G2 such that adjacency is preserved:\n"
            "For any pair (u, v), (u, v) is an edge in G1 if and only if (p(u), p(v)) is an edge in G2.\n"
            "Output Format: Provide your final answer as space-separated integers p(0) p(1) ... p(N1-1) inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate parameters
        N2 = self.N2_default
        assert isinstance(N2, int), "N2 must be an integer"
        assert N2 >= 3, "N2 should be greater than or equal to 3"

        edge_density = self.edge_density
        assert isinstance(edge_density, float) or isinstance(edge_density, int), "edge_density must be a number"
        edge_density = float(edge_density)
        assert 0.0 < edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"
        total_possible_edges = N2 * (N2 - 1) // 2
        num_edges_G2 = int(edge_density * total_possible_edges)
        assert num_edges_G2 > 0, "edge_density too small for given N2; results in zero edges"

        # Generate G2 edges
        all_pairs_G2 = [(u, v) for u in range(N2) for v in range(u + 1, N2)]
        G2_edges = random.sample(all_pairs_G2, num_edges_G2)
        random.shuffle(G2_edges)

        # Generate N1 and mapping from G1 to G2
        N1 = random.randint(3, N2)
        mapping = random.sample(range(N2), N1)
        random.shuffle(mapping)

        # Generate G1 edges induced by mapping from G2 edges
        G1_edges: List[Tuple[int, int]] = []
        G2_edges_set = set(G2_edges)
        for u in range(N1):
            for v in range(u + 1, N1):
                G2_u, G2_v = mapping[u], mapping[v]
                if G2_u > G2_v:
                    G2_u, G2_v = G2_v, G2_u
                if (G2_u, G2_v) in G2_edges_set:
                    G1_edges.append((u, v))
        random.shuffle(G1_edges)

        # Sanity checks on edges
        for edges, N in zip((G1_edges, G2_edges), (N1, N2)):
            for u, v in edges:
                assert 0 <= u < v < N, "Edge endpoints out of valid range"
            assert len(edges) == len(set(edges)), "Edges should be unique"

        # Store internal state
        self.N1 = N1
        self.N2 = N2
        self.G1_edges = G1_edges
        self.G2_edges = G2_edges
        self.reference_mapping = mapping
        self.reference_answer = " ".join(map(str, mapping))

        # Build the problem statement
        N1_minus_1, N2_minus_1 = N1 - 1, N2 - 1
        problem_text = (
            "You are given two undirected graphs, G1 and G2.\n\n"
            f"- G1 has {N1} vertices labeled from 0 to {N1_minus_1}. It has the following edge set E1:\n"
            + "\n".join(f"({u}, {v})" for u, v in G1_edges) + "\n\n"
            f"- G2 has {N2} vertices labeled from 0 to {N2_minus_1}. It has the following edge set E2:\n"
            + "\n".join(f"({u}, {v})" for u, v in G2_edges) + "\n\n"
            "Please find an injection p from the vertices of G1 to the vertices of G2. "
            "This mapping p must satisfy the following condition: for every pair (u, v), "
            "the edge (u, v) exists in E1 if and only if the edge (p(u), p(v)) exists in E2.\n\n"
            "Output Format: Your final answer should be a single line containing p(0), p(1), ..., p("
            f"{N1_minus_1}) separated by spaces, wrapped in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the submitted mapping."""
        if self.N1 is None or self.N2 is None or self.G1_edges is None or self.G2_edges is None:
            # Environment was not properly reset before calling step
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse mapping list from boxed content
        user_mapping_list: List[int]
        try:
            tokens = boxed_content.strip().split()
            user_mapping_list = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Basic validation
        if len(user_mapping_list) != self.N1:
            info = {
                "correct": False,
                "error": "invalid_length",
                "expected_length": self.N1,
                "user_length": len(user_mapping_list),
                "reference_answer": self.reference_answer,
                "user_answer": boxed_content,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if len(set(user_mapping_list)) != self.N1:
            info = {
                "correct": False,
                "error": "non_injective_mapping",
                "reference_answer": self.reference_answer,
                "user_answer": user_mapping_list,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(0 <= i < self.N2 for i in user_mapping_list):
            info = {
                "correct": False,
                "error": "out_of_range_vertices",
                "reference_answer": self.reference_answer,
                "user_answer": user_mapping_list,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Verify adjacency preservation for all pairs
        G1_edge_set = set(self.G1_edges)
        G2_edge_set = set(self.G2_edges)

        correct = True
        for u in range(self.N1):
            for v in range(u + 1, self.N1):
                G1_has = (u, v) in G1_edge_set
                a, b = user_mapping_list[u], user_mapping_list[v]
                if a > b:
                    a, b = b, a
                G2_has = (a, b) in G2_edge_set
                if G1_has != G2_has:
                    correct = False
                    break
            if not correct:
                break

        reward: float = 1.0 if correct else 0.0
        info = {
            "correct": correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_mapping_list,
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
        """Sample a random action: a random injective mapping wrapped in \\boxed{...}."""
        if self.N2 is None:
            N2 = self.N2_default
        else:
            N2 = self.N2

        if self.N1 is None:
            N1 = random.randint(3, N2)
        else:
            N1 = self.N1

        mapping = random.sample(range(N2), N1)
        random.shuffle(mapping)
        mapping_str = " ".join(map(str, mapping))
        return f"\\boxed{{{mapping_str}}}"