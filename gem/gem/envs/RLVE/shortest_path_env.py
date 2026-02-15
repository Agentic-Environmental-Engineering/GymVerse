import random
import networkx
import re
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ShortestPathEnv(Env):
    """Shortest path problem environment - single-turn Q&A."""

    def __init__(
        self,
        N: int = 6,
        edge_density: float = 0.4,
        **kwargs
    ):
        super().__init__()
        self.N = N
        self.edge_density = edge_density

        # Internal state
        self.current_problem: Optional[str] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.reference_answer_weight: Optional[int] = None
        self.reference_path: Optional[List[int]] = None
        self.reference_path_str: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving shortest path problems on directed weighted graphs.\n"
            "Please provide your final path as space-separated vertex indices, enclosed in \\boxed{...}.\n"
            "Example: \\boxed{0 1 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Parameter validation
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 4, "N should be greater than or equal to 4"

        assert isinstance(self.edge_density, (int, float)), "edge_density must be a number"
        assert 0.0 <= float(self.edge_density) <= 1.0, "edge_density should be between 0.0 and 1.0"

        N = self.N
        edge_density = float(self.edge_density)
        edges: List[Tuple[int, int, int]] = []

        # Construct a guaranteed path from 0 to N-1 through a shuffled sequence of intermediate vertices
        constructed_path = list(range(1, N - 1))
        random.shuffle(constructed_path)
        constructed_path = [0] + constructed_path + [N - 1]
        for s, t in zip(constructed_path, constructed_path[1:]):
            w = random.randint(1, max(1, N // 3))
            edges.append((s, t, w))

        # Add additional random edges according to edge density (without duplications and avoiding (0, N-1))
        num_edges = int(edge_density * N * (N - 1))
        if len(edges) < num_edges:
            existing_pairs = set((s, t) for s, t, _ in edges)
            all_possible_pairs = set((s, t) for s in range(N) for t in range(N) if s != t)
            remaining_pairs = list(all_possible_pairs - existing_pairs - {(0, N - 1)})
            remaining_pairs = random.sample(remaining_pairs, min(len(remaining_pairs), num_edges - len(edges)))
            for s, t in remaining_pairs:
                edges.append((s, t, random.randint(max(1, N // 2), N)))
        random.shuffle(edges)

        # Remove certain edges to avoid trivial two-edge path 0 -> t -> N-1 except for the constructed final step
        starting = {t: (s, t, w) for s, t, w in edges if s == 0}
        ending = {s: (s, t, w) for s, t, w in edges if t == N - 1}
        for s, t, w in list(starting.values()):
            if t in ending:
                if t == constructed_path[-2]:
                    assert t != constructed_path[1]
                    if starting[t] in edges:
                        edges.remove(starting[t])
                else:
                    if ending[t] in edges:
                        edges.remove(ending[t])

        # Sanity checks: unique edges and valid ranges
        assert len(edges) == len(set((s, t) for s, t, w in edges)), "edges should be unique"
        for s, t, w in edges:
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"

        # Build graph and compute reference shortest path using Dijkstra
        G = networkx.DiGraph()
        G.add_weighted_edges_from(edges)
        shortest_path_length, shortest_path = networkx.single_source_dijkstra(G, 0, N - 1)

        # Save internal state
        self.edges = edges
        self.reference_answer_weight = int(shortest_path_length)
        self.reference_path = list(shortest_path)
        self.reference_path_str = " ".join(map(str, shortest_path))

        # Build problem prompt
        edges_str = "\n".join("({}, {}, {})".format(s, t, w) for s, t, w in edges)
        problem_text = (
            f"You are given a directed graph with {N} vertices, labeled from 0 to {N - 1}.\n\n"
            "The graph contains the following directed edges. Each edge is represented as a tuple (s, t, w), "
            "meaning there is a directed edge from vertex s to vertex t with weight w:\n"
            f"{edges_str}\n\n"
            "Your task is to find a path p1, p2, ..., pk such that:\n"
            f"- p1 = 0 (the path starts at vertex 0) and pk = {N - 1} (the path ends at vertex {N - 1})\n"
            "- Try your best to minimize the total weight of the path (i.e., the sum of all edge weights used).\n\n"
            "Output Format:\n"
            "Your final answer should be a single line containing the path in order: p1 p2 ... pk, separated by spaces, "
            "and enclosed in \\boxed{...}.\n"
            f"Example: \\boxed{{0 1 {N - 1}}} (do NOT include quotes)."
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error: no \boxed{...} content
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the vertex path from boxed content
        path_tokens = boxed_content.strip().split()
        try:
            path = list(map(int, path_tokens))
            if not path:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "detail": "empty_path"}
        except ValueError:
            # Not all tokens are integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "detail": "non_integer_tokens"}

        # Validate path vertices are within range
        N = self.N
        for vertex in path:
            if not (0 <= vertex < N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "detail": "vertex_out_of_range"}

        # Validate start and end vertices
        if not (path[0] == 0 and path[-1] == N - 1):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "detail": "wrong_start_or_end"}

        # Validate edges and compute total weight
        edge2weight: Dict[Tuple[int, int], int] = {(s, t): w for s, t, w in self.edges}
        answer_weight = 0
        for s, t in zip(path, path[1:]):
            if (s, t) not in edge2weight:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "detail": "missing_edge"}
            answer_weight += edge2weight[(s, t)]

        gold = self.reference_answer_weight
        assert gold is not None, "Reference answer not computed; call reset() first."
        assert 0 < gold <= answer_weight, "Answer weight should be greater than or equal to reference."

        is_correct = (answer_weight == gold)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer_weight": gold,
            "reference_path": self.reference_path,
            "reference_path_str": self.reference_path_str,
            "user_path": path,
            "user_weight": answer_weight,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (returns the reference path if available)."""
        if self.reference_path_str is not None:
            return f"\\boxed{{{self.reference_path_str}}}"
        # Fallback: trivial attempt (likely invalid before reset)
        return "\\boxed{0}"