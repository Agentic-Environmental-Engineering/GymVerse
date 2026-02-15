import random
from typing import Any, Optional, SupportsFloat, Tuple, List
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MultiDrinkEnv(Env):
    """Tree permutation environment: find a vertex permutation where consecutive vertices are at distance at most 2."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 100,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, use this fixed number of vertices. Must be at least 4.
        - min_N: Minimum number of vertices if N is not provided. Must be at least 4.
        - max_N: Maximum number of vertices if N is not provided. Must be at least min_N.
        """
        super().__init__()
        assert min_N >= 4, "min_N must be at least 4"
        assert max_N >= min_N, "max_N must be at least min_N"
        if N is not None:
            assert N >= 4, "N must be at least 4"

        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N

        self.current_problem: Optional[str] = None
        self.reference_answer_seq: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.edges: Optional[List[tuple[int, int]]] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a tree permutation problem.\n"
            "Given a tree with vertices labeled from 0 to N-1 and the list of edges, "
            "find a permutation p[0], p[1], ..., p[N-1] such that for every i, the distance "
            "between p[i] and p[i+1] (in number of edges) is at most 2.\n"
            "Output Format: Provide the permutation as space-separated integers inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
        assert N >= 4, "N must be at least 4"
        self.N = N

        # Generate a random tree with the special construction ensuring a valid permutation
        edges: List[tuple[int, int]] = []
        neighbors: List[List[int]] = [[] for _ in range(N)]

        def add_edge(u: int, v: int) -> None:
            edges.append((min(u, v), max(u, v)))
            neighbors[u].append(v)
            neighbors[v].append(u)

        # Initialize paths (each vertex starts as a separate path)
        paths: List[List[int]] = [[u] for u in range(N)]
        while len(paths) > 1:
            # Select two distinct paths with probability proportional to their lengths
            while True:
                i, j = random.choices(range(len(paths)), k=2, weights=[len(path) for path in paths])
                if i != j:
                    break
            path_i, path_j = paths[i], paths[j]
            a, b = path_i[-1], path_j[0]
            # Connect a to b or to a neighbor of b (or symmetrically) to ensure distance <= 2
            if random.random() < 0.5:
                add_edge(a, random.choice([b] + neighbors[b]))
            else:
                add_edge(b, random.choice([a] + neighbors[a]))
            # Merge the two paths
            paths = [path for index, path in enumerate(paths) if index not in (i, j)] + [path_i + path_j]

        # Reference answer is the merged path
        reference_seq = paths[0]
        self.reference_answer_seq = reference_seq
        self.reference_answer_str = " ".join(map(str, reference_seq))

        # Shuffle edges for presentation
        random.shuffle(edges)

        # Validate edges
        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"

        # Validate it is a tree
        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree)

        self.edges = edges

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        self.current_problem = (
            f"There is a tree (i.e., a connected undirected graph with no cycles) with {N} vertices "
            f"labeled from 0 to {N - 1}. Its edges are:\n{edges_str}\n\n"
            f"Please find a permutation of the vertices p[0], p[1], ..., p[{N - 1}] such that for every pair "
            f"(p[i], p[i + 1]) with 0 ≤ i < {N - 1}, the distance between p[i] and p[i + 1] in the tree "
            f"(measured in number of edges) is at most 2.\n\n"
            f"Output Format: Your final answer should be the permutation as space-separated integers inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step by verifying the submitted answer."""
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to list of integers
        try:
            tokens = boxed.strip().split()
            user_seq = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.N
        edges = self.edges
        assert N is not None and edges is not None

        # Validate permutation length and content
        if len(user_seq) != N or set(user_seq) != set(range(N)):
            info = {
                "error": "invalid_solution",
                "expected_length": N,
                "unique_vertices_required": True
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Build neighbor sets
        neighbor_sets: List[set[int]] = [set() for _ in range(N)]
        for u, v in edges:
            neighbor_sets[u].add(v)
            neighbor_sets[v].add(u)

        # Check distance <= 2 for each consecutive pair
        violations: List[int] = []
        for i in range(N - 1):
            a, b = user_seq[i], user_seq[i + 1]
            direct = (b in neighbor_sets[a])
            two_step = len(neighbor_sets[a] & neighbor_sets[b]) > 0
            if not (direct or two_step):
                violations.append(i)

        is_correct = (len(violations) == 0)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_seq,
            "reference_answer_str": self.reference_answer_str,
            "user_answer": user_seq,
            "violations_indices": violations,
            "N": N,
            "edges": edges,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action. For convenience, return the reference answer if available, otherwise a random permutation."""
        if self.reference_answer_str is not None:
            return f"\\boxed{{{self.reference_answer_str}}}"
        if self.N is not None:
            perm = list(range(self.N))
            random.shuffle(perm)
            return f"\\boxed{{{' '.join(map(str, perm))}}}"
        # Fallback: empty boxed
        return "\\boxed{}"