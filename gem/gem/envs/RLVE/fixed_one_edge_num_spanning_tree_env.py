from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FixedOneEdgeNum_SpanningTreeEnv(Env):
    """Environment for selecting a spanning tree with a fixed number of weight-1 edges.

    The environment generates an undirected graph with N vertices (0..N-1) and edges labeled with weight 0 or 1.
    The task is to select exactly N-1 edges that:
      - connect all vertices (form a spanning tree, i.e., connected and acyclic with N-1 edges),
      - contain exactly K edges with weight 1 (K is given in the prompt).

    The agent must output the answer in \\boxed{...} format as a space-separated list:
      u1 v1 u2 v2 ... u_{N-1} v_{N-1}
    """

    def __init__(
        self,
        N: int = 5,
        edge_ratio: float = 2.0,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: Number of vertices in the graph (must be >= 3).
            edge_ratio: Controls the total number of edges generated. Approximately int(edge_ratio * N) edges are created.
        """
        super().__init__()
        self.N: int = N
        self.edge_ratio: float = edge_ratio

        # Problem state
        self.edges: List[Tuple[int, int, int]] = []  # List of (u, v, w)
        self.K: int = 0  # Number of weight-1 edges required in the spanning tree
        self.reference_answer: Optional[str] = None  # One valid spanning tree as reference (not required for scoring)
        self.current_problem: Optional[str] = None

        # Validation
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 3, "N should be greater than or equal to 3"
        assert isinstance(self.edge_ratio, (int, float)), "edge_ratio must be a number"
        assert self.edge_ratio > 0, "edge_ratio must be positive"

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an undirected graph. Your task is to select a subset of edges that forms a spanning tree "
            "with exactly K edges having weight 1.\n"
            "Please provide your final answer in \\boxed{...} format containing the endpoints of the selected edges as "
            "space-separated integers: u1 v1 u2 v2 ... u_{N-1} v_{N-1}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: A string containing instructions and the problem statement.
            info: An empty dictionary.
        """
        super().reset(seed)

        # Generate graph with at least a spanning tree and additional edges
        N = self.N
        edge_ratio = self.edge_ratio

        edges: List[Tuple[int, int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)

        # Build a random spanning tree structure first
        one_probability = random.random()
        self.K = 0
        reference_pairs: List[str] = []
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            reference_pairs.append(f"{u} {v}")
            uu, vv = min(u, v), max(u, v)
            w = int(random.random() < one_probability)
            edges.append((uu, vv, w))
            self.K += w
        self.reference_answer = " ".join(reference_pairs)

        # Add additional edges up to approximately edge_ratio * N
        num_edges_target = int(edge_ratio * N)
        if len(edges) < num_edges_target:
            existing = set((u, v) for u, v, _ in edges)
            all_pairs = set((u, v) for u in range(N) for v in range(u + 1, N))
            remaining_pairs = list(all_pairs - existing)
            random.shuffle(remaining_pairs)
            remaining_pairs = remaining_pairs[: max(0, num_edges_target - len(edges))]

            one_probability = random.random()
            for u, v in remaining_pairs:
                w = int(random.random() < one_probability)
                edges.append((u, v, w))

        random.shuffle(edges)

        # Validations
        for u, v, w in edges:
            assert 0 <= u < v < N
            assert w in (0, 1), "edge weight should be either 0 or 1"
        assert len(edges) == len(set((u, v) for u, v, _ in edges)), "edges should be unique"

        self.edges = edges

        # Build prompt
        edges_str = "\n".join(f"({u}, {v}, {w})" for u, v, w in self.edges)
        problem = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}. "
            f"The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w) "
            f"(w is either 0 or 1), meaning an undirected edge connecting vertex u to vertex v with weight w:\n"
            f"{edges_str}\n\n"
            f"Please select a subset of edges T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)] such that:\n"
            f"- k = {N - 1} (i.e., you select exactly {N - 1} edges),\n"
            f"- The selected edges form a spanning tree — that is, they connect all {N} vertices without forming any cycles,\n"
            f"- There are exactly {self.K} edges with weight 1 in the selected edges.\n\n"
            f"Output Format: Your final answer should be a single line containing the endpoints of the selected edges in order, "
            f"enclosed in \\boxed{{...}}: u_1 v_1 u_2 v_2 ... u_k v_k, separated by spaces.\n"
            f"Example: \\boxed{{0 1 1 2 2 3}}"
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the user's answer.

        Args:
            action: The agent's answer text, expected to contain \\boxed{...} with a space-separated list of integers.

        Returns:
            observation: TERMINAL_STATE
            reward: 1.0 if correct, 0.0 otherwise; -0.1 for format errors.
            terminated: True (single-turn)
            truncated: False
            info: Additional info including correctness and diagnostics.
        """
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from the boxed content
        try:
            tokens = answer_text.strip().split()
            ints = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Build edge list from integer sequence
        if len(ints) % 2 != 0:
            info = {"error": "invalid_solution", "reason": "odd_number_of_integers"}
            return TERMINAL_STATE, 0.0, True, False, info

        mst_pairs = [(ints[i], ints[i + 1]) for i in range(0, len(ints), 2)]

        # Validate size
        if len(mst_pairs) != self.N - 1:
            info = {
                "error": "invalid_solution",
                "reason": "wrong_number_of_edges",
                "expected": self.N - 1,
                "got": len(mst_pairs)
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate vertex coverage
        vertices_used = set(u for u, _ in mst_pairs) | set(v for _, v in mst_pairs)
        if vertices_used != set(range(self.N)):
            info = {
                "error": "invalid_solution",
                "reason": "not_spanning_all_vertices",
                "expected_vertices": list(range(self.N)),
                "used_vertices": sorted(vertices_used)
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate edges exist and check connectivity
        edge2weight: Dict[Tuple[int, int], int] = {(u, v): w for u, v, w in self.edges}
        # Ensure mapping is normalized
        edge2weight.update({(v, u): w for (u, v), w in list(edge2weight.items())})

        subgraph = networkx.Graph()
        answer_weight = 0
        for u, v in mst_pairs:
            uu, vv = min(u, v), max(u, v)
            if (uu, vv) not in edge2weight:
                info = {
                    "error": "invalid_solution",
                    "reason": "edge_not_in_graph",
                    "missing_edge": (uu, vv)
                }
                return TERMINAL_STATE, 0.0, True, False, info
            answer_weight += edge2weight[(uu, vv)]
            subgraph.add_edge(uu, vv)

        # Ensure all nodes included (coverage already ensures this)
        if not networkx.is_connected(subgraph):
            info = {
                "error": "invalid_solution",
                "reason": "not_connected"
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check weight constraint
        is_correct = (answer_weight == self.K)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "required_weight_1_edges": self.K,
            "user_weight_1_edges": answer_weight,
            "num_vertices": self.N,
            "num_edges_required": self.N - 1,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the correct \\boxed{...} format."""
        # Randomly pick N-1 edges from the graph (not guaranteed to be a valid spanning tree)
        if not self.edges or self.N < 2:
            return r"\boxed{}"
        chosen = random.sample(self.edges, k=min(len(self.edges), self.N - 1))
        flat = []
        for u, v, _ in chosen:
            flat.extend([u, v])
        answer = " ".join(map(str, flat))
        return f"\\boxed{{{answer}}}"