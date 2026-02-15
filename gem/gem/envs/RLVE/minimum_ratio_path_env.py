from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumRatioPathEnv(Env):
    """Environment for the Minimum Ratio Path problem (single-turn Q&A).

    The task is:
    - Given an undirected graph with N vertices labeled 0 to N-1, and a list of edges (u, v, w).
    - Find a path from 0 to N-1 that minimizes the ratio max(w) / min(w) along the path.

    The agent must return the path vertices in order, in \\boxed{...} format, separated by spaces.
    Example: \\boxed{0 1 4 7}
    """

    def __init__(
        self,
        N: Optional[int] = None,
        edge_ratio: Optional[float] = None,
        n_min: int = 3,
        n_max: int = 25,
        edge_ratio_min: float = 1.0,
        edge_ratio_max: float = 3.0,
        weight_range_multiple: int = 5,
        **kwargs
    ):
        super().__init__()
        # Parameter configuration
        self.N_fixed = N
        self.edge_ratio_fixed = edge_ratio
        self.n_min = n_min
        self.n_max = n_max
        self.edge_ratio_min = edge_ratio_min
        self.edge_ratio_max = edge_ratio_max
        self.weight_range_multiple = weight_range_multiple

        # Runtime state
        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.gold_answer: Optional[Tuple[int, int]] = None  # (numerator, denominator)

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a graph path optimization problem.\n"
            "Task: Find a path from vertex 0 to vertex N-1 that minimizes the ratio of the maximum edge weight to the minimum edge weight along the path.\n"
            "Answer Format: Provide your path as a sequence of vertex indices separated by spaces, wrapped in \\boxed{...}.\n"
            "Example: \\boxed{0 3 5 7}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N and edge_ratio
        if self.N_fixed is not None:
            self.N = self.N_fixed
        else:
            self.N = random.randint(self.n_min, self.n_max)
        assert self.N is not None and self.N >= 3, "N should be greater than or equal to 3"

        if self.edge_ratio_fixed is not None:
            edge_ratio = self.edge_ratio_fixed
        else:
            edge_ratio = random.uniform(self.edge_ratio_min, self.edge_ratio_max)

        # Generate edges
        edges: List[Tuple[int, int, int]] = []
        constructed_path = list(range(1, (self.N - 2) + 1))
        random.shuffle(constructed_path)
        constructed_path = [0] + constructed_path + [self.N - 1]
        assert set(constructed_path) == set(range(self.N)), "constructed_path should contain all vertices from 0 to N-1"

        # Add edges along a shuffled Hamiltonian-like path to ensure connectivity
        max_weight = max(1, int(self.N * edge_ratio) * self.weight_range_multiple)
        for u, v in zip(constructed_path, constructed_path[1:]):
            w = random.randint(1, max_weight)
            edges.append((min(u, v), max(u, v), w))

        # Add extra edges up to num_edges
        num_edges = int(self.N * edge_ratio)
        if len(edges) < num_edges:
            existing_pairs = set((u, v) for u, v, _ in edges)
            all_pairs = set((u, v) for u in range(self.N) for v in range(u + 1, self.N) if (u, v) != (0, self.N - 1))
            remaining_edges = list(all_pairs - existing_pairs)
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges:
                edges.append((u, v, random.randint(1, max_weight)))

        random.shuffle(edges)

        # Validate edges
        for u, v, w in edges:
            assert 0 <= u < v < self.N, "Edge endpoints must satisfy 0 <= u < v < N"
        assert len(edges) == len(set((u, v) for u, v, _ in edges)), "edges should be unique"

        # Compute the optimal ratio using union-find over sorted edges by weight
        edges_sorted = sorted([(w, u, v) for (u, v, w) in edges], key=lambda x: x[0])
        M = len(edges_sorted)
        S, T = 0, self.N - 1

        ans_num = 0
        ans_den = 1
        found_any = False

        def find(parent: List[int], x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent, parent[x])
            return parent[x]

        for i in range(M):
            parent = list(range(self.N))
            wj = edges_sorted[i][0]
            for j in range(i, M):
                wj, uj, vj = edges_sorted[j]
                fu = find(parent, uj)
                fv = find(parent, vj)
                if fu != fv:
                    parent[fu] = fv
                if find(parent, S) == find(parent, T):
                    break

            if find(parent, S) != find(parent, T):
                if i == 0:
                    # If even after adding all edges from i onward s and t aren't connected, something is wrong.
                    # For robustness in GEM, do not assert here; just stop the search.
                    pass
                break

            wi = edges_sorted[i][0]
            if not found_any or ans_num * wi >= ans_den * wj:
                ans_num = wj
                ans_den = wi
                found_any = True

        self.edges = edges
        self.gold_answer = (ans_num, ans_den)

        # Build problem statement
        edges_text = "\n".join(f"({u}, {v}, {w})" for (u, v, w) in self.edges)
        self.current_problem = (
            f"You are given an undirected graph with {self.N} vertices, labeled from 0 to {self.N - 1}.\n\n"
            f"The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w), "
            f"meaning there is an undirected edge connecting vertex u and vertex v with weight w:\n"
            f"{edges_text}\n\n"
            "Your task is to find a path p1, p2, ..., pk such that:\n"
            f"- p1 = 0 (the path starts at vertex 0)\n"
            f"- pk = {self.N - 1} (the path ends at vertex {self.N - 1})\n"
            "- Try your best to minimize the ratio of the maximum edge weight to the minimum edge weight along the path "
            "(i.e., minimize max(w) / min(w), where w are the edge weights on the path).\n\n"
            "Output Format: Your final answer should be the path in order: p1 p2 ... pk, separated by spaces, "
            "wrapped in \\boxed{...}. Example: \\boxed{0 1 7}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure problem is available
        if self.N is None or not self.edges or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem"}

        # Parse path
        try:
            path = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate path
        if len(path) == 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "empty_path"}

        # Check vertex range
        for vertex in path:
            if not (0 <= vertex < self.N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "vertex_out_of_range"}

        # Check start and end
        if not (path[0] == 0 and path[-1] == self.N - 1):
            return TERMINAL_STATE, 0.0, True, False, {"error": "wrong_endpoints"}

        # Build edge weight map
        edge2weight: Dict[Tuple[int, int], int] = {(u, v): w for (u, v, w) in self.edges}

        # Verify path continuity and compute ratio
        all_weights = list(edge2weight.values())
        if not all_weights:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_edges"}

        answer_num = min(all_weights)
        answer_den = max(all_weights)

        for s, t in zip(path, path[1:]):
            u, v = (s, t) if s < t else (t, s)
            if (u, v) not in edge2weight:
                return TERMINAL_STATE, 0.0, True, False, {"error": "edge_missing"}
            w = edge2weight[(u, v)]
            answer_num = max(answer_num, w)
            answer_den = min(answer_den, w)

        gold_num, gold_den = self.gold_answer
        is_correct = (gold_num * answer_den == answer_num * gold_den)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "gold_ratio_numerator": gold_num,
            "gold_ratio_denominator": gold_den,
            "user_ratio_numerator": answer_num,
            "user_ratio_denominator": answer_den,
            "user_path": path,
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
        """Sample a random action (path). This is a naive sampler for demonstration."""
        if self.N is None:
            return "\\boxed{0}"
        # Simple attempt: direct path 0 N-1 if edge exists, otherwise fallback to 0 ... N-1
        edge_set = {(u, v) for (u, v, _) in self.edges}
        if (0, self.N - 1) in edge_set:
            return f"\\boxed{{0 {self.N - 1}}}"
        else:
            # Fallback: sequential vertices (may be invalid for some instances)
            seq_path = " ".join(str(i) for i in range(self.N))
            return f"\\boxed{{{seq_path}}}"