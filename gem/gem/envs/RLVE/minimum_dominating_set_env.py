import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Minimum_DominatingSetEnv(Env):
    """Minimum Cost Dominating Set environment - single-turn Q&A.

    The agent is given an undirected graph with vertex costs and must output
    a set of vertices forming a dominating set with minimum total cost.
    The answer must be submitted in \\boxed{...} format, where the content is
    a space-separated list of vertex indices.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        edge_density: Optional[float] = None,
        cost_range: int = 10,
        **kwargs
    ):
        """
        Initialize the Minimum_DominatingSetEnv instance.

        Parameters:
        - N: Optional[int] = None
            If provided, fixes the number of vertices; must satisfy N >= 2.
            If None, N will be randomly sampled on each reset.
        - edge_density: Optional[float] = None
            If provided, fixes the edge density; must satisfy 0.0 <= edge_density <= 1.0.
            If None, edge density will be randomly sampled on each reset.
        - cost_range: int = 10
            Maximum cost for each vertex (costs are sampled uniformly from [1, cost_range]).
        """
        super().__init__()
        self.N_fixed = N
        self.edge_density_fixed = edge_density
        self.cost_range = cost_range

        # Problem state
        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int]] = []
        self.C: List[int] = []
        self.covering: List[int] = []
        self.reference_vertices: List[int] = []
        self.reference_cost: int = 0
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an undirected graph with vertex costs.\n"
            "Your task is to select a set of distinct vertices such that every vertex "
            "is either selected or has at least one selected neighbor, and the total "
            "cost of selected vertices is minimized.\n\n"
            "Answer Format: Provide the selected vertices as space-separated integers "
            "inside \\boxed{...}.\n"
            "Example: \\boxed{0 1 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            self.N = self.N_fixed
        else:
            # Sample N if not fixed
            self.N = random.randint(2, 10)

        assert self.N is not None
        assert self.N >= 2, "N should be greater than or equal to 1"

        # Determine edge density
        if self.edge_density_fixed is not None:
            edge_density = self.edge_density_fixed
        else:
            edge_density = random.random()  # uniform in [0.0, 1.0]

        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        # Generate edges
        all_possible_edges = [(u, v) for u in range(self.N) for v in range(u + 1, self.N)]
        num_edges = int(edge_density * self.N * (self.N - 1) / 2)
        self.edges = random.sample(all_possible_edges, num_edges)
        random.shuffle(self.edges)

        for u, v in self.edges:
            assert 0 <= u < v < self.N
        assert len(self.edges) == len(set(self.edges)), "edges should be unique"

        # Generate costs
        self.C = [random.randint(1, self.cost_range) for _ in range(self.N)]

        # Build covering bitmasks
        self.covering = [1 << u for u in range(self.N)]
        for u, v in self.edges:
            self.covering[u] |= 1 << v
            self.covering[v] |= 1 << u

        # Compute reference (optimal) answer via DFS with pruning
        self.reference_vertices = list(range(self.N))
        self.reference_cost = sum(self.C)

        selected: List[int] = []

        def DFS(u: int, now_covering: int, sumC: int) -> None:
            if sumC >= self.reference_cost:
                return
            if u == self.N:
                if now_covering == (1 << self.N) - 1:
                    assert sumC < self.reference_cost
                    self.reference_vertices, self.reference_cost = selected.copy(), sumC
                return
            DFS(u + 1, now_covering, sumC)
            if (now_covering | self.covering[u]) > now_covering:
                selected.append(u)
                DFS(u + 1, now_covering | self.covering[u], sumC + self.C[u])
                selected.pop()

        DFS(0, 0, 0)

        assert self.reference_cost > 0, "gold_answer must be greater than 0"

        # Build problem string
        edges_str = "\n".join(f"({u}, {v})" for u, v in self.edges)
        costs_str = "\n".join(f"C[{i}]={Ci}" for i, Ci in enumerate(self.C))

        self.current_problem = (
            f"You are given an undirected graph with {self.N} vertices labeled from 0 to {self.N - 1}. "
            "The graph contains the following undirected edges:\n"
            f"{edges_str}\n\n"
            f"Each vertex has a cost, given as a list C of length {self.N}, where C[i] is the cost of vertex i:\n"
            f"{costs_str}\n\n"
            "Your task is to select a set of distinct vertices x_1, x_2, ..., x_k (you determine k), "
            "such that every vertex is either selected or has at least one selected neighbor. "
            "Try your best to minimize the total cost: C[x_1] + C[x_2] + ... + C[x_k].\n\n"
            "Output Format: Your final answer should be the selected vertices in any order, "
            "separated by spaces, inside \\boxed{...}.\n"
            f"Example: \\boxed{{0 1 {self.N - 1}}}"
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse vertices list from boxed content
        boxed_content = boxed_content.strip()
        if boxed_content == "":
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            selected_vertices = list(map(int, boxed_content.split()))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate solution
        info: dict[str, Any] = {
            "user_vertices": selected_vertices,
            "graph_N": self.N,
            "graph_edges": self.edges,
            "vertex_costs": self.C,
        }

        # Check distinctness
        if len(selected_vertices) != len(set(selected_vertices)):
            info["error"] = "invalid_solution_duplicates"
            return TERMINAL_STATE, 0.0, True, False, info

        # Check range
        if any(not (0 <= u < self.N) for u in selected_vertices):
            info["error"] = "invalid_solution_out_of_range"
            return TERMINAL_STATE, 0.0, True, False, info

        # Check dominating property
        all_covering = 0
        for u in selected_vertices:
            all_covering |= self.covering[u]
        if all_covering != (1 << self.N) - 1:
            info["error"] = "invalid_solution_not_dominating"
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute cost and compare with optimal
        user_cost = sum(self.C[u] for u in selected_vertices)
        info["user_cost"] = user_cost
        info["reference_vertices"] = self.reference_vertices
        info["reference_cost"] = self.reference_cost

        is_correct = (user_cost == self.reference_cost)
        info["correct"] = is_correct
        reward = 1.0 if is_correct else 0.0

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random subset of vertices inside \\boxed{...}."""
        if self.N is None:
            # Fallback if called before reset
            N = 5
        else:
            N = self.N
        k = random.randint(0, N)
        subset = sorted(random.sample(range(N), k))
        content = " ".join(map(str, subset))
        return f"\\boxed{{{content}}}"