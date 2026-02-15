from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Maximum_IndependentSet_TreeEnv(Env):
    """Environment for Maximum Weight Independent Set on a tree - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 50,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, fixes the number of vertices. Must be >= 3.
        - min_N: Minimum number of vertices when sampling N randomly. Must be >= 3.
        - max_N: Maximum number of vertices when sampling N randomly. Must be >= min_N.
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N

        # Problem state
        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int]] = []
        self.weights: List[int] = []
        self.reference_weight: Optional[int] = None
        self.reference_vertices: List[int] = []
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Maximum Weight Independent Set problem on a tree.\n"
            "Provide your selected vertex indices as space-separated integers inside \\boxed{...}.\n"
            "Example: \\boxed{0 2 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            self.N = self.fixed_N
        else:
            self.N = random.randint(self.min_N, self.max_N)
        assert self.N is not None and self.N >= 3

        N = self.N

        # Generate a random rooted tree using a shuffled permutation and random parent selection
        permutations = list(range(N))
        random.shuffle(permutations)
        root = permutations[0]

        childrens: List[List[int]] = [[] for _ in range(N)]
        edges: List[Tuple[int, int]] = []
        for idx, child in enumerate(permutations):
            if idx == 0:
                continue
            parent = random.choice(permutations[:idx])
            childrens[parent].append(child)
            u, v = min(parent, child), max(parent, child)
            edges.append((u, v))

        # Validate edges
        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1

        # Validate it is a tree using networkx
        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree)

        # Generate vertex weights
        weights = [random.randint(1, N) for _ in range(N)]

        # Compute reference maximum weight via DP on tree
        dpF: List[Optional[List[int]]] = [None] * N

        def dp(u: int) -> None:
            dpF[u] = [0, weights[u]]
            for child in childrens[u]:
                dp(child)
                assert dpF[child] is not None
                dpF[u][0] += max(dpF[child][0], dpF[child][1])
                dpF[u][1] += dpF[child][0]

        dp(root)
        assert dpF[root] is not None
        reference_weight = max(dpF[root][0], dpF[root][1])

        # Reconstruct one optimal set of vertices
        picked: List[int] = []

        def pick_set(u: int, pick: bool) -> None:
            if pick:
                picked.append(u)
            for child in childrens[u]:
                if pick:
                    pick_set(child, False)
                else:
                    assert dpF[child] is not None
                    pick_set(child, bool(dpF[child][0] < dpF[child][1]))

        pick_set(root, bool(dpF[root][0] < dpF[root][1]))

        # Store state
        self.edges = edges
        self.weights = weights
        self.reference_weight = reference_weight
        self.reference_vertices = sorted(picked)
        self.current_problem = self._build_problem_prompt(N, edges, weights)

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def _build_problem_prompt(
        self,
        N: int,
        edges: List[Tuple[int, int]],
        weights: List[int]
    ) -> str:
        """Build the problem statement string."""
        edges_str = "\n".join(f"({u}, {v})" for (u, v) in edges)
        weights_str = "\n".join(f"R[{i}] = {weights[i]}" for i in range(N))
        return (
            f"You are given a tree (a connected undirected graph with no cycles) with {N} vertices, "
            f"labeled from 0 to {N - 1}.\n\n"
            f"The tree contains the following {N - 1} undirected edges:\n"
            f"{edges_str}\n\n"
            f"Each vertex has a weight, given as a list R of length {N}:\n"
            f"{weights_str}\n\n"
            "Your task is to select a set of distinct vertices such that no two selected vertices are adjacent. "
            "Your goal is to maximize the total weight sum of selected vertices.\n\n"
            "Output Format:\n"
            "Your final answer should be the selected vertices in any order, separated by spaces, inside \\boxed{...}.\n"
            "Example: \\boxed{0 1 4}\n"
        )

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step, verify the answer, and terminate."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to list of integers (allow empty set)
        try:
            content = boxed_content.strip()
            if content == "":
                picked_list: List[int] = []
            else:
                picked_list = list(map(int, content.split()))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate problem state
        if self.N is None or self.reference_weight is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        N = self.N
        edges_set = set(self.edges)

        # Validate distinctness
        if len(set(picked_list)) != len(picked_list):
            info = {
                "error": "invalid_solution",
                "reason": "duplicate_vertices",
                "user_vertices": picked_list,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate vertex range
        if not all(0 <= v < N for v in picked_list):
            info = {
                "error": "invalid_solution",
                "reason": "vertex_out_of_range",
                "user_vertices": picked_list,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        picked_set = set(picked_list)

        # Validate independent set property
        for u, v in edges_set:
            if u in picked_set and v in picked_set:
                info = {
                    "error": "invalid_solution",
                    "reason": "adjacent_vertices_selected",
                    "edge": (u, v),
                    "user_vertices": picked_list,
                }
                return TERMINAL_STATE, 0.0, True, False, info

        # Compute user's total weight
        user_weight = sum(self.weights[u] for u in picked_set)
        gold_weight = self.reference_weight

        is_correct = (user_weight == gold_weight)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_weight": gold_weight,
            "user_weight": user_weight,
            "user_vertices": sorted(picked_list),
            "reference_vertices": self.reference_vertices,
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
        """Sample a random valid independent set action as a baseline."""
        if self.N is None:
            return "\\boxed{}"

        # Greedy sampling to form an independent set
        vertices = list(range(self.N))
        random.shuffle(vertices)
        picked: List[int] = []
        picked_set: set[int] = set()
        adj: List[set[int]] = [set() for _ in range(self.N)]
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)

        for u in vertices:
            if all((nbr not in picked_set) for nbr in adj[u]):
                picked.append(u)
                picked_set.add(u)

        return "\\boxed{" + " ".join(map(str, sorted(picked))) + "}"