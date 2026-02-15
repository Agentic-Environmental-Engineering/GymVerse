import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumTreeWeightedDominatingAncestorEnv(Env):
    """
    Minimum Tree Weighted Dominating Ancestor problem environment.

    You are given a rooted tree with nodes labeled 0..N (N+1 nodes in total). Node 0 is the root
    and is already selected. Each non-root node i (1..N) has:
    - a parent with an edge weight,
    - a cost C[i].

    You must select exactly K additional nodes among 1..N to minimize the total cost defined as:
    For each node u (1..N), let D[u] be the distance (sum of edge weights) from u to its nearest
    selected ancestor (including possibly itself or the root 0). The contribution of node u is
    C[u] * D[u]. Minimize the sum over all u in 1..N.

    The environment provides the tree structure, node costs, and K. The answer must be K integers
    (the selected nodes excluding 0) submitted in \\boxed{...} format, separated by spaces.

    Reward:
    - 1.0 if the submitted set achieves the optimal total cost,
    - 0.0 if the submission is well-formed but suboptimal or invalid (wrong nodes/count),
    - -0.1 if the submission format is wrong (e.g., not using \\boxed{...}).
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 30,
        weight_range: int = 10,
        **kwargs
    ) -> None:
        """
        Initialize the environment.

        Args:
            N: Optional fixed N (number of non-root nodes). If None, N is sampled in reset().
            min_N: Minimum N when sampling (inclusive). Must be >= 2.
            max_N: Maximum N when sampling (inclusive). Must be >= min_N.
            weight_range: Range for random edge weights and costs (1..weight_range for weights,
                          and 0..weight_range for costs, with C[0] = 0).
            **kwargs: Unused, for forward compatibility.
        """
        super().__init__()
        assert min_N >= 2, "min_N should be greater than or equal to 2"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"
        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N
        self.weight_range = weight_range

        # Problem state
        self.N: Optional[int] = None  # number of non-root nodes, total nodes = N + 1
        self.parents: Optional[List[Optional[Tuple[int, int]]]] = None  # (parent, weight) for nodes 1..N
        self.C: Optional[List[int]] = None  # costs, length N+1, with C[0] = 0
        self.K: Optional[int] = None  # number of additional selected nodes (excluding root)
        self.graph: Optional[List[List[Tuple[int, int]]]] = None  # adjacency list
        self.gold_cost: Optional[int] = None  # optimal minimal total cost

        # Texts
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "Task: Minimum Tree Weighted Dominating Ancestor.\n"
            "You are given a rooted tree with nodes 0..N (0 is the root). "
            "The root (node 0) is already selected. You must select exactly K additional nodes among 1..N.\n"
            "For each node u, D[u] is the distance to its nearest selected ancestor (including itself if selected, or the root 0). "
            "The total cost is sum over u=1..N of C[u] × D[u]. Minimize this cost.\n"
            "Answer Format: Provide exactly K distinct integers (node labels in [1..N]) separated by single spaces, "
            "enclosed in \\boxed{...}. Example: \\boxed{3 5 7}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            assert self.fixed_N >= 2, "N should be greater than or equal to 2"
            self.N = int(self.fixed_N)
        else:
            self.N = random.randint(self.min_N, self.max_N)

        N = self.N

        # Generate random tree structure
        parents: List[Optional[Tuple[int, int]]] = [None] * (N + 1)
        permutations = list(range(1, N + 1))
        random.shuffle(permutations)
        permutations = [0] + permutations
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            parent = random.choice(permutations[:index])
            parents[vertex] = (parent, random.randint(1, self.weight_range))

        # Costs
        C = [0] + [random.randint(0, self.weight_range) for _ in range(1, N + 1)]

        # K
        K = random.randint(1, N - 1)

        # Build adjacency list
        graph: List[List[Tuple[int, int]]] = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            parent, dist = parents[i]
            graph[parent].append((i, dist))

        # Compute optimal minimal cost using DP (as in the original environment)
        depth = [0] * (N + 1)

        # f[p][j][l]: minimum cost in subtree of p, for wood from all nodes
        # that are descendants of p but no closer ancestor than j,
        # using l new sawmills in that subtree
        # g[p][j][l]: same but requiring that none of those l sawmills lies
        # on the path from p up to j (i.e., the first mill is strictly below p)
        f = [[[0] * (K + 1) for _ in range(N + 1)] for _ in range(N + 1)]
        g = [[[0] * (K + 1) for _ in range(N + 1)] for _ in range(N + 1)]

        st: List[int] = []

        def dfs(p: int) -> None:
            st.append(p)
            # Process children
            for to, dist in graph[p]:
                depth[to] = depth[p] + dist
                dfs(to)
                # Merge DP from child 'to' into p
                for j in st:
                    for l in range(K, -1, -1):
                        f[p][j][l] += f[to][j][0]
                        g[p][j][l] += f[to][p][0]
                        best_fpjl = f[p][j][l]
                        best_gpjl = g[p][j][l]
                        for x in range(1, l + 1):
                            cost_f = f[p][j][l - x] + f[to][j][x]
                            if cost_f < best_fpjl:
                                best_fpjl = cost_f
                            cost_g = g[p][j][l - x] + f[to][p][x]
                            if cost_g < best_gpjl:
                                best_gpjl = cost_g
                        f[p][j][l] = best_fpjl
                        g[p][j][l] = best_gpjl

            # Account for p's own wood
            for j in st:
                dist_up = depth[p] - depth[j]
                f[p][j][0] += C[p] * dist_up
                for l in range(1, K + 1):
                    f[p][j][l] = min(
                        f[p][j][l] + C[p] * dist_up,
                        g[p][j][l - 1]
                    )
            st.pop()

        dfs(0)
        gold_answer = f[0][0][K]

        # Store problem state
        self.parents = parents
        self.C = C
        self.K = K
        self.graph = graph
        self.gold_cost = gold_answer

        # Build problem text
        edges_desc = "\n".join(
            f"`{i + 1}`'s parent is `{parents[i + 1][0]}` with weight `{parents[i + 1][1]}`"
            for i in range(N)
        )
        C_desc = " ".join(f"C[{i}]={C[i]}" for i in range(1, N + 1))

        problem_text = (
            f"You are given a tree (connected acyclic undirected graph) with {N} + 1 = {N + 1} vertices, "
            f"labeled from 0 to {N}.\n\n"
            f"0 is the root of the tree. Each non-root vertex has a parent, and the edge connecting it with its parent has a weight. "
            f"The edges are given as follows:\n{edges_desc}\n\n"
            f"Each non-root vertex also has a cost, given as a list C of length {N}, where C[i] is the cost of vertex i:\n{C_desc}\n\n"
            f"The root (vertex 0) is already selected. Your task is to select exactly {K} additional non-root vertices. "
            f"The total cost of the selection is defined as follows:\n"
            f"- For every vertex u, let D[u] be the distance from u to its nearest selected ancestor, where a selected ancestor includes 0 or the vertex itself (if selected). "
            f"The distance between two vertices is the sum of weights along the unique path between them.\n"
            f"- The cost contributed by vertex u is C[u] × D[u] for all non-root vertices u.\n"
            f"- Try your best to minimize the total cost.\n\n"
            f"Output Format: Output a single line containing {K} integers — the selected vertices (excluding 0), separated by spaces. "
            f"Your final answer must be enclosed as \\boxed{{v1 v2 ... v{K}}}."
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted selection and assign reward."""
        # Parse answer from boxed format
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Environment must be initialized
        assert self.N is not None and self.K is not None and self.C is not None and self.parents is not None and self.graph is not None and self.gold_cost is not None

        # Validate and score
        N = self.N
        K = self.K
        selected_vertices = parsed

        info: dict[str, Any] = {}

        # Check count
        if len(selected_vertices) != K:
            info.update({"error": "invalid_solution", "reason": "wrong_count"})
            return TERMINAL_STATE, 0.0, True, False, info

        # Check range
        for v in selected_vertices:
            if not (1 <= v <= N):
                info.update({"error": "invalid_solution", "reason": "vertex_out_of_range"})
                return TERMINAL_STATE, 0.0, True, False, info

        # Build selected array (root is always selected)
        selected = [False] * (N + 1)
        selected[0] = True
        for v in selected_vertices:
            selected[v] = True

        # Compute total cost for the submission
        total_cost = 0

        def dfs_cost(vertex: int, dist: int) -> None:
            nonlocal total_cost
            if selected[vertex]:
                dist = 0
            total_cost += self.C[vertex] * dist
            for neighbor, w in self.graph[vertex]:
                dfs_cost(neighbor, dist + w)

        dfs_cost(0, 0)

        is_optimal = (total_cost == self.gold_cost)
        reward = 1.0 if is_optimal else 0.0

        info.update({
            "correct": is_optimal,
            "gold_cost": self.gold_cost,
            "user_cost": total_cost,
            "K": K,
            "N": N,
            "selection": selected_vertices
        })

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[List[int]]:
        """
        Extract the answer from \\boxed{...} and parse as a list of integers.
        Returns None if not found or parsing fails.
        """
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if not matches:
            return None
        content = matches[-1].strip()
        if not content:
            return None
        tokens = content.split()
        try:
            return [int(tok) for tok in tokens]
        except ValueError:
            return None

    def sample_random_action(self) -> str:
        """Sample a random valid action (uniformly selects K nodes)."""
        if self.N is None or self.K is None:
            # Fallback: random small example
            k = 1
            vals = [1]
        else:
            # Sample without replacement for a valid example
            vals = random.sample(range(1, self.N + 1), self.K)
        vals_str = " ".join(str(v) for v in vals)
        return f"\\boxed{{{vals_str}}}"