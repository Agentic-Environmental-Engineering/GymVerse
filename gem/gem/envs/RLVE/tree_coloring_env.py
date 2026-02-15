from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TreeColoringEnv(Env):
    """
    Tree Coloring environment (single-turn Q&A).

    Task:
    - Given a weighted tree with N vertices labeled 0..N-1 and exactly N-1 edges.
    - Choose exactly K distinct vertices to be "colored".
    - The objective is to maximize:
        sum of pairwise distances between colored vertices
        + sum of pairwise distances between uncolored vertices.

    Answer format:
    - Provide exactly K distinct vertex indices separated by spaces, inside \\boxed{...}.
      Example: \\boxed{0 2 5}

    Reward:
    - Correct (optimal value achieved): 1.0
    - Wrong (suboptimal or invalid selection): 0.0
    - Format error (no boxed content): -0.1
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 30,
        K: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.N_fixed: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N
        self.K_fixed: Optional[int] = K

        # Problem state
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.current_problem: Optional[str] = None

        # Reference solution
        self.reference_answer_vertices: List[int] = []
        self.reference_answer: Optional[str] = None
        self.reference_answer_distance: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions shown before the problem."""
        return (
            "You are solving a tree coloring optimization problem.\n"
            "Task: Given a weighted tree with N vertices labeled 0..N-1 and exactly N-1 edges,\n"
            "select exactly K distinct vertices (colored). The score is:\n"
            "- sum of pairwise distances between colored vertices\n"
            "+ sum of pairwise distances between uncolored vertices.\n"
            "Your goal is to maximize this score.\n\n"
            "Output Format:\n"
            "- Provide exactly K vertex indices separated by single spaces inside \\boxed{...}.\n"
            "  Example: \\boxed{0 2 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new tree problem."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            N = self.N_fixed
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"

        # Determine K
        if self.K_fixed is not None:
            K = self.K_fixed
            assert 1 <= K <= N - 1, "K must be in [1, N-1]"
        else:
            K = random.randint(1, N - 1)

        # Generate a random tree with weights in [1, N]
        edges: List[Tuple[int, int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            edges.append((u, v, random.randint(1, N)))
        random.shuffle(edges)

        for u, v, w in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, _ in edges)) == N - 1

        tree = networkx.Graph()
        tree.add_weighted_edges_from(edges)
        assert networkx.is_tree(tree)

        # Compute reference optimal solution using DP
        adjacency_list: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
        for u, v, w in edges:
            adjacency_list[u].append((v, w))
            adjacency_list[v].append((u, w))

        dpF: List[List[Optional[int]]] = [[None] * (K + 1) for _ in range(N)]
        decisions: List[List[Tuple[int, int, List[Optional[int]]]]] = [[] for _ in range(N)]
        Size: List[int] = [0] * N

        def DP(u: int, parent: int) -> None:
            Size[u] = 1
            dpF[u][0] = 0
            if K:
                dpF[u][1] = 0
            for v, w in adjacency_list[u]:
                if v == parent:
                    continue
                DP(v, u)
                decision_list = decisions[u]
                decision_list.append((v, w, [None] * (min(Size[u] + Size[v], K) + 1)))
                decision = decision_list[-1][2]
                for uk in range(min(Size[u], K), -1, -1):
                    for vk in range(min(Size[v], K - uk), -1, -1):
                        if dpF[u][uk] is None or dpF[v][vk] is None:
                            continue
                        if (N - K) < (Size[v] - vk):
                            continue
                        val = dpF[u][uk] + dpF[v][vk] + w * (
                            vk * (K - vk) + (Size[v] - vk) * ((N - K) - (Size[v] - vk))
                        )
                        if dpF[u][uk + vk] is None or dpF[u][uk + vk] <= val:
                            dpF[u][uk + vk] = val
                            decision[uk + vk] = vk
                Size[u] += Size[v]

        DP(0, -1)
        assert dpF[0][K] is not None
        reference_answer_distance = dpF[0][K]

        reference_vertices: List[int] = []

        def DFS_build(u: int, k: int) -> None:
            if Size[u] == 1:
                assert len(decisions[u]) == 0
            decisions[u].reverse()
            for dec in decisions[u]:
                v, _, dec_arr = dec
                vk = dec_arr[k]
                assert vk is not None
                k -= vk
                DFS_build(v, vk)
            assert k in (0, 1)
            if k == 1:
                reference_vertices.append(u)

        DFS_build(0, K)
        reference_vertices_str = " ".join(map(str, reference_vertices))

        # Store state
        self.N = N
        self.K = K
        self.edges = edges
        self.reference_answer_vertices = reference_vertices
        self.reference_answer = reference_vertices_str
        self.reference_answer_distance = reference_answer_distance

        # Build problem statement
        problem_text = (
            f"You are given a tree (a connected undirected graph with no cycles) with {N} vertices, "
            f"labeled from 0 to {N - 1}.\n\n"
            f"The tree contains the following {N} - 1 = {N - 1} undirected edges. Each edge is a tuple (u, v, w), "
            f"meaning there is an undirected edge connecting vertex u to vertex v with weight w:\n"
            + "\n".join(f"({u}, {v}, {w})" for (u, v, w) in edges)
            + "\n\n"
            f"Your task is to select exactly {K} distinct vertices. These selected vertices are called colored, "
            f"and the remaining {N - K} vertices are uncolored. Try your best to maximize the total distance, defined as:\n"
            f"- The sum of all pairwise distances between colored vertices,\n"
            f"- Plus the sum of all pairwise distances between uncolored vertices.\n\n"
            f"Note: Since the graph is a tree, there is exactly one unique path between any two vertices.\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single line containing the {K} selected (colored) vertices in any order, "
            f"separated by spaces, and wrapped in \\boxed{{...}}.\n"
            f"Example: \\boxed{{{' '.join(map(str, range(min(K, 5))))}}}\n"
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Parse and verify the answer, then terminate."""
        # Ensure problem is initialized
        if self.N is None or self.K is None or self.reference_answer_distance is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: no boxed content
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse vertex list from boxed content
        try:
            tokens = boxed_content.strip().split()
            user_vertices = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate selection
        if len(user_vertices) != self.K:
            info = {"error": "invalid_solution_length", "expected_K": self.K, "provided_len": len(user_vertices)}
            return TERMINAL_STATE, 0.0, True, False, info
        if len(set(user_vertices)) != self.K:
            info = {"error": "duplicate_vertices"}
            return TERMINAL_STATE, 0.0, True, False, info
        if not all(0 <= v < self.N for v in user_vertices):
            info = {"error": "vertex_out_of_range"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute user's achieved objective
        adjacency_list: List[List[Tuple[int, int]]] = [[] for _ in range(self.N)]
        for u, v, w in self.edges:
            adjacency_list[u].append((v, w))
            adjacency_list[v].append((u, w))

        colored = [0] * self.N
        for c in user_vertices:
            colored[c] = 1

        Size = [0] * self.N
        user_distance = 0

        def DFS(u: int, parent: int) -> None:
            nonlocal user_distance
            Size[u] = 1
            for v, w in adjacency_list[u]:
                if v == parent:
                    continue
                DFS(v, u)
                user_distance += w * (
                    colored[v] * (self.K - colored[v]) +
                    (Size[v] - colored[v]) * ((self.N - self.K) - (Size[v] - colored[v]))
                )
                Size[u] += Size[v]
                colored[u] += colored[v]

        DFS(0, -1)

        gold = self.reference_answer_distance
        assert gold is not None
        is_correct = (user_distance == gold)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_distance": gold,
            "user_distance": user_distance,
            "N": self.N,
            "K": self.K,
            "edges": self.edges,
            "reference_answer": self.reference_answer,
            "user_vertices": user_vertices,
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
        """Sample a random feasible action for the current problem."""
        if self.K is None or self.N is None:
            # Fallback random sample if not initialized
            k = 1
            n = 3
            selection = random.sample(range(n), k)
            return f"\\boxed{{{' '.join(map(str, selection))}}}"
        selection = random.sample(range(self.N), self.K)
        return f"\\boxed{{{' '.join(map(str, selection))}}}"