import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

import networkx as nx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GasFireExtinguishersEnv(Env):
    """
    Gas Fire Extinguishers Environment (GEM version).

    Task:
    - You are given a tree with N vertices labeled 0..N-1 and edges.
    - Choose, for each vertex u, a vertex P[u] within distance <= K.
    - Let C[x] be the count of u such that P[u] = x.
    - Objective value is sum over x of ceil(C[x] / S).
    - Your goal is to minimize this objective.
    - Output P[0], P[1], ..., P[N-1] (space-separated) inside \\boxed{...}.

    The environment generates a random tree and parameters, computes the minimal
    achievable objective value (the "gold answer") via a tree DP, and validates
    the user's assignment P:
      - Valid if each P[u] is within K of u, and the resulting objective equals gold.
      - Reward: 1.0 if valid and optimal; 0.0 otherwise; -0.1 for format errors.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 50,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: If provided, fixes the number of nodes in the tree. Must be >= 3.
               If None, a random N in [min_n, max_n] is chosen during reset().
            min_n: Minimum N when sampling N randomly (inclusive), must be >= 3.
            max_n: Maximum N when sampling N randomly (inclusive), must be >= min_n.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__()
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")

        self.fixed_N: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        # Problem state
        self.N: Optional[int] = None
        self.edges: Optional[List[Tuple[int, int]]] = None
        self.K: Optional[int] = None
        self.S: Optional[int] = None
        self.valid_P: Optional[List[List[int]]] = None
        self.gold_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """
        Return task instructions for the agent.
        """
        return (
            "You are solving a tree assignment optimization problem.\n"
            "For each vertex u, choose a vertex P[u] within distance at most K.\n"
            "Let C[x] be the number of vertices u such that P[u] = x. The objective is:\n"
            "  sum over x of ceil(C[x] / S)\n"
            "Your goal is to minimize this objective.\n\n"
            "Answer Format:\n"
            "- Provide N integers P[0], P[1], ..., P[N-1] (space-separated) inside \\boxed{...}.\n"
            "- Example: \\boxed{0 0 1 2}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: The task instructions plus the problem description.
            info: An empty dict (no additional info required).
        """
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        self.N = N

        # Generate a random tree with N vertices
        edges: List[Tuple[int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            edges.append((u, v))
        random.shuffle(edges)

        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1
        self.edges = edges

        # Compute all-pairs shortest path distances
        G = nx.Graph()
        G.add_edges_from(edges)
        distances = dict(nx.all_pairs_shortest_path_length(G))
        maxdist = max(distances[u][v] for u in range(N) for v in range(N))

        # Choose K and S
        K = random.randint(1, max(1, maxdist // 2))
        self.K = K
        valid_P = [[v for v in range(N) if distances[u][v] <= K] for u in range(N)]
        self.valid_P = valid_P
        S = random.randint(2, max(2, N // K))
        self.S = S

        # Compute gold answer using DP on the tree
        self.gold_answer = self._compute_gold_answer(N, edges, K, S)

        # Build problem prompt
        edges_str = "\n".join(f"{u} {v}" for u, v in edges)
        self.current_problem = (
            f"You are given a tree (a connected undirected graph with no cycles) "
            f"with {N} vertices labeled from 0 to {N-1}. The tree has the following "
            f"{N-1} undirected edges (u v):\n{edges_str}\n\n"
            f"There is an array C[0], C[1], ..., C[{N-1}], all initially set to 0. "
            f"For each vertex u (0 ≤ u < {N}), you must choose a vertex P[u] such that the distance "
            f"(in number of edges) from u to P[u] is at most {K}; then, increment C[P[u]] by 1.\n"
            f"Try your best to minimize the total value of ceil(C[0] / {S}) + ceil(C[1] / {S}) + ... + "
            f"ceil(C[{N-1}] / {S}), where ceil(x) means rounding x up to the nearest integer.\n\n"
            f"Output Format: Provide P[0], P[1], ..., P[{N-1}] separated by spaces inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_gold_answer(self, N: int, edges: List[Tuple[int, int]], K: int, S: int) -> int:
        """
        Compute the minimal objective value using a DP over the tree.
        Mirrors the original algorithm to determine the gold answer.
        """
        # Build adjacency list
        graph = [[] for _ in range(N)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # f[u][i]: number of rooms in subtree u at distance exactly i that still need an extinguisher
        # g[u][i]: capacity of extinguishers at u that can serve rooms at distance exactly i
        f = [[0] * (K + 1) for _ in range(N)]
        g = [[0] * (K + 1) for _ in range(N)]
        ans = 0

        def dfs(u: int, parent: int) -> None:
            nonlocal ans
            f[u][0] = 1
            # accumulate from children
            for v in graph[u]:
                if v == parent:
                    continue
                dfs(v, u)
                for i in range(K):
                    f[u][i + 1] += f[v][i]
                    g[u][i + 1] += g[v][i]
            # place new extinguishers for rooms at distance K in subtree
            need = (f[u][K] + S - 1) // S
            ans += need
            # capacity left in newly placed extinguishers
            leftover = need * S - f[u][K]
            f[u][K] = 0
            g[u][0] += leftover
            # match needs and capacities within K
            # first for exact K distance pairs
            for i in range(K + 1):
                j = K - i
                d = min(f[u][i], g[u][j])
                f[u][i] -= d
                g[u][j] -= d
            # then for distance K-1 pairs
            for i in range(K):
                j = K - 1 - i
                d = min(f[u][i], g[u][j])
                f[u][i] -= d
                g[u][j] -= d

        # run DFS from root 0
        dfs(0, -1)

        # final matching at root
        for i in range(K + 1):
            for j in range(K + 1):
                if i + j <= K:
                    d = min(f[0][i], g[0][j])
                    f[0][i] -= d
                    g[0][j] -= d
        # remaining rooms need extinguishers
        tot = sum(f[0][i] for i in range(K + 1))
        ans += (tot + S - 1) // S

        if ans <= 0:
            raise AssertionError("The answer should be greater than 0")
        return ans

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Validate the user's answer.

        Args:
            action: A string containing the user's proposed P array inside \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE (single-turn environment).
            reward: 1.0 if correct, 0.0 if incorrect, -0.1 if format error.
            terminated: True (always a single-turn environment).
            truncated: False.
            info: Additional information about validation.
        """
        # Parse the boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure problem is initialized
        if self.N is None or self.valid_P is None or self.S is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        # Parse the list of integers
        try:
            parsed = list(map(int, boxed.strip().split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length
        if len(parsed) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "length_mismatch"}

        # Validate each P[u] is within allowed set (distance <= K)
        for u, p_u in enumerate(parsed):
            if p_u not in self.valid_P[u]:
                return TERMINAL_STATE, 0.0, True, False, {
                    "error": "invalid_solution",
                    "reason": f"P[{u}] not within distance <= K",
                }

        # Compute objective value
        C = [0] * self.N
        for p_u in parsed:
            C[p_u] += 1
        objective_value = sum((c + self.S - 1) // self.S for c in C)

        is_optimal = (objective_value == self.gold_answer)
        reward = 1.0 if is_optimal else 0.0

        info = {
            "correct": is_optimal,
            "reference_answer": self.gold_answer,
            "user_objective": objective_value,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside the last \\boxed{...}.
        """
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random valid assignment P[u] for demonstration or exploration.
        """
        if self.N is None or self.valid_P is None:
            # If not initialized, produce an empty action (format will be invalid)
            return "\\boxed{}"
        P = [random.choice(self.valid_P[u]) for u in range(self.N)]
        return "\\boxed{" + " ".join(map(str, P)) + "}"