from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxTreeXorPathEnv(Env):
    """Environment for the Max XOR Path in a Tree problem - single-turn Q&A.

    The task: Given a tree with N vertices labeled 0..N-1 and weighted edges,
    find a pair of vertices (u, v) that maximizes the bitwise XOR of edge weights along the unique path between u and v.
    The agent must respond with two integers 'u v' enclosed in \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 50,
        lower_max_weight: int = 2 ** 4,
        **kwargs
    ):
        """Initialize the MaxTreeXorPathEnv.

        Args:
            N: If provided, use this fixed number of vertices (must be >= 3).
            min_n: Minimum number of vertices when sampling N randomly (inclusive).
            max_n: Maximum number of vertices when sampling N randomly (inclusive).
            lower_max_weight: Lower bound (power-of-two seed) for sampling edge weights.
        """
        super().__init__()
        self.N_fixed = N
        self.min_n = min_n
        self.max_n = max_n
        self.lower_max_weight = lower_max_weight

        # Runtime state
        self.N: Optional[int] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.Xor: List[int] = []
        self.reference_pair: Optional[Tuple[int, int]] = None
        self.reference_xor: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a tree (connected acyclic undirected graph) with weighted edges.\n"
            "Your goal is to select two vertices (u, v) to maximize the XOR of all edge weights along the unique path between them.\n"
            "Please provide your answer in \\boxed{u v} format, where u and v are integers separated by a single space.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            self.N = self.N_fixed
        else:
            self.N = random.randint(self.min_n, self.max_n)

        assert self.N is not None
        assert self.N >= 3, "N should be greater than or equal to 3"

        # Determine maximum weight bound
        max_weight = self.lower_max_weight
        while max_weight <= self.N * 2:
            max_weight *= 2

        # Generate a random tree with weighted edges
        edges: List[Tuple[int, int, int]] = []
        permutations = list(range(self.N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            w = random.randint(1, max_weight - 2)
            edges.append((u, v, w))
        random.shuffle(edges)

        for u, v, _ in edges:
            assert 0 <= u < v < self.N
        assert len(edges) == len(set((u, v) for u, v, _ in edges)) == self.N - 1

        # Validate tree structure
        tree = networkx.Graph()
        tree.add_weighted_edges_from(edges)
        assert networkx.is_tree(tree)

        # Build adjacency list
        adj: List[List[Tuple[int, int]]] = [[] for _ in range(self.N)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        # Compute XOR from root 0 to every node
        Xor = [0] * self.N

        def dfs(u: int, parent: int) -> None:
            for vv, ww in adj[u]:
                if vv == parent:
                    continue
                Xor[vv] = Xor[u] ^ ww
                dfs(vv, u)

        dfs(0, -1)

        # Find the best pair (u, v) maximizing XOR
        Ans_u, Ans_v = 0, 1
        for u in range(self.N):
            for v in range(u + 1, self.N):
                if (Xor[u] ^ Xor[v]) > (Xor[Ans_u] ^ Xor[Ans_v]):
                    Ans_u, Ans_v = u, v

        self.edges = edges
        self.Xor = Xor
        self.reference_pair = (Ans_u, Ans_v)
        self.reference_xor = Xor[Ans_u] ^ Xor[Ans_v]

        # Build problem prompt
        edges_str = "\n".join(f"({u}, {v}, {w})" for u, v, w in self.edges)
        self.current_problem = (
            f"You are given a tree with {self.N} vertices labeled from 0 to {self.N - 1}.\n"
            f"The tree has {self.N - 1} undirected weighted edges given below, each as (u, v, w):\n"
            f"{edges_str}\n\n"
            "Find two vertices u and v (0-based indices) that maximize the XOR of edge weights on the unique path connecting them.\n\n"
            "Output Format: Provide two integers u and v (separated by a single space) inside \\boxed{...}. Example: \\boxed{0 1}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse two integers u and v
        parts = boxed.strip().split()
        if len(parts) != 2:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            u = int(parts[0])
            v = int(parts[1])
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate range
        if self.N is None or not (0 <= u < self.N and 0 <= v < self.N):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Compute user's XOR
        user_xor = self.Xor[u] ^ self.Xor[v]
        correct = (self.reference_xor is not None and user_xor == self.reference_xor)

        reward: float = 1.0 if correct else 0.0
        info = {
            "correct": correct,
            "reference_pair": self.reference_pair,
            "reference_xor": self.reference_xor,
            "user_pair": (u, v),
            "user_xor": user_xor,
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
        """Sample a random action in the required \\boxed{u v} format."""
        if self.N is None or self.N < 2:
            # Fallback to a generic random pair
            return "\\boxed{0 1}"
        u = random.randint(0, self.N - 1)
        v = random.randint(0, self.N - 1)
        while v == u:
            v = random.randint(0, self.N - 1)
        return f"\\boxed{{{u} {v}}}"