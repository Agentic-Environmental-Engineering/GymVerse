import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxXorPathEnv(Env):
    """Max XOR Path problem environment - single turn Q&A.

    You are given an undirected weighted graph. Find a path from 0 to N-1
    that maximizes the XOR of edge weights along the path.
    """

    def __init__(
        self,
        N: int = 8,
        edge_density: float = 0.5,
        MAX_bit_length: int = 10,
        **kwargs
    ):
        super().__init__()
        # Parameter validation as in the original environment
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"

        assert isinstance(edge_density, (int, float)), "edge_density must be a number"
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        assert isinstance(MAX_bit_length, int), "MAX_bit_length must be an integer"
        assert MAX_bit_length >= 2, "MAX_bit_length should be greater than or equal to 2"

        self.N: int = N
        self.edge_density: float = float(edge_density)
        self.MAX_bit_length: int = MAX_bit_length

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.edges: List[Tuple[int, int, int]] = []

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a maximum XOR path problem on an undirected weighted graph.\n"
            "Please provide your final answer as a single integer enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        MAX_bit_length = self.MAX_bit_length
        edge_density = self.edge_density

        # Generate a random graph and the corresponding edges and reference answer
        while True:
            # Build random adjacency based on density
            adjacent: List[List[int]] = [[] for _ in range(N)]
            all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
            m = int(edge_density * N * (N - 1) / 2)
            m = min(m, len(all_pairs))
            chosen_pairs = random.sample(all_pairs, m) if m > 0 else []

            for u, v in chosen_pairs:
                adjacent[u].append(v)
                adjacent[v].append(u)

            base_size_upper = random.randint(0, MAX_bit_length - 1)

            edges: List[Tuple[int, int, int]] = []
            P: List[int] = [0] * MAX_bit_length
            base_size: int = 0

            def insert_into_basis(x: int) -> None:
                """Insert x into the XOR linear basis P."""
                nonlocal P, base_size
                cur = x
                for i in range(MAX_bit_length - 1, -1, -1):
                    if ((cur >> i) & 1) == 0:
                        continue
                    if P[i] == 0:
                        P[i] = cur
                        base_size += 1
                        return
                    cur ^= P[i]

            def maximize_with_basis(x: int) -> int:
                """Maximize x XOR any combination of basis vectors."""
                res = x
                for i in range(MAX_bit_length - 1, -1, -1):
                    if P[i] != 0 and (res ^ P[i]) > res:
                        res ^= P[i]
                return res

            visited: List[bool] = [False] * N
            xor_to: List[int] = [0] * N
            edge2weight: Dict[Tuple[int, int], int] = {}

            def DFS(u: int) -> None:
                visited[u] = True
                for nbr in adjacent[u]:
                    a, b = (u, nbr)
                    mn, mx = (a, b) if a < b else (b, a)
                    if not visited[nbr]:
                        w = random.randint(0, (1 << MAX_bit_length) - 1)
                        if (mn, mx) not in edge2weight:
                            edges.append((mn, mx, w))
                            edge2weight[(mn, mx)] = w
                        xor_to[nbr] = xor_to[u] ^ w
                        DFS(nbr)
                    else:
                        if (mn, mx) not in edge2weight:
                            if base_size < base_size_upper:
                                w = random.randint(0, (1 << MAX_bit_length) - 1)
                            else:
                                # Create a weight correlated with current XOR-to values and basis
                                w = xor_to[u] ^ xor_to[nbr]
                                for i in range(MAX_bit_length - 1, -1, -1):
                                    if random.random() < 0.5:
                                        w ^= P[i]
                            edges.append((mn, mx, w))
                            edge2weight[(mn, mx)] = w
                        else:
                            w = edge2weight[(mn, mx)]
                        cycle_xor = xor_to[u] ^ w ^ xor_to[nbr]
                        insert_into_basis(cycle_xor)

            # Start DFS from node 0
            if N > 0:
                DFS(0)

            # Ensure node N-1 is reachable
            if not visited[N - 1]:
                continue

            reference_answer = maximize_with_basis(xor_to[N - 1])

            # Ensure the answer is not trivially the maximum possible value
            if reference_answer < (1 << MAX_bit_length) - 1:
                random.shuffle(edges)
                # Perform final assertions to ensure edge uniqueness and ordering
                assert len(edges) == len(set((u, v) for u, v, _ in edges)), "edges should be unique"
                for u, v, _w in edges:
                    assert 0 <= u < v < N
                # Save state
                self.edges = edges
                self.reference_answer = reference_answer
                break

        # Build problem prompt
        edges_text = "\n".join(f"({u}, {v}, {w})" for u, v, w in self.edges)
        problem_text = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}.\n\n"
            f"The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w), "
            f"meaning an undirected edge connecting vertex u to vertex v with weight w:\n{edges_text}\n\n"
            f"Find a path from vertex 0 to vertex {N - 1} such that the XOR of the weights of the edges in the path "
            f"is maximized. Output the maximum XOR value.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        # Extract answer in boxed format
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare with reference answer
        try:
            user_answer = int(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "MAX_bit_length": self.MAX_bit_length,
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
        """Sample a random action in \\boxed{...} format."""
        rnd = random.randint(0, (1 << self.MAX_bit_length) - 1)
        return f"\\boxed{{{rnd}}}"