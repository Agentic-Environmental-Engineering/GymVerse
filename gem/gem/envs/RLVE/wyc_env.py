from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class WYCEnv(Env):
    """Directed graph path weight ordering environment - single-turn Q&A."""

    def __init__(
        self,
        N: int = 5,
        MAX_K: int = 100,
        **kwargs
    ):
        """
        Initialize the WYCEnv instance.

        Parameters:
        - N: Number of vertices in the directed graph (must be >= 2).
        - MAX_K: Upper bound for K (must be >= 1).
        """
        super().__init__()
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")
        if MAX_K < 1:
            raise ValueError("MAX_K should be greater than or equal to 1")

        self.N = N
        self.MAX_K = MAX_K

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are given a directed graph problem.\n"
            "Your task is to compute the total weight of the K-th path when all possible paths "
            "with at least one edge are sorted by total weight in non-decreasing order.\n"
            "Paths may start and end at any vertex, and may revisit vertices or edges multiple times.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        N = self.N
        MAX_K = self.MAX_K

        # Generate a valid instance that yields a computable answer
        while True:
            edges: List[Tuple[int, int, int]] = []
            num_edges = random.randint(1, N * (N - 1))
            for _ in range(num_edges):
                s, t = random.sample(range(1, N + 1), 2)
                w = random.randint(1, 3)
                edges.append((s, t, w))
            random.shuffle(edges)

            # Validate edge bounds and that no self-loops in input
            for s, t, w in edges:
                assert 1 <= s <= N and 1 <= t <= N and s != t and w in (1, 2, 3)

            K = random.randint(1, MAX_K)
            ans = self._compute_answer(N, edges, K)
            if ans != -1:
                self.edges = edges
                self.K = K
                self.reference_answer = ans
                break

        # Build problem prompt
        edges_str = "\n".join(f"({s}, {t}, {w})" for s, t, w in self.edges)
        self.current_problem = (
            f"You are given a directed graph with {N} vertices (labeled from 1 to {N}). "
            f"Each edge is represented as a tuple (s, t, w), meaning there is a directed edge "
            f"from vertex s to vertex t with weight w. It is guaranteed that each weight w is either 1, 2, or 3. "
            f"The list of edges is:\n{edges_str}\n\n"
            f"Considering all possible paths in this graph that consist of at least one edge "
            f"(a path may start and end at any vertex, and may visit vertices or edges multiple times), "
            f"sort all such paths by their total edge weight in non-decreasing order. "
            f"Output a single integer - the total weight of the {self.K}-th path in the sorted list.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Verify answer
        try:
            user_answer = int(parsed)
            is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
            reward = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...} from the provided text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random boxed integer)."""
        upper = max(10, (self.K or 10) + self.N)
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"

    def _compute_answer(self, N: int, edges: List[Tuple[int, int, int]], K: int) -> int:
        """
        Compute the total weight of the K-th path when all paths are sorted by total edge weight.
        Returns -1 if the count never reaches K within the explored bounds.
        """

        def mat_mult(X: List[List[int]], Y: List[List[int]], cap: int) -> List[List[int]]:
            """
            Multiply two square matrices X and Y of the same dimension, capping all entries at `cap`.
            """
            D = len(X)
            Z = [[0] * D for _ in range(D)]
            for i in range(D):
                Xi = X[i]
                Zi = Z[i]
                for k, Xik in enumerate(Xi):
                    if Xik:
                        Yk = Y[k]
                        for j in range(D):
                            Zi[j] += Xik * Yk[j]
                            if Zi[j] > cap:
                                Zi[j] = cap
            return Z

        def vec_mat_mult(v: List[int], M: List[List[int]], cap: int) -> List[int]:
            """
            Multiply a row vector v by matrix M, capping all entries at `cap`.
            Returns a new row vector.
            """
            D = len(v)
            w = [0] * D
            for k, vk in enumerate(v):
                if vk:
                    Mk = M[k]
                    for j in range(D):
                        w[j] += vk * Mk[j]
                        if w[j] > cap:
                            w[j] = cap
            return w

        # Dimension of the expanded state space
        D = 3 * N + 1
        # Cap counts at K + N so we never need values above that
        cap = K + N

        # Build the base adjacency matrix g0 (size D x D)
        g0 = [[0] * D for _ in range(D)]
        # Self-loop at state 0
        g0[0][0] = 1

        # Initial row-vector A of length D
        A = [0] * D
        # Set up waiting chains and finishing transitions
        for i in range(N):
            idx1 = i * 3 + 1
            idx2 = idx1 + 1
            idx3 = idx1 + 2
            A[idx1] = 1           # can start at any vertex
            g0[idx1][0] = 1       # from "just arrived" to finish
            g0[idx2][idx1] = 1    # wait one unit
            g0[idx3][idx2] = 1    # wait two units

        # Read the edges and add the entry-point transitions
        for u, v, w in edges:
            u_idx = (u - 1) * 3 + 1
            v_idx = (v - 1) * 3 + w
            g0[u_idx][v_idx] += 1

        # Store powers g[d] = g0^(2^d)
        g = [g0]

        # Determine how many bits are needed instead of a fixed 64
        max_bits = max(1, K.bit_length()) * 2

        # Find highest d such that number of paths of length ≤ 2^d is ≥ K
        d = 0
        while True:
            if d >= max_bits:
                # Even at length 2^max_bits we don't reach K paths
                return -1
            g.append(mat_mult(g[d], g[d], cap))
            d += 1
            tmp = vec_mat_mult(A, g[d], cap)
            # Subtract N trivial finishes
            if tmp[0] - N >= K:
                break

        # Binary-lift to find exact length
        ans = 0
        for bit in range(d, -1, -1):
            tmp = vec_mat_mult(A, g[bit], cap)
            if tmp[0] - N < K:
                A = tmp
                ans += 1 << bit

        return ans