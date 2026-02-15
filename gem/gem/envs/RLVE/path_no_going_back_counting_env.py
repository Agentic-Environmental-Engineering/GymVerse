from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Path_NoGoingBack_CountingEnv(Env):
    """Environment for counting paths in an undirected graph with a 'no immediate return' constraint."""

    def __init__(
        self,
        max_m: int = 50,
        modulo: int = 10000,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_m: Maximum number of undirected edges allowed when generating the graph (must be at least 3).
        - modulo: The modulus for the final answer.
        """
        super().__init__()
        self.max_m = max_m
        self.modulo = modulo

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Store last generated parameters for info/debug
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.T: Optional[int] = None
        self.edges: Optional[List[Tuple[int, int]]] = None

    def _get_instructions(self) -> str:
        """Return task description."""
        return (
            "You are solving a graph path counting problem.\n"
            "Given an undirected graph with N vertices labeled from 0 to N-1 and a list of edges, "
            "count the number of length-T paths from vertex 0 to vertex N-1 under the restriction "
            "that you may not immediately return to the previous vertex (i.e., no backtracking on consecutive steps).\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Parameter validation
        assert self.max_m >= 3, "max_m must be at least 3"

        # Randomly choose M and N following the original logic
        M = random.randint(3, self.max_m)

        valid_N = [N for N in range(3, (M + 1) + 1) if M <= N * (N - 1) // 2]
        N = random.choice(valid_N)
        assert N - 1 <= M <= N * (N - 1) // 2, "M must be at least N - 1 and at most N * (N - 1) / 2"

        T = random.randint(1, 2 ** N)

        # Generate edges ensuring uniqueness and basic connectivity
        edges: List[Tuple[int, int]] = []
        initial_permutation = list(range(N))
        random.shuffle(initial_permutation)
        for u, v in zip(initial_permutation, initial_permutation[1:]):
            edges.append((min(u, v), max(u, v)))
        if len(edges) < M:
            all_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set(edges))
            edges += random.sample(all_edges, M - len(edges))
        random.shuffle(edges)

        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"

        # Compute the reference answer using the original algorithm
        reference_answer = self._compute_reference_answer(N, edges, T, self.modulo)

        # Build the problem prompt
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        problem_prompt = (
            f"You are given an undirected graph with {N} vertices labeled from 0 to {N - 1}. "
            f"The graph contains the following undirected edges (no repeated edges):\n"
            f"{edges_str}\n\n"
            f"Please count the number of paths from vertex 0 to vertex {N - 1} that satisfy the following conditions:\n"
            f"- The path has exactly {T} edges.\n"
            f"- You may not immediately return to the previous vertex. That is, if you move along edge (u, v) from u to v, "
            f"you cannot move back to u in the very next step.\n\n"
            f"Output Format: Your final answer should be a single integer — the number of valid paths, modulo {self.modulo}. "
            f"Please provide your answer in \\boxed{{...}} format."
        )

        # Save state
        self.current_problem = problem_prompt
        self.reference_answer = reference_answer
        self.N = N
        self.M = M
        self.T = T
        self.edges = edges

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the action (answer) and return the result."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Reference answer not set. Call reset() before step()."

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "T": self.T,
            "modulo": self.modulo,
            "edges": self.edges,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within \\boxed{...} from the provided text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action (boxed integer)."""
        random_answer = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, N: int, edges: List[Tuple[int, int]], T: int, mod: int) -> int:
        """
        Compute the number of valid paths using the original 'edge-graph' adjacency matrix method
        with modular matrix exponentiation.
        """
        Start, End = 0, N - 1

        # Build directed edge representation with a pseudo-start edge
        x = [-1]        # x[i] = source vertex of the i-th "edge"
        y = [Start]     # y[i] = destination vertex of the i-th "edge"
        for u, v in edges:
            x.append(u)
            y.append(v)
            x.append(v)
            y.append(u)

        cnt = len(x)

        # Precompute reversal-pair for each directed edge
        pair = [-1] * cnt
        for j in range(1, cnt):
            if j % 2 == 1:
                pair[j] = j + 1
            else:
                pair[j] = j - 1

        # Build the adjacency matrix A of the "edge-graph"
        A = [[0] * cnt for _ in range(cnt)]
        for i in range(cnt):
            yi = y[i]
            Ai = A[i]
            for j in range(cnt):
                if yi == x[j] and i != j and i != pair[j]:
                    Ai[j] = 1

        # Matrix multiplication (modular), optimized for adjacency-like sparsity
        def mat_mult(M1: List[List[int]], M2: List[List[int]]) -> List[List[int]]:
            n = len(M1)
            C = [[0] * n for _ in range(n)]
            for i in range(n):
                Ai = M1[i]
                Ci = C[i]
                for k in range(n):
                    if Ai[k]:
                        aik = Ai[k]
                        Bk = M2[k]
                        for j in range(n):
                            Ci[j] = (Ci[j] + aik * Bk[j]) % mod
            return C

        # Fast exponentiation of matrix A^power
        def mat_pow(mat: List[List[int]], power: int) -> List[List[int]]:
            n = len(mat)
            # Identity matrix
            res = [[0] * n for _ in range(n)]
            for i in range(n):
                res[i][i] = 1
            base = mat
            while power:
                if power & 1:
                    res = mat_mult(res, base)
                base = mat_mult(base, base)
                power >>= 1
            return res

        # Compute A^T
        A_exp = mat_pow(A, T)

        # Sum the number of walks of length T from S to T over all directed edges ending at End
        ans = 0
        row0 = A_exp[0]
        for i in range(cnt):
            if y[i] == End:
                ans = (ans + row0[i]) % mod

        return ans