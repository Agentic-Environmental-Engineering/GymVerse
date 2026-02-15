from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import networkx as nx
import re

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FractionalProgramming_BipartiteGraphMatchingEnv(Env):
    """Fractional Programming Bipartite Graph Matching environment - single-turn QA.

    Task:
    Given two N x N integer matrices A and B (0-indexed), find a permutation P of [0..N-1]
    that maximizes (sum_i A[i][P[i]]) / (sum_i B[i][P[i]]).

    Answer format:
    The final answer must be provided in \\boxed{...} format, containing N integers
    P[0], P[1], ..., P[N-1] separated by spaces.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 8,
        max_proportion: int = 2,
        **kwargs: Any
    ) -> None:
        super().__init__()
        # Problem size configuration
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if N is None:
            if min_n < 3:
                raise ValueError("min_n should be greater than or equal to 3")
            if min_n > max_n:
                raise ValueError("min_n should be less than or equal to max_n")

        self.N_fixed: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        # Problem generation configuration
        self.max_proportion: int = max_proportion

        # Runtime state
        self.N: Optional[int] = None
        self.A: Optional[List[List[int]]] = None
        self.B: Optional[List[List[int]]] = None
        self.gold_P: Optional[List[int]] = None
        self.gold_SumA: Optional[int] = None
        self.gold_SumB: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a fractional programming bipartite graph matching problem.\n"
            "Given two matrices A and B (N x N, 0-indexed), find a permutation P of [0..N-1]\n"
            "that maximizes (sum_i A[i][P[i]]) / (sum_i B[i][P[i]]).\n"
            "Please provide your final answer in \\boxed{...} format, containing N integers\n"
            "P[0], P[1], ..., P[N-1] separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Choose N
        if self.N_fixed is not None:
            N = self.N_fixed
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"

        # Generate matrices B and A
        B = [[random.randint(1, N) for _ in range(N)] for _ in range(N)]
        A = [[random.randint(1, self.max_proportion * b) for b in B_row] for B_row in B]

        # Compute optimal permutation using binary search + maximum weight matching
        P_opt, gold_SumA, gold_SumB = self._compute_optimal_permutation(A, B)

        # Save state
        self.N = N
        self.A = A
        self.B = B
        self.gold_P = P_opt
        self.gold_SumA = gold_SumA
        self.gold_SumB = gold_SumB
        self.reference_answer = " ".join(map(str, P_opt))

        # Build problem statement
        problem = (
            f"You are given two matrices A and B of size {N} × {N} (0-indexed).\n\n"
            f"Matrix A is (given in row-major order, with each row represented as a list of integers separated by spaces):\n"
            f"{self._format_matrix(A)}\n\n"
            f"Matrix B is (given in row-major order, with each row represented as a list of integers separated by spaces):\n"
            f"{self._format_matrix(B)}\n\n"
            f"Please find a permutation P of indices from 0 to {N-1}, i.e., P[0], P[1], ..., P[{N-1}], "
            f"such that the following value is maximized: "
            f"(A[0][P[0]] + A[1][P[1]] + ... + A[{N-1}][P[{N-1}]]) / "
            f"(B[0][P[0]] + B[1][P[1]] + ... + B[{N-1}][P[{N-1}]])\n\n"
            f"Output Format: A single line containing P[0], P[1], ..., P[{N-1}], separated by spaces, "
            f"wrapped in \\boxed{{...}}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted permutation."""
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse permutation
        try:
            tokens = boxed.strip().split()
            user_P = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate problem state
        if self.N is None or self.A is None or self.B is None or self.gold_P is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error"}

        N = self.N

        # Validate permutation
        if len(user_P) != N:
            info = {
                "error": "invalid_solution_length",
                "expected_length": N,
                "got_length": len(user_P),
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if set(user_P) != set(range(N)):
            info = {
                "error": "invalid_permutation",
                "expected_set": list(range(N)),
                "got_set": sorted(set(user_P)),
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute sums for user's permutation
        answer_SumA = sum(self.A[i][user_P[i]] for i in range(N))
        answer_SumB = sum(self.B[i][user_P[i]] for i in range(N))

        # Compute correctness by comparing ratios via cross multiplication
        gold_SumA = self.gold_SumA if self.gold_SumA is not None else 0
        gold_SumB = self.gold_SumB if self.gold_SumB is not None else 1

        # Avoid division; all values are positive
        is_correct = (answer_SumA * gold_SumB == answer_SumB * gold_SumA)

        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, user_P)),
            "gold_SumA": gold_SumA,
            "gold_SumB": gold_SumB,
            "answer_SumA": answer_SumA,
            "answer_SumB": answer_SumB,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random permutation action in boxed format."""
        if self.N is None:
            # Provide a fallback in case called before reset
            n = self.N_fixed if self.N_fixed is not None else max(self.min_n, 3)
        else:
            n = self.N
        perm = list(range(n))
        random.shuffle(perm)
        return f"\\boxed{{{' '.join(map(str, perm))}}}"

    def _format_matrix(self, M: List[List[int]]) -> str:
        """Format a matrix into row-major string with space-separated rows."""
        return "\n".join(" ".join(map(str, row)) for row in M)

    def _compute_optimal_permutation(
        self, A: List[List[int]], B: List[List[int]]
    ) -> Tuple[List[int], int, int]:
        """Compute the optimal permutation maximizing sum(A[i][P[i]])/sum(B[i][P[i]])."""
        N = len(A)

        # Initial search bounds for ratio using entries of A and B
        r = max(
            max(a / b for a, b in zip(A_row, B_row))
            for A_row, B_row in zip(A, B)
        )
        l = 0.0
        P_best: Optional[List[int]] = None

        # Binary search over ratio
        for _ in range(256):
            mid = (l + r) / 2.0
            W = [[A[i][j] - mid * B[i][j] for j in range(N)] for i in range(N)]
            tempP, total_weight = self._max_weight_matching_networkx(W)

            if total_weight >= 0.0:
                l = mid
                P_best = tempP[:]
            else:
                r = mid

        assert P_best is not None, "Failed to compute an optimal permutation"
        gold_SumA = sum(A[i][P_best[i]] for i in range(N))
        gold_SumB = sum(B[i][P_best[i]] for i in range(N))
        return P_best, gold_SumA, gold_SumB

    def _max_weight_matching_networkx(self, W: List[List[float]]) -> Tuple[List[int], float]:
        """Solve maximum weight perfect matching on a bipartite graph defined by weight matrix W."""
        N = len(W)
        G = nx.Graph()
        left_nodes = list(range(N))
        right_nodes = list(range(N, 2 * N))
        G.add_nodes_from(left_nodes, bipartite=0)
        G.add_nodes_from(right_nodes, bipartite=1)

        # Add all edges with weights
        for i in range(N):
            for j in range(N):
                G.add_edge(i, N + j, weight=W[i][j])

        matching = nx.max_weight_matching(G, maxcardinality=True)
        P = [-1] * N
        for u, v in matching:
            if u < N:
                P[u] = v - N
            else:
                P[v] = u - N

        # Compute total weight (should be perfect matching; if not, unmatched remain -1)
        total_weight = sum(W[i][P[i]] for i in range(N) if P[i] != -1)
        return P, total_weight