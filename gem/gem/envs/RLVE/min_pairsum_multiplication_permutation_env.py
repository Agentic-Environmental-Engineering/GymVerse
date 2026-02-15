import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinPairSumMultiplicationPermutationEnv(Env):
    """Environment for minimizing the product of pairwise sums under a permutation using Hungarian algorithm and convex hull traversal."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed size of the matrices. If None, a random N in [min_N, max_N] will be used on reset.
        - min_N: Minimum size for N when sampling.
        - max_N: Maximum size for N when sampling.
        """
        super().__init__()
        self.N = N
        self.min_N = min_N
        self.max_N = max_N

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_value: Optional[int] = None
        self.matrix_A: Optional[List[List[int]]] = None
        self.matrix_B: Optional[List[List[int]]] = None
        self.current_N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given two N×N matrices A and B.\n"
            "Find a permutation P of indices 0..N-1 that minimizes "
            "(sum of A[i][P[i]] for i=0..N-1) × (sum of B[i][P[i]] for i=0..N-1).\n"
            "Output Format: Provide the permutation as space-separated integers inside \\boxed{...}.\n"
            "Example: \\boxed{0 2 1}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem, and return the observation."""
        super().reset(seed)

        # Determine N
        if self.N is not None:
            assert self.N >= 3, "N should be greater than or equal to 3"
            N = self.N
        else:
            assert self.min_N >= 3, "min_N should be greater than or equal to 3"
            assert self.max_N >= self.min_N, "max_N should be greater than or equal to min_N"
            N = random.randint(self.min_N, self.max_N)

        # Generate matrices A and B with entries in [1, N]
        A = [[random.randint(1, N) for _ in range(N)] for _ in range(N)]
        B = [[random.randint(1, N) for _ in range(N)] for _ in range(N)]

        # Store for later use
        self.current_N = N
        self.matrix_A = A
        self.matrix_B = B

        # Compute the minimal product using Hungarian-based approach
        def hungarian(CX: int, CY: int, A_local, B_local, N_local, BIG_val):
            """
            Minimise Σ ( A[i][j]*CX + B[i][j]*CY ), i,j forming a permutation.
            Returns the permutation as row_match[i] = chosen column.
            """
            U = [0] * (N_local + 1)
            V = [0] * (N_local + 1)
            P = [0] * (N_local + 1)
            WAY = [0] * (N_local + 1)

            for i in range(1, N_local + 1):
                P[0] = i
                j0 = 0
                MINV = [BIG_val] * (N_local + 1)
                USED = [False] * (N_local + 1)
                USED[0] = True
                while True:
                    USED[j0] = True
                    i0 = P[j0]
                    delta = BIG_val
                    j1 = 0
                    for j in range(1, N_local + 1):
                        if not USED[j]:
                            cur = (A_local[i0 - 1][j - 1] * CX + B_local[i0 - 1][j - 1] * CY) - U[i0] - V[j]
                            if cur < MINV[j]:
                                MINV[j] = cur
                                WAY[j] = j0
                            if MINV[j] < delta:
                                delta = MINV[j]
                                j1 = j
                    for j in range(N_local + 1):
                        if USED[j]:
                            U[P[j]] += delta
                            V[j] -= delta
                        else:
                            MINV[j] -= delta
                    j0 = j1
                    if P[j0] == 0:
                        break
                while True:
                    j1 = WAY[j0]
                    P[j0] = P[j1]
                    j0 = j1
                    if j0 == 0:
                        break

            row_match = [-1] * N_local
            for j in range(1, N_local + 1):
                if P[j] != 0:
                    row_match[P[j] - 1] = j - 1
            return row_match

        class Point:
            __slots__ = ("x", "y")

            def __init__(self, x: int = 0, y: int = 0):
                self.x = x
                self.y = y

            def calc(self, A_point, B_point) -> int:
                # <self , (A.y - B.y , B.x - A.x)>
                return self.x * (A_point.y - B_point.y) + self.y * (B_point.x - A_point.x)

        def compute_min_product():
            MAX_A = max(max(row) for row in A)
            MAX_B = max(max(row) for row in B)
            SUM_BOUND = N * max(MAX_A, MAX_B)
            BIG = (MAX_A + MAX_B) * SUM_BOUND + 1

            def MM(cx: int, cy: int) -> Point:
                match = hungarian(cx, cy, A, B, N, BIG)
                sx = 0
                sy = 0
                for i in range(N):
                    j = match[i]
                    sx += A[i][j]
                    sy += B[i][j]
                return Point(sx, sy)

            POINT_A = MM(1, 0)  # minimal ΣA
            POINT_B = MM(0, 1)  # minimal ΣB
            best_val = min(POINT_A.x * POINT_A.y, POINT_B.x * POINT_B.y)

            def recurse(P_point: Point, Q_point: Point):
                nonlocal best_val
                C_point = MM(P_point.y - Q_point.y, Q_point.x - P_point.x)
                best_val = min(best_val, C_point.x * C_point.y)
                if C_point.calc(P_point, Q_point) >= P_point.calc(P_point, Q_point):
                    return
                recurse(P_point, C_point)
                recurse(C_point, Q_point)

            recurse(POINT_A, POINT_B)
            return best_val

        gold_value = compute_min_product()
        assert gold_value > 0, "The minimal product should be positive."

        self.reference_value = gold_value

        matrix_A_str = "\n".join(
            " ".join(f"A[{i}][{j}]={A[i][j]}" for j in range(N)) for i in range(N)
        )
        matrix_B_str = "\n".join(
            " ".join(f"B[{i}][{j}]={B[i][j]}" for j in range(N)) for i in range(N)
        )

        self.current_problem = (
            f"You are given two matrices A and B, each of size {N} × {N}:\n"
            f"{matrix_A_str}\n"
            f"{matrix_B_str}\n\n"
            f"You need to find a permutation P of indices from 0 to {N - 1} such that the value "
            f"(sum of A[0][P[0]], A[1][P[1]], ..., A[{N - 1}][P[{N - 1}]]) multiplied by "
            f"(sum of B[0][P[0]], B[1][P[1]], ..., B[{N - 1}][P[{N - 1}]]) is minimized.\n\n"
            f"Output Format: Provide P[0], P[1], ..., P[{N - 1}] separated by spaces, inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {"N": N}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided permutation and compute reward."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse permutation from boxed content
        try:
            tokens = boxed_content.replace(",", " ").split()
            P = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.current_N
        A = self.matrix_A
        B = self.matrix_B
        gold_value = self.reference_value

        if N is None or A is None or B is None or gold_value is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        # Validate permutation
        if len(P) != N or set(P) != set(range(N)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "permutation": P}

        # Compute user's product
        sum_A = sum(A[i][P[i]] for i in range(N))
        sum_B = sum(B[i][P[i]] for i in range(N))
        user_value = sum_A * sum_B

        is_correct = (user_value == gold_value)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_value": gold_value,
            "user_value": user_value,
            "N": N,
            "permutation": P,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random permutation action."""
        if self.current_N is None:
            # Default to a small N if not initialized; this is only for sampling purposes.
            N = 3
            perm = list(range(N))
            random.shuffle(perm)
        else:
            perm = list(range(self.current_N))
            random.shuffle(perm)
        return f"\\boxed{{{' '.join(map(str, perm))}}}"