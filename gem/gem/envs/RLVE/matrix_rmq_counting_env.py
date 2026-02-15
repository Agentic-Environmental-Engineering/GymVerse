from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MatrixRMQCountingEnv(Env):
    """Matrix RMQ counting environment - single-turn Q&A."""

    def __init__(
        self,
        min_N: int = 2,
        max_N: int = 5,
        H_W_range: int = 2,
        max_MOD: int = 1000000,
        **kwargs
    ):
        """
        Initialize the MatrixRMQCountingEnv instance.

        Args:
            min_N: Minimum number of constraints (must be >= 2).
            max_N: Maximum number of constraints (must be >= min_N).
            H_W_range: Scale factor for random H and W relative to N.
            max_MOD: Maximum modulus value (inclusive upper bound for sampling).
        """
        super().__init__()
        assert isinstance(min_N, int) and isinstance(max_N, int), "min_N and max_N must be integers"
        assert min_N >= 2, "min_N should be greater than or equal to 2"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"
        assert isinstance(H_W_range, int) and H_W_range >= 1, "H_W_range must be an integer >= 1"
        assert isinstance(max_MOD, int) and max_MOD >= 2, "max_MOD must be an integer >= 2"

        self.min_N = min_N
        self.max_N = max_N
        self.H_W_range = H_W_range
        self.max_MOD = max_MOD

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Store parameters of the current instance for info/debug
        self.current_H: Optional[int] = None
        self.current_W: Optional[int] = None
        self.current_M: Optional[int] = None
        self.current_N: Optional[int] = None
        self.current_constraints: Optional[List[tuple]] = None
        self.current_MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instruction string."""
        return (
            "You are solving a matrix counting problem with RMQ constraints.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample problem parameters
        N = random.randint(self.min_N, self.max_N)
        H = random.randint(1, N * self.H_W_range)
        W = random.randint(1, N * self.H_W_range)
        M = random.randint(1, (N * self.H_W_range) ** 2)

        # Create a random matrix A with entries in [1, M]
        A = [[random.randint(1, M) for _ in range(W)] for _ in range(H)]

        # Generate constraints from random submatrices based on A's maxima
        constraints: List[tuple] = []
        for _ in range(N):
            row_length = random.randint(1, H)
            col_length = random.randint(1, W)
            x1 = random.randint(1, H - row_length + 1)
            y1 = random.randint(1, W - col_length + 1)
            x2 = x1 + row_length - 1
            y2 = y1 + col_length - 1
            v = max(A[i - 1][j - 1] for i in range(x1, x2 + 1) for j in range(y1, y2 + 1))
            constraints.append((x1, y1, x2, y2, v))

        MOD = random.randint(2, self.max_MOD)

        # Compute the reference answer using inclusion-exclusion with coordinate compression
        reference_answer = self._compute_reference_answer(H, W, M, N, constraints, MOD)

        # Construct the problem prompt
        constraints_str_lines = [
            f"max(A[{x1} : {x2} + 1, {y1} : {y2} + 1]) = {v}"
            for (x1, y1, x2, y2, v) in constraints
        ]
        constraints_str = "\n".join(constraints_str_lines)

        problem = (
            f"Count the number of matrices A of size {H} × {W} (1-indexed, meaning row indices range "
            f"from 1 to {H} and column indices from 1 to {W}) such that:\n"
            f"1. Each element of A is an integer between 1 and {M}, inclusive.\n"
            f"2. The matrix satisfies the following {N} constraints, where max(A[x1 : x2 + 1, y1 : y2 + 1]) "
            f"denotes the maximum value in the contiguous submatrix defined by the corners (x1, y1) and "
            f"(x2, y2) (inclusive):\n{constraints_str}\n\n"
            f"Output a single integer — the number of such matrices modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Persist current problem state
        self.current_problem = problem
        self.reference_answer = reference_answer
        self.current_H = H
        self.current_W = W
        self.current_M = M
        self.current_N = N
        self.current_constraints = constraints
        self.current_MOD = MOD

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(
        self,
        H: int,
        W: int,
        M: int,
        N: int,
        constraints: List[tuple],
        MOD: int
    ) -> int:
        """Compute the reference answer using inclusion-exclusion with coordinate compression."""
        pos = []
        X = [1, H + 1]
        Y = [1, W + 1]

        # Read constraints and collect coordinates for compression
        for x1, y1, x2, y2, v in constraints:
            assert 1 <= x1 <= x2 <= H, "Invalid x1, x2 range"
            assert 1 <= y1 <= y2 <= W, "Invalid y1, y2 range"
            assert 1 <= v <= M, "Invalid value v"
            pos.append((x1, y1, x2 + 1, y2 + 1, v))
            X.append(x1)
            X.append(x2 + 1)
            Y.append(y1)
            Y.append(y2 + 1)

        # Coordinate compression
        X = sorted(set(X))
        Y = sorted(set(Y))
        xi = {x: i for i, x in enumerate(X)}
        yi = {y: i for i, y in enumerate(Y)}

        # Precompute block ranges for each constraint
        ranges = []
        for x1, y1, x2p, y2p, v in pos:
            xl = xi[x1]
            xr = xi[x2p]
            yl = yi[y1]
            yr = yi[y2p]
            ranges.append((xl, xr, yl, yr, v))

        # Number of blocks in compressed grid
        Wb = len(X) - 1
        Hb = len(Y) - 1

        ans = 0
        # Inclusion-exclusion over subsets of constraints
        for mask in range(1 << N):
            # Initialize each block's max allowed value to M
            arr = [[M] * Hb for _ in range(Wb)]
            # Apply each constraint, reducing allowed max by 1 if in the subset
            for j in range(N):
                bit = (mask >> j) & 1
                xl, xr, yl, yr, v = ranges[j]
                limit = v - bit
                for xi_ in range(xl, xr):
                    row = arr[xi_]
                    for yi_ in range(yl, yr):
                        if row[yi_] > limit:
                            row[yi_] = limit

            # Compute number of fillings for this configuration
            tmp = 1
            for xi_ in range(Wb):
                dx = X[xi_ + 1] - X[xi_]
                for yi_ in range(Hb):
                    dy = Y[yi_ + 1] - Y[yi_]
                    area = dx * dy
                    val = arr[xi_][yi_]
                    tmp = (tmp * pow(val, area, MOD)) % MOD
                    if tmp == 0:
                        break
                if tmp == 0:
                    break

            # Inclusion-exclusion sign
            if bin(mask).count('1') & 1:
                ans = (ans - tmp) % MOD
            else:
                ans = (ans + tmp) % MOD

        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        answer_text = self._parse_answer(action)

        if answer_text is None:
            # Format error: missing or malformed \boxed{...}
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            # Not a valid integer
            info = {"error": "invalid_answer", "reference_answer": self.reference_answer}
            return TERMINAL_STATE, 0.0, True, False, info

        # Check correctness
        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "H": self.current_H,
            "W": self.current_W,
            "M": self.current_M,
            "N": self.current_N,
            "constraints": self.current_constraints,
            "MOD": self.current_MOD,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        mod = self.current_MOD if self.current_MOD is not None else max(2, self.max_MOD)
        random_answer = random.randint(0, mod - 1)
        return f"\\boxed{{{random_answer}}}"