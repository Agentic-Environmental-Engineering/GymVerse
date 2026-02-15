from typing import Any, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class XorEquationCountingEnv(Env):
    """
    Environment for counting the number of solutions to the XOR equation:
    X[1] XOR X[2] XOR ... XOR X[N] = K with L <= X[i] <= R, counting modulo 'modulo'.
    The agent must provide the answer in \\boxed{...} format.

    Single-turn environment: one reset() to obtain the problem, one step() to submit the answer.
    """

    def __init__(
        self,
        N: int = 3,
        range_limit: int = 10,
        modulo: int = 10000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: Number of variables in the XOR equation (must be >= 2).
            range_limit: Upper bound for generating R in [0, range_limit] and L in [0, R] (must be >= 1).
            modulo: Modulo for the answer (default 10000).
        """
        super().__init__()
        self.N: int = N
        self.range_limit: int = range_limit
        self.modulo: int = modulo

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Parameters generated per episode
        self.L: Optional[int] = None
        self.R: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are given an equation: X[1] XOR ... XOR X[N] = K. Each variable X[i] must satisfy L <= X[i] <= R.\n"
            "Please compute how many combinations of values satisfy the equation, modulo the specified value.\n"
            "Output Format: Provide a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate configuration parameters
        assert isinstance(self.N, int) and self.N >= 2, "N should be an integer >= 2"
        assert isinstance(self.range_limit, int) and self.range_limit >= 1, "range_limit should be an integer >= 1"
        assert isinstance(self.modulo, int) and self.modulo >= 2, "modulo should be an integer >= 2"

        # Generate L and R
        R = random.randint(0, self.range_limit)
        L = random.randint(0, R)

        # Generate K as XOR of N random values within [L, R]
        K = 0
        for _ in range(1, self.N + 1):
            K ^= random.randint(L, R)

        self.L = L
        self.R = R
        self.K = K

        # Helper arithmetic functions modulo self.modulo
        def mult(a: int, b: int) -> int:
            return (a * b) % self.modulo

        def add(a: int, b: int) -> int:
            s = a + b
            return s - self.modulo if s >= self.modulo else s

        def sub(a: int, b: int) -> int:
            d = a - b
            return d + self.modulo if d < 0 else d

        def power(a: int, n: int) -> int:
            result = 1
            base = a % self.modulo
            exp = n
            while exp > 0:
                if exp & 1:
                    result = mult(result, base)
                base = mult(base, base)
                exp >>= 1
            return result

        def idx3(v0: int, v1: int, v2: int) -> int:
            return v0 + (v1 << 1) + (v2 << 2)

        def idx2(v0: int, v1: int) -> int:
            return v0 + (v1 << 1)

        class Matrix:
            """8x8 matrix with modulo arithmetic."""
            def __init__(self) -> None:
                self.v = [[0] * 8 for _ in range(8)]

            def __mul__(self, other: "Matrix") -> "Matrix":
                temp = [[0] * 8 for _ in range(8)]
                for k in range(8):
                    for i in range(8):
                        aik = self.v[i][k]
                        if aik:
                            for j in range(8):
                                temp[i][j] += aik * other.v[k][j]
                c = Matrix()
                for i in range(8):
                    for j in range(8):
                        c.v[i][j] = temp[i][j] % self.modulo
                return c

            def __pow__(self, n: int) -> "Matrix":
                result = Matrix()
                for i in range(8):
                    result.v[i][i] = 1
                base = self
                exp = n
                while exp > 0:
                    if exp & 1:
                        result = result * base
                    base = base * base
                    exp >>= 1
                return result

        # Bind modulo for Matrix class
        Matrix.modulo = self.modulo  # type: ignore[attr-defined]

        def work4(c: int, a: int, b: int, k: int, N: int) -> int:
            if a > b:
                a, b = b, a
                c ^= (N & 1)
            if b == 0:
                return power(2, N - 1) if k == 0 else 0
            w = 1 << (b.bit_length() - 1)
            if (w << 1) - 1 < k:
                return 0

            zy = Matrix()
            for v0 in (0, 1):
                for v1 in (0, 1):
                    for v2 in (0, 1):
                        row = idx3(v0, v1, v2)
                        zy.v[row][idx3(v0 ^ 1, v1, v2)] = add(zy.v[row][idx3(v0 ^ 1, v1, v2)], b - w + 1)
                        zy.v[row][idx3(v0, 1, v2)] = add(zy.v[row][idx3(v0, 1, v2)], w if v1 else 1)
                        if a & w:
                            zy.v[row][idx3(v0 ^ 1, v1, v2 ^ 1)] = add(zy.v[row][idx3(v0 ^ 1, v1, v2 ^ 1)], a - w + 1)
                            zy.v[row][idx3(v0, 1, v2 ^ 1)] = add(zy.v[row][idx3(v0, 1, v2 ^ 1)], w if v1 else 1)
                        else:
                            zy.v[row][idx3(v0, v1, v2 ^ 1)] = add(zy.v[row][idx3(v0, v1, v2 ^ 1)], a + 1)

            zy = zy ** N
            bit = 1 if (k & w) else 0
            base_count = zy.v[idx3(0, 0, 0)][idx3(bit, 1, c)]

            next_a = (a ^ w) if (a & w) else a
            next_b = b ^ w
            next_k = k ^ ((a & w) * c) ^ (w * (c ^ (N & 1)))

            return add(base_count, work4(c, next_a, next_b, next_k, N))

        def work2(b: int, k: int, N: int) -> int:
            if b == 0:
                return 1 if k == 0 else 0
            w = 1 << (b.bit_length() - 1)
            if (w << 1) - 1 < k:
                return 0
            zy = Matrix()
            for v0 in (0, 1):
                for v1 in (0, 1):
                    row = idx2(v0, v1)
                    zy.v[row][idx2(v0 ^ 1, v1)] = add(zy.v[row][idx2(v0 ^ 1, v1)], b - w + 1)
                    zy.v[row][idx2(v0, 1)] = add(zy.v[row][idx2(v0, 1)], w if v1 else 1)
            zy = zy ** N
            bit = 1 if (k & w) else 0
            base_count = zy.v[idx2(0, 0)][idx2(bit, 1)]
            next_b = b ^ w
            next_k = k ^ (w * (N & 1))
            return add(base_count, work2(next_b, next_k, N))

        # Compute reference answer
        if L == 0:
            reference_answer = work2(R, K, self.N)
        else:
            reference_answer = sub(work4(0, L - 1, R, K, self.N), work4(1, L - 1, R, K, self.N))

        self.reference_answer = reference_answer % self.modulo

        # Build problem statement
        self.current_problem = (
            f"You are given an equation: X[1] XOR ... XOR X[{self.N}] = {K}\n"
            f"Each variable X[i] must satisfy: {L} <= X[i] <= {R} for all i = 1, ..., {self.N}.\n"
            f"Please compute how many such combinations of values satisfy the equation. "
            f"Give the result modulo {self.modulo}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.N,
            "L": L,
            "R": R,
            "K": K,
            "modulo": self.modulo
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the submitted answer and return the outcome."""
        parsed = self._parse_answer(action)
        if parsed is None:
            # Format error: no boxed answer found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to interpret the boxed content as an integer
        try:
            user_answer = int(parsed)
        except ValueError:
            # Not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check with respect to modulo
        if not (0 <= user_answer < self.modulo):
            info = {
                "error": "out_of_range",
                "user_answer": user_answer,
                "reference_answer": self.reference_answer
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (self.reference_answer == user_answer)
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "user_answer": user_answer,
            "reference_answer": self.reference_answer
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the latest occurrence of \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer in boxed format)."""
        random_answer = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{random_answer}}}"