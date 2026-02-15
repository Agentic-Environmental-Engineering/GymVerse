from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FibonacciEnv(Env):
    """Single-turn environment for computing terms of a linear recurrence modulo m."""

    def __init__(
        self,
        max_n: int = 1000000,
        modulo: int = 10000,
        **kwargs
    ):
        """
        Initialize the FibonacciEnv instance.

        Args:
            max_n: Maximum value for N (must be >= 3).
            modulo: Modulus for computations (default: 10000).
            **kwargs: Additional keyword arguments (unused, kept for compatibility).
        """
        super().__init__()
        self.max_n = max_n
        self.modulo = modulo
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.parameters: dict[str, Any] = {}

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a linear recurrence problem.\n"
            "Please provide your final answer as a single integer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Args:
            seed: Optional random seed.

        Returns:
            A tuple of (observation, info), where observation is the problem statement.
        """
        super().reset(seed)

        if self.max_n < 3:
            raise ValueError("max_n should be greater than or equal to 3")

        # Generate problem parameters
        N = random.randint(3, self.max_n)
        A1 = random.randint(0, self.modulo - 1)
        A2 = random.randint(0, self.modulo - 1)
        P = random.randint(1, self.modulo - 1)
        Q = random.randint(1, self.modulo - 1)

        # Build the problem prompt
        self.current_problem = (
            f"We have a sequence A, where A[1] = {A1}, A[2] = {A2}, "
            f"and for n > 2 the recurrence is defined as A[n] = {P} × A[n - 1] + {Q} × A[n - 2]. "
            f"Please compute A[{N}] mod {self.modulo}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Store parameters for info/debug
        self.parameters = {
            "N": N,
            "A1": A1,
            "A2": A2,
            "P": P,
            "Q": Q,
            "modulo": self.modulo,
        }

        # Compute reference answer using matrix exponentiation
        self.reference_answer = self._solve_recurrence(P, Q, A1, A2, N, self.modulo)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _solve_recurrence(self, p: int, q: int, a1: int, a2: int, n: int, m: int) -> int:
        """
        Solve the linear recurrence using matrix exponentiation:
        A[n] = p * A[n-1] + q * A[n-2], with given A[1], A[2], modulo m.
        """
        if n == 1:
            return a1 % m
        if n == 2:
            return a2 % m

        def matrix_multiply(A: list[list[int]], B: list[list[int]], mod: int) -> list[list[int]]:
            """Multiply two square matrices modulo mod using a cache-friendly approach."""
            size = len(A)
            C = [[0] * size for _ in range(size)]
            # Transpose B for better cache locality
            B_T = [[B[j][i] for j in range(size)] for i in range(size)]
            for i in range(size):
                for j in range(size):
                    s = 0
                    for k in range(size):
                        s += A[i][k] * B_T[j][k]
                    C[i][j] = s % mod
            return C

        def matrix_power(A: list[list[int]], k: int, mod: int) -> list[list[int]]:
            """Compute A^k modulo mod using binary exponentiation."""
            size = len(A)
            result = [[0] * size for _ in range(size)]
            for i in range(size):
                result[i][i] = 1
            base = [row[:] for row in A]
            while k > 0:
                if k & 1:
                    result = matrix_multiply(result, base, mod)
                base = matrix_multiply(base, base, mod)
                k >>= 1
            return result

        T = [
            [p % m, q % m],
            [1,     0    ],
        ]
        Tn = matrix_power(T, n - 2, m)
        return (Tn[0][0] * (a2 % m) + Tn[0][1] * (a1 % m)) % m

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step by evaluating the provided action (answer).

        Args:
            action: The model's output text, expected to contain \\boxed{...} with the integer answer.

        Returns:
            A tuple (observation, reward, terminated, truncated, info).
        """
        # Parse boxed answer
        parsed = self._parse_answer(action)

        if parsed is None:
            # Format error: no boxed answer found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            # Boxed content is not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "parameters": self.parameters,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the answer from \\boxed{...} in the provided text.

        Args:
            text: The model's output text.

        Returns:
            The content inside \\boxed{...} if present, otherwise None.
        """
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer answer) in boxed format."""
        random_answer = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{random_answer}}}"