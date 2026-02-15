from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
import re


class PrefixConcatenationEnv(Env):
    """Prefix Concatenation problem environment - single-turn Q&A.

    Task:
      Define Concatenate(n) as the number formed by concatenating all positive integers from 1 to n in order.
      For example, when n = 12, Concatenate(12) = 123456789101112.
      Compute Concatenate(N) mod M.
    """

    def __init__(
        self,
        max_n: int = 1_000_000,
        max_modulo: int = 1_000_000,
        **kwargs: Any
    ):
        super().__init__()
        self.max_n = max_n
        self.max_modulo = max_modulo

        # Runtime variables
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a prefix concatenation modular arithmetic problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem, and return the observation and info."""
        super().reset(seed)

        # Parameter validation
        if self.max_n < 2:
            raise ValueError("max_n should be greater than or equal to 2")
        if self.max_modulo < 3:
            raise ValueError("max_modulo should be greater than or equal to 3")

        # Sample parameters
        self.N = random.randint(2, self.max_n)
        self.M = random.randint(3, self.max_modulo)

        # Build problem statement
        self.current_problem = (
            "Define Concatenate(n) as the number formed by concatenating all positive integers from 1 to n in order. "
            "For example, when n = 12, Concatenate(12) = 123456789101112.\n\n"
            f"Your task is to compute Concatenate({self.N}) mod {self.M}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_concatenate_mod(self.N, self.M)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted answer and return the terminal transition."""
        # Extract boxed answer
        extracted = self._parse_answer(action)
        if extracted is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integer answer
        try:
            user_answer = int(extracted)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content as the answer."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        upper = (self.M - 1) if (self.M is not None and self.M > 0) else (self.max_modulo - 1)
        random_answer = random.randint(0, max(0, upper))
        return f"\\boxed{{{random_answer}}}"

    def _compute_concatenate_mod(self, N: int, M: int) -> int:
        """Compute Concatenate(N) mod M using block processing with matrix exponentiation."""
        # 3x3 matrix multiplication modulo M
        def mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
            return [
                [
                    (A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]) % M
                    for j in range(3)
                ]
                for i in range(3)
            ]

        # Fast exponentiation of a 3x3 matrix
        def mat_pow(base: list[list[int]], exp: int) -> list[list[int]]:
            R = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
            Bm = [row[:] for row in base]
            e = exp
            while e:
                if e & 1:
                    R = mat_mul(R, Bm)
                Bm = mat_mul(Bm, Bm)
                e >>= 1
            return R

        # Multiply 3x3 matrix with a 3x1 vector modulo M
        def mat_vec_mul(A: list[list[int]], v: list[int]) -> list[int]:
            return [
                (A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2]) % M,
                (A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2]) % M,
                (A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]) % M,
            ]

        # Initial state corresponds to the start before appending any numbers
        state = [0, 1, 1]
        start = 1
        power_of_10 = 10

        while start <= N:
            end = min(N, power_of_10 - 1)
            block_size = end - start + 1

            B = [
                [power_of_10 % M, 1, 0],
                [0,               1, 1],
                [0,               0, 1],
            ]

            Bk = mat_pow(B, block_size)
            state = mat_vec_mul(Bk, state)

            start = power_of_10
            power_of_10 *= 10

        return state[0] % M