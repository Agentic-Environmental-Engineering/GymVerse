from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NotContainingStringCountingEnv(Env):
    """Environment for counting binary strings that do NOT contain a given forbidden substring (single-turn)."""

    def __init__(
        self,
        max_n: int = 100,
        max_m: int = 20,
        max_mod: int = 10000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: Maximum length N of the binary string (must be >= 3).
            max_m: Maximum length M of the forbidden pattern (must be >= 2).
            max_mod: Maximum modulo value (must be >= 2). Default is 10000 to match the original environment.
        """
        super().__init__()
        self.max_n = max_n
        self.max_m = max_m
        self.max_mod = max_mod

        # Current instance parameters
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.pattern: Optional[str] = None
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics counting problem on binary strings.\n"
            "Task: Count the number of binary (0/1) strings of a given length that do NOT contain a given substring.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate configuration
        assert self.max_n >= 3, "max_n should be greater than or equal to 3"
        assert self.max_m >= 2, "max_m should be greater than or equal to 2"
        assert self.max_mod >= 2, "max_mod should be greater than or equal to 2"

        # Sample parameters
        N = random.randint(3, self.max_n)
        M_upper = min(N - 1, self.max_m)
        # Ensure M_upper >= 2 (since N >= 3, N-1 >= 2, and max_m >= 2, this holds)
        M = random.randint(2, M_upper)

        one_probability = random.random()
        pattern = "".join("1" if random.random() < one_probability else "0" for _ in range(M))
        MOD = random.randint(2, self.max_mod)

        # Store for later use
        self.N = N
        self.M = M
        self.pattern = pattern
        self.MOD = MOD

        # Build the problem statement
        self.current_problem = (
            f"Please count the number of binary (0/1) strings of length {N} that do NOT contain the substring '{pattern}'.\n"
            f"Output the result modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single non-negative integer strictly less than {MOD}, in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference(N, pattern, MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted answer and terminate immediately."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate range: must be in [0, MOD)
        if self.MOD is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error"}
        if not (0 <= user_answer < self.MOD):
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "out_of_range",
                "mod": self.MOD
            }

        # Check correctness
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "pattern": self.pattern,
            "MOD": self.MOD
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in required format."""
        if self.MOD is None:
            # Fallback: default to a small modulo if not initialized
            random_answer = random.randint(0, 9999)
        else:
            random_answer = random.randint(0, self.MOD - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference(self, N: int, pattern: str, MOD: int) -> int:
        """Compute the reference answer using KMP-based automaton and matrix exponentiation."""

        def build_prefix(pat: str) -> list[int]:
            """Build the KMP prefix function for the pattern."""
            m = len(pat)
            pi = [0] * m
            j = 0
            for i in range(1, m):
                while j > 0 and pat[i] != pat[j]:
                    j = pi[j - 1]
                if pat[i] == pat[j]:
                    j += 1
                pi[i] = j
            return pi

        def multiply_matrices(A: list[list[int]], B: list[list[int]], mod: int) -> list[list[int]]:
            """Multiply two square matrices A and B under modulo mod."""
            size = len(A)
            C = [[0] * size for _ in range(size)]
            for i in range(size):
                for k in range(size):
                    aik = A[i][k]
                    if aik:
                        for j in range(size):
                            C[i][j] = (C[i][j] + aik * B[k][j]) % mod
            return C

        def matrix_power(matrix: list[list[int]], exponent: int, mod: int) -> list[list[int]]:
            """Binary exponentiation of a square matrix under modulo mod."""
            size = len(matrix)
            result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
            base = matrix
            e = exponent
            while e > 0:
                if e & 1:
                    result = multiply_matrices(result, base, mod)
                base = multiply_matrices(base, base, mod)
                e >>= 1
            return result

        M = len(pattern)
        pi = build_prefix(pattern)

        # Build transition matrix of size (M+1) with an absorbing forbidden state M
        size = M + 1
        B = [[0] * size for _ in range(size)]

        # States 0..M-1: matched prefix length so far
        for state in range(M):
            for digit in ("0", "1"):
                k = state
                while k > 0 and digit != pattern[k]:
                    k = pi[k - 1]
                if digit == pattern[k]:
                    k += 1
                B[state][k] += 1

        # State M is absorbing with both digits
        B[M][M] = 2

        # Compute B^N mod MOD
        Bn = matrix_power(B, N, MOD)

        # Sum over non-forbidden states 0..M-1
        result = sum(Bn[0][j] for j in range(M)) % MOD
        return result