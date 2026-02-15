import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Matrix_BinaryExponentiationEnv(Env):
    """Matrix exponentiation environment - single-turn Q&A.

    Task: Given a square matrix A (size N x N) and an exponent K, compute A^K modulo M.
    The answer must be provided as N lines (each with N integers separated by spaces)
    inside a single \\boxed{...} block.
    """

    def __init__(
        self,
        max_n: int = 5,
        max_k: int = 10,
        modulo: int = 10000,
        **kwargs: Any,
    ) -> None:
        """Initialize the environment.

        Args:
            max_n: Maximum matrix dimension N. N will be sampled uniformly from [1, max_n].
            max_k: Maximum exponent K. K will be sampled uniformly from [2, max_k].
            modulo: Modulus used for computations and final results.
        """
        super().__init__()
        assert max_n >= 1, "max_n should be greater than or equal to 1"
        assert max_k >= 2, "max_k should be greater than or equal to 2"

        self.max_n = max_n
        self.max_k = max_k
        self.modulo = modulo

        # Problem state
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.A: Optional[List[List[int]]] = None

        # Reference answers
        self.gold_answer_matrix: Optional[List[List[int]]] = None
        self.reference_answer_text: Optional[str] = None

        # Cached prompt
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving matrix exponentiation problems.\n"
            "Given a square matrix A of size N×N and an integer K, compute A^K modulo M.\n"
            "Output Format: Provide your final answer as N lines, each containing N integers separated by spaces.\n"
            "You must wrap the entire matrix inside a single \\boxed{...} block.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: The problem description and instructions.
            info: Additional information (empty for this environment).
        """
        super().reset(seed)

        # Sample parameters
        N = random.randint(1, self.max_n)
        K = random.randint(2, self.max_k)
        A = [[random.randint(0, self.modulo - 1) for _ in range(N)] for _ in range(N)]

        # Store problem
        self.N = N
        self.K = K
        self.A = A

        # Compute gold answer
        self.gold_answer_matrix = self._matrix_power(A, K, self.modulo)
        self.reference_answer_text = "\n".join(
            " ".join(map(str, row)) for row in self.gold_answer_matrix
        )

        # Build problem prompt
        matrix_text = "\n".join(" ".join(map(str, row)) for row in A)
        all_zeros_example = "\n".join(" ".join("0" for _ in range(N)) for _ in range(N))

        self.current_problem = (
            "We use the integer in the i-th row and j-th column to represent the element A[i][j] of a matrix.\n\n"
            f"You are given a square matrix A of size {N}×{N}:\n{matrix_text}\n\n"
            f"Please compute the matrix A^{K} (i.e., matrix A raised to the power of {K}). "
            f"Since the values may become very large, take each element modulo {self.modulo}.\n\n"
            "Output Format:\n"
            f"Your final answer — the matrix A^{K} — should be printed as {N} lines separated by line breaks. "
            f"Each line should contain {N} integers separated by spaces. "
            "You must put the entire matrix inside a single \\boxed{...} block.\n"
            "Example (do NOT include backticks or quotes):\n"
            f"\\boxed{{\n{all_zeros_example}\n}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the provided answer.

        Args:
            action: The agent's answer text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE since this is single-turn.
            reward: 1.0 if correct; 0.0 if wrong; -0.1 on format error.
            terminated: Always True for single-turn environment.
            truncated: Always False.
            info: Additional debugging info including correctness and reference answer.
        """
        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse matrix from boxed content
        try:
            lines = [ln.strip() for ln in boxed_content.splitlines() if ln.strip() != ""]
            if self.N is None:
                return TERMINAL_STATE, -0.1, True, False, {"error": "internal_error"}
            if len(lines) != self.N:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            parsed_matrix: List[List[int]] = []
            for ln in lines:
                row = [int(x) for x in ln.split()]
                parsed_matrix.append(row)
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate shape
        if any(len(row) != self.N for row in parsed_matrix):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Compare with gold answer
        assert self.gold_answer_matrix is not None
        is_correct = parsed_matrix == self.gold_answer_matrix
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer_matrix": self.gold_answer_matrix,
            "reference_answer_text": self.reference_answer_text,
            "user_answer_matrix": parsed_matrix,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random matrix answer in boxed format."""
        if self.N is None:
            # Fallback to a 1x1 matrix if not initialized
            n = 1
        else:
            n = self.N
        random_matrix = [
            [random.randint(0, self.modulo - 1) for _ in range(n)] for _ in range(n)
        ]
        body = "\n".join(" ".join(map(str, row)) for row in random_matrix)
        return f"\\boxed{{\n{body}\n}}"

    def _matrix_multiply(
        self, A: List[List[int]], B: List[List[int]], mod: int
    ) -> List[List[int]]:
        """Multiply two square matrices modulo mod using a transposed B optimization."""
        n = len(A)
        C = [[0] * n for _ in range(n)]
        # Transpose B for cache-friendly access
        B_T = [[B[j][i] for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0
                Ai = A[i]
                BTj = B_T[j]
                for k in range(n):
                    s += Ai[k] * BTj[k]
                C[i][j] = s % mod
        return C

    def _matrix_power(
        self, A: List[List[int]], k: int, mod: int
    ) -> List[List[int]]:
        """Binary exponentiation for matrices modulo mod."""
        n = len(A)
        # Identity matrix
        result = [[0] * n for _ in range(n)]
        for i in range(n):
            result[i][i] = 1

        base = [row[:] for row in A]
        while k > 0:
            if k & 1:
                result = self._matrix_multiply(result, base, mod)
            base = self._matrix_multiply(base, base, mod)
            k >>= 1
        return result