import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MatrixPoolingEnv(Env):
    """Max pooling problem environment - single-turn Q&A in GEM format."""

    def __init__(
        self,
        max_n_m: int = 10,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "mean([gold=answer])^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 5.0,
        **kwargs
    ):
        """
        Initialize the MatrixPoolingEnv instance.

        Parameters:
            max_n_m: Maximum value for N and M (inclusive). Must be >= 3.
            wrong_format: Preserved parameter from RLVE (not used in GEM rewards).
            rewarding_strategy: Preserved parameter from RLVE (not used in GEM rewards).
            rewarding_weight: Preserved parameter from RLVE (not used in GEM rewards).
            rewarding_beta: Preserved parameter from RLVE (not used in GEM rewards).
        """
        super().__init__()
        if max_n_m < 3:
            raise ValueError("max_n_m should be greater than or equal to 3")

        # Difficulty/control parameters
        self.max_n_m = max_n_m

        # Preserved RLVE reward parameters (not used by GEM reward logic)
        self.wrong_format = wrong_format
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None
        self.matrix: Optional[List[List[int]]] = None
        self.gold_answer: Optional[List[List[int]]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a matrix and must perform a max pooling operation.\n"
            "Please provide your answer wrapped in \\boxed{...}.\n"
            "Inside the \\boxed{...}, output the pooled matrix exactly as plain text with lines separated by newlines.\n"
            "Each line should contain integers separated by single spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        self.N = random.randint(3, self.max_n_m)
        self.M = random.randint(3, self.max_n_m)
        self.K = random.randint(2, min(self.N, self.M) - 1)

        # Generate input matrix with integers in [0, N*M]
        self.matrix = [
            [random.randint(0, self.N * self.M) for _ in range(self.M)]
            for _ in range(self.N)
        ]

        # Compute gold answer via max pooling with K x K kernel
        pooled_rows = self.N - self.K + 1
        pooled_cols = self.M - self.K + 1
        self.gold_answer = [
            [
                max(
                    self.matrix[i + di][j + dj]
                    for di in range(self.K)
                    for dj in range(self.K)
                )
                for j in range(pooled_cols)
            ]
            for i in range(pooled_rows)
        ]

        # Reference answer as plain text (lines with space-separated integers)
        self.reference_answer = self._format_matrix(self.gold_answer)

        # Build problem prompt
        matrix_text = self._format_matrix(self.matrix)
        self.current_problem = (
            f"You are given a matrix of size {self.N} × {self.M}. Perform a max pooling operation "
            f"with a kernel size of {self.K} × {self.K}. In max pooling, each output cell contains "
            f"the maximum value in the corresponding {self.K} × {self.K} submatrix of the input.\n\n"
            f"The matrix is:\n{matrix_text}\n\n"
            f"Output Format: Your output should contain {pooled_rows} lines, each with {pooled_cols} integers separated by spaces. "
            f"Wrap the entire output in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse the boxed answer, validate shape, and check correctness."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: boxed content missing
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse boxed content into matrix of integers
        parsed_matrix = self._parse_matrix_text(boxed_content)
        if parsed_matrix is None:
            # Format error: parsing failed
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate shape
        assert self.N is not None and self.M is not None and self.K is not None
        expected_rows = self.N - self.K + 1
        expected_cols = self.M - self.K + 1

        if len(parsed_matrix) != expected_rows:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
        if not all(len(row) == expected_cols for row in parsed_matrix):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Check correctness
        assert self.gold_answer is not None
        is_correct = parsed_matrix == self.gold_answer
        reward: float = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": parsed_matrix,
            "N": self.N,
            "M": self.M,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} (supports multiline)."""
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _parse_matrix_text(text: str) -> Optional[List[List[int]]]:
        """Parse a plaintext matrix from lines of space-separated integers."""
        try:
            lines = text.splitlines()
            matrix: List[List[int]] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                row = list(map(int, line.split()))
                matrix.append(row)
            return matrix
        except ValueError:
            return None

    @staticmethod
    def _format_matrix(mat: List[List[int]]) -> str:
        """Format a matrix as lines with space-separated integers."""
        return "\n".join(" ".join(map(str, row)) for row in mat)

    def sample_random_action(self) -> str:
        """Sample a random action: a random matrix with the expected shape, wrapped in \\boxed{...}."""
        if self.N is None or self.M is None or self.K is None:
            # If not initialized, return a boxed empty content
            return "\\boxed{}"

        rows = self.N - self.K + 1
        cols = self.M - self.K + 1
        random_matrix = [
            [random.randint(0, self.N * self.M) for _ in range(cols)]
            for _ in range(rows)
        ]
        content = self._format_matrix(random_matrix)
        return f"\\boxed{{\n{content}\n}}"