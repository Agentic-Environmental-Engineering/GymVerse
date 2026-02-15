import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MatrixPermutationEquivalenceEnv(Env):
    """Environment for the Matrix Permutation Equivalence problem - single-turn Q&A.

    The task is to find permutations of row and column indices such that
    permuting the rows and columns of matrix A results in matrix B.
    The agent must return the solution in a single \\boxed{...} block
    containing two lines:
      - First line: row permutation (space-separated integers)
      - Second line: column permutation (space-separated integers)
    """

    def __init__(self, max_n_m: int = 8, **kwargs) -> None:
        """Initialize the environment.

        Args:
            max_n_m: Maximum dimension size for both N and M (must be >= 2).
        """
        super().__init__()
        if max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")
        self.max_n_m: int = max_n_m

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.A: Optional[List[List[int]]] = None
        self.B: Optional[List[List[int]]] = None
        self.reference_row_permutation: Optional[List[int]] = None
        self.reference_column_permutation: Optional[List[int]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task description."""
        return (
            "You are given two matrices A and B of size N × M, where each element is either 0 or 1. "
            "Both matrices are 0-indexed.\n\n"
            "Please find:\n"
            "- a permutation of the row indices a[0], ..., a[N-1] (a reordering of 0 to N-1), and\n"
            "- a permutation of the column indices b[0], ..., b[M-1] (a reordering of 0 to M-1),\n"
            "such that after permuting the rows and columns of matrix A accordingly, the resulting matrix matches B. "
            "Formally, for all 0 ≤ i < N and 0 ≤ j < M, it must hold that A[a[i]][b[j]] = B[i][j].\n\n"
            "Output Format: Provide your answer inside a single \\boxed{...} block, containing exactly two lines:\n"
            "- First line: a[0] ... a[N-1]\n"
            "- Second line: b[0] ... b[M-1]\n"
            "Use spaces to separate the integers on each line. Do not include extra text outside the box.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: The instruction and the generated problem as a string.
            info: An empty dict (no extra info on reset).
        """
        super().reset(seed)

        # Generate N, M in [2, max_n_m]
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Generate A with a random probability of ones
        one_probability = random.random()
        self.A = [
            [1 if random.random() < one_probability else 0 for _ in range(self.M)]
            for _ in range(self.N)
        ]

        # Generate random permutations for rows and columns
        self.reference_row_permutation = list(range(self.N))
        random.shuffle(self.reference_row_permutation)
        self.reference_column_permutation = list(range(self.M))
        random.shuffle(self.reference_column_permutation)

        # Construct B by permuting A
        self.B = [
            [
                self.A[self.reference_row_permutation[i]][self.reference_column_permutation[j]]
                for j in range(self.M)
            ]
            for i in range(self.N)
        ]

        # Reference answer as two lines
        row_line = " ".join(map(str, self.reference_row_permutation))
        col_line = " ".join(map(str, self.reference_column_permutation))
        self.reference_answer = f"{row_line}\n{col_line}"

        # Build the problem statement
        assert self.N is not None and self.M is not None and self.A is not None and self.B is not None
        a_str = "\n".join("".join(map(str, row)) for row in self.A)
        b_str = "\n".join("".join(map(str, row)) for row in self.B)
        self.current_problem = (
            f"You are given two matrices A and B of size {self.N} × {self.M}, where each element is either 0 or 1. "
            f"Both matrices are 0-indexed.\n\n"
            f"Please find:\n"
            f"- a permutation of the row indices a[0], ..., a[{self.N - 1}] (a reordering of 0 to {self.N - 1}), and\n"
            f"- a permutation of the column indices b[0], ..., b[{self.M - 1}] (a reordering of 0 to {self.M - 1}),\n"
            f"such that after permuting the rows and columns of matrix A accordingly, the resulting matrix matches B. "
            f"Formally, for all 0 ≤ i < {self.N} and 0 ≤ j < {self.M}, it must hold that A[a[i]][b[j]] = B[i][j].\n\n"
            f"A is given as follows:\n{a_str}\n\n"
            f"B is given as follows:\n{b_str}\n\n"
            f"Output Format: Provide your answer inside a single \\boxed{{...}} block, containing exactly two lines:\n"
            f"- First line: a[0] ... a[{self.N - 1}]\n"
            f"- Second line: b[0] ... b[{self.M - 1}]\n"
            f"Use spaces to separate the integers on each line."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the user's answer.

        Args:
            action: The agent's response text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE since this is a single-turn environment.
            reward: 1.0 if correct, 0.0 if incorrect, -0.1 if format error.
            terminated: True (always single-turn).
            truncated: False.
            info: Additional information such as correctness and reference answer.
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            parsed = self._process_answer_text(boxed_content)
            if parsed is None:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

            row_perm, col_perm = parsed
        except Exception:
            # Any parsing errors are treated as format errors
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate permutations
        assert self.N is not None and self.M is not None and self.A is not None and self.B is not None

        if not (len(row_perm) == self.N and set(row_perm) == set(range(self.N))):
            info = {
                "error": "invalid_solution",
                "reason": "row_permutation_invalid",
                "user_row_permutation": row_perm,
                "user_column_permutation": col_perm,
                "reference_row_permutation": self.reference_row_permutation,
                "reference_column_permutation": self.reference_column_permutation,
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if not (len(col_perm) == self.M and set(col_perm) == set(range(self.M))):
            info = {
                "error": "invalid_solution",
                "reason": "column_permutation_invalid",
                "user_row_permutation": row_perm,
                "user_column_permutation": col_perm,
                "reference_row_permutation": self.reference_row_permutation,
                "reference_column_permutation": self.reference_column_permutation,
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Apply permutations and compare
        candidate_B = [
            [self.A[row_perm[i]][col_perm[j]] for j in range(self.M)]
            for i in range(self.N)
        ]
        is_correct = (candidate_B == self.B)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "user_row_permutation": row_perm,
            "user_column_permutation": col_perm,
            "reference_row_permutation": self.reference_row_permutation,
            "reference_column_permutation": self.reference_column_permutation,
            "reference_answer": self.reference_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence.

        Args:
            text: The full response text.

        Returns:
            The content inside \\boxed{...} or None if not found.
        """
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _process_answer_text(self, answer: str) -> Optional[Tuple[List[int], List[int]]]:
        """Process the boxed content to extract the two permutations.

        The content should contain exactly two non-empty lines:
          - First line: row permutation (space-separated integers)
          - Second line: column permutation (space-separated integers)

        Args:
            answer: The text extracted from inside \\boxed{...}.

        Returns:
            A tuple (row_permutation, column_permutation) or None if parsing fails.
        """
        lines = [line.strip() for line in answer.splitlines() if line.strip() != ""]
        if len(lines) != 2:
            return None
        try:
            row_perm = list(map(int, lines[0].split()))
            col_perm = list(map(int, lines[1].split()))
            return row_perm, col_perm
        except ValueError:
            return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format.

        If the environment has not been reset yet, defaults to N = M = 2.
        """
        n = self.N if self.N is not None else 2
        m = self.M if self.M is not None else 2
        row_perm = list(range(n))
        col_perm = list(range(m))
        random.shuffle(row_perm)
        random.shuffle(col_perm)
        row_line = " ".join(map(str, row_perm))
        col_line = " ".join(map(str, col_perm))
        return f"\\boxed{{{row_line}\n{col_line}}}"