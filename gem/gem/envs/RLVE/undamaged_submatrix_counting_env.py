from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class UndamagedSubmatrixCountingEnv(Env):
    """Environment for counting the number of all-ones submatrices in a binary matrix (single-turn)."""

    def __init__(
        self,
        max_n_m: int = 20,
        fixed_n: Optional[int] = None,
        fixed_m: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n_m: Maximum value for both N and M when randomly generated. Must be >= 2.
            fixed_n: If provided, fixes the number of rows N to this value (must be >= 2).
            fixed_m: If provided, fixes the number of columns M to this value (must be >= 2).
        """
        super().__init__()
        if max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")
        if fixed_n is not None and fixed_n < 2:
            raise ValueError("fixed_n should be greater than or equal to 2")
        if fixed_m is not None and fixed_m < 2:
            raise ValueError("fixed_m should be greater than or equal to 2")

        self.max_n_m = max_n_m
        self.fixed_n = fixed_n
        self.fixed_m = fixed_m

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.matrix: Optional[List[List[int]]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a binary matrix counting problem.\n"
            "Task: Count the number of contiguous non-empty submatrices consisting entirely of 1s.\n"
            "Please provide your answer in \\boxed{...} format, as a single non-negative integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine matrix dimensions
        N = self.fixed_n if self.fixed_n is not None else random.randint(2, self.max_n_m)
        M = self.fixed_m if self.fixed_m is not None else random.randint(2, self.max_n_m)

        # Generate random matrix with a random probability of 1s
        one_probability = random.random()
        A = [[1 if random.random() < one_probability else 0 for _ in range(M)] for _ in range(N)]

        # Compute the reference answer: number of all-ones submatrices
        f = [-1] * M  # last row index where column j had a 0
        ans = 0

        for i in range(N):
            stack: List[Tuple[int, int]] = []  # (column_index, height)
            sum_arr: List[int] = []

            for j in range(M):
                if A[i][j] == 0:
                    f[j] = i
                height = i - f[j]

                while stack and stack[-1][1] > height:
                    stack.pop()
                    sum_arr.pop()

                if not stack:
                    total = height * (j + 1)
                else:
                    prev_total = sum_arr[-1]
                    prev_idx, _ = stack[-1]
                    total = prev_total + height * (j - prev_idx)

                stack.append((j, height))
                sum_arr.append(total)
                ans += total

        self.N = N
        self.M = M
        self.matrix = A
        self.reference_answer = ans

        matrix_str = "\n".join("".join(map(str, row)) for row in A)
        self.current_problem = (
            f"You are given a matrix of size {N} × {M}, where each element is either 0 or 1. "
            f"Please count the number of contiguous non-empty submatrices that consist entirely of 1s. "
            f"The matrix is:\n{matrix_str}\n\n"
            f"Note:\n"
            f"- Two submatrices are considered different if they differ in position, even if they contain identical elements.\n"
            f"- The whole matrix itself is also considered a submatrix.\n"
            f"- Output Format: A single non-negative integer — the total number of all-one submatrices.\n\n"
            f"Your final answer must be provided in \\boxed{{...}} format."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and terminate."""
        answer_str = self._parse_answer(action)

        if answer_str is None:
            info = {"error": "format_error", "message": "Answer must be provided in \\boxed{...}."}
            return TERMINAL_STATE, -0.1, True, False, info

        try:
            user_answer = int(answer_str)
        except ValueError:
            info = {"error": "invalid_answer", "message": "The boxed content is not an integer."}
            return TERMINAL_STATE, 0.0, True, False, info

        if user_answer < 0:
            info = {"error": "negative_answer", "message": "The answer must be a non-negative integer."}
            return TERMINAL_STATE, -0.1, True, False, info

        is_correct = (self.reference_answer == user_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random boxed integer action."""
        # Heuristic range for random answer
        n = self.N if self.N is not None else max(2, self.fixed_n or 2)
        m = self.M if self.M is not None else max(2, self.fixed_m or 2)
        upper = max(1, n * m * (n + m))
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"