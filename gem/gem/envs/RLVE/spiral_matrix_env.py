from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SpiralMatrixEnv(Env):
    """Spiral Matrix problem environment - single-turn Q&A.

    The task is to output all elements of a given M x N integer matrix in clockwise spiral order,
    starting from the top-left corner. The answer should be provided in \\boxed{...} format,
    containing exactly M*N integers separated by spaces.
    """

    def __init__(
        self,
        max_m_n: int = 10,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "mean([gold=answer])^beta",
        rewarding_beta: float = 5.0,
        rewarding_weight: float = +1.0,
        **kwargs
    ):
        """Initialize the SpiralMatrixEnv.

        Parameters:
            max_m_n: Maximum dimension for both M and N. Must be >= 2.
            wrong_format: Preserved parameter from the original environment (unused in GEM step).
            invalid_solution: Preserved parameter from the original environment (unused in GEM step).
            rewarding_strategy: Preserved parameter from the original environment (unused in GEM step).
            rewarding_beta: Preserved parameter from the original environment (unused in GEM step).
            rewarding_weight: Preserved parameter from the original environment (unused in GEM step).
        """
        super().__init__()
        if not isinstance(max_m_n, int) or max_m_n < 2:
            raise ValueError("max_m_n must be an integer >= 2")
        self.max_m_n = max_m_n

        # Preserve original parameters for compatibility, though they are not used in GEM reward settings.
        self.rewards_config = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_beta": rewarding_beta,
            "rewarding_weight": rewarding_weight,
        }

        # State variables
        self.M: Optional[int] = None
        self.N: Optional[int] = None
        self.matrix: Optional[List[List[int]]] = None
        self.gold_answer: Optional[List[int]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a 2D integer matrix. Return all elements of the matrix in a clockwise spiral order, "
            "starting from the top-left corner. The spiral traversal proceeds by moving right, then down, then left, "
            "then up, and continues inward until all elements are visited exactly once.\n\n"
            "Output requirement:\n"
            "- Your final answer must be provided inside \\boxed{...}.\n"
            "- Inside the box, output exactly M*N integers separated by spaces, in the correct spiral order.\n\n"
            "Example formatting:\n"
            "\\boxed{1 2 3 6 9 8 7 4 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new spiral matrix problem."""
        super().reset(seed)

        # Generate matrix dimensions
        self.M = random.randint(2, self.max_m_n)
        self.N = random.randint(2, self.max_m_n)

        # Generate matrix with integers in range [1, M*N]
        self.matrix = [[random.randint(1, self.M * self.N) for _ in range(self.N)] for _ in range(self.M)]

        # Compute gold answer and reference answer string
        self.gold_answer = self._compute_spiral(self.matrix)
        self.reference_answer = " ".join(map(str, self.gold_answer))

        # Build problem prompt
        matrix_str = "\n".join(" ".join(map(str, row)) for row in self.matrix)
        self.current_problem = (
            f"You are given a 2D integer matrix of size {self.M} x {self.N}:\n"
            f"{matrix_str}\n\n"
            f"Return all elements of the matrix in a clockwise spiral order, starting from the top-left corner.\n\n"
            f"Output Format: Your final answer should be a single line of {self.M * self.N} integers separated by spaces, "
            f"enclosed in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "M": self.M,
            "N": self.N,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step to verify the user's answer.
        
        Reward settings:
        - Correct answer: 1.0
        - Wrong answer: 0.0
        - Format error (no \\boxed{...}): -0.1
        """
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error: boxed content not found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse the boxed content into a list of integers
        tokens = boxed_content.strip().split()
        try:
            user_answer_list = list(map(int, tokens))
        except ValueError:
            # Non-integer tokens inside the box
            info = {
                "error": "invalid_answer",
                "user_answer_raw": boxed_content
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate length
        assert self.M is not None and self.N is not None and self.gold_answer is not None
        expected_len = self.M * self.N
        if len(user_answer_list) != expected_len:
            info = {
                "error": "invalid_length",
                "expected_length": expected_len,
                "received_length": len(user_answer_list),
                "reference_answer": self.reference_answer,
                "user_answer": user_answer_list
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compare with gold answer
        is_correct = (user_answer_list == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer_list
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_spiral(self, matrix: List[List[int]]) -> List[int]:
        """Compute the spiral order traversal of the given matrix."""
        res: List[int] = []
        if not matrix:
            return res

        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1

        while top <= bottom and left <= right:
            # Traverse from left to right
            for i in range(left, right + 1):
                res.append(matrix[top][i])
            top += 1

            # Traverse downwards
            for i in range(top, bottom + 1):
                res.append(matrix[i][right])
            right -= 1

            # Traverse from right to left
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    res.append(matrix[bottom][i])
                bottom -= 1

            # Traverse upwards
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    res.append(matrix[i][left])
                left += 1

        return res

    def sample_random_action(self) -> str:
        """Sample a random action: a random sequence of integers inside \\boxed{...}."""
        if self.M is None or self.N is None:
            # Fallback if reset has not been called
            return "\\boxed{0}"
        count = self.M * self.N
        # Randomly sample integers; not necessarily correct
        random_sequence = [str(random.randint(1, self.M * self.N)) for _ in range(count)]
        return f"\\boxed{{{' '.join(random_sequence)}}}"