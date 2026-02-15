import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class KakurasuEnv(Env):
    """Kakurasu puzzle environment - single-turn Q&A.

    The task is to fill an N × M binary grid so that:
    - For each row i, the sum of the column indices (1-indexed) of cells with value 1 equals A[i].
    - For each column j, the sum of the row indices (1-indexed) of cells with value 1 equals B[j].

    Submit the grid inside \\boxed{...} with exactly N lines, each of length M, consisting only of '0' and '1'.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(satisfied/all)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 10.0,
        min_one_rate: float = 0.1,
        max_one_rate: float = 0.9,
        **kwargs
    ):
        super().__init__()
        # Parameters controlling instance generation and legacy reward config (kept for compatibility)
        self.max_n_m = max_n_m
        self.wrong_format = wrong_format
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta
        self.min_one_rate = min_one_rate
        self.max_one_rate = max_one_rate

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.B: Optional[List[int]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Kakurasu-like binary grid puzzle.\n"
            "Fill an N × M grid with 0s and 1s so that row and column weighted sums match given arrays A and B.\n"
            "Please provide your answer inside \\boxed{...}.\n"
            "The boxed content must contain exactly N lines, each with M characters ('0' or '1', no separators).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate parameters
        assert self.max_n_m >= 3, "max_n_m should be greater than or equal to 3"

        # Sample dimensions
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Generate a random solution grid and compute A, B from it
        one_rate = random.uniform(self.min_one_rate, self.max_one_rate)
        grid = [["1" if random.random() < one_rate else "0" for _ in range(self.M)] for _ in range(self.N)]
        self.reference_answer = "\n".join("".join(row) for row in grid)

        # Compute row-weighted sums (A) and column-weighted sums (B)
        self.A = [sum((j + 1) for j in range(self.M) if grid[i][j] == "1") for i in range(self.N)]
        self.B = [sum((i + 1) for i in range(self.N) if grid[i][j] == "1") for j in range(self.M)]

        # Build problem statement
        a_str = " ".join(f"A[{i + 1}]={a}" for i, a in enumerate(self.A))
        b_str = " ".join(f"B[{j + 1}]={b}" for j, b in enumerate(self.B))
        self.current_problem = (
            f"You are given a {self.N} × {self.M} grid (1-indexed). Fill the grid with '0's and '1's such that:\n"
            f"- For each row i, the sum of the column indices where there are '1's is equal to A[i]. Array A is given as: {a_str}\n"
            f"- For each column j, the sum of the row indices where there are '1's is equal to B[j]. Array B is given as: {b_str}\n\n"
            f"Output Format: Your final answer must be inside \\boxed{{...}}. "
            f"The boxed content should contain exactly {self.N} lines, each containing {self.M} characters ('0' or '1', with no separators)."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted grid and return reward and termination."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: no boxed content
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "No \\boxed{...} content found."}

        # Process boxed content into grid lines
        lines = [line.strip() for line in boxed_content.splitlines() if line.strip()]
        if self.N is None or self.M is None or self.A is None or self.B is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error", "message": "Problem not initialized."}

        # Validate grid dimensions and content
        if len(lines) != self.N:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Incorrect number of lines."}
        if not all(len(row) == self.M for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Incorrect row length."}
        if not all(ch in "01" for row in lines for ch in row):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "message": "Grid must contain only '0' and '1'."}

        # Compute user A and B arrays from the submitted grid
        user_A = [sum((j + 1) for j in range(self.M) if lines[i][j] == "1") for i in range(self.N)]
        user_B = [sum((i + 1) for i in range(self.N) if lines[i][j] == "1") for j in range(self.M)]

        # Compare against target A and B
        satisfied_rows = sum(int(a == gold_a) for a, gold_a in zip(user_A, self.A))
        satisfied_cols = sum(int(b == gold_b) for b, gold_b in zip(user_B, self.B))
        is_correct = (satisfied_rows + satisfied_cols) == (self.N + self.M)

        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied_rows": satisfied_rows,
            "satisfied_cols": satisfied_cols,
            "required_rows": self.N,
            "required_cols": self.M,
            "A": self.A,
            "B": self.B,
            "reference_answer": self.reference_answer,
            "user_grid": "\n".join(lines),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the latest \\boxed{...} content as the user's answer."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action that matches the generated reference grid."""
        # Return the reference grid wrapped in boxed format
        if self.reference_answer is None:
            # If not initialized, sample a trivial small grid
            random_grid = "0"
            return f"\\boxed{{{random_grid}}}"
        return f"\\boxed{{{self.reference_answer}}}"