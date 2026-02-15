import math
import random
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PanSolarPanelsEnv(Env):
    """Pan Solar Panels problem environment - Single-turn Q&A.

    Task:
    Given two integer intervals [A, B] and [C, D], choose integers X and Y such that:
      - A ≤ X ≤ B
      - C ≤ Y ≤ D
    and the greatest common divisor gcd(X, Y) is maximized.

    The answer must be provided as two integers separated by a space, wrapped in \\boxed{...}, e.g., \\boxed{12 18}.
    """

    def __init__(
        self,
        max_a_b_c_d: int = 1000,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 3.0,
        **kwargs: Any,
    ):
        super().__init__()
        # Difficulty/control parameter
        self.max_a_b_c_d = max_a_b_c_d
        assert self.max_a_b_c_d >= 4, "MAX_A_B_C_D should be greater than or equal to 4"

        # Legacy reward settings from original environment (kept for compatibility; not used in GEM scoring)
        self.wrong_format = wrong_format
        self.invalid_solution = invalid_solution
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # Runtime state
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.C: Optional[int] = None
        self.D: Optional[int] = None
        self.reference_max_gcd: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory optimization task.\n"
            "Given two integer intervals [A, B] and [C, D], choose integers X and Y such that:\n"
            "- A ≤ X ≤ B\n"
            "- C ≤ Y ≤ D\n"
            "- gcd(X, Y) is maximized\n\n"
            "Output Format: Provide two integers X and Y separated by a space, wrapped in \\boxed{...}.\n"
            "Example: \\boxed{12 18}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate valid intervals
        while True:
            numbers = [random.randint(1, self.max_a_b_c_d) for _ in range(4)]
            numbers.sort()
            A, B, C, D = numbers
            if A <= B < C <= D:
                break

        # Randomly swap the intervals to avoid positional bias
        if random.random() < 0.5:
            A, B, C, D = C, D, A, B

        self.A, self.B, self.C, self.D = A, B, C, D

        # Compute the reference maximal gcd
        self.reference_max_gcd = self._solve_max_gcd(self.A, self.B, self.C, self.D)

        # Build problem description
        self.current_problem = (
            f"Output two integers X and Y (separated by a space), such that:\n"
            f"- {self.A} ≤ X ≤ {self.B}\n"
            f"- {self.C} ≤ Y ≤ {self.D}\n"
            f"- gcd(X, Y) is maximized\n\n"
            f"Your answer must be in \\boxed{{X Y}} format."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer.

        Rewards:
        - Correct answer: 1.0
        - Wrong answer or invalid solution: 0.0
        - Format error: -0.1
        """
        # Ensure problem is initialized
        if self.A is None or self.B is None or self.C is None or self.D is None or self.reference_max_gcd is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse two integers X and Y from boxed content
        parts = boxed.strip().split()
        if len(parts) != 2:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        try:
            X = int(parts[0])
            Y = int(parts[1])
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range validation
        in_range = (self.A <= X <= self.B) and (self.C <= Y <= self.D)
        if not in_range:
            info = {
                "error": "out_of_range",
                "A": self.A,
                "B": self.B,
                "C": self.C,
                "D": self.D,
                "user_X": X,
                "user_Y": Y,
                "reference_max_gcd": self.reference_max_gcd,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        user_gcd = math.gcd(X, Y)
        is_correct = (user_gcd == self.reference_max_gcd)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "user_X": X,
            "user_Y": Y,
            "user_gcd": user_gcd,
            "reference_max_gcd": self.reference_max_gcd,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re

        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _solve_max_gcd(self, A: int, B: int, C: int, D: int) -> int:
        """Compute the maximal gcd achievable with X in [A, B] and Y in [C, D].

        This preserves the core algorithm from the original RLVE environment.
        """
        res = 1
        m = min(B, D)
        p = 1
        while p <= m:
            t1 = B // p
            t2 = D // p
            r1 = B // t1
            r2 = D // t2
            r = min(r1, r2)
            x = (B // r) * r
            y = (D // r) * r
            if x >= A and y >= C:
                res = r
            p = r + 1
        return res

    def sample_random_action(self) -> str:
        """Sample a random feasible action in \\boxed{X Y} format."""
        if self.A is None or self.B is None or self.C is None or self.D is None:
            # If not initialized, sample a generic pair
            X = random.randint(1, max(10, self.max_a_b_c_d))
            Y = random.randint(1, max(10, self.max_a_b_c_d))
        else:
            X = random.randint(self.A, self.B)
            Y = random.randint(self.C, self.D)
        return f"\\boxed{{{X} {Y}}}"