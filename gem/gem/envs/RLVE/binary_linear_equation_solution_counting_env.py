from typing import Any, Optional, Tuple, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BinaryLinearEquation_SolutionCountingEnv(Env):
    """
    Single-turn environment for counting integer solutions to a binary linear equation:
    Find the number of integer pairs (x, y) within given bounds such that A*x + B*y + C = 0.
    The agent must output the result in \\boxed{...} format.
    """

    def __init__(
        self,
        max_range: int = 20,
        not_guaranteed_probability: float = 0.05,
    ):
        """
        Initialize the environment.

        Args:
            max_range: Maximum absolute value range for generating coefficients and bounds (must be >= 8).
            not_guaranteed_probability: Probability to generate a problem that may have zero solutions.
        """
        super().__init__()
        assert isinstance(max_range, int), "max_range must be an integer"
        assert max_range >= 8, "max_range must be at least 8"
        assert 0.0 <= not_guaranteed_probability <= 1.0, "not_guaranteed_probability must be in [0, 1]"

        self.max_range: int = max_range
        self.not_guaranteed_probability: float = not_guaranteed_probability

        # Problem state
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.C: Optional[int] = None
        self.X1: Optional[int] = None
        self.X2: Optional[int] = None
        self.Y1: Optional[int] = None
        self.Y2: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving linear Diophantine counting problems.\n"
            "Task: Count the number of integer pairs (x, y) that satisfy a linear equation within given ranges.\n"
            "Output Format: Your final answer must be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation (str): A description of the problem to solve.
            info (dict): Additional information (empty for this environment).
        """
        super().reset(seed)

        MAX_RANGE = self.max_range
        # Generate coefficients A, B
        A = random.randint(-MAX_RANGE, +MAX_RANGE)
        B = random.randint(-MAX_RANGE, +MAX_RANGE)
        not_guaranteed = random.random() < self.not_guaranteed_probability

        if not_guaranteed:
            X1 = random.randint(-MAX_RANGE, +MAX_RANGE)
            X2 = random.randint(X1, +MAX_RANGE)
            Y1 = random.randint(-MAX_RANGE, +MAX_RANGE)
            Y2 = random.randint(Y1, +MAX_RANGE)
            C = random.randint(-2 * (MAX_RANGE ** 2), +2 * (MAX_RANGE ** 2))
        else:
            # Ensure at least one solution by choosing (x, y) first
            x = random.randint(-MAX_RANGE, +MAX_RANGE)
            y = random.randint(-MAX_RANGE, +MAX_RANGE)
            C = -(A * x + B * y)
            X1 = random.randint(-MAX_RANGE, x)
            X2 = random.randint(x, +MAX_RANGE)
            Y1 = random.randint(-MAX_RANGE, y)
            Y2 = random.randint(y, +MAX_RANGE)

        # Save parameters
        self.A, self.B, self.C = A, B, C
        self.X1, self.X2, self.Y1, self.Y2 = X1, X2, Y1, Y2

        # Compute reference answer
        self.reference_answer = self._compute_number_of_solutions(A, B, C, X1, X2, Y1, Y2)

        # Build problem statement
        prompt = (
            f"What is the number of integer solution pairs (x, y) such that "
            f"({A}) * x + ({B}) * y + ({C}) = 0, with {X1} <= x <= {X2} and {Y1} <= y <= {Y2}?\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = prompt

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Evaluate the agent's answer.

        Args:
            action (str): The agent's output text. The answer must be in \\boxed{...} format.

        Returns:
            observation (str): TERMINAL_STATE since this is a single-turn task.
            reward (float): 1.0 if correct, 0.0 if incorrect, -0.1 for format error.
            terminated (bool): Always True for single-turn.
            truncated (bool): Always False for this environment.
            info (dict): Additional feedback information.
        """
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Reference answer is not computed. Call reset() first."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "params": {
                "A": self.A,
                "B": self.B,
                "C": self.C,
                "X1": self.X1,
                "X2": self.X2,
                "Y1": self.Y1,
                "Y2": self.Y2,
            },
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside \\boxed{...}.

        Args:
            text (str): The full output text from the agent.

        Returns:
            The extracted string inside the last \\boxed{...} if present, else None.
        """
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random action in the required \\boxed{...} format.

        Returns:
            A randomly guessed integer within a plausible range, boxed.
        """
        # The maximum possible count of solutions is the number of grid points
        # in the rectangle, which is at most (2*max_range + 1)^2.
        max_possible = (2 * self.max_range + 1) ** 2
        guess = random.randint(0, max_possible)
        return f"\\boxed{{{guess}}}"

    @staticmethod
    def _gcd(a: int, b: int) -> int:
        """Compute the greatest common divisor."""
        while b:
            a, b = b, a % b
        return abs(a)

    @staticmethod
    def _extended_gcd_positive(a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean algorithm for non-negative a, b.
        Returns (g, x, y) such that a*x + b*y = g.
        """
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = BinaryLinearEquation_SolutionCountingEnv._extended_gcd_positive(b, a % b)
        return (g, y1, x1 - (a // b) * y1)

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        """Ceil division that works for any sign of b."""
        return -((-a) // b)

    @staticmethod
    def _floor_div(a: int, b: int) -> int:
        """Floor division (Python's // already floors)."""
        return a // b

    @classmethod
    def _k_range(cls, a0: int, step: int, L: int, R: int) -> Tuple[int, int]:
        """
        From constraint: L <= a0 + step*k <= R
        Return [lo, hi] for integer k. If empty, lo > hi.
        """
        if step > 0:
            lo = cls._ceil_div(L - a0, step)
            hi = cls._floor_div(R - a0, step)
        else:
            # step < 0: inequality reverses when dividing by a negative
            lo = cls._ceil_div(R - a0, step)
            hi = cls._floor_div(L - a0, step)
        return lo, hi

    @classmethod
    def _compute_number_of_solutions(cls, A: int, B: int, C: int, X1: int, X2: int, Y1: int, Y2: int) -> int:
        """
        Compute the number of integer solutions (x, y) within [X1, X2] x [Y1, Y2]
        that satisfy A*x + B*y + C = 0.
        """
        # Normalize ranges
        if X1 > X2:
            X1, X2 = X2, X1
        if Y1 > Y2:
            Y1, Y2 = Y2, Y1

        # Degenerate cases
        if A == 0 and B == 0:
            return (X2 - X1 + 1) * (Y2 - Y1 + 1) if C == 0 else 0

        if A == 0:
            # B*y + C = 0
            if B != 0 and C % B == 0:
                y = -C // B
                return (X2 - X1 + 1) if (Y1 <= y <= Y2) else 0
            else:
                return 0

        if B == 0:
            # A*x + C = 0
            if A != 0 and C % A == 0:
                x = -C // A
                return (Y2 - Y1 + 1) if (X1 <= x <= X2) else 0
            else:
                return 0

        # General case
        d = cls._gcd(A, B)
        if C % d != 0:
            return 0

        # Find one solution to A*x + B*y = -C
        _, xg, yg = cls._extended_gcd_positive(abs(A), abs(B))  # axg + byg = gcd(|A|,|B|)
        if A < 0:
            xg = -xg
        if B < 0:
            yg = -yg

        mult = (-C) // d
        x0 = xg * mult
        y0 = yg * mult

        # Parametric form
        step_x = B // d
        step_y = -A // d  # can be negative

        # k-range from x and y intervals
        kx_lo, kx_hi = cls._k_range(x0, step_x, X1, X2)
        ky_lo, ky_hi = cls._k_range(y0, step_y, Y1, Y2)

        lo = max(kx_lo, ky_lo)
        hi = min(kx_hi, ky_hi)

        return 0 if lo > hi else hi - lo + 1