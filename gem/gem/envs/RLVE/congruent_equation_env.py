import random
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CongruentEquationEnv(Env):
    """Environment for solving modular inverse problems (single-turn Q&A).

    Task:
    Find the smallest positive integer solution x to the congruence:
        A * x ≡ 1 (mod B)

    The agent must respond with the answer formatted as \\boxed{...}.
    """

    def __init__(
        self,
        max_a_b: int = 1000,
        **kwargs
    ):
        super().__init__()
        self.max_a_b: int = max_a_b
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving modular inverse problems.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        assert isinstance(self.max_a_b, int), "max_a_b must be an integer"
        assert self.max_a_b >= 2, "max_a_b must be greater than or equal to 2"

        # Generate A and B such that gcd(A, B) == 1 and B >= 2
        while True:
            A = random.randint(1, self.max_a_b)
            B = random.randint(2, self.max_a_b)

            d, x, _ = self._exgcd(A, B)
            if d == 1:
                # Compute the minimal positive solution
                x = (x % B + B) % B
                # Sanity checks
                assert x > 0, f"x should be positive, but got {x}"
                assert (A * x) % B == 1, f"A * x % B should be 1, but got {(A * x) % B}"
                self.A = A
                self.B = B
                self.reference_answer = x
                break

        # Build the problem description
        self.current_problem = (
            "Find the smallest positive integer solution x to the following congruence equation:\n\n"
            f"{self.A} * x ≡ 1 (mod {self.B})\n\n"
            "Output Format:\n"
            "Your final answer should be a single positive integer in \\boxed{...}.\n"
            "Example: \\boxed{17}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate positivity
        if user_answer <= 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "non_positive"}

        assert self.A is not None and self.B is not None and self.reference_answer is not None, "Environment not initialized properly."

        # Check congruence and minimality
        valid_congruence = (self.A * user_answer) % self.B == 1
        is_smallest = valid_congruence and (user_answer == self.reference_answer)
        is_correct = is_smallest

        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "valid_congruence": valid_congruence,
            "is_smallest": is_smallest if valid_congruence else False,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "A": self.A,
            "B": self.B,
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

    @staticmethod
    def _exgcd(a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean algorithm: returns (gcd, x, y) such that a*x + b*y = gcd."""
        if b == 0:
            return a, 1, 0
        d, x1, y1 = CongruentEquationEnv._exgcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return d, x, y

    def sample_random_action(self) -> str:
        """Sample a random plausible action formatted in \\boxed{...}."""
        if self.B is not None and self.B > 2:
            random_answer = random.randint(1, self.B - 1)
        else:
            random_answer = random.randint(1, max(2, self.max_a_b))
        return f"\\boxed{{{random_answer}}}"