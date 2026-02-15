import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PolynomialInterpolationEnv(Env):
    """Polynomial interpolation environment - single-turn Q&A.

    The task is to recover integer coefficients of a degree-N polynomial given N+1 points.
    The answer must be provided in \\boxed{...} format as a space-separated list of coefficients.
    """

    def __init__(
        self,
        N: int = 2,
        max_weight: int = 5,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            N: Degree of the polynomial. Must be >= 2.
            max_weight: Maximum absolute value for coefficients (except the leading coefficient which is positive).
        """
        super().__init__()
        assert N >= 2, "N should be greater than or equal to 2"
        self.N = N
        self.max_weight = max_weight

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_coeffs: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.X: Optional[List[int]] = None
        self.Y: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a polynomial interpolation problem.\n"
            "Please provide your final answer as the coefficients in \\boxed{a_0 a_1 ... a_N} format, separated by spaces.\n\n"
        )

    def compute(self, x: int, coeffs: List[int]) -> int:
        """Compute f(x) = sum_{i=0..N} a_i * x^i for the given coefficients."""
        return sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new polynomial interpolation problem.

        Returns:
            observation: A string containing instructions and the problem statement.
            info: An empty dict or additional metadata.
        """
        super().reset(seed)

        # Generate coefficients: a_0..a_{N-1} in [-max_weight, max_weight], a_N in [1, max_weight]
        coeffs = [random.randint(-self.max_weight, self.max_weight) for _ in range(self.N)]
        coeffs.append(random.randint(1, self.max_weight))
        self.reference_coeffs = coeffs
        self.reference_answer_str = " ".join(map(str, coeffs))

        # Generate N+1 distinct x-values in [-N, N] and corresponding y-values
        X = random.sample(range(-self.N, self.N + 1), self.N + 1)
        Y = [self.compute(x, coeffs) for x in X]
        self.X = X
        self.Y = Y

        # Build problem statement
        points_str = "\n".join(f"f({x}) = {y}" for x, y in zip(X, Y))
        self.current_problem = (
            f"You are given a polynomial of degree {self.N} in the form:\n"
            f"f(x) = a_0 * x^0 + a_1 * x^1 + ... + a_{self.N} * x^{self.N}, where the coefficients a_0, a_1, ..., a_{self.N} are integers.\n\n"
            f"It is known that the polynomial passes through the following {self.N + 1} points:\n"
            f"{points_str}\n\n"
            f"Please determine the coefficients a_0, a_1, ..., a_{self.N}.\n\n"
            f"Output Format: Your final answer should be a single line containing a_0 a_1 ... a_{self.N} in \\boxed{{...}}, separated by spaces."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer.

        Args:
            action: The user's response text.

        Returns:
            observation: TERMINAL_STATE since this is single-turn.
            reward: 1.0 if correct; 0.0 if wrong; -0.1 if format error.
            terminated: True for single-turn environments.
            truncated: False (no truncation logic).
            info: Additional information about correctness and answers.
        """
        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from boxed content
        try:
            user_coeffs = list(map(int, boxed_content.split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length
        expected_len = self.N + 1
        if len(user_coeffs) != expected_len:
            info = {
                "error": "invalid_length",
                "expected_length": expected_len,
                "received_length": len(user_coeffs),
                "reference_answer": self.reference_answer_str,
                "user_answer": " ".join(map(str, user_coeffs)),
                "correct": False,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check correctness
        is_correct = user_coeffs == self.reference_coeffs
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "user_answer": " ".join(map(str, user_coeffs)),
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within \\boxed{...} from the user's text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: random coefficients in boxed format."""
        coeffs = [random.randint(-self.max_weight, self.max_weight) for _ in range(self.N)]
        coeffs.append(random.randint(1, self.max_weight))
        return f"\\boxed{{{' '.join(map(str, coeffs))}}}"