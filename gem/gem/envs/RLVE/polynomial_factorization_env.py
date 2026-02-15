import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PolynomialFactorizationEnv(Env):
    """Polynomial factorization environment - single-turn Q&A.

    The task: Given a degree-N polynomial expressed by its coefficients,
    find any valid set of integers a_1, ..., a_N (not necessarily distinct)
    such that (x - a_1) ... (x - a_N) expands to the given polynomial.

    Output must be provided as a single line of space-separated integers enclosed in \\boxed{...}.
    """

    def __init__(
        self,
        degree: Optional[int] = None,
        min_degree: int = 3,
        max_degree: int = 8,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - degree: If provided, fixes the polynomial degree N. Must be >= 3.
        - min_degree: Minimum degree for random problem generation when degree is None. Must be >= 3.
        - max_degree: Maximum degree for random problem generation when degree is None. Must be >= min_degree.

        Note: This environment is single-turn. The reward scheme is:
        - Correct answer: 1.0
        - Wrong answer: 0.0
        - Format error: -0.1
        """
        super().__init__()

        # Validate parameters
        if degree is not None and degree < 3:
            raise ValueError("degree should be greater than or equal to 3")
        if min_degree < 3:
            raise ValueError("min_degree should be greater than or equal to 3")
        if max_degree < min_degree:
            raise ValueError("max_degree should be greater than or equal to min_degree")

        self.degree: Optional[int] = degree
        self.min_degree: int = min_degree
        self.max_degree: int = max_degree

        # Problem state
        self.N: Optional[int] = None
        self.gold_factors: Optional[List[int]] = None
        self.coefficients: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions and output format requirements."""
        return (
            "You are solving a polynomial factorization problem.\n"
            "Given a degree-N polynomial, find any valid integers a_1, ..., a_N (not necessarily distinct) "
            "such that (x - a_1) ... (x - a_N) expands to the given polynomial.\n"
            "Output Format: Provide your final answer as space-separated integers inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine degree N
        if self.degree is not None:
            N = self.degree
        else:
            N = random.randint(self.min_degree, self.max_degree)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Generate random integer roots in range [-N, N]
        gold_factors = [random.randint(-N, N) for _ in range(N)]

        # Compute polynomial coefficients for (x - a_1) ... (x - a_N)
        coefficients = [1] + [0] * N
        for a in gold_factors:
            for i in range(N, 0, -1):
                coefficients[i] = coefficients[i - 1] - a * coefficients[i]
            coefficients[0] *= -a

        polynomial_str = " + ".join(
            f"({coef}) * x^{i}" for i, coef in enumerate(coefficients) if coef != 0
        )

        self.N = N
        self.gold_factors = gold_factors
        self.coefficients = coefficients
        self.reference_answer_str = " ".join(map(str, gold_factors))

        self.current_problem = (
            f"You are given a degree-{N} polynomial: (x - a_1)...(x - a_{N}) = {polynomial_str}\n\n"
            f"Your task is to find any valid set of integers a_1, ..., a_{N} (not necessarily distinct) "
            f"such that the product of the linear factors on the left expands to match the given polynomial.\n\n"
            f"Output Format: Your final answer should be a single line containing a_1, ..., a_{N}, "
            f"separated by spaces, enclosed in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "degree": N,
            "polynomial_coefficients": coefficients,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step: parse the answer, verify it, and return reward."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from boxed content
        tokens = boxed_content.strip().split()
        try:
            user_answer_list = list(map(int, tokens))
        except ValueError:
            # Content inside box is not all integers
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate length
        N = self.N if self.N is not None else 0
        if len(user_answer_list) != N:
            info = {
                "correct": False,
                "reason": "wrong_length",
                "expected_length": N,
                "received_length": len(user_answer_list),
                "reference_answer": self.reference_answer_str,
                "user_answer": user_answer_list,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Verify using multiset comparison (order does not matter)
        gold_multiset: Dict[int, int] = {}
        for a in self.gold_factors or []:
            gold_multiset[a] = gold_multiset.get(a, 0) + 1

        satisfied = 0
        for a in user_answer_list:
            if gold_multiset.get(a, 0) > 0:
                satisfied += 1
                gold_multiset[a] -= 1

        is_correct = (satisfied == N)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "user_answer": user_answer_list,
            "degree": N,
            "polynomial_coefficients": self.coefficients,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: random list of integers inside \\boxed{...}."""
        N = self.N if self.N is not None else (self.degree if self.degree is not None else self.min_degree)
        # Sample integers in the same range used for generation
        vals = [random.randint(-N, N) for _ in range(N)]
        return f"\\boxed{{{' '.join(map(str, vals))}}}"