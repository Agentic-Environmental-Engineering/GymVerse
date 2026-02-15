import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GaussianEliminationEnv(Env):
    """Single-turn environment for solving a system of linear equations with integer solutions."""

    def __init__(
        self,
        N: int = 3,
        M: int = 2,
        coefficient_non_zero_probability: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.N = N
        self.M = M
        self.coefficient_non_zero_probability = coefficient_non_zero_probability

        # Internal state variables
        self.current_problem: Optional[str] = None
        self.x: Optional[List[int]] = None
        self.equations: Optional[List[List[int]]] = None
        self.results: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a system of linear equations with integer variables.\n"
            "Please provide your answer in \\boxed{...} format.\n"
            "Inside the box, write the N integers separated by spaces (no commas).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Validate parameters
        assert isinstance(self.N, int) and self.N >= 2, "N should be an integer >= 2"
        assert isinstance(self.M, int) and self.M >= 1, "M should be an integer >= 1"
        assert (
            0.0 <= self.coefficient_non_zero_probability <= 1.0
        ), "coefficient_non_zero_probability should be in [0.0, 1.0]"

        # Generate a random integer solution x
        self.x = [random.randint(1, self.N) for _ in range(self.N)]
        self.reference_answer_str = " ".join(map(str, self.x))

        # Generate equations using random coefficients, ensuring at least one non-zero per equation
        self.equations = []
        self.results = []
        for _ in range(self.M):
            while True:
                equation = []
                for i in range(self.N):
                    if random.random() < self.coefficient_non_zero_probability:
                        coefficient = random.randint(1, max(1, self.N // 5))
                    else:
                        coefficient = 0
                    equation.append(coefficient)
                if any(equation):
                    break
            self.equations.append(equation)
            self.results.append(sum(coef * xi for coef, xi in zip(equation, self.x)))

        # Build the problem statement
        equations_str_lines = []
        for eq, res in zip(self.equations, self.results):
            terms = " + ".join(f"{coef} * x[{i}]" for i, coef in enumerate(eq) if coef != 0)
            equations_str_lines.append(f"{terms} = {res}")
        equations_block = "\n".join(equations_str_lines)

        one_to_N = " ".join(map(str, range(1, self.N + 1)))
        self.current_problem = (
            f"There are {self.N} integers x[0], x[1], ..., x[{self.N - 1}]. "
            f"They satisfy the following {self.M} equations:\n"
            f"{equations_block}\n\n"
            f"Please find any solution x[0], x[1], ..., x[{self.N - 1}] that satisfies the equations.\n\n"
            f"Output Format: Your final answer should be a single line containing x[0], x[1], ..., x[{self.N - 1}], "
            f"separated by spaces, and enclosed in \\boxed{{...}}.\n"
            f"Example: \\boxed{{{one_to_N}}}; this means: x[0] = 1, x[1] = 2, ..., x[{self.N - 1}] = {self.N}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process one action (answer) and return the evaluation result."""
        # Parse the boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse integers from the boxed content
        try:
            user_x = list(map(int, boxed_content.strip().split()))
        except ValueError:
            # Content inside box is not valid integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check length matches N; treat length mismatch as format error
        if len(user_x) != self.N:
            return (
                TERMINAL_STATE,
                -0.1,
                True,
                False,
                {
                    "error": "wrong_length",
                    "expected_length": self.N,
                    "received_length": len(user_x),
                },
            )

        # Verify equations
        assert self.equations is not None and self.results is not None
        satisfied = sum(
            int(sum(coef * xi for coef, xi in zip(eq, user_x)) == res)
            for eq, res in zip(self.equations, self.results)
        )
        is_correct = satisfied == len(self.equations)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied_count": satisfied,
            "total_equations": len(self.equations),
            "reference_answer": self.reference_answer_str,
            "user_answer": user_x,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re

        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the expected format."""
        random_answer = [random.randint(1, self.N) for _ in range(self.N)]
        return f"\\boxed{{{' '.join(map(str, random_answer))}}}"