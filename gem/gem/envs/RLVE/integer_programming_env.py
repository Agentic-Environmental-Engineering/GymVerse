from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class IntegerProgrammingEnv(Env):
    """Single-turn environment for solving a system of linear inequalities over integers.

    The task provides N variables x[0], x[1], ..., x[N-1] and M linear inequalities of the form:
        sum_i (a_i * x[i]) >= b

    The agent must return any integer vector satisfying all inequalities.
    The final answer must be enclosed in \\boxed{...} and contain N integers separated by spaces.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        M: Optional[int] = None,
        N_min: int = 2,
        N_max: int = 10,
        M_min: int = 1,
        M_max: int = 10,
        number_range: int = 4,
        coefficient_non_zero_probability: float = 0.5,
        **kwargs
    ):
        super().__init__()
        # Problem size configuration
        self.N_fixed = N
        self.M_fixed = M
        self.N_min = N_min
        self.N_max = N_max
        self.M_min = M_min
        self.M_max = M_max

        # Generation parameters
        self.number_range = number_range
        self.coefficient_non_zero_probability = coefficient_non_zero_probability

        # Internal state
        self.N: int = 0
        self.M: int = 0
        self.inequations: List[List[int]] = []
        self.results: List[int] = []
        self.solution_x: List[int] = []
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an integer programming feasibility problem with linear inequalities.\n"
            "Please provide any integer vector x[0], x[1], ..., x[N-1] that satisfies all given inequalities.\n"
            "Output Format: Your final answer must be N integers separated by spaces, enclosed in \\boxed{...}.\n"
            "Example: \\boxed{1 2 3} for N = 3.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate and determine N and M
        if self.N_fixed is not None:
            if self.N_fixed < 2:
                raise ValueError("N should be greater than or equal to 2")
            self.N = self.N_fixed
        else:
            if self.N_min < 2:
                raise ValueError("N_min should be >= 2")
            if self.N_max < self.N_min:
                raise ValueError("N_max should be >= N_min")
            self.N = random.randint(self.N_min, self.N_max)

        if self.M_fixed is not None:
            if self.M_fixed < 1:
                raise ValueError("M should be greater than or equal to 1")
            self.M = self.M_fixed
        else:
            if self.M_min < 1:
                raise ValueError("M_min should be >= 1")
            if self.M_max < self.M_min:
                raise ValueError("M_max should be >= M_min")
            self.M = random.randint(self.M_min, self.M_max)

        # Generate a guaranteed feasible solution vector
        self.solution_x = [random.randint(-self.N, +self.N) for _ in range(self.N)]
        self.reference_answer = " ".join(map(str, self.solution_x))

        # Generate inequalities and right-hand sides b such that solution_x satisfies them
        self.inequations = []
        self.results = []
        for _ in range(self.M):
            # Ensure at least one non-zero coefficient
            while True:
                inequation = []
                for i in range(self.N):
                    if random.random() < self.coefficient_non_zero_probability:
                        coefficient = random.randint(1, self.number_range)
                        if random.random() < 0.5:
                            coefficient = -coefficient
                    else:
                        coefficient = 0
                    inequation.append(coefficient)
                if any(inequation):
                    break
            self.inequations.append(inequation)  # left >= right form

            left_value = sum(c * xi for c, xi in zip(inequation, self.solution_x))
            slack = random.randint(0, max(0, self.number_range // 2))
            self.results.append(left_value - slack)

        # Build problem description
        inequations_str_lines = []
        for inequation, result in zip(self.inequations, self.results):
            lhs_terms = [f"({coef}) * x[{i}]" for i, coef in enumerate(inequation) if coef != 0]
            lhs_str = " + ".join(lhs_terms) if lhs_terms else "0"
            inequations_str_lines.append(f"{lhs_str} >= {result}")
        inequations_block = "\n".join(inequations_str_lines)

        self.current_problem = (
            f"There are {self.N} integers x[0], x[1], ..., x[{self.N - 1}]. "
            f"They satisfy the following {self.M} inequalities (in the form of left >= right):\n"
            f"{inequations_block}\n\n"
            f"Please find any solution x[0], x[1], ..., x[{self.N - 1}] that satisfies all inequalities.\n\n"
            f"Output Format: Your final answer must be N integers separated by spaces, enclosed in \\boxed{{...}}.\n"
            f"Example: \\boxed{{{' '.join(map(str, range(1, self.N + 1)))}}} "
            f"(do NOT include quotes or backticks)."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.N,
            "M": self.M,
            "reference_answer": self.reference_answer,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        # Parse boxed content
        content = self._parse_answer(action)
        if content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from boxed content
        tokens = content.strip().split()
        try:
            x = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Length check
        if len(x) != self.N:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "expected_length": self.N, "received_length": len(x)}

        # Verify inequalities
        satisfied_count = 0
        satisfied_list = []
        for inequation, result in zip(self.inequations, self.results):
            lhs = sum(c * xi for c, xi in zip(inequation, x))
            is_satisfied = lhs >= result
            satisfied_list.append(is_satisfied)
            if is_satisfied:
                satisfied_count += 1

        is_correct = (satisfied_count == self.M)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied": satisfied_count,
            "total": self.M,
            "satisfied_list": satisfied_list,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, x)),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by generating a random integer vector enclosed in \\boxed{...}."""
        random_vec = [random.randint(-self.N, self.N) for _ in range(self.N)] if self.N > 0 else [0]
        return f"\\boxed{{{' '.join(map(str, random_vec))}}}"