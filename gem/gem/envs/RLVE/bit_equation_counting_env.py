from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BitEquationCountingEnv(Env):
    """Single-turn environment for counting satisfying assignments of a Boolean expression.

    The expression consists of N variables represented by '_' (each can be 0 or 1), combined
    using bitwise operators: '&' (AND), '|' (OR), and '^' (XOR). The task is to count how many
    of the 2^N possible assignments make the expression evaluate to true.
    """

    def __init__(
        self,
        N: int = 3,
        wrong_format: float = -1.0,
        wrong_range: float = -0.5,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        """Initialize the BitEquationCountingEnv instance.

        Parameters:
        - N: number of variables used in the expression (must be >= 2)
        - wrong_format, wrong_range, rewarding_strategy, rewarding_weight, rewarding_beta:
          preserved from the original environment for compatibility, but not used in reward calculation.
        """
        super().__init__()
        assert N >= 2, "N should be greater than or equal to 2"
        self.N = N

        # Preserve original reward parameter fields for compatibility (not used in GEM reward scheme)
        self.wrong_format = wrong_format
        self.wrong_range = wrong_range
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # State
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.expression: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Boolean expression counting problem.\n"
            "Operators: '&' is bitwise AND, '|' is bitwise OR, '^' is bitwise XOR. "
            "Each '_' denotes a Boolean variable that can be 0 or 1.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Build a random Boolean expression with exactly N variables and compute the true count
        def build_expression(n: int) -> Tuple[str, int, int]:
            """Recursively build an expression over n variables and compute counts of true/false evaluations."""
            if n == 1:
                # Single variable: '_' is true in exactly 1 assignment, false in 1 assignment
                return "_", 1, 1

            left_n = random.randint(1, n - 1)
            right_n = n - left_n

            left_expr, left_true, left_false = build_expression(left_n)
            right_expr, right_true, right_false = build_expression(right_n)

            op = random.choice(("&", "|", "^"))

            if op == "&":
                true_count = left_true * right_true
                false_count = (2 ** n) - true_count
            elif op == "|":
                false_count = left_false * right_false
                true_count = (2 ** n) - false_count
            elif op == "^":
                true_count = left_true * right_false + left_false * right_true
                false_count = left_true * right_true + left_false * right_false
                assert true_count + false_count == 2 ** n, "XOR operation should cover all cases"
            else:
                raise ValueError("Invalid operator")

            expr = f"({left_expr} {op} {right_expr})"
            return expr, true_count, false_count

        full_expr, true_count, _ = build_expression(self.N)
        # Remove the outermost parentheses to mimic original formatting
        if full_expr.startswith("(") and full_expr.endswith(")"):
            formatted_expr = full_expr[1:-1]
        else:
            formatted_expr = full_expr

        self.expression = formatted_expr
        self.reference_answer = true_count

        # Build the problem statement
        self.current_problem = (
            f"Given a Boolean expression (where '_' represents a variable that can be 0 or 1, "
            f"'&' is bitwise AND, '|' is bitwise OR, and '^' is bitwise XOR): {self.expression}\n\n"
            f"There are 2^{self.N} possible combinations of values for the variables. "
            f"Your task is to find how many of these combinations make the expression evaluate to true.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}. "
            f"Example: \\boxed{{15}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the provided answer."""
        # Extract boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate range: must be between 0 and 2^N (inclusive)
        max_count = 2 ** self.N
        if not (0 <= user_answer <= max_count):
            info = {
                "error": "out_of_range",
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "N": self.N,
                "expression": self.expression,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check correctness
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "expression": self.expression,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        random_answer = random.randint(0, 2 ** self.N)
        return f"\\boxed{{{random_answer}}}"