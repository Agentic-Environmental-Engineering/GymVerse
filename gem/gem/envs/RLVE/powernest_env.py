import random
import re
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PowerNestEnv(Env):
    """Environment for converting a positive integer to the nested power-of-two expression format."""

    def __init__(
        self,
        max_number: int = 1000000,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_number: The maximum value for the randomly generated positive integer.
        """
        super().__init__()
        assert max_number >= 1, "max_number should be greater than or equal to 1"
        self.max_number = max_number
        self.current_number: Optional[int] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a positive integer and must represent it as a sum of powers of 2 using a nested format.\n"
            "Formatting rules:\n"
            "- Write a^b as 2(b).\n"
            "- Concatenate terms with '+'.\n"
            "- Exponents themselves are expressed recursively in the same format.\n"
            "- Special cases: 2(0) represents 1, and 2 represents 2.\n"
            "Examples:\n"
            "137 = 2^7 + 2^3 + 2^0 → 2(2(2)+2+2(0))+2(2+2(0))+2(0)\n"
            "1315 = 2^10 + 2^8 + 2^5 + 2 + 1 → 2(2(2+2(0))+2)+2(2(2+2(0)))+2(2(2)+2(0))+2+2(0)\n\n"
            "Output Format: Your final answer must be the expression wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem."""
        super().reset(seed)

        # Generate the problem
        self.current_number = random.randint(1, self.max_number)
        self.reference_answer = self._convert_to_powernest(self.current_number)

        # Build the prompt
        self.current_problem = (
            f"You are given a positive integer {self.current_number}.\n"
            "Write this number in the power-of-two expression form as described above.\n\n"
            "Output Format: Provide only the final expression wrapped in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the provided answer and return the result."""
        # Parse boxed answer
        answer = self._parse_answer(action)

        if answer is None:
            # Format error: no boxed content found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        answer = answer.strip()

        # Validate the expression format
        is_valid_expression = self._check_powernest(answer)
        if not is_valid_expression:
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "invalid_solution",
                "user_answer": answer,
                "reference_answer": self.reference_answer,
                "number": self.current_number,
            }

        # Check correctness
        is_correct = (answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": answer,
            "number": self.current_number,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the input text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _convert_to_powernest(self, n: int) -> str:
        """Convert a positive integer to the nested power-of-two expression."""
        assert n > 0, "n should be greater than 0"
        memo: dict[int, str] = {}

        def helper(x: int) -> str:
            assert x > 0, "x should be greater than 0"
            if x in memo:
                return memo[x]
            power = 0
            parts: list[str] = []
            y = x
            while y:
                if y & 1:
                    if power == 0:
                        parts.append("2(0)")
                    elif power == 1:
                        parts.append("2")
                    else:
                        parts.append(f"2({helper(power)})")
                y //= 2
                power += 1
            parts.reverse()
            expr = "+".join(parts)
            memo[x] = expr
            return expr

        return helper(n)

    def _check_powernest(self, expression: str) -> bool:
        """Validate whether the given expression follows the powernest grammar."""
        if expression == "":
            return False

        # Identify top-level '+' separators
        intervals: list[tuple[int, int]] = []
        stack_count = 0
        for i, char in enumerate(expression):
            if char == "(":
                stack_count += 1
            elif char == ")":
                if stack_count > 0:
                    stack_count -= 1
                else:
                    return False
            elif char == "+":
                if stack_count == 0:
                    if not intervals:
                        intervals.append((0, i))
                    else:
                        intervals.append((intervals[-1][1] + 1, i))
            else:
                # Other characters are allowed in context (digits and '2')
                pass

        if stack_count != 0:
            return False

        if intervals:
            intervals.append((intervals[-1][1] + 1, len(expression)))
            for interval in intervals:
                if interval[0] < interval[1]:
                    if not self._check_powernest(expression[interval[0]:interval[1]]):
                        return False
                else:
                    return False
            return True
        else:
            # Base cases
            if expression == "2":
                return True
            elif expression.startswith("2(") and expression.endswith(")"):
                inner = expression[2:-1]
                if inner == "0":
                    return True
                return self._check_powernest(inner)
            else:
                return False

    def sample_random_action(self) -> str:
        """Sample a random action by returning a simple valid expression wrapped in \\boxed{...}."""
        # This is a simple valid expression (represents the number 1).
        return "\\boxed{2(0)}"