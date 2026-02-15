from typing import Any, Optional, SupportsFloat, Tuple, List, Set
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Expression_AddingParenthese_CountingEnv(Env):
    """Environment for counting distinct values from inserting parentheses in an arithmetic expression.

    Single-turn Q&A environment. The agent must provide the number of distinct values obtainable by
    inserting parentheses into the given expression without rearranging terms, using \\boxed{...} format.
    """

    operation_options = ("+", "-", "*")

    def __init__(
        self,
        num_operands: int = 3,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
            num_operands: Number of operands in the expression (must be >= 3).
        """
        super().__init__()
        if num_operands < 3:
            raise ValueError("num_operands should be greater than or equal to 3")
        self.num_operands: int = num_operands

        # State holders
        self.operands: List[int] = []
        self.operations: List[str] = []
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an arithmetic expression problem.\n"
            "Given an expression with +, -, * operations, please count the number of distinct values "
            "that can be obtained by inserting parentheses into the expression. Rearranging terms is NOT allowed.\n"
            "Answer format: provide a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # Generate operands and operations
        self.operands = [random.randint(1, self.num_operands * self.num_operands) for _ in range(self.num_operands)]
        self.operations = [random.choice(self.operation_options) for _ in range(self.num_operands - 1)]

        # Build the expression string
        expression_parts: List[str] = []
        for i in range(2 * self.num_operands - 1):
            if i % 2 == 0:
                expression_parts.append(str(self.operands[i // 2]))
            else:
                expression_parts.append(self.operations[i // 2])
        expression_str = " ".join(expression_parts)

        # Compute reference answer using DP over all parenthesizations
        n = self.num_operands
        dp_cache: List[List[Set[int]]] = [[set() for _ in range(n)] for _ in range(n)]

        def dp(l: int, r: int) -> Set[int]:
            if l == r:
                dp_cache[l][r] = {self.operands[l]}
                return dp_cache[l][r]
            if dp_cache[l][r]:
                return dp_cache[l][r]
            for i in range(l, r):
                left_values = dp(l, i)
                right_values = dp(i + 1, r)
                op = self.operations[i]
                for lv in left_values:
                    for rv in right_values:
                        if op == "+":
                            dp_cache[l][r].add(lv + rv)
                        elif op == "-":
                            dp_cache[l][r].add(lv - rv)
                        elif op == "*":
                            dp_cache[l][r].add(lv * rv)
                        else:
                            raise NotImplementedError(f"Operation {op} is not implemented")
            return dp_cache[l][r]

        self.reference_answer = len(dp(0, n - 1))

        # Construct problem statement
        self.current_problem = (
            f"Given the expression {expression_str}, please count the number of distinct values that can be "
            f"obtained by inserting parentheses in the expression (do NOT rearrange terms).\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "operands": self.operands[:],
            "operations": self.operations[:],
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the provided answer."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(boxed_content.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Enforce positive integer as per original environment logic
        if user_answer <= 0:
            return TERMINAL_STATE, -0.1, True, False, {"error": "non_positive_answer"}

        is_correct = (self.reference_answer is not None) and (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "operands": self.operands[:],
            "operations": self.operations[:],
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer contained in \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random positive integer wrapped in \\boxed{...}."""
        # Since the number of distinct values is at least 1, choose a random positive integer
        random_answer = random.randint(1, max(1, self.num_operands * self.num_operands))
        return f"\\boxed{{{random_answer}}}"