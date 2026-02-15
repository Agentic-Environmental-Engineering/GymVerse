from typing import Any, Optional, SupportsFloat, Tuple, Dict, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CircuitEnv(Env):
    """Boolean circuit environment - single-turn Q&A."""

    def __init__(
        self,
        N: int,
        M: Optional[int] = None,
        binary_ops_probs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize the CircuitEnv instance.

        Parameters:
            N: Number of boolean variables (must be >= 2).
            M: Size parameter for the expression tree (must be >= N). If None, defaults to N.
            binary_ops_probs: Probabilities for choosing binary operators. Keys are one of "&", "|", "^".
                              Values must sum to 1. Defaults to {"&": 0.25, "|": 0.25, "^": 0.5}.
        """
        super().__init__()

        assert N >= 2, "N should be greater than or equal to 2"
        self.N = N
        self.M = M if M is not None else N
        assert self.M >= self.N, "M should be greater than or equal to N"

        if binary_ops_probs is None:
            binary_ops_probs = {
                "&": 0.25,
                "|": 0.25,
                "^": 0.5,
            }
        assert abs(sum(binary_ops_probs.values()) - 1.0) < 1e-8, "binary_ops_probs values should sum to 1"
        self.binary_ops_probs = binary_ops_probs

        # State holders
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.tree: Optional[Any] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Boolean expression satisfiability problem with bitwise operators.\n"
            "Please submit your answer inside \\boxed{...} as N integers (0 or 1) separated by spaces.\n"
            "Example: \\boxed{0 1 0 1}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        binary_ops, binary_probs = list(self.binary_ops_probs.keys()), list(self.binary_ops_probs.values())

        # Keep generating until the constructed expression evaluates to 1 for a sampled assignment
        while True:
            x = [random.randint(0, 1) for _ in range(self.N)]

            def build_tree(n: int) -> Tuple[Any, int]:
                """Build a random binary expression tree and evaluate it under assignment x."""
                if n == 1:
                    index = random.randint(0, self.N - 1)
                    return index, x[index]
                left_n = random.randint(1, n - 1)
                right_n = n - left_n
                left_tree, left_value = build_tree(left_n)
                right_tree, right_value = build_tree(right_n)
                op = random.choices(binary_ops, weights=binary_probs, k=1)[0]
                if op == "&":
                    value = left_value & right_value
                elif op == "|":
                    value = left_value | right_value
                elif op == "^":
                    value = left_value ^ right_value
                else:
                    raise ValueError("Invalid operator")
                return (left_tree, op, right_tree), value

            tree, value = build_tree(self.M)
            if value == 1:
                self.reference_answer = " ".join(map(str, x))
                self.tree = tree
                break

        # Build the expression string
        expression = self._build_expression(self.tree)
        if expression.startswith("(") and expression.endswith(")"):
            expression = expression[1:-1]

        # Construct the problem text
        example = " ".join(str(i % 2) for i in range(self.N))
        self.current_problem = (
            f"There are {self.N} boolean (0/1) values x[0], x[1], ..., x[{self.N - 1}].\n\n"
            f"Given a Boolean expression (where `&` is bitwise AND, `|` is bitwise OR, and `^` is bitwise XOR): {expression}\n"
            f"Please find any solution x[0], x[1], ..., x[{self.N - 1}] that makes the expression evaluate to 1.\n\n"
            f"Output Format: Your final answer should be inside \\boxed{{...}} as {self.N} integers separated by spaces.\n"
            f"Example: \\boxed{{{example}}} (do NOT include quotes or backticks)."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        tokens = boxed_content.strip().split()
        if not tokens:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        try:
            x = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if len(x) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_length"}

        if not all(xi in (0, 1) for xi in x):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_bits"}

        def compute(tree: Any) -> int:
            if isinstance(tree, int):
                return x[tree]
            left_tree, op, right_tree = tree
            left_value = compute(left_tree)
            right_value = compute(right_tree)
            if op == "&":
                return left_value & right_value
            elif op == "|":
                return left_value | right_value
            elif op == "^":
                return left_value ^ right_value
            else:
                raise ValueError("Invalid operator")

        is_correct = compute(self.tree) == 1
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, x)),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _build_expression(self, tree: Any) -> str:
        """Build the infix expression string from the tree."""
        if isinstance(tree, int):
            return f"x[{tree}]"
        left_tree, op, right_tree = tree
        return f"({self._build_expression(left_tree)} {op} {self._build_expression(right_tree)})"

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        random_bits = [str(random.randint(0, 1)) for _ in range(self.N)]
        return f"\\boxed{{{ ' '.join(random_bits) }}}"