import math
import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict

import sympy

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


def generate_test_points(num: int, low: float, high: float) -> List[float]:
    """Generate evenly spaced test points."""
    assert num >= 2, "num should be greater than or equal to 2"
    return [low + (high - low) * i / (num - 1) for i in range(num)]


class DifferentiateEnv(Env):
    """Single-turn environment for symbolic differentiation in SymPy syntax."""

    def __init__(
        self,
        node_num: int = 3,
        node_type_probs: Optional[List[float]] = None,
        unary_ops_probs: Optional[Dict[str, float]] = None,
        binary_ops_probs: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-5,
        max_val: float = 1e4,
        test_points: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize DifferentiateEnv.

        Parameters:
        - node_num: number of nodes to control the complexity of generated function tree.
        - node_type_probs: probabilities for choosing unary vs binary node types (length 2, sums to 1).
        - unary_ops_probs: probabilities for unary operations (sums to 1).
        - binary_ops_probs: probabilities for binary operations (sums to 1).
        - epsilon: numeric tolerance for validation.
        - max_val: maximum allowed absolute value during numeric checks to avoid blow-ups.
        - test_points: list of points where numeric checks are performed.
        """
        super().__init__()
        assert isinstance(node_num, int) and node_num >= 1, "node_num should be a positive integer"
        self.node_num = node_num

        if node_type_probs is None:
            node_type_probs = [0.5, 0.5]
        assert len(node_type_probs) == 2 and abs(sum(node_type_probs) - 1.0) < 1e-8, \
            "node_type_probs should have length 2 and sum to 1"
        self.node_type_probs = node_type_probs

        if unary_ops_probs is None:
            unary_ops_probs = {
                "sin": 0.1,
                "cos": 0.1,
                "exp": 0.05,
                "log": 0.05,
                "const_pow": 0.1,
                "const_add": 0.25,
                "const_mul": 0.25,
                "const_div": 0.1,
            }
        assert abs(sum(unary_ops_probs.values()) - 1.0) < 1e-8, "unary_ops_probs values should sum to 1"
        self.unary_ops_probs = unary_ops_probs

        if binary_ops_probs is None:
            binary_ops_probs = {
                "+": 0.2,
                "-": 0.2,
                "*": 0.3,
                "/": 0.2,
                "**": 0.1,
            }
        assert abs(sum(binary_ops_probs.values()) - 1.0) < 1e-8, "binary_ops_probs values should sum to 1"
        self.binary_ops_probs = binary_ops_probs

        self.epsilon = epsilon
        self.max_val = max_val
        self.test_points = test_points if test_points is not None else generate_test_points(1024, -2.0, 2.0)

        # Internal state
        self.current_problem: Optional[str] = None
        self.function_str: Optional[str] = None
        self.reference_answer: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a function F(x).\n"
            "Your task is to compute its derivative with respect to x (i.e., F'(x)).\n"
            "Output Format: Provide the expression for F'(x) in SymPy syntax inside \\boxed{...}.\n"
            "Do not omit any symbols (e.g., always use '*' for multiplication).\n"
            "Example: \\boxed{sin(2*x)/2}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new differentiation problem."""
        super().reset(seed)

        unary_ops, unary_probs = zip(*self.unary_ops_probs.items())
        binary_ops, binary_probs = zip(*self.binary_ops_probs.items())

        x = sympy.symbols("x")

        def build_expr(n: int) -> sympy.Expr:
            """Recursively build a random SymPy expression."""
            assert n >= 1, "n should be greater than or equal to 1"
            if n == 1:
                return x

            # Choose node type: unary or binary
            is_unary = (random.choices(("unary", "binary"), weights=self.node_type_probs, k=1)[0] if n >= 3 else "unary") == "unary"
            if is_unary:
                op = random.choices(unary_ops, weights=unary_probs, k=1)[0]
                sub = build_expr(n - 1)
                if op == "sin":
                    return sympy.sin(sub)
                elif op == "cos":
                    return sympy.cos(sub)
                elif op == "exp":
                    return sympy.exp(sub)
                elif op == "log":
                    return sympy.log(sub)
                elif op == "const_pow":
                    try:
                        if random.random() < 0.5:
                            # Fractional power 1/k
                            return sub ** (1 / sympy.Integer(random.randint(2, 4)))
                        else:
                            # Integer power k
                            return sub ** sympy.Integer(random.randint(2, 4))
                    except Exception:
                        # Fall back to a safer integer power if fractional power fails
                        return sub ** sympy.Integer(random.randint(2, 4))
                elif op == "const_add":
                    return sub + sympy.Integer(random.choice([-2, -1, 1, 2]))
                elif op == "const_mul":
                    if random.random() < 0.5:
                        return sub * -sympy.Integer(random.randint(2, 4))
                    else:
                        return sub * sympy.Integer(random.randint(2, 4))
                elif op == "const_div":
                    return sub / sympy.Integer(random.randint(2, 4))
                else:
                    raise NotImplementedError(f"Unknown unary op: {op}")
            else:
                op = random.choices(binary_ops, weights=binary_probs, k=1)[0]
                assert 1 <= (n - 1) - 1
                left_n = random.randint(1, (n - 1) - 1)
                left = build_expr(left_n)
                right = build_expr((n - 1) - left_n)
                if op == "+":
                    return left + right
                elif op == "-":
                    return left - right
                elif op == "*":
                    return left * right
                elif op == "/":
                    return left / right
                elif op == "**":
                    return left ** right
                else:
                    raise NotImplementedError(f"Unknown binary op: {op}")

        # Generate a valid function and its derivative
        while True:
            try:
                f_expr = build_expr(self.node_num)
                if sympy.count_ops(f_expr) > 1000:
                    continue

                # Ensure it depends on x
                if not f_expr.free_symbols:
                    continue

                # Exclude problematic atoms
                if sympy.zoo in f_expr.atoms() or sympy.nan in f_expr.atoms():
                    continue

                f_prime = sympy.diff(f_expr, x)
                if sympy.count_ops(f_prime) > 1000:
                    continue

                if sympy.zoo in f_prime.atoms() or sympy.nan in f_prime.atoms():
                    continue

                # Numeric sanity check on derivative
                f_prime_compute = sympy.lambdify(x, f_prime, modules=["math"])
                valid_count = 0
                for pt in self.test_points:
                    try:
                        val = float(f_prime_compute(pt))
                    except Exception:
                        continue
                    if not math.isfinite(val):
                        continue
                    if abs(val) > self.max_val:
                        valid_count = 0
                        break
                    valid_count += 1
                if valid_count >= len(self.test_points) // 2:
                    # Accept this function
                    self.function_str = str(f_expr)
                    self.reference_answer = str(f_prime)
                    break
                else:
                    continue
            except Exception:
                continue

        # Build problem prompt
        problem_prompt = f"You are given a function: F(x) = {self.function_str}\n\n" \
                         f"Your task is to compute its derivative with respect to x (i.e., F'(x)).\n" \
                         f"Output Format: Your answer should be the expression for F'(x), written in SymPy syntax inside \\boxed{{...}}."
        self.current_problem = problem_prompt

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse, verify the answer, and return terminal state."""
        if self.reference_answer is None or self.function_str is None:
            # Environment was not properly reset
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_ready"}

        # Extract boxed content
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Limit input length to avoid parsing explosion
        answer_text = answer_text.strip()
        if len(answer_text) > 10000:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse answer as SymPy expression
        try:
            processed_result = sympy.sympify(answer_text)
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        x = sympy.symbols("x")

        # Check that only x is used
        try:
            if processed_result.free_symbols - {x}:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Complexity check relative to reference
        try:
            ref_expr = sympy.sympify(self.reference_answer)
            if sympy.count_ops(processed_result) > 4 * sympy.count_ops(ref_expr):
                info = {
                    "correct": False,
                    "function": self.function_str,
                    "reference_answer": self.reference_answer,
                    "user_answer": answer_text
                }
                return TERMINAL_STATE, 0.0, True, False, info
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Compare expressions
        try:
            diff_expr = processed_result - ref_expr
            if sympy.count_ops(diff_expr) > 5000:
                info = {
                    "correct": False,
                    "function": self.function_str,
                    "reference_answer": self.reference_answer,
                    "user_answer": answer_text
                }
                return TERMINAL_STATE, 0.0, True, False, info
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try symbolic equality
        eq = diff_expr.is_zero
        if eq is not None:
            is_correct = bool(eq)
            reward = 1.0 if is_correct else 0.0
            info = {
                "correct": is_correct,
                "function": self.function_str,
                "reference_answer": self.reference_answer,
                "user_answer": answer_text
            }
            return TERMINAL_STATE, reward, True, False, info

        # Numeric fallback check
        try:
            diff_compute = sympy.lambdify(x, diff_expr, modules=["math"])
        except Exception:
            info = {
                "correct": False,
                "function": self.function_str,
                "reference_answer": self.reference_answer,
                "user_answer": answer_text
            }
            return TERMINAL_STATE, 0.0, True, False, info

        zero_count = 0
        for pt in self.test_points:
            try:
                val = float(diff_compute(pt))
            except Exception:
                continue
            if not math.isfinite(val):
                continue
            if abs(val) > self.epsilon:
                info = {
                    "correct": False,
                    "function": self.function_str,
                    "reference_answer": self.reference_answer,
                    "user_answer": answer_text
                }
                return TERMINAL_STATE, 0.0, True, False, info
            else:
                zero_count += 1

        is_correct = zero_count >= len(self.test_points) // 4
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "function": self.function_str,
            "reference_answer": self.reference_answer,
            "user_answer": answer_text
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
        """Sample a random action; here we return a simple derivative-like expression."""
        # This does not guarantee correctness; it is for sampling/demo only.
        return "\\boxed{x}"