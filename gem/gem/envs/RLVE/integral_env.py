import math
import random
import re
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

import sympy
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


def generate_test_points(num: int, low: float, high: float) -> List[float]:
    """Generate evenly spaced test points between low and high inclusive."""
    assert num >= 2, "num should be greater than or equal to 2"
    return [low + (high - low) * i / (num - 1) for i in range(num)]


class IntegralEnv(Env):
    """Single-turn environment for finding an antiderivative given its derivative in SymPy syntax."""

    def __init__(
        self,
        node_num: int = 5,
        node_type_probs: Optional[List[float]] = None,
        unary_ops_probs: Optional[Dict[str, float]] = None,
        binary_ops_probs: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-5,
        max_val: float = 1e4,
        test_points_low: float = -2.0,
        test_points_high: float = 2.0,
        test_points_num: int = 1024,
        **kwargs,
    ):
        """
        Initialize the IntegralEnv instance.

        Parameters:
        - node_num: Number of nodes for the randomly generated expression tree (must be >= 1).
        - node_type_probs: Probabilities for choosing unary vs binary node types.
        - unary_ops_probs: Probabilities for unary operations.
        - binary_ops_probs: Probabilities for binary operations.
        - epsilon: Numerical tolerance for validation.
        - max_val: Maximum absolute value allowed during function evaluation to avoid overflow.
        - test_points_low/high/num: Parameters for generating evaluation points.
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

        self.epsilon = float(epsilon)
        self.max_val = float(max_val)
        self.test_points = generate_test_points(test_points_num, test_points_low, test_points_high)

        # State variables for the current problem
        self.current_problem: Optional[str] = None
        self.reference_answer_str: Optional[str] = None
        self.reference_expr: Optional[sympy.Expr] = None
        self.f_prime_str: Optional[str] = None
        self.f_prime_expr: Optional[sympy.Expr] = None

    def _get_instructions(self) -> str:
        """Return task instructions for the environment."""
        return (
            "You are given the derivative of a function: F'(x) = f_prime(x).\n"
            "Your task is to find an antiderivative F(x) such that its derivative equals the given expression.\n\n"
            "Answer format requirements:\n"
            "- Provide your final answer as a SymPy expression wrapped in \\boxed{...}.\n"
            "- Always use explicit symbols (e.g., always use '*' for multiplication).\n"
            "Example: \\boxed{sin(2*x)/2}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        x = sympy.symbols("x")

        unary_ops, unary_probs = zip(*self.unary_ops_probs.items())
        binary_ops, binary_probs = zip(*self.binary_ops_probs.items())

        def build_expr(n: int) -> sympy.Expr:
            """Recursively build a random SymPy expression with n nodes."""
            assert n >= 1, "n should be greater than or equal to 1"
            if n == 1:
                return x

            # Choose unary or binary node; ensure feasibility of binary nodes
            if (random.choices(("unary", "binary"), weights=self.node_type_probs, k=1)[0] if n >= 3 else "unary") == "unary":
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
                            return sub ** (1 / sympy.Integer(random.randint(2, 4)))
                        else:
                            return sub ** sympy.Integer(random.randint(2, 4))
                    except Exception:
                        # Fall back to a safer option if fractional power fails
                        return sub ** sympy.Integer(random.randint(2, 4))
                elif op == "const_add":
                    return sub + sympy.Integer(random.choice([-2, -1, +1, +2]))
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
                # Binary node
                op = random.choices(binary_ops, weights=binary_probs, k=1)[0]
                assert 1 <= (n - 1) - 1, "Binary split requires at least two children"
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

        # Generate a valid problem
        while True:
            try:
                f_expr = build_expr(self.node_num)
                if sympy.count_ops(f_expr) > 1000:
                    continue

                f_prime = sympy.diff(f_expr, x)
                if sympy.count_ops(f_prime) > 1000:
                    continue

                # Basic validity checks
                if not f_prime.free_symbols:
                    continue
                if sympy.zoo in f_expr.atoms() or sympy.nan in f_expr.atoms():
                    continue
                if sympy.zoo in f_prime.atoms() or sympy.nan in f_prime.atoms():
                    continue

                # Numerical sanity checks
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
                    # Accept this problem
                    self.reference_expr = f_expr
                    self.reference_answer_str = str(f_expr)
                    self.f_prime_expr = f_prime
                    self.f_prime_str = str(f_prime)
                    break
                else:
                    continue
            except Exception:
                continue

        # Build problem prompt
        self.current_problem = (
            f"You are given the derivative of a function: F'(x) = {self.f_prime_str}\n\n"
            "Your task is to find an antiderivative F(x) such that its derivative is equal to the given expression.\n\n"
            "Output Format: Write the expression for F(x) in SymPy syntax, and wrap it in \\boxed{...}.\n"
            "Do not omit any symbols (e.g., always use '*' for multiplication).\n"
            "Example: \\boxed{sin(2*x)/2}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the provided antiderivative answer."""
        # Ensure a problem exists
        if self.f_prime_expr is None or self.reference_expr is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "no_active_problem"}

        raw_answer = self._parse_answer(action)
        if raw_answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Process the answer string as a SymPy expression
        try:
            answer_str = raw_answer.strip()
            if len(answer_str) > 10000:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            processed_expr = sympy.sympify(answer_str)
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        x = sympy.symbols("x")

        # Enforce using only variable x
        try:
            if processed_expr.free_symbols - {x}:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Complexity check relative to the reference solution
        try:
            if sympy.count_ops(processed_expr) > 4 * sympy.count_ops(self.reference_expr):
                info = {
                    "correct": False,
                    "reference_answer": self.reference_answer_str,
                    "user_answer": str(processed_expr),
                    "f_prime": self.f_prime_str,
                    "reason": "excessive_complexity",
                }
                return TERMINAL_STATE, 0.0, True, False, info
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Verify by differentiation
        try:
            expr = sympy.diff(processed_expr, x) - self.f_prime_expr
            if sympy.count_ops(expr) > 5000:
                info = {
                    "correct": False,
                    "reference_answer": self.reference_answer_str,
                    "user_answer": str(processed_expr),
                    "f_prime": self.f_prime_str,
                    "reason": "post_diff_excessive_complexity",
                }
                return TERMINAL_STATE, 0.0, True, False, info
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        eq = expr.is_zero
        if eq is not None:
            assert isinstance(eq, bool), "eq should be a boolean value"
            is_correct = bool(eq)
            reward = 1.0 if is_correct else 0.0
            info = {
                "correct": is_correct,
                "reference_answer": self.reference_answer_str,
                "user_answer": str(processed_expr),
                "f_prime": self.f_prime_str,
            }
            return TERMINAL_STATE, reward, True, False, info

        # Numerical fallback check on test points
        try:
            expr_compute = sympy.lambdify(x, expr, modules=["math"])
        except Exception:
            info = {
                "correct": False,
                "reference_answer": self.reference_answer_str,
                "user_answer": str(processed_expr),
                "f_prime": self.f_prime_str,
                "reason": "lambdify_failed",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        zero_count = 0
        for pt in self.test_points:
            try:
                val = float(expr_compute(pt))
            except Exception:
                continue
            if not math.isfinite(val):
                continue
            if abs(val) > self.epsilon:
                info = {
                    "correct": False,
                    "reference_answer": self.reference_answer_str,
                    "user_answer": str(processed_expr),
                    "f_prime": self.f_prime_str,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            else:
                zero_count += 1

        is_correct = (zero_count >= len(self.test_points) // 4)
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "user_answer": str(processed_expr),
            "f_prime": self.f_prime_str,
        }
        return TERMINAL_STATE, reward, True, False, info

    def sample_random_action(self) -> str:
        """Sample a simple random action (heuristic)."""
        # A simple guess could be x, which differentiates to 1; not necessarily correct but valid format.
        return r"\boxed{x}"