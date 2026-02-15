import math
import random
from typing import Any, Optional, SupportsFloat, Tuple

import sympy
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PolynomialMinimumEnv(Env):
    """Environment for finding the minimizing x0 of a generated polynomial f(x)."""

    def __init__(
        self,
        N: int = 4,
        max_weight: int = 2,
        **kwargs,
    ):
        """
        Initialize the PolynomialMinimumEnv instance.

        Args:
            N: The maximum degree of the polynomial (must be even and >= 2).
            max_weight: The maximum absolute value for weights and shifts used in polynomial generation.
        """
        super().__init__()
        assert N >= 2 and N % 2 == 0, "N should be greater than or equal to 2 and even"
        self.N = N
        self.max_weight = max_weight

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[float] = None
        self.reference_value: Optional[float] = None
        self.worst_value: Optional[float] = None
        self.coeffs: Optional[list[int]] = None
        self.poly_expr_str: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Given a polynomial f(x), find the value of x0 that minimizes f(x).\n"
            "Answer Format: Provide a single real number in \\boxed{...} representing x0.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new polynomial problem."""
        super().reset(seed)

        # Generate degrees: include N and all even degrees below N in random order
        N = self.N
        available_degrees = list(range(2, N, 2))
        random.shuffle(available_degrees)
        degrees = [N] + available_degrees

        # Build polynomial with random weights and shifts
        x = sympy.Symbol("x")
        terms = []
        for deg in degrees:
            a = random.randint(1, self.max_weight)
            s = random.choice(range(-self.max_weight, self.max_weight + 1))
            term = a * ((x - s) ** deg)
            terms.append(term)

        poly = sum(terms)
        poly_expanded = sympy.expand(poly)
        coeffs = [int(poly_expanded.coeff(x, i)) for i in range(N + 1)]

        # Validations consistent with original environment
        assert len(coeffs) == N + 1, "coeffs should have length N + 1"
        assert coeffs[N] > 0.0, "leading coefficient should be positive"
        self.coeffs = coeffs

        # Build the polynomial expression for the prompt
        poly_expr = sum(c * (x ** i) for i, c in enumerate(self.coeffs))
        self.poly_expr_str = str(sympy.simplify(poly_expr))

        # Compute reference minimizing x0 using candidate points and stationary points
        real_roots = [0.0] + [random.uniform(-self.max_weight, self.max_weight) for _ in range(5)]
        try:
            d_expr = sympy.diff(poly_expr, x)
            roots = sympy.nroots(d_expr)
            real_roots += [float(sympy.re(r)) for r in roots if abs(sympy.im(r)) < 1e-6]
        except Exception:
            # If derivative root finding fails, continue with sampled points
            pass

        def f_eval(x_val: float) -> float:
            return float(poly_expr.evalf(subs={x: x_val}))

        f_vals = [f_eval(xr) for xr in real_roots]
        min_idx = f_vals.index(min(f_vals))
        x0 = real_roots[min_idx]
        self.reference_answer = float(x0)
        self.reference_value = float(f_vals[min_idx])
        self.worst_value = f_vals[0]

        # Construct problem statement
        self.current_problem = (
            f"Given f(x) = {self.poly_expr_str}, find the value of x0 that minimizes f(x).\n"
            f"Your final answer should be a single real number in decimal form, representing the value of x0.\n\n"
            f"Output Format: Provide your answer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided answer and return the result."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to float
        try:
            user_x = float(boxed_content.strip())
            if not math.isfinite(user_x):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Compute f at user_x
        x = sympy.Symbol("x")
        poly_expr = sum(c * (x ** i) for i, c in enumerate(self.coeffs or []))
        f_user = float(poly_expr.evalf(subs={x: user_x})) if self.coeffs is not None else float("inf")

        # Determine correctness
        tol_x = 1e-6
        tol_f = 1e-8
        ref_x = self.reference_answer if self.reference_answer is not None else float("inf")
        ref_val = self.reference_value if self.reference_value is not None else float("inf")

        is_correct = (abs(user_x - ref_x) <= tol_x) or (f_user <= ref_val + tol_f)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "reference_value": self.reference_value,
            "user_answer": user_x,
            "user_value": f_user,
            "polynomial": self.poly_expr_str,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the response text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random real number within a reasonable range."""
        random_answer = random.uniform(-self.max_weight, self.max_weight)
        # Format with limited precision to avoid overly long decimals
        return f"\\boxed{{{random_answer:.6f}}}"