from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumManhattan_CurvedSurfaceEnv(Env):
    """Environment for computing the sum of P(k) over a range [A, B],
    where P(k) is the sum of (|x| + |y| + |z|)^2 over all integer triples (x, y, z)
    such that x * y * z = k.
    Single-turn Q&A environment.
    """

    prompt_template = (
        "Define P(k) as the sum of (|x| + |y| + |z|)^2 over all integer triples (x, y, z) "
        "such that x × y × z = k. Compute the sum of P(k) for all integers k in the range "
        "[{A}, {B}] (inclusive)."
    )

    def __init__(self, max_a_b: int = 100000, **kwargs):
        """
        Initialize the environment.

        Parameters:
        - max_a_b: Maximum bound for A and B. Must be >= 1.

        Note:
        This environment is single-turn; the result is verified in one step.
        """
        super().__init__()
        assert max_a_b >= 1, "max_a_b should be greater than or equal to 1"
        self.max_a_b: int = max_a_b

        # Current problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.A: Optional[int] = None
        self.B: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory problem over integer triples (x, y, z) with x*y*z = k.\n"
            "Your goal is to compute the sum of P(k) = sum over all (x, y, z) with x*y*z = k of (|x| + |y| + |z|)^2,\n"
            "for all integers k in a given interval [A, B].\n\n"
            "Output Format: Your final answer must be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        self.A = random.randint(1, self.max_a_b)
        self.B = random.randint(self.A, self.max_a_b)

        # Build problem statement
        self.current_problem = self.prompt_template.format(A=self.A, B=self.B) + (
            "\n\nOutput Format: Provide your final answer in \\boxed{...}."
        )

        # Compute reference answer using the core algorithm
        result = self._work(self.B) - self._work(self.A - 1)
        result = result * 4
        assert result > 0, "Result should be positive"
        self.reference_answer = result

        obs = self._get_instructions() + self.current_problem
        return obs, {}

        # Note: The environment is single-turn and expects an answer via step().

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Reference answer was not computed."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "A": self.A,
            "B": self.B,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random boxed integer as an action."""
        # Provide a random large integer; not necessarily correct.
        random_answer = random.randint(0, 10**12)
        return f"\\boxed{{{random_answer}}}"

    # ---------- Core algorithm helpers ----------

    def _funa(self, l: int, r: int) -> int:
        """Sum of i for i in [l..r]."""
        cnt = r - l + 1
        return (l + r) * cnt // 2

    def _ready(self, x: int) -> int:
        """Sum of i^2 for i in [1..x]."""
        return x * (x + 1) * (2 * x + 1) // 6

    def _funb(self, l: int, r: int) -> int:
        """Sum of i^2 for i in [l..r]."""
        return self._ready(r) - self._ready(l - 1)

    def _work2(self, n: int) -> Tuple[int, int, int]:
        """
        Compute helper sums for a given n using divisor grouping:
        ans1 = sum_{i=1..n} floor(n/i)
        ans2 = sum_{i=1..n} [ sum_{j=1..i} j + i * sum_{j=1..floor(n/i)} j ]
        ans3 = sum_{i=1..n} [ sum_{j=1..i} j^2 + i * sum_{j=1..floor(n/i)} j^2 + 2 * (sum_{j=1..i} j) * (sum_{k=1..floor(n/i)} k) ]
        """
        ans1 = 0
        ans2 = 0
        ans3 = 0
        l = 1
        while l <= n:
            d = n // l
            r = n // d
            cnt = r - l + 1

            ans1 += cnt * d
            ans2 += self._funa(l, r) * d + cnt * self._funa(1, d)
            ans3 += (
                self._funb(l, r) * d
                + cnt * self._funb(1, d)
                + 2 * self._funa(l, r) * self._funa(1, d)
            )

            l = r + 1

        return ans1, ans2, ans3

    def _work(self, n: int) -> int:
        """
        Compute the cumulative sum S(n) = sum_{k=1..n} P(k)/4,
        where P(k) is the squared Manhattan distance sum over triples with product k.
        Uses divisor grouping to achieve ~O(sqrt(n)).
        """
        if n <= 0:
            return 0

        ans = 0
        l = 1
        while l <= n:
            d = n // l
            r = n // d
            cnt = r - l + 1

            a1, a2, a3 = self._work2(d)
            ans += self._funb(l, r) * a1 + self._funa(l, r) * 2 * a2 + cnt * a3

            l = r + 1

        return ans