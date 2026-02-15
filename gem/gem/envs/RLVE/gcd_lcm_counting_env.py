from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GcdLcmCountingEnv(Env):
    """Environment for counting pairs (P, Q) with given GCD and LCM - single-turn Q&A."""

    def __init__(
        self,
        max_lcm: int = 1000,
        answer_being_0_probability: float = 0.01,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_lcm: Maximum value for the LCM to be generated (must be >= 3).
            answer_being_0_probability: Probability to generate an instance where no valid pairs exist (answer = 0).
        """
        super().__init__()
        if max_lcm < 3:
            raise ValueError("max_lcm should be greater than or equal to 3")
        if not (0.0 <= answer_being_0_probability <= 1.0):
            raise ValueError("answer_being_0_probability must be in [0, 1]")
        self.max_lcm = max_lcm
        self.answer_being_0_probability = answer_being_0_probability

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.gcd_value: Optional[int] = None
        self.lcm_value: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory counting problem about GCD and LCM.\n"
            "Your task: Given gcd(P, Q) and lcm(P, Q), count the number of pairs of positive integers (P, Q)\n"
            "that satisfy both values simultaneously.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate LCM and GCD according to the original logic
        def all_factors(n: int) -> set[int]:
            factors = set()
            i = 1
            while i * i <= n:
                if n % i == 0:
                    factors.add(i)
                    factors.add(n // i)
                i += 1
            return factors

        if random.random() < self.answer_being_0_probability:
            # Force an impossible case with LCM % GCD != 0
            while True:
                lcm = random.randint(1, self.max_lcm)
                gcd = random.randint(1, lcm)
                if lcm % gcd != 0:
                    break
        else:
            lcm = random.randint(1, self.max_lcm)
            factors = list(all_factors(lcm))
            gcd = random.choice(factors)

        self.gcd_value = gcd
        self.lcm_value = lcm

        # Compute reference answer using the original algorithm
        def prime_factorization(n: int) -> dict[int, int]:
            prime2count: dict[int, int] = {}
            x = 2
            while x * x <= n:
                while n % x == 0:
                    prime2count[x] = prime2count.get(x, 0) + 1
                    n //= x
                x += 1
            if n > 1:
                prime2count[n] = prime2count.get(n, 0) + 1
            return prime2count

        def solve(g: int, l: int) -> int:
            gcd_pf = prime_factorization(g)
            lcm_pf = prime_factorization(l)
            count = 1
            for p in set(gcd_pf.keys()) | set(lcm_pf.keys()):
                x_count = gcd_pf.get(p, 0)
                y_count = lcm_pf.get(p, 0)
                if x_count > y_count:
                    count = 0
                    break
                if x_count == y_count:
                    count *= 1
                else:
                    count *= 2
            return count

        self.reference_answer = solve(self.gcd_value, self.lcm_value)
        assert (self.reference_answer == 0) == (self.lcm_value % self.gcd_value != 0)

        # Build problem statement
        self.current_problem = (
            "Find the number of pairs of positive integers (P, Q) that satisfy the following conditions:\n"
            f"1) gcd(P, Q) = {self.gcd_value}\n"
            f"2) lcm(P, Q) = {self.lcm_value}\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        # Parse the boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Negative answers are considered format errors in the original environment
        if user_answer < 0:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Check correctness
        assert self.reference_answer is not None
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "gcd": self.gcd_value,
            "lcm": self.lcm_value,
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
        """Sample a random action in the expected \\boxed{...} format."""
        # Sample 0 or a power of two for variety, both are plausible in this task
        if random.random() < 0.5:
            val = 0
        else:
            val = 1 << random.randint(0, 10)
        return f"\\boxed{{{val}}}"