import math
import random
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DiscreteLogarithmEnv(Env):
    """Discrete logarithm (modular) environment - single-turn Q&A.

    Task:
      Find the smallest non-negative integer y such that (X^y) mod Z = K mod Z.

    Answer format:
      The answer must be submitted in \\boxed{...} containing a single non-negative integer.
    """

    def __init__(
        self,
        max_z: int = 10**6,
        **kwargs,
    ):
        super().__init__()
        self.max_z = max_z
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.X: Optional[int] = None
        self.Z: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a discrete logarithm problem modulo Z.\n"
            "Find the smallest non-negative integer y such that (X^y) mod Z = K mod Z.\n"
            "Please provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)
        assert self.max_z >= 2, "max_z should be greater than or equal to 2"

        Z = random.randint(2, self.max_z)
        X = random.randint(2, Z)
        Y = random.randint(2, Z)
        K = pow(X, Y, Z)

        self.Z = Z
        self.X = X
        self.K = K

        # Build the problem prompt
        self.current_problem = (
            f"Please find the smallest non-negative integer y such that "
            f"({X}^y) mod {Z} = {K} mod {Z}.\n\n"
            f"Output Format: Your final answer should be a single non-negative integer in \\boxed{{...}}."
        )

        # Compute the reference (minimal) solution
        self.reference_answer = self._modular_log_solver(X, Z, K)
        assert self.reference_answer is not None and self.reference_answer >= 0, "Reference answer should be non-negative"

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer and terminate the episode."""
        # Parse \\boxed{...}
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate integer and non-negative
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if user_answer < 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None
        assert self.X is not None and self.Z is not None and self.K is not None

        # Must be the smallest y; equality to reference answer is required
        is_congruence = (pow(self.X, user_answer, self.Z) == self.K)
        is_correct = is_congruence and (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "X": self.X,
            "Z": self.Z,
            "K": self.K,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within \\boxed{...} from text."""
        import re

        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _modular_log_solver(self, a: int, mod: int, r: int) -> int:
        """Solve for minimal x such that a^x ≡ r (mod mod) using extended BSGS."""
        def adjust(x: int, m: int) -> int:
            return (x % m + m) % m

        def power(base: int, exp: int, m: int) -> int:
            s = 1
            x = base % m
            n = exp
            while n:
                if n & 1:
                    s = (s * x) % m
                x = (x * x) % m
                n >>= 1
            return s

        def gcd(x: int, y: int) -> int:
            return math.gcd(x, y)

        def exgcd(x: int, y: int) -> Tuple[int, int]:
            if y == 0:
                return (1, 0)
            else:
                x1, y1 = exgcd(y, x % y)
                return (y1, x1 - (x // y) * y1)

        def BSGS(base: int, rem: int, m: int) -> int:
            a = base % m
            r_local = rem % m
            T = int(round(math.sqrt(m)))
            a_T = power(a, T, m)
            table = {}
            cur = r_local
            for i in range(1, T + 1):
                cur = (cur * a) % m
                table[cur] = i
            cur = a_T
            for i in range(1, T + 2):
                val = cur
                if val in table:
                    return i * T - table[val]
                cur = (cur * a_T) % m
            return -1

        def exBSGS(base: int, rem: int, m: int) -> int:
            a_local = base % m
            r_local = rem % m
            g = gcd(m, a_local)
            if r_local % g != 0:
                if r_local == 1:
                    return 0
                else:
                    return -1
            if g == 1:
                return BSGS(a_local, r_local, m)
            else:
                iv, y = exgcd(a_local // g, m // g)
                iv = adjust(iv, m // g)
                res = exBSGS(a_local, (r_local // g) * iv % (m // g), m // g)
                if res < 0:
                    return -1
                return res + 1

        return exBSGS(a, r, mod)

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        # If modulus is available, sample within a reasonable range; otherwise fallback.
        if self.Z is not None and self.Z > 0:
            val = random.randint(0, max(0, self.Z - 1))
        else:
            val = random.randint(0, 1000)
        return f"\\boxed{{{val}}}"