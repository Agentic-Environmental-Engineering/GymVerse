from math import gcd
from functools import lru_cache
import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CirculatingDecimalCountingEnv(Env):
    """Environment for counting distinct pure repeating decimals in base K."""

    def __init__(
        self,
        max_n: int = 100000,
        max_m: int = 100000,
        max_k: int = 100,
        **kwargs
    ):
        super().__init__()
        assert max_n >= 1, "max_n should be greater than or equal to 1"
        assert max_m >= 1, "max_m should be greater than or equal to 1"
        assert max_k >= 2, "max_k should be greater than or equal to 2"
        self.max_n = max_n
        self.max_m = max_m
        self.max_k = max_k

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a number theory counting problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate parameters
        N = random.randint(1, self.max_n)
        M = random.randint(1, self.max_m)
        K = random.randint(2, self.max_k)

        self.N, self.M, self.K = N, M, K

        # Build problem description
        problem = (
            f"Please count how many distinct pure repeating decimals (in terms of numeric value) exist in base {K}, "
            f"that can be written as a reduced fraction x/y where 1 ≤ x ≤ {N} and 1 ≤ y ≤ {M}, with x and y being integers.\n"
            "A number is called a pure repeating decimal if and only if it can be written in the form of "
            "$$a.\\dot{c_1} c_2 c_3 \\dots c_{p - 1} \\dot{c_p}$$, where a is an integer, p ≥ 1, and each c_i (1 ≤ i ≤ p) "
            f"is a digit in base {K}.\n\n"
            "Examples:\n"
            "- In base 10, 0.454545... = 0.\\dot{4}\\dot{5} is a pure repeating decimal; it can be written as 5/11 or 10/22.\n"
            "- In contrast, 0.166666... = 0.1\\dot{6} is not pure repeating in base 10; it can be written as 1/6.\n\n"
            "Note:\n"
            "- Integers are considered pure repeating, because their decimal part can be represented as a repeating sequence of 0s.\n"
            "- Finite decimals with non-zero fractional parts are not considered pure repeating.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )
        self.current_problem = problem

        # Compute reference answer using the original algorithm
        ans = self._compute_answer(N, M, K)
        assert ans > 0
        self.reference_answer = ans

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_answer(self, N: int, M: int, K: int) -> int:
        """Compute the number of distinct pure repeating decimals using the original algorithm."""
        LIM = min(M, max(K, int(M ** 0.5) + 1))

        g = [0] * (K + 1)
        for i in range(1, K + 1):
            g[i] = g[i - 1] + (1 if gcd(i, K) == 1 else 0)

        mu = [0] * (LIM + 1)
        is_comp = [False] * (LIM + 1)
        f = [0] * (LIM + 1)
        primes = []

        mu[1] = 1
        f[1] = 1

        def G(x: int) -> int:
            """Count numbers in [1..x] that are coprime to K using block decomposition."""
            return (x // K) * g[K] + g[x % K]

        for i in range(2, LIM + 1):
            if not is_comp[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                ip = i * p
                if ip > LIM:
                    break
                is_comp[ip] = True
                if i % p == 0:
                    mu[ip] = 0
                    break
                else:
                    mu[ip] = -mu[i]
            f[i] = f[i - 1] + mu[i] * (G(i) - G(i - 1))

        @lru_cache(None)
        def F(x: int) -> int:
            """Compute the summatory function via divide-and-conquer with caching."""
            if x <= LIM:
                return f[x]
            res = 1
            l = 2
            while l <= x:
                t = x // l
                r = x // t
                res -= F(t) * (G(r) - G(l - 1))
                l = r + 1
            return res

        ans = 0
        l = 1
        up = min(N, M)
        while l <= up:
            n_div = N // l
            m_div = M // l
            r = min(N // n_div, M // m_div)
            ans += n_div * G(m_div) * (F(r) - F(l - 1))
            l = r + 1

        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the user's answer."""
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Reference answer must be computed before step."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "K": self.K,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the expected boxed format."""
        random_answer = random.randint(0, max(self.max_n, self.max_m))
        return f"\\boxed{{{random_answer}}}"