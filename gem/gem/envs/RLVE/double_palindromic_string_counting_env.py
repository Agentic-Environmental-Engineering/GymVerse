from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DoublePalindromicStringCountingEnv(Env):
    """Environment for counting double palindromic strings - single-turn Q&A.

    A string S is double palindromic if:
    - Each character in S is an integer between 1 and C (inclusive).
    - S can be written as the concatenation of two non-empty palindromic strings, S1 and S2, i.e., S = S1 + S2.

    The task is to count the number of distinct double palindromic strings of length at most N.
    """

    def __init__(
        self,
        max_n: int = 2000,
        C: int = 10,
        **kwargs
    ):
        super().__init__()
        self.max_n = max_n
        self.C = C
        self.N: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem about double palindromic strings.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate parameters
        assert self.max_n >= 2, "max_n should be greater than or equal to 2"
        assert self.C >= 1, "C should be greater than or equal to 1"

        # Sample N
        self.N = random.randint(2, self.max_n)

        # Build problem statement
        self.current_problem = (
            "We define a string S as double palindromic if it satisfies all of the following conditions:\n"
            f"- Each character in S is an integer between 1 and {self.C} (inclusive).\n"
            "- S can be written as the concatenation of two non-empty palindromic strings, S1 and S2, such that S = S1 + S2.\n\n"
            f"Please count the number of distinct double palindromic strings of length at most {self.N}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_answer(self.N, self.C)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to check the submitted answer."""
        # Parse answer from \boxed{...}
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # Random small integer as a guess
        random_answer = random.randint(0, 10**6)
        return f"\\boxed{{{random_answer}}}"

    def _compute_answer(self, N: int, C: int) -> int:
        """Compute the number of distinct double palindromic strings of length at most N."""
        # Special-case C == 1 to avoid division by zero in formulas.
        # For alphabet size 1, every string is a palindrome.
        # A double palindromic string must be of length at least 2 (split into two non-empty palindromes).
        # Therefore, the number of distinct strings of length at most N is exactly N-1 (for lengths 2..N).
        if C == 1:
            return max(0, N - 1)

        def pre(limit_n: int):
            mu = [0] * (limit_n + 1)
            f_pref = [0] * (limit_n + 1)
            is_comp = [False] * (limit_n + 1)
            primes = []

            mu[1] = 1
            f_pref[1] = 1

            for i in range(2, limit_n + 1):
                if not is_comp[i]:
                    primes.append(i)
                    mu[i] = -1
                    f_pref[i] = 1 - i
                for p in primes:
                    ip = i * p
                    if ip > limit_n:
                        break
                    is_comp[ip] = True
                    if i % p == 0:
                        f_pref[ip] = f_pref[i]
                        break
                    mu[ip] = -mu[i]
                    f_pref[ip] = f_pref[i] * (1 - p)

            for i in range(1, limit_n + 1):
                mu[i] += mu[i - 1]
                f_pref[i] += f_pref[i - 1]
            return mu, f_pref

        def S(x: int) -> int:
            return x * (x + 1) // 2

        def make_calc1(f_pref_local, limit_n: int):
            memo: dict[int, int] = {}

            def calc1(n: int) -> int:
                if n <= limit_n:
                    return f_pref_local[n]
                if n in memo:
                    return memo[n]
                res = n
                i = 2
                while i <= n:
                    t = n // i
                    last = n // t
                    res -= (S(last) - S(i - 1)) * calc1(t)
                    i = last + 1
                memo[n] = res
                return res

            return calc1

        def make_calc2(mu_pref_local, limit_n: int):
            memo: dict[int, int] = {}

            def calc2(n: int) -> int:
                if n <= limit_n:
                    return mu_pref_local[n]
                if n in memo:
                    return memo[n]
                res = 1
                i = 2
                while i <= n:
                    t = n // i
                    last = n // t
                    res -= (last - i + 1) * calc2(t)
                    i = last + 1
                memo[n] = res
                return res

            return calc2

        def query1(n: int, CC: int, den: int) -> int:
            # ((t*(4n-2) - 4*(t-C)/(C-1)) / (C-1))
            t = pow(CC, n + 1)
            part = 4 * (t - CC) // den
            return (t * (4 * n - 2) - part) // den

        def querysum(n: int, CC: int, den: int) -> int:
            half = n // 2
            s_half = query1(half, CC, den)
            t = pow(CC, half + 1)
            extra = (n + half) if (n & 1) else half
            return s_half + t * extra

        def solve1(limit_n: int, CC: int, calc1_fn, den: int) -> int:
            ans = 0
            i = 1
            while i <= limit_n:
                t = limit_n // i
                last = limit_n // t
                ans += (querysum(last, CC, den) - querysum(i - 1, CC, den)) * calc1_fn(t)
                i = last + 1
            return ans

        def query2(n: int, CC: int, den: int) -> int:
            half = n // 2
            t = pow(CC, half + 1)
            # 2*(t-C)/(C-1)  +  (t if odd)
            base = 2 * (t - CC) // den
            return base + (t if (n & 1) else 0)

        def solve2(limit_n: int, CC: int, calc2_fn, den: int) -> int:
            ans = 0
            i = 1
            while i <= limit_n:
                t = limit_n // i
                last = limit_n // t
                ans += (query2(last, CC, den) - query2(i - 1, CC, den)) * calc2_fn(t)
                i = last + 1
            return ans

        den = C - 1
        mu_pref, f_pref = pre(N)
        calc1_fn = make_calc1(f_pref, N)
        calc2_fn = make_calc2(mu_pref, N)
        answer = solve1(N, C, calc1_fn, den) - solve2(N, C, calc2_fn, den)
        return answer