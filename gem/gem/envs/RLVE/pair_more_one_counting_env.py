import random
import re
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PairMoreOneCountingEnv(Env):
    """
    Environment for counting the number of pairs of binary strings (S, T) under constraints:
    - len(S) = N = M + delta, len(T) = M
    - number of 1s in S is strictly greater than number of 1s in T
    The answer should be reported modulo 10^K.
    Single-turn Q&A environment.
    """

    def __init__(
        self,
        max_M: int = 100000,
        max_delta: int = 100000,
        max_K: int = 5,
        **kwargs,
    ):
        super().__init__()
        assert max_M >= 1, "max_M must be at least 1"
        assert max_delta >= 0, "max_delta must be at least 0"
        assert max_K >= 1, "max_K must be at least 1"

        self.max_M = max_M
        self.max_delta = max_delta
        self.max_K = max_K

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_M: Optional[int] = None
        self.current_delta: Optional[int] = None
        self.current_N: Optional[int] = None
        self.current_K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics counting problem about binary strings.\n"
            "Task: Count the number of pairs of binary strings (S, T) such that:\n"
            "- The length of S is N = M + delta, and the length of T is M.\n"
            "- The number of 1s in S is strictly greater than the number of 1s in T.\n"
            "You must output the result modulo 10^K.\n"
            "Output Format: Your final answer must be a single integer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample parameters
        M = random.randint(1, self.max_M)
        delta = random.randint(0, self.max_delta)
        N = M + delta
        K = random.randint(1, self.max_K)

        self.current_M = M
        self.current_delta = delta
        self.current_N = N
        self.current_K = K

        # Compute the reference answer
        self.reference_answer = self._compute_answer(M, delta, K)

        # Build problem statement
        self.current_problem = (
            f"Please count the number of pairs of binary strings (S, T) such that:\n"
            f"- The length of S is N = {N} = {M} + {delta}, and the length of T is {M}.\n"
            f"- The number of 1s in S is strictly greater than the number of 1s in T.\n\n"
            f"Please output the result modulo 10^{K}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "M": M,
            "delta": delta,
            "N": N,
            "K": K,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer."""
        # Parse answer from \boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.current_K is not None, "Environment must be reset before step."
        assert self.reference_answer is not None, "Environment must be reset before step."

        modulus = 10 ** self.current_K
        info: dict[str, Any] = {}

        # Range check
        if not (0 <= user_answer < modulus):
            info["error"] = "out_of_range"
            is_correct = False
        else:
            is_correct = (user_answer == self.reference_answer)

        reward = 1.0 if is_correct else 0.0
        info.update(
            {
                "correct": is_correct,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "M": self.current_M,
                "delta": self.current_delta,
                "N": self.current_N,
                "K": self.current_K,
            }
        )
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...} within the valid range."""
        K = self.current_K if self.current_K is not None else 1
        modulus = 10 ** K
        random_answer = random.randint(0, modulus - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_answer(self, M: int, delta: int, K: int) -> int:
        """
        Compute the number of valid pairs (S, T) modulo 10^K using extended Lucas and CRT.
        This preserves the core logic from the original environment.
        """
        N = M + delta

        MOD10 = 10 ** K
        MOD2 = 2 ** (K + 1)
        MOD5 = 5 ** K
        MOD_ALL = MOD10 * 2  # 2 * 10^K

        # Precompute factorial products excluding multiples of 2 up to MOD2
        s2 = [1] * (MOD2 + 1)
        for i in range(1, MOD2 + 1):
            if (i & 1) == 0:
                s2[i] = s2[i - 1]
            else:
                s2[i] = (s2[i - 1] * i) % MOD2

        # Precompute factorial products excluding multiples of 5 up to MOD5
        s5 = [1] * (MOD5 + 1)
        for i in range(1, MOD5 + 1):
            if i % 5 == 0:
                s5[i] = s5[i - 1]
            else:
                s5[i] = (s5[i - 1] * i) % MOD5

        def solve_fact(n: int, p: int, modp: int) -> int:
            """Recursive factorial modulo p^c excluding multiples of p."""
            if n <= 1:
                return 1
            sub = solve_fact(n // p, p, modp)
            if p == 2:
                sp_mod = s2[modp]
                sp_rem = s2[n % modp]
            else:
                sp_mod = s5[modp]
                sp_rem = s5[n % modp]
            return sub * pow(sp_mod, n // modp, modp) % modp * sp_rem % modp

        def count_p(n: int, p: int) -> int:
            """Count exponent of prime p in n!."""
            cnt = 0
            while n:
                n //= p
                cnt += n
            return cnt

        def lucas(n: int, m: int) -> int:
            """
            Extended Lucas theorem to compute C(n, m) modulo 2 * 10^K (i.e., 2^(K+1) * 5^K),
            followed by CRT combination.
            """
            # 2-adic part
            c2 = count_p(n, 2) - count_p(m, 2) - count_p(n - m, 2)
            if c2 <= K:
                a2 = solve_fact(n, 2, MOD2)
                b2 = solve_fact(m, 2, MOD2)
                inv_b2 = pow(b2, -1, MOD2)
                a2 = a2 * inv_b2 % MOD2
                c2part = solve_fact(n - m, 2, MOD2)
                inv_c2 = pow(c2part, -1, MOD2)
                a2 = a2 * inv_c2 % MOD2 * pow(2, c2, MOD2) % MOD2
            else:
                a2 = 0

            # 5-adic part
            c5 = count_p(n, 5) - count_p(m, 5) - count_p(n - m, 5)
            if c5 < K:
                a5 = solve_fact(n, 5, MOD5)
                b5 = solve_fact(m, 5, MOD5)
                inv_b5 = pow(b5, -1, MOD5)
                a5 = a5 * inv_b5 % MOD5
                c5part = solve_fact(n - m, 5, MOD5)
                inv_c5 = pow(c5part, -1, MOD5)
                a5 = a5 * inv_c5 % MOD5 * pow(5, c5, MOD5) % MOD5
            else:
                a5 = 0

            # Combine via CRT: x ≡ a2 (mod MOD2), x ≡ a5 (mod MOD5)
            t = (a5 - a2) * pow(MOD2, -1, MOD5) % MOD5
            return (a2 + MOD2 * t) % (MOD2 * MOD5)

        # Main computation
        if N == M:
            total = pow(2, 2 * N, MOD_ALL)
            comb = lucas(2 * N, N)
            ans = (total - comb) % MOD_ALL
            ans = (ans // 2) % MOD10
        else:
            total = pow(2, N + M, MOD_ALL)
            diff = N - M
            for i in range(1, diff):
                total = (total + lucas(N + M, M + i)) % MOD_ALL
            ans = (total // 2) % MOD10

        return ans