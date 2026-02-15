from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GCDFibonacciProductEnv(Env):
    """
    Single-turn environment for computing the product of Fibonacci numbers over gcd pairs.

    Task:
    Given positive integers N, M, and a modulus MOD, compute:
      Product over all pairs (i, j) with 1 <= i <= N and 1 <= j <= M of f(gcd(i, j)) modulo MOD,
    where the Fibonacci sequence is defined as f(0) = 0, f(1) = 1, f(n) = f(n - 1) + f(n - 2) for n >= 2.

    The environment presents one problem per reset, and expects the final answer in \\boxed{...} format.
    """

    def __init__(
        self,
        max_n_m: int = 100000,
        allowed_mods: Tuple[int, ...] = (666623333, 998244353, 10**9 + 7),
        **kwargs,
    ):
        super().__init__()
        assert max_n_m >= 3, "max_n_m should be greater than or equal to 3"
        self.max_n_m: int = max_n_m
        self.allowed_mods: Tuple[int, ...] = allowed_mods

        # State variables for the current episode
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.MOD: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a number theory problem involving Fibonacci numbers and gcd.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem, and return the observation."""
        super().reset(seed)

        # Sample parameters
        self.N = random.randint(3, self.max_n_m)
        self.M = random.randint(3, self.max_n_m)
        self.MOD = random.choice(self.allowed_mods)

        # Build the problem statement
        self.current_problem = (
            "The Fibonacci sequence is defined as follows: f(0) = 0, f(1) = 1, and f(n) = f(n - 1) + f(n - 2) for all n ≥ 2.\n"
            f"Please compute the product of all f(gcd(i, j)) for all pairs (i, j) such that 1 ≤ i ≤ {self.N} and 1 ≤ j ≤ {self.M}. "
            f"Output the result modulo {self.MOD}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(self.N, self.M, self.MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the submitted answer and return the result."""
        # Ensure that a problem has been generated
        if self.reference_answer is None or self.MOD is None:
            # No active problem; terminate with zero reward
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Parse answer from \\boxed{...}
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and score
        info: dict[str, Any] = {}
        try:
            user_answer = int(answer_str)
        except ValueError:
            info.update({"error": "invalid_answer"})
            return TERMINAL_STATE, 0.0, True, False, info

        # Range check (preserved from original logic). Out of range counts as wrong.
        if not (0 <= user_answer < self.MOD):
            info.update({"error": "out_of_range", "user_answer": user_answer, "reference_answer": self.reference_answer})
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info.update(
            {
                "correct": is_correct,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "N": self.N,
                "M": self.M,
                "MOD": self.MOD,
            }
        )

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
        """Sample a random action in the required format."""
        mod = self.MOD if self.MOD is not None else random.choice(self.allowed_mods)
        random_answer = random.randint(0, mod - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, N: int, M: int, MOD: int) -> int:
        """
        Compute the reference answer using the same algorithm as the original environment.
        """
        # Precompute f and fr arrays
        max_n = max(N, M)
        f, fr = self._precompute_f_fr(max_n, MOD)

        # Ensure N <= M
        if N > M:
            N, M = M, N

        # Compute the product using division-block technique
        ans = 1
        i = 1
        while i <= N:
            divN = N // i
            divM = M // i
            j = min(N // divN, M // divM)
            base = f[j] * fr[i - 1] % MOD
            exponent = divN * divM
            ans = ans * pow(base, exponent, MOD) % MOD
            i = j + 1
        return ans

    def _precompute_f_fr(self, max_n: int, MOD: int) -> Tuple[list[int], list[int]]:
        """
        Precompute f and fr arrays using a linear sieve for the Möbius function and
        the alternating Fibonacci-like sequence transformation, then compute prefix products.
        """
        # Linear sieve to compute mu[1..max_n]
        is_composite = [False] * (max_n + 1)
        primes: list[int] = []
        mu = [0] * (max_n + 1)
        mu[1] = 1
        for i in range(2, max_n + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > max_n:
                    break
                is_composite[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]

        # Arrays f and fr
        f = [1] * (max_n + 1)
        fr = [1] * (max_n + 1)

        # Generate sequence values and apply Möbius-weighted updates
        A, B = 1, 0
        for i in range(1, max_n + 1):
            # Update the alternating Fibonacci-like sequence terms
            B = (A + B) % MOD
            A = (B - A) % MOD

            # Compute modular inverse of B (assumes B invertible modulo MOD)
            invB = pow(B, MOD - 2, MOD)

            # Apply contributions to f and fr using mu
            for j in range(i, max_n + 1, i):
                k = j // i
                m = mu[k]
                # Update f[j]
                if m == -1:
                    f[j] = f[j] * invB % MOD
                elif m == 0:
                    pass
                else:  # m == 1
                    f[j] = f[j] * B % MOD

                # Update fr[j]: note fr uses G[1 - mu[k]]
                if m == 1:
                    fr[j] = fr[j] * invB % MOD
                elif m == 0:
                    pass
                else:  # m == -1
                    fr[j] = fr[j] * B % MOD

        # Prefix products
        for i in range(1, max_n + 1):
            f[i] = f[i - 1] * f[i] % MOD
            fr[i] = fr[i - 1] * fr[i] % MOD

        return f, fr