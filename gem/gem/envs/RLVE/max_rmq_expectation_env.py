from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxRMQExpectationEnv(Env):
    """Environment for computing the expected value of the maximum of range minimum queries over a random array."""

    MODS: Tuple[int, int, int] = (666623333, 998244353, 10**9 + 7)

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 200,
        mods: Optional[Tuple[int, int, int]] = None,
        **kwargs
    ):
        """
        Initialize the MaxRMQExpectationEnv instance.

        Parameters:
        - N: If provided, the array size will be fixed to this value (must be >= 2).
        - min_N: Minimum value for N when sampled randomly (must be >= 2).
        - max_N: Maximum value for N when sampled randomly.
        - mods: Tuple of candidate prime moduli to sample from.
        """
        super().__init__()
        if N is not None and N < 2:
            raise ValueError("N should be greater than or equal to 2")
        if min_N < 2:
            raise ValueError("min_N should be greater than or equal to 2")
        if min_N > max_N:
            raise ValueError("min_N should be less than or equal to max_N")

        self.N_fixed: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N
        self.mods: Tuple[int, int, int] = mods if mods is not None else self.MODS

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Generated parameters
        self.N: Optional[int] = None
        self.X: Optional[int] = None
        self.Q: Optional[int] = None
        self.intervals: List[Tuple[int, int]] = []
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Compute the expected value of the maximum of range minimum queries over a random array.\n"
            "You must output a single integer modulo the given MOD.\n"
            "Output Format: Provide your final answer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            self.N = self.N_fixed
        else:
            self.N = random.randint(self.min_N, self.max_N)
        assert self.N >= 2, "N should be greater than or equal to 2"

        # Generate X, Q, intervals, and MOD
        self.X = random.randint(2, self.N)
        self.Q = random.randint(1, self.N)

        self.intervals = []
        for _ in range(self.Q):
            L = random.randint(1, self.N)
            R = random.randint(1, self.N)
            if L > R:
                L, R = R, L
            self.intervals.append((L, R))

        self.MOD = random.choice(self.mods)

        # Build problem prompt
        intervals_str = "\n".join(f"[{L}, {R}]" for L, R in self.intervals)
        self.current_problem = (
            f"Let's randomly generate an array A[1], ..., A[{self.N}], where each A[i] is independently and "
            f"uniformly chosen from the integers 1 to {self.X} (so there are {self.X}^{self.N} possible arrays in total). "
            f"You are also given {self.Q} intervals [L[i], R[i]] (1 ≤ i ≤ {self.Q}):\n"
            f"{intervals_str}\n\n"
            f"For each interval [L[i], R[i]], define M[i] = min(A[j]) for L[i] ≤ j ≤ R[i]. "
            f"Please compute the expected value of max(M[1], ..., M[{self.Q}]) and output the result modulo {self.MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference(self.N, self.X, self.intervals, self.MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": self.N,
            "X": self.X,
            "Q": self.Q,
            "intervals": self.intervals,
            "MOD": self.MOD,
            "reference_answer": self.reference_answer,
        }

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate a single-step answer submission."""
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to convert to integer
        try:
            user_answer = int(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check: 0 <= answer < MOD
        assert self.MOD is not None, "Environment must be reset before calling step."
        if not (0 <= user_answer < self.MOD):
            return TERMINAL_STATE, 0.0, True, False, {"error": "range_error"}

        # Verify correctness
        assert self.reference_answer is not None, "Environment must be reset before calling step."
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "X": self.X,
            "Q": self.Q,
            "intervals": self.intervals,
            "MOD": self.MOD,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the valid range."""
        if self.MOD is None:
            # If environment not reset yet, default to a reasonable modulus
            mod = self.mods[0]
        else:
            mod = self.MOD
        random_answer = random.randint(0, mod - 1)
        return f"\\boxed{{{random_answer}}}"

    @staticmethod
    def _modinv(a: int, mod: int) -> int:
        """Modular inverse using Fermat's little theorem (mod must be prime)."""
        return pow(a, mod - 2, mod)

    def _compute_reference(
        self, N: int, X: int, intervals: List[Tuple[int, int]], MOD: int
    ) -> int:
        """Compute the reference answer using the original algorithm."""
        # ar[i] will store the maximum l among all queries whose r+1 == i
        ar = [0] * (N + 2)
        for l, r in intervals:
            ar[r + 1] = max(ar[r + 1], l)
        # prefix max so that ar[j] = max_{i ≤ j}(ar[i])
        for i in range(1, N + 2):
            if ar[i] < ar[i - 1]:
                ar[i] = ar[i - 1]

        # ix = 1/X mod
        ix = self._modinv(X % MOD, MOD)
        ans = 0

        # loop over possible threshold i1 = 1..X
        for i1 in range(1, X + 1):
            # p = (i1 - 1) / X  (mod)
            p = (i1 - 1) * ix % MOD
            one_minus_p = (1 - p) % MOD
            # ip = (1 - p)^{-1} mod
            ip = self._modinv(one_minus_p, MOD)

            # precompute ff0[j] = (1-p)^j, ff1[j] = ip^j
            ff0 = [1] * (N + 1)
            ff1 = [1] * (N + 1)
            for j in range(1, N + 1):
                ff0[j] = ff0[j - 1] * one_minus_p % MOD
                ff1[j] = ff1[j - 1] * ip % MOD

            # f0[j], f1[j] DP arrays
            f0 = [0] * (N + 1)
            f1 = [0] * (N + 1)
            f1[0] = 1
            for j in range(1, N + 1):
                if ar[j] > 0:
                    prev = (f1[j - 1] - f1[ar[j] - 1]) % MOD
                else:
                    prev = f1[j - 1]
                # f0[j] = p * prev * (1-p)^(j-1)
                f0[j] = p * prev % MOD * ff0[j - 1] % MOD
                # f1[j] = f1[j-1] + f0[j]*(ip^j)
                f1[j] = (f1[j - 1] + f0[j] * ff1[j]) % MOD

            # sum up contributions from j = ar[N+1]..N
            Lmax = ar[N + 1]
            s = 0
            for j in range(Lmax, N + 1):
                s = (s + f0[j] * ff0[N - j]) % MOD

            # accumulate into answer: ans += 1 - s
            ans = (ans + 1 - s) % MOD

        return ans