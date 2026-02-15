from typing import Any, Optional, SupportsFloat, Tuple, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumSetMultiplicationEnv(Env):
    """
    Environment for the problem:
    Consider all sequences A[1..N] of distinct integers chosen from [1, K].
    Compute the sum of (A[1] × A[2] × ... × A[N]) over all such sequences, modulo MOD.
    Single-turn Q&A environment.
    """

    def __init__(
        self,
        max_n: int = 10,
        max_k: int = 20,
        mods: Tuple[int, ...] = (666623333, 998244353, 10**9 + 7),
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: Maximum value for N (must be >= 3).
            max_k: Maximum value for K (must be > max_n).
            mods: A tuple of possible mod values to choose from.
        """
        super().__init__()
        if max_n < 3:
            raise ValueError("max_n should be greater than or equal to 3")
        if max_k <= max_n:
            raise ValueError("max_k should be greater than max_n")
        if not mods or not isinstance(mods, tuple):
            raise ValueError("mods must be a non-empty tuple of integers")

        self.max_n: int = max_n
        self.max_k: int = max_k
        self.mods: Tuple[int, ...] = mods

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_params: Dict[str, int] = {}

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a combinatorics problem.\n"
            "Task: Consider all sequences A[1..N] of distinct integers chosen from [1, K]. "
            "Compute the sum of (A[1] × A[2] × ... × A[N]) over all such sequences, modulo MOD.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n"
            "Your answer should be in the range [0, MOD-1].\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample parameters
        N = random.randint(3, self.max_n)
        K = random.randint(N + 1, self.max_k)
        MOD = random.choice(self.mods)

        # Build the problem prompt
        self.current_problem = (
            f"Consider all sequences A[1..{N}] of distinct integers chosen from [1, {K}]. "
            f"Compute the sum of (A[1] × A[2] × ... × A[{N}]) over all such sequences, modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Store current params
        self.current_params = {"N": N, "K": K, "MOD": MOD}

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(N, K, MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Check the submitted answer and return the outcome."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to convert to integer
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        MOD = self.current_params.get("MOD", None)
        if MOD is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_error"}

        # Range check
        if not (0 <= user_answer < MOD):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "error": "out_of_range"
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compare with reference
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of boxed content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...}."""
        MOD = self.current_params.get("MOD", None)
        if MOD is None:
            # Fallback if called before reset
            MOD = random.choice(self.mods)
        random_answer = random.randint(0, MOD - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, N: int, K: int, MOD: int) -> int:
        """Compute the reference answer using the original algorithm."""
        # Dynamic sizing based on N
        size = 2 * N + 3  # to safely index up to 2N+1 and use i+1 at i=2N
        F = [0] * size
        C = [0] * size

        def mod_pow(a: int, b: int) -> int:
            a %= MOD
            res = 1
            while b:
                if b & 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b >>= 1
            return res

        INX = K if (2 * N + 1) > K else (2 * N + 1)
        C[INX] = 1
        F[0] = 1

        for i in range(1, N + 1):
            for j in range(2 * i, 1, -1):
                F[j] = (F[j - 1] * j + F[j - 2] * (2 * i - j)) % MOD
            F[1] = F[0]
            F[0] = 0

        if INX == 2 * N + 1:
            for i in range(1, 2 * N + 1):
                C[INX] = (C[INX] * ((K - i) % MOD)) % MOD
                C[INX] = (C[INX] * mod_pow(i % MOD, MOD - 2)) % MOD

        for i in range(INX - 1, -1, -1):
            numerator = (K + 2 * N - i) % MOD
            denom = (K - i) % MOD
            C[i] = C[i + 1] * numerator % MOD * mod_pow(denom, MOD - 2) % MOD

        ans = 0
        for i in range(0, 2 * N + 1):
            ans = (ans + C[i] * F[i]) % MOD
        for i in range(1, N + 1):
            ans = ans * i % MOD

        return ans