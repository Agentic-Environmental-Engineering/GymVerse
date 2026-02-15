import random
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FixedModK_Selection_CountingEnv(Env):
    """Environment for computing a fixed-mod-k selection counting problem."""

    def __init__(
        self,
        max_n: int = 1000,
        max_k: int = 1000,
        mod_range: int = 1000000,
        **kwargs
    ):
        super().__init__()
        self.max_n = max_n
        self.max_k = max_k
        self.mod_range = mod_range

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.R: Optional[int] = None
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem involving binomial coefficients modulo a prime.\n"
            "You must compute the sum of binomial coefficients at indices congruent to r modulo k.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem."""
        super().reset(seed)

        # Parameter validation (preserving original logic)
        assert self.max_n >= 1, "max_n should be greater than or equal to 1"
        assert self.max_k >= 1, "max_k should be greater than or equal to 1"

        # Generate problem parameters
        self.N = random.randint(1, self.max_n)
        # In original environment, K is sampled from [2, MAX_K]
        self.K = random.randint(2, self.max_k) if self.max_k >= 2 else 2
        self.R = random.randint(0, self.K - 1)
        self.MOD = random.randint(2, self.mod_range)

        # Build problem statement (English only)
        self.current_problem = (
            "Please compute the following value:\n"
            f"Sum_{{i=0}}^{{\\infty}} C_{{{self.N} \\times {self.K}}}^{{i \\times {self.K} + {self.R}}} modulo {self.MOD}.\n"
            "In other words, sum all binomial coefficients of (n*k choose i*k + r) for i >= 0, and output the result modulo p.\n"
            f"Here: n = {self.N}, k = {self.K}, r = {self.R}, p = {self.MOD}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer using fast exponentiation with convolution modulo k
        self.reference_answer = self._solve(self.N, self.K, self.R, self.MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _multiply(self, lhs: List[int], rhs: List[int], P: int, K: int) -> List[int]:
        """Convolution modulo K with coefficients modulo P."""
        result = [0] * K
        for i in range(K):
            for j in range(K):
                result[(i + j) % K] = (result[(i + j) % K] + lhs[i] * rhs[j]) % P
        return result

    def _solve(self, n: int, k: int, r: int, m: int) -> int:
        """Compute the R-th entry after raising the base vector to the power n*k with convolution modulo k."""
        # Prepare base vector a
        a = [0] * k
        if k == 1:
            a[0] = 2 % m
        else:
            a[0] = 1
            a[1] = 1

        # Identity vector for convolution exponentiation
        ans = [0] * k
        ans[0] = 1

        # Exponent: n * k
        e = n * k

        # Fast exponentiation by squaring
        while e > 0:
            if e & 1:
                ans = self._multiply(ans, a, m, k)
            a = self._multiply(a, a, m, k)
            e >>= 1

        # Output the r-th entry
        return ans[r]

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the answer."""
        # Parse the boxed answer
        parsed = self._parse_answer(action)

        if parsed is None:
            # Format error: no \\boxed{...} found
            info = {
                "error": "format_error",
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": None,
                "N": self.N,
                "K": self.K,
                "R": self.R,
                "MOD": self.MOD,
            }
            return TERMINAL_STATE, -0.1, True, False, info

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            info = {
                "error": "invalid_answer",
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": None,
                "N": self.N,
                "K": self.K,
                "R": self.R,
                "MOD": self.MOD,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compare with reference
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "error": None,
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
            "R": self.R,
            "MOD": self.MOD,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        modulo = self.MOD if self.MOD is not None else 100
        random_answer = random.randint(0, max(0, modulo - 1))
        return f"\\boxed{{{random_answer}}}"