from typing import Any, Optional, SupportsFloat, Tuple, Dict, List
import random
from bisect import bisect_left
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SegmentMinLengthEqual_CountingEnv(Env):
    """Environment for counting arrays where each segment's minimum equals its length."""

    def __init__(
        self,
        min_N: int = 3,
        max_N: int = 50,
        max_mod: int = 1000000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            min_N: Minimum value for N (inclusive), must be >= 3.
            max_N: Maximum value for N (inclusive), must be >= min_N.
            max_mod: Maximum modulo value (exclusive upper bound for random generation, at least 2).
        """
        super().__init__()
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")
        if max_mod < 2:
            raise ValueError("max_mod should be at least 2")

        self.min_N = min_N
        self.max_N = max_N
        self.max_mod = max_mod

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.S: Optional[List[int]] = None
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a counting problem about array segment partitions.\n"
            "An array x[1], x[2], ..., x[N] is called valid if and only if there exists a partition of it into intervals\n"
            "such that the minimum value in each interval is exactly equal to the interval’s length.\n"
            "Equivalently, there exist indices 0 = x_1 < x_2 < ... < x_m = N, such that for every 1 ≤ i < m,\n"
            "we have min_{j = x_i + 1}^{x_{i+1}} x[j] = x_{i+1} - x_i.\n\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new single-turn problem."""
        super().reset(seed)

        # Generate parameters
        N = random.randint(self.min_N, self.max_N)
        S_list = sorted(random.sample(range(1, N + 1), random.randint(2, N)))
        MOD = random.randint(2, self.max_mod)

        self.N = N
        self.S = S_list
        self.MOD = MOD

        # Build problem statement
        set_str = "{" + ", ".join(map(str, S_list)) + "}"
        self.current_problem = (
            f"An array x[1], x[2], ..., x[{N}] is called valid if and only if there exists a partition of it into intervals\n"
            f"such that the minimum value in each interval is exactly equal to the interval’s length.\n"
            f"Equivalently, there exist indices 0 = x_1 < x_2 < ... < x_m = {N}, such that for every 1 ≤ i < m,\n"
            f"we have min_{{j = x_i + 1}}^{{x_{{i+1}}}} x[j] = x_{{i+1}} - x_i.\n"
            f"What is the number of such valid arrays x, where each element x[i] must belong to the set S = {set_str}?\n"
            f"Output the answer modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(N, S_list, MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, N: int, S: List[int], MOD: int) -> int:
        """Compute the number of valid arrays modulo MOD using dynamic programming."""

        def quick_power(a: int, e: int, m: int) -> int:
            # Fast modular exponentiation
            res = 1
            a %= m
            while e:
                if e & 1:
                    res = (res * a) % m
                a = (a * a) % m
                e >>= 1
            return res

        M = len(S)
        exist_set = set(S)

        # C[i] = count of elements in S >= i, for i in 1..N
        C = [0] * (N + 1)
        for i in range(1, N + 1):
            C[i] = M - bisect_left(S, i)

        # DP: F[i] = number of valid arrays of length i
        F = [0] * (N + 1)
        F[0] = 1

        for i in range(1, N + 1):
            total = 0
            for j in range(i):
                L = i - j  # length of the last segment
                if L in exist_set:
                    cL = C[L]
                    # ways to fill a segment of length L with min exactly L:
                    ways = (quick_power(cL, L, MOD) - quick_power(cL - 1, L, MOD) + MOD) % MOD
                    total = (total + F[j] * ways) % MOD
            F[i] = total

        return F[N] % MOD

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        if self.reference_answer is None or self.MOD is None:
            # Environment not properly reset before step
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_initialized"}

        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        info: Dict[str, Any] = {}
        try:
            user_answer = int(answer_str)
        except ValueError:
            info.update({
                "error": "invalid_answer",
                "reference_answer": self.reference_answer
            })
            return TERMINAL_STATE, 0.0, True, False, info

        # Range check: must be 0 <= answer < MOD
        if not (0 <= user_answer < self.MOD):
            info.update({
                "error": "out_of_range",
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "N": self.N,
                "S": self.S,
                "MOD": self.MOD
            })
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info.update({
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "S": self.S,
            "MOD": self.MOD
        })

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
        """Sample a random action in the correct format."""
        if self.MOD is None:
            # Fallback if called before reset
            random_answer = 0
        else:
            random_answer = random.randint(0, self.MOD - 1)
        return f"\\boxed{{{random_answer}}}"