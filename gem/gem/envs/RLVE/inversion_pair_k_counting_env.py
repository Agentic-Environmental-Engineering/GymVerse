from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class InversionPairK_CountingEnv(Env):
    """Environment for counting permutations with exactly K inversion pairs - single-turn Q&A."""

    def __init__(
        self,
        N: int,
        max_MOD: int = 1000000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: The size of the permutation (must be >= 1).
            max_MOD: The maximum modulo value (inclusive upper bound for random selection).
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 1, "N should be greater than or equal to 1"
        assert isinstance(max_MOD, int) and max_MOD >= 2, "max_MOD must be an integer >= 2"

        self.N = N
        self.max_MOD = max_MOD

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_K: Optional[int] = None
        self.current_MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem about inversion pairs in permutations.\n"
            "Please provide your final answer as a single integer enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        N = self.N
        K = random.randint(0, N * (N - 1) // 2)
        MOD = random.randint(2, self.max_MOD)

        # Build problem statement
        self.current_problem = (
            f"Consider all permutations of the numbers 1 through {N}. "
            f"Your task is to count how many of them have exactly {K} inversion pairs. "
            f"Since the number may be large, output the result modulo {MOD}.\n\n"
            "Definitions:\n"
            f"- A permutation of 1 to {N} is an arrangement of the numbers 1 through {N}, "
            "where each number appears exactly once.\n"
            "- An inversion pair in a permutation a_1, a_2, ..., a_N is a pair of indices (i, j) such that i < j and a_i > a_j.\n\n"
            "Output Format:\n"
            "Your final answer should be a single integer enclosed in \\boxed{...}.\n"
            "Example: \\boxed{9999}\n"
        )

        # Compute reference answer
        self.reference_answer = self._count_exact_inversion_permutations(N, K, MOD)
        self.current_K = K
        self.current_MOD = MOD

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        # Parse the boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return (
                TERMINAL_STATE,
                -0.1,
                True,
                False,
                {"error": "format_error"}
            )

        # Validate numeric answer
        try:
            user_answer = int(answer_text.strip())
        except ValueError:
            return (
                TERMINAL_STATE,
                0.0,
                True,
                False,
                {"error": "invalid_answer"}
            )

        # Ensure range: 0 <= answer < MOD
        if self.current_MOD is None or self.reference_answer is None:
            return (
                TERMINAL_STATE,
                0.0,
                True,
                False,
                {"error": "environment_not_initialized"}
            )

        if not (0 <= user_answer < self.current_MOD):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "N": self.N,
                "K": self.current_K,
                "MOD": self.current_MOD,
                "error": "out_of_range",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.current_K,
            "MOD": self.current_MOD,
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

    def _count_exact_inversion_permutations(self, N: int, K: int, MOD: int) -> int:
        """Compute the number of permutations of [1..N] with exactly K inversions modulo MOD."""
        dpF = [0] * (K + 1)
        dpF[0] = 1
        for i in range(1, N + 1):
            prefix_sum = [0] * (K + 1)
            prefix_sum[0] = dpF[0]
            for k in range(1, K + 1):
                prefix_sum[k] = prefix_sum[k - 1] + dpF[k]

            def get_sum(l: int, r: int) -> int:
                l = max(l, 0)
                if r < 0:
                    return 0
                return prefix_sum[r] - (prefix_sum[l - 1] if l > 0 else 0)

            limit = min(K, i * (i - 1) // 2)
            for k in range(limit + 1):
                dpF[k] = get_sum(k - (i - 1), k) % MOD

        return dpF[K] % MOD

    def sample_random_action(self) -> str:
        """Sample a random action by proposing a random boxed integer."""
        mod = self.current_MOD if self.current_MOD is not None else max(2, self.max_MOD)
        random_answer = random.randint(0, mod - 1)
        return f"\\boxed{{{random_answer}}}"