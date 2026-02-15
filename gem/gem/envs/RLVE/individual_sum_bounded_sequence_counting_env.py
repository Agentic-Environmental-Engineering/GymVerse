import math
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class IndividualSumBounded_SequenceCountingEnv(Env):
    """Single-turn environment for counting sequences with individual bounds and sum constraint.

    Problem:
    Count the number of sequences X[1], ..., X[K] such that:
    - X[1] >= 1
    - For all i in [2, K]: 1 <= X[i] <= M
    - The total sum X[1] + X[2] + ... + X[K] <= N

    Output the count modulo MOD.

    Answer format: provide a single integer in \\boxed{...}.
    """

    def __init__(
        self,
        max_n: int,
        max_mod: int = 1_000_000_000,
        **kwargs: Any,
    ):
        """Initialize the environment.

        Args:
            max_n: Maximum value for N (must be >= 2).
            max_mod: Upper bound for the random modulus MOD (inclusive upper bound is handled in sampling).
            **kwargs: Placeholder for compatibility with extended configurations.
        """
        super().__init__()
        if max_n < 2:
            raise ValueError("max_n should be greater than or equal to 2")
        self.max_n = max_n
        self.max_mod = max_mod

        # Problem instance variables
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.M: Optional[int] = None
        self.MOD: Optional[int] = None

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a combinatorics counting problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Args:
            seed: Optional seed for deterministic behavior.

        Returns:
            A tuple containing the observation string and an info dict.
        """
        super().reset(seed)

        # Generate parameters following the original logic and validations
        N = random.randint(2, self.max_n)
        # Ensure K in [2, N] distributed by 2 ** U(1, log2 N)
        K = int(2 ** random.uniform(1.0, math.log2(N)))
        if K < 2:
            K = 2
        if K > N:
            K = N
        # M such that 1 + M * (K - 1) <= N
        upper_M = max(1, (N - 1) // (K - 1))
        M = random.randint(1, upper_M)
        assert K >= 2, "K should be at least 2"
        assert 1 + M * (K - 1) <= N, "N should be at least 1 + M * (K - 1)"
        MOD = random.randint(2, self.max_mod)

        # Store parameters
        self.N = N
        self.K = K
        self.M = M
        self.MOD = MOD

        # Compute reference answer with original formula
        pow1 = pow(M, K - 1, MOD)
        pow2 = pow(M, K - 2, MOD)
        term1 = (N % MOD) * pow1 % MOD
        x = (M * (M + 1) // 2) % MOD
        term2 = ((K - 1) % MOD) * x % MOD * pow2 % MOD
        ans = (term1 - term2) % MOD
        self.reference_answer = ans

        # Build problem statement
        problem = (
            f"Count the number of sequences X[1], ..., X[{K}] such that:\n"
            f"- X[1] >= 1\n"
            f"- For all i in [2, {K}]: 1 <= X[i] <= {M}\n"
            f"- The total sum X[1] + X[2] + ... + X[{K}] <= {N}\n\n"
            f"Output the count modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the user's answer.

        Args:
            action: The model's response containing the answer in \\boxed{...}.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_str.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Ensure we have a reference to compare
        assert self.reference_answer is not None, "Environment not properly initialized. Call reset() first."
        assert self.MOD is not None, "Environment not properly initialized. Call reset() first."

        # Optional range check as per original scorer logic
        info: dict[str, Any] = {
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "parameters": {
                "N": self.N,
                "K": self.K,
                "M": self.M,
                "MOD": self.MOD,
            },
        }

        if not (0 <= user_answer < self.MOD):
            info.update({"correct": False, "error": "out_of_range"})
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0
        info["correct"] = is_correct

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence in the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...}."""
        modulo = self.MOD if self.MOD is not None else max(2, self.max_mod)
        random_answer = random.randint(0, modulo - 1)
        return f"\\boxed{{{random_answer}}}"