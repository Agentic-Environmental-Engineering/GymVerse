from typing import Any, Optional, SupportsFloat, Tuple, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CanonEnv(Env):
    """Single-turn environment for counting sequences of distinct non-empty subsets with even coverage."""

    def __init__(
        self,
        max_mod: int = 1000000,
        max_n_m: int = 100,
        **kwargs
    ):
        """
        Initialize the CanonEnv instance.

        Parameters:
        - max_mod: Upper bound for the modulo value (MOD) to be randomly sampled.
        - max_n_m: Upper bound for N and M values to be randomly sampled (must be >= 2).
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        self.max_mod = max_mod
        self.max_n_m = max_n_m

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem about sequences of distinct non-empty subsets.\n"
            "Please provide your final answer in \\boxed{...} format containing a single integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)
        self.MOD = random.randint(2, self.max_mod)

        # Build problem prompt
        self.current_problem = (
            f"Let S be the set of integers from 1 to {self.N} ({self.N} integers in total).\n\n"
            f"Please count the number of sequences T[1], ..., T[{self.M}] such that:\n"
            f"- Each T[i] is a non-empty subset of S.\n"
            f"- For each integer x in [1, {self.N}], the total number of subsets T[i] that contain x is an even number (including 0).\n"
            f"- T[1], ..., T[{self.M}] are distinct subsets.\n\n"
            f"Output Format: Output a single integer — the number of valid sequences T, modulo {self.MOD}.\n"
            f"Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer using the original algorithm
        tot = pow(2, self.N, self.MOD) - 1

        # Precompute A[i] = tot * (tot - 1) * ... * (tot - (i - 1)) mod MOD
        A = [0] * (self.M + 1)
        A[0] = 1
        for i in range(1, self.M + 1):
            A[i] = A[i - 1] * ((tot - (i - 1)) % self.MOD) % self.MOD

        # f[i] counts (up to multiplying by i!) the number of valid sequences of i distinct subsets
        f = [0] * (self.M + 1)
        f[0] = 1
        for i in range(2, self.M + 1):
            val = A[i - 1]
            val = (val - f[i - 1]) % self.MOD
            correction = f[i - 2] * (i - 1) % self.MOD * ((tot - (i - 2)) % self.MOD) % self.MOD
            val = (val - correction) % self.MOD
            f[i] = val

        self.reference_answer = f[self.M] % self.MOD

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute a single step to verify the submitted answer."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and score answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            # Not a valid integer inside the box
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": answer_text,
                "N": self.N,
                "M": self.M,
                "MOD": self.MOD,
                "error": "invalid_answer",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check range consistency: answer should be in [0, MOD-1]
        if not (0 <= user_answer < (self.MOD if self.MOD is not None else 1)):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "N": self.N,
                "M": self.M,
                "MOD": self.MOD,
                "error": "range_error",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "MOD": self.MOD,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the substring inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random boxed answer."""
        if self.MOD is None:
            # Default to a reasonable range if MOD has not been set yet.
            random_answer = random.randint(0, 9999)
        else:
            random_answer = random.randint(0, self.MOD - 1)
        return f"\\boxed{{{random_answer}}}"