import math
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RootExtractionEnv(Env):
    """Root extraction environment - single-turn Q&A.

    Task: Compute the K-th root of N (i.e., N^(1/K)) and provide the result
    as a decimal number accurate up to 5 decimal places. The answer must be
    submitted in \\boxed{...} format.

    Correctness criterion: The answer is considered correct if, when rounded
    to 5 decimal places, it equals the reference value rounded to 5 decimals.
    """

    def __init__(
        self,
        max_n: int = 1_000_000,
        max_k: int = 10,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(max_n, int) and max_n >= 1, "max_n should be an integer >= 1"
        assert isinstance(max_k, int) and max_k >= 1, "max_k should be an integer >= 1"

        self.max_n = max_n
        self.max_k = max_k

        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.reference_answer: Optional[float] = None
        self.decimal_places: int = 5

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "Your task is to compute the K-th root of N, that is, find the value of N^(1/K).\n"
            f"Please output the value in decimal form, as accurate as possible, up to {self.decimal_places} decimal places.\n"
            "If the result has fewer than the maximum decimal digits, you may omit trailing zeros.\n\n"
            "Output Format: Your final answer must be a single decimal number in \\boxed{...} format.\n"
            "Example: \\boxed{2.24573}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate parameters
        self.N = random.randint(1, self.max_n)
        self.K = random.randint(1, self.max_k)

        # Compute the reference answer rounded to specified decimal places
        self.reference_answer = round(self.N ** (1.0 / self.K), self.decimal_places)

        # Build problem description
        self.current_problem = (
            f"Compute the {self.K}-th root of {self.N}, i.e., {self.N}^(1/{self.K}).\n"
            f"Provide your answer as a decimal number up to {self.decimal_places} decimal places in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": self.N,
            "K": self.K,
            "decimal_places": self.decimal_places,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the outcome."""
        raw_answer = self._parse_answer(action)
        if raw_answer is None:
            # Format error (no or invalid \\boxed{...})
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to parse the numeric value
        try:
            user_value = float(raw_answer.strip())
            if not math.isfinite(user_value):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check correctness: compare after rounding to required decimal places
        rounded_user = round(user_value, self.decimal_places)
        is_correct = (rounded_user == self.reference_answer)

        reward: float = 1.0 if is_correct else 0.0
        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_value,
            "rounded_user_answer": rounded_user,
            "N": self.N,
            "K": self.K,
            "decimal_places": self.decimal_places,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random decimal with up to 5 decimal places)."""
        # Generate a random float with up to 5 decimal places
        value = random.randint(0, 10_000) / 100.0
        return f"\\boxed{{{value}}}"