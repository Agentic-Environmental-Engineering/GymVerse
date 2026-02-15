from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SameAdjacencyCountingEnv(Env):
    """Environment for counting sequences with at least one pair of adjacent equal elements (single-turn Q&A)."""

    def __init__(
        self,
        max_n: int = 1_000_000,
        max_m: int = 1_000_000,
        **kwargs
    ):
        """
        Initialize the SameAdjacencyCountingEnv instance.

        Args:
            max_n: Maximum value for N (sequence length). Must be >= 2.
            max_m: Maximum value for M (alphabet size). Must be >= 2.
        """
        super().__init__()
        assert max_n >= 2, "max_n should be greater than or equal to 2"
        assert max_m >= 2, "max_m should be greater than or equal to 2"
        self.max_n = max_n
        self.max_m = max_m

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a combinatorics counting problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n"
            "Your answer must be an integer in the range [0, MOD - 1].\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        self.N = random.randint(2, self.max_n)
        self.M = random.randint(2, self.max_m)
        self.MOD = random.randint(self.M, 2 * self.M)

        # Compute reference answer:
        # Count of sequences of length N over [1..M] with at least one adjacent equal pair.
        # Total sequences: M^N
        # Sequences with all adjacent distinct: M * (M - 1)^(N - 1)
        # Desired count = total - distinct_adjacent
        total_mod = pow(self.M, self.N, self.MOD)
        distinct_adjacent_mod = (self.M % self.MOD) * pow(self.M - 1, self.N - 1, self.MOD) % self.MOD
        self.reference_answer = (total_mod - distinct_adjacent_mod + self.MOD) % self.MOD

        # Build problem prompt
        self.current_problem = (
            f"Count the number of length-{self.N} sequences using integers from 1 to {self.M} "
            f"such that at least one pair of adjacent elements is equal. "
            f"Output the result modulo {self.MOD}.\n\n"
            f"Output Format: Provide a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None and self.MOD is not None

        # Range check: answer should be in [0, MOD)
        if not (0 <= user_answer < self.MOD):
            info = {
                "error": "out_of_range",
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "modulo": self.MOD,
                "N": self.N,
                "M": self.M
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Correctness check
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "modulo": self.MOD,
            "N": self.N,
            "M": self.M
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
        """Sample a random action formatted as \\boxed{...}."""
        if self.MOD is None:
            # If called before reset, default to 0
            return "\\boxed{0}"
        random_answer = random.randint(0, self.MOD - 1)
        return f"\\boxed{{{random_answer}}}"