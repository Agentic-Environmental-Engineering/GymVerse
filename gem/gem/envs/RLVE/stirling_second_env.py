from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class StirlingSecondEnv(Env):
    """Environment for counting surjective placements of distinct balls into distinct boxes.

    The task: given R distinct boxes and N distinct balls, count the number of ways to place all N balls
    into the boxes such that no box is empty. Two arrangements are different if at least one ball is placed
    into a different box. The answer should be given modulo `modulo`.
    """

    def __init__(
        self,
        max_n: int = 1000000,
        max_r: int = 1000000,
        modulo: int = 10**9 + 7,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
            max_n: maximum N to sample (N >= 2).
            max_r: maximum R to sample (R >= 2).
            modulo: modulus for the answer (default 1_000_000_007).
        """
        super().__init__()
        if max_n < 2:
            raise ValueError("max_n should be greater than or equal to 2")
        if max_r < 2:
            raise ValueError("max_r should be greater than or equal to 2")

        self.max_n = max_n
        self.max_r = max_r
        self.modulo = modulo

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.R: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem about surjective mappings.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem.

        Returns:
            observation: The instruction and problem statement.
            info: An empty info dict.
        """
        super().reset(seed)

        # Generate problem parameters
        self.N = random.randint(2, self.max_n)
        self.R = random.randint(2, min(self.N, self.max_r))

        # Build problem prompt
        self.current_problem = (
            f"There are {self.R} distinct boxes and {self.N} distinct balls. "
            f"Count the number of ways to place all {self.N} balls into the boxes such that no box is empty. "
            f"Two arrangements are different if at least one ball is placed into a different box. "
            f"Output the result modulo {self.modulo}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer using inclusion-exclusion:
        # count = sum_{k=0}^{R} (-1)^k * C(R, k) * (R - k)^N
        # The k=R term is zero for N >= 1, so we iterate k in [0, R-1].
        self.reference_answer = self._solve_surjective_count(self.R, self.N, self.modulo)

        observation = self._get_instructions() + self.current_problem
        return observation, {}

    def _solve_surjective_count(self, R: int, N: int, M: int) -> int:
        """Compute the number of surjective mappings from N balls to R boxes modulo M using inclusion-exclusion."""
        ans = 0
        c = 1  # C(R, 0)
        for k in range(R):
            term = c * pow(R - k, N, M) % M
            if k & 1:
                ans = (ans - term) % M
            else:
                ans = (ans + term) % M
            # Update binomial coefficient: C(R, k+1) = C(R, k) * (R - k) // (k + 1)
            c = c * (R - k) // (k + 1)
        return ans % M

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer.

        Returns:
            observation: TERMINAL_STATE (single-turn environment).
            reward: 1.0 if correct; 0.0 if wrong or invalid; -0.1 for format error.
            terminated: Always True (single-turn).
            truncated: Always False.
            info: Dict with details (correctness, reference_answer, user_answer, and errors if any).
        """
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check similar to original environment
        if not (0 <= user_answer < self.modulo):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
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
            "R": self.R,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action in \\boxed{...} format."""
        random_answer = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{random_answer}}}"