import random
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LDSTwo_CountingEnv(Env):
    """
    Single-turn environment for counting permutations A[1..N] of {1..N} that:
    - Are permutations (each integer appears exactly once),
    - Satisfy a fixed position constraint A[X] = Y,
    - Contain no decreasing subsequence of length 3.

    The agent must output the count as an integer within \\boxed{...}.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 2000,
        **kwargs
    ):
        super().__init__()
        # Validate parameters
        if n is not None:
            if n < 3:
                raise ValueError("n should be greater than or equal to 3")
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if min_n > max_n:
            raise ValueError("min_n should be less than or equal to max_n")

        self.fixed_n: Optional[int] = n
        self.min_n: int = min_n
        self.max_n: int = max_n

        # State variables
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.X: Optional[int] = None
        self.Y: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics counting problem.\n"
            "Task: Count permutations A[1..N] of {1..N} with A[X] = Y that contain no decreasing subsequence of length 3.\n"
            "Please provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample parameters
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
        X = random.randint(1, N)
        Y = random.randint(1, N)

        # Build problem prompt
        prompt = (
            f"Consider a permutation A[1], A[2], ..., A[{N}] of the integers 1 through {N} that satisfies the following conditions:\n"
            f"- A is a permutation, meaning each integer from 1 to {N} appears exactly once.\n"
            f"- The value at position {X} is fixed: A[{X}] = {Y}.\n"
            f"- The permutation must not contain any decreasing subsequence of length 3. That is, there must not exist indices 1 <= a < b < c <= {N} such that A[a] > A[b] > A[c].\n\n"
            f"Please count the number of such permutations.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer using the original algorithmic logic
        def C(n: int, m: int) -> int:
            if n < m or m < 0:
                return 0
            result = 1
            for i in range(m):
                result = result * (n - i) // (i + 1)
            return result

        def go(sx: int, sy: int, tx: int, ty: int) -> int:
            return C((tx - sx) + (ty - sy), tx - sx)

        def solve(sx: int, sy: int, tx: int, ty: int) -> int:
            return go(sx, sy, tx, ty) - go(sx, sy, ty + 1, tx - 1)

        # The original implementation swaps X and Y for computation if Y < X
        comp_X, comp_Y = X, Y
        if comp_Y < comp_X:
            comp_X, comp_Y = comp_Y, comp_X

        reference_answer = solve(0, 0, comp_X - 1, comp_Y - 1) * solve(comp_X, comp_Y, N, N)

        # Save state
        self.N = N
        self.X = X
        self.Y = Y
        self.current_problem = prompt
        self.reference_answer = reference_answer

        obs = self._get_instructions() + prompt
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the agent's answer and return reward."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and score
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        correct = (self.reference_answer == user_answer)
        reward: float = 1.0 if correct else 0.0

        info: dict[str, Any] = {
            "correct": correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "X": self.X,
            "Y": self.Y,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        # If we have a reference answer, sample near it; otherwise, sample a small random integer
        if self.reference_answer is not None:
            low = max(0, self.reference_answer - 10)
            high = self.reference_answer + 10
            guess = random.randint(low, high)
        else:
            guess = random.randint(0, 100)
        return f"\\boxed{{{guess}}}"