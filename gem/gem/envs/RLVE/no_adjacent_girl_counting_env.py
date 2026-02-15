from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NoAdjacentGirlCountingEnv(Env):
    """
    Environment for counting the number of ways to arrange N distinct boys, M distinct girls,
    and 2 distinct teachers in a line such that no two girls are adjacent and the two teachers
    are not adjacent. Single-turn Q&A environment.
    """

    prompt_template = (
        "Please count the number of ways to arrange {N} distinct boys, {M} distinct girls, "
        "and 2 distinct teachers in a line such that no two girls are adjacent and the two "
        "teachers are not adjacent."
    )

    def __init__(self, max_n_m: int = 10, **kwargs) -> None:
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        self.max_n_m: int = max_n_m

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a combinatorics counting problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate a valid problem instance
        while True:
            N = random.randint(1, self.max_n_m)
            M = random.randint(1, self.max_n_m)
            ans = self._compute_answer(N, M)
            if ans > 0:
                self.N = N
                self.M = M
                self.reference_answer = ans
                break

        # Build the problem statement
        problem = self.prompt_template.format(N=self.N, M=self.M)
        problem += "\n\nOutput Format: Your final answer should be a single integer in \\boxed{...}."

        self.current_problem = problem
        obs = self._get_instructions() + problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and provide a reward."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Compare with reference answer
        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_answer(self, N: int, M: int) -> int:
        """
        Compute the number of valid arrangements using the formula:
        Ans = [A(N+3, M) * A(N+2, N+2)] if N+3 >= M
              minus 2 * [A(N+2, M) * A(N+1, N+1)] if N+2 >= M
        where A(x, y) = x * (x-1) * ... * (x - y + 1) is the falling factorial.
        """
        def A(x: int, y: int) -> int:
            res = 1
            for i in range(y):
                res *= (x - i)
            return res

        ans = 0
        if N + 3 >= M:
            ans += A(N + 3, M) * A(N + 2, N + 2)
        if N + 2 >= M:
            ans -= 2 * A(N + 2, M) * A(N + 1, N + 1)
        return ans

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...}."""
        # If reference_answer known, sample around its scale; otherwise default small range
        if self.reference_answer is not None and self.reference_answer >= 0:
            upper = max(1, min(self.reference_answer * 2, self.reference_answer + 1000))
        else:
            upper = 1000
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"