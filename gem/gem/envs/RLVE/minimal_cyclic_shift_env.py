from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimalCyclicShiftEnv(Env):
    """Minimal Cyclic Shift problem environment - single-turn Q&A.

    Given a binary string S of length N, the task is to compute the lexicographically
    smallest string obtainable by performing any number of cyclic shifts (one shift
    moves the leftmost character to the rightmost position). The answer must be
    provided in \\boxed{...} format containing a binary string of length N.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 4,
        max_n: int = 1024,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
            N: Optional fixed length of the binary string. If None, N will be sampled in reset().
            min_n: Minimum allowed value of N. Must be >= 4.
            max_n: Maximum allowed value of N.
        """
        super().__init__()
        assert min_n >= 4, "N should be greater than or equal to 4"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"

        if N is not None:
            assert N >= 4, "N should be greater than or equal to 4"

        self.N_fixed: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.N: Optional[int] = None
        self.S: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving minimal cyclic shift problems.\n"
            "Please provide your answer in \\boxed{...} format.\n"
            "The boxed content must be a binary string of length N.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            self.N = self.N_fixed
        else:
            self.N = random.randint(self.min_n, self.max_n)
        assert self.N is not None and self.N >= 4, "N should be greater than or equal to 4"

        # Generate binary string S with a shared probability of '1'
        one_probability = random.random()
        self.S = "".join("1" if random.random() < one_probability else "0" for _ in range(self.N))

        # Compute reference answer using Booth's algorithm for minimal rotation
        self.reference_answer = self._minimal_cyclic_shift(self.S)

        # Build problem description
        self.current_problem = (
            f"Here is a binary string S of length {self.N}: {self.S}\n"
            "You may perform any number of cyclic shifts on S, where one shift moves the leftmost "
            "character to the rightmost position. Output the lexicographically smallest string obtainable "
            "after any number of shifts.\n\n"
            "Output Format: Provide the final binary string in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        # Parse boxed answer
        answer = self._parse_answer(action)
        if answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate format: must be binary and of correct length
        if self.N is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        if len(answer) != self.N or not all(c in "01" for c in answer):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check correctness
        is_correct = (answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": answer,
            "N": self.N,
            "S": self.S,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _minimal_cyclic_shift(self, s: str) -> str:
        """Compute the lexicographically smallest rotation of s using Booth's algorithm."""
        n = len(s)
        i, j, k = 0, 1, 0
        while i < n and j < n and k < n:
            c1 = s[(i + k) % n]
            c2 = s[(j + k) % n]
            if c1 == c2:
                k += 1
            else:
                if c1 > c2:
                    i += k + 1
                else:
                    j += k + 1
                if i == j:
                    i += 1
                k = 0
        start = min(i, j)
        return "".join(s[(start + t) % n] for t in range(n))

    def sample_random_action(self) -> str:
        """Sample a random action: a random binary string of length N in \\boxed{...} format."""
        if self.N is None:
            # Default to a short random length if reset has not been called
            length = 8
        else:
            length = self.N
        random_answer = "".join(random.choice("01") for _ in range(length))
        return f"\\boxed{{{random_answer}}}"