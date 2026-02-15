import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ZeroPrefixSubsetCountingEnv(Env):
    """Environment for counting non-empty subsets where no string is a prefix of another string."""

    def __init__(
        self,
        fixed_n: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 30,
        alphabet: str = "ab",
        length_min: int = 2,
        length_max: Optional[int] = None,
        proportion_prefix_low: float = 0.1,
        proportion_prefix_high: float = 0.9,
        **kwargs,
    ):
        """
        Initialize the environment with configurable parameters.

        Parameters:
            fixed_n: If provided, the environment will always generate problems with exactly this N (must be >= 3).
            min_n: Minimum N when randomly sampled (inclusive).
            max_n: Maximum N when randomly sampled (inclusive).
            alphabet: Alphabet used to generate base strings.
            length_min: Minimum length of base strings.
            length_max: Maximum length of base strings; if None, it defaults to N at generation time.
            proportion_prefix_low: Lower bound for the proportion of strings that are prefixes.
            proportion_prefix_high: Upper bound for the proportion of strings that are prefixes.
        """
        super().__init__()
        if fixed_n is not None and fixed_n < 3:
            raise ValueError("fixed_n must be >= 3")
        if min_n < 3:
            raise ValueError("min_n must be >= 3")
        if max_n < min_n:
            raise ValueError("max_n must be >= min_n")
        if length_min < 1:
            raise ValueError("length_min must be >= 1")
        if proportion_prefix_low < 0.0 or proportion_prefix_high > 1.0 or proportion_prefix_low >= proportion_prefix_high:
            raise ValueError("Invalid prefix proportion bounds")

        self.fixed_n = fixed_n
        self.min_n = min_n
        self.max_n = max_n
        self.alphabet = alphabet
        self.length_min = length_min
        self.length_max = length_max
        self.proportion_prefix_low = proportion_prefix_low
        self.proportion_prefix_high = proportion_prefix_high

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_strings: List[str] = []
        self.current_n: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task description and answer format."""
        return (
            "You are given a list of strings.\n"
            "Your task is to count the number of non-empty subsets such that no string is a prefix of any other string within the subset.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
        if N < 3:
            raise ValueError("N must be >= 3")

        # Generate strings according to the original logic
        array: List[str] = []
        while True:
            proportion_being_prefix = random.uniform(self.proportion_prefix_low, self.proportion_prefix_high)
            M = N - int(N * proportion_being_prefix)
            if M < 1:
                continue

            array = []
            effective_length_max = self.length_max if self.length_max is not None else N
            effective_length_max = max(effective_length_max, self.length_min)

            # Generate M distinct base strings
            for _ in range(M):
                while True:
                    length = random.randint(self.length_min, effective_length_max)
                    s = "".join(random.choices(self.alphabet, k=length))
                    if s not in array:
                        array.append(s)
                        break

            # Generate prefixed strings from the base ones
            for _ in range(N - M):
                prefix = random.choice(array[:M])
                # Choose a proper prefix length (at least 1 and strictly less than len(prefix))
                pref_len = random.randint(1, len(prefix) - 1)
                array.append(prefix[:pref_len])

            assert len(array) == N
            # Ensure uniqueness across all strings
            if len(array) == len(set(array)):
                random.shuffle(array)
                break

        # Compute the reference answer using the original algorithm
        A = [""] + array.copy()
        A = [""] + sorted(A[1:])  # sort A[1..N]

        f = [[False] * (N + 1) for _ in range(N + 1)]
        dp = [0] * (N + 1)

        def calc(i: int, j: int) -> bool:
            # Ensure the shorter (or equal length) string is at i
            ii, jj = i, j
            if len(A[ii]) > len(A[jj]):
                ii, jj = jj, ii
            # Return True iff A[ii] is NOT a prefix of A[jj]
            return A[jj].find(A[ii]) != 0

        for i in range(1, N + 1):
            dp[i] = 1
            for j in range(1, N + 1):
                f[i][j] = calc(i, j)

        for i in range(1, N + 1):
            for j in range(i, N + 1):
                if f[i][j]:
                    dp[j] += dp[i]

        ret = sum(dp[1:])

        self.reference_answer = ret
        self.current_strings = array
        self.current_n = N

        # Build the problem prompt
        strings_block = "\n".join(f"String {idx}: {s}" for idx, s in enumerate(array, start=1))
        self.current_problem = (
            f"You are given {N} strings:\n"
            f"{strings_block}\n\n"
            f"How many non-empty subsets such that no string is a prefix of another string within the subset?\n\n"
            f"Output Format: Provide a single integer in \\boxed{{...}}."
        )

        observation = self._get_instructions() + self.current_problem
        return observation, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted answer and return the result."""
        # Parse the answer from \boxed{...}
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == user_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_n,
            "strings": self.current_strings,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by generating a plausible integer answer."""
        # Generate a random non-negative integer as a guess
        random_answer = random.randint(0, max(1, (self.current_n or self.max_n)))
        return f"\\boxed{{{random_answer}}}"