import random
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Kth_SemiBalancedBracketSequenceEnv(Env):
    """Environment for finding the K-th semi-balanced bracket sequence of a given odd length."""

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 101,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - min_n: Minimum length of the bracket sequence (must be >= 3).
        - max_n: Maximum length of the bracket sequence.
                 The environment will choose an odd N uniformly at random within [min_n, max_n].
        """
        super().__init__()
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        self.min_n = min_n
        self.max_n = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a bracket sequence enumeration problem.\n"
            "Provide your final answer exactly in \\boxed{...} format, containing only '(' and ')'.\n"
            "Do not include any additional text.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Choose an odd N within [min_n, max_n]
        def _random_odd_in_range(a: int, b: int) -> int:
            start = a if a % 2 == 1 else a + 1
            if start > b:
                raise ValueError("No odd number available in the given range")
            count = ((b - start) // 2) + 1
            return start + 2 * random.randrange(count)

        N = _random_odd_in_range(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"
        assert N % 2 == 1, "N should be odd"

        # Precompute counts for balanced bracket sequences (cbs DP)
        cbs = [[0] * (N + 2) for _ in range(N + 2)]
        cbs[0][0] = 1
        for i in range(1, N + 1):
            cbs[i][0] = cbs[i - 1][1]
            for j in range(1, N + 1):
                cbs[i][j] = cbs[i - 1][j - 1] + cbs[i - 1][j + 1]

        total = 0
        for i in range(0, N + 1, 2):
            total += 2 * cbs[i][0] * cbs[N - 1 - i][0]

        K = random.randint(1, total)

        # Construct the K-th semi-balanced bracket sequence using greedy algorithm
        K_internal = K - 1
        s = ["("] * N
        b = [0] * (N + 2)
        good = [[False] * (N + 2) for _ in range(N + 2)]
        for i in range(1, N + 2):
            good[i][i - 1] = True

        for i in range(1, N + 1):
            b[i] = b[i - 1] + 1
            for j in range(1, i + 1):
                good[j][i] = good[j][i - 1] and (b[i] - b[j - 1] >= 0)

            cur = 0
            for j in range(1, i + 1):
                if good[1][j - 1] and b[j - 1] == 0 and good[j + 1][i]:
                    cur += cbs[N - i][b[i] - b[j]]
            if good[1][i]:
                for j in range(i + 1, N + 1):
                    cur += 2 * cbs[j - i - 1][b[i]] * cbs[N - j][0]

            if cur <= K_internal:
                K_internal -= cur
                s[i - 1] = ")"
                b[i] = b[i - 1] - 1
                for j in range(1, i + 1):
                    good[j][i] = good[j][i - 1] and (b[i] - b[j - 1] >= 0)

        assert len(s) == N and all([c in "()" for c in s]), "The generated sequence is not valid"

        self.N = N
        self.K = K
        self.reference_answer = "".join(s)

        # Build problem description
        self.current_problem = (
            "Consider strings that only contain the characters '(' and ')':\n"
            "- A string is called a balanced bracket sequence if, after inserting digits and operators,\n"
            "  it can form a valid arithmetic expression. For example, '(())' is balanced, while ')(()' is not.\n"
            "- A string is called a semi-balanced bracket sequence if removing exactly one bracket from it\n"
            "  can result in a balanced bracket sequence.\n\n"
            "We define the lexicographical order such that '(' comes before ')'.\n"
            f"Please find the {self.K}-th semi-balanced bracket sequence of length {self.N},\n"
            "when all such sequences are sorted in lexicographical order.\n\n"
            "Output Format: Your final answer should be a single line containing the semi-balanced bracket sequence in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {"N": self.N, "K": self.K}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the user's answer."""
        if self.reference_answer is None or self.N is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Parse answer from \\boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate the answer format: length and characters
        user_answer = parsed.strip()
        if len(user_answer) != self.N or any(c not in "()"
                                             for c in user_answer):
            reward = 0.0
            info = {
                "error": "invalid_answer",
                "expected_length": self.N,
                "allowed_characters": "()",
                "user_answer": user_answer,
                "reference_answer": self.reference_answer,
                "correct": False,
            }
            return TERMINAL_STATE, reward, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
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
        """Sample a random bracket sequence of the correct length as a boxed answer."""
        if self.N is None:
            # Default to a small odd number if not initialized
            n = 3
        else:
            n = self.N
        seq = "".join(random.choice("()") for _ in range(n))
        return f"\\boxed{{{seq}}}"