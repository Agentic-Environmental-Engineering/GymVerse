from typing import Any, Optional, SupportsFloat, Tuple
import random
import functools
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ThreeStringCommonSubsequenceCountingEnv(Env):
    """Environment for counting the number of non-empty common subsequences among three binary strings."""

    def __init__(
        self,
        max_n: int = 50,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: Maximum length for each of the three generated strings. Must be >= 3.
        """
        super().__init__()
        if max_n < 3:
            raise ValueError("max_n should be greater than or equal to 3")
        self.max_n = max_n
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.strings: Optional[tuple[str, str, str]] = None

    def _get_instructions(self):  # type: ignore[no-untyped-def]
        """Return general task instructions."""
        return (
            "Task: Count non-empty common subsequences.\n"
            "A string T is a subsequence of S if T can be obtained from S by deleting zero or more characters "
            "without changing the order of the remaining characters.\n"
            "You will be given three strings A, B, and C over the alphabet {a, b}.\n"
            "Your task is to compute the number of non-empty strings that are subsequences of A, B, and C simultaneously.\n"
            "Output Format: Provide a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate three binary strings with a shared bias towards 'a'
        a_probability = random.random()
        strings = []
        for _ in range(3):
            length = random.randint(3, self.max_n)
            s = "".join("a" if random.random() < a_probability else "b" for _ in range(length))
            strings.append(s)

        A, B, C = strings[0], strings[1], strings[2]
        self.strings = (A, B, C)

        # Compute the reference answer using next-occurrence DP + DFS with memoization
        n, m, k = len(A), len(B), len(C)

        # 1-based padding for easy next occurrence lookup
        A1 = "#" + A
        B1 = "#" + B
        C1 = "#" + C

        # Build next-occurrence tables for alphabet {'a','b'}, indexed as 0:'a', 1:'b'
        nextA = [[0] * 2 for _ in range(n + 1)]
        nextB = [[0] * 2 for _ in range(m + 1)]
        nextC = [[0] * 2 for _ in range(k + 1)]

        for u in range(n - 1, -1, -1):
            nextA[u] = nextA[u + 1].copy()
            nextA[u][ord(A1[u + 1]) - ord('a')] = u + 1

        for v in range(m - 1, -1, -1):
            nextB[v] = nextB[v + 1].copy()
            nextB[v][ord(B1[v + 1]) - ord('a')] = v + 1

        for w in range(k - 1, -1, -1):
            nextC[w] = nextC[w + 1].copy()
            nextC[w][ord(C1[w + 1]) - ord('a')] = w + 1

        @functools.lru_cache(maxsize=None)
        def dfs(u: int, v: int, w: int) -> int:
            # Count all distinct common subsequences starting from positions (u,v,w),
            # including the empty subsequence (to be subtracted at the end).
            total = 1
            for ch in range(2):
                nu = nextA[u][ch]
                nv = nextB[v][ch]
                nw = nextC[w][ch]
                if nu and nv and nw:
                    total += dfs(nu, nv, nw)
            return total

        self.reference_answer = dfs(0, 0, 0) - 1

        # Build the problem statement
        self.current_problem = (
            f"There are three strings A, B, and C:\n"
            f"A: {A}\n"
            f"B: {B}\n"
            f"C: {C}\n\n"
            f"What is the number of non-empty strings that are subsequences of A, B, and C simultaneously?\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the outcome."""
        # Extract answer from \boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer and compare
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "strings": self.strings
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...}."""
        # A crude upper bound: number of non-empty binary subsequences <= 2^min(n,m,k) - 1
        # Use a capped range to avoid extremely large integers for random sampling.
        if self.strings is not None:
            n = min(len(self.strings[0]), len(self.strings[1]), len(self.strings[2]))
            try:
                upper = min((1 << n) - 1, 10**6)
            except OverflowError:
                upper = 10**6
            upper = max(1, upper)
        else:
            upper = 10**6
        guess = random.randint(0, upper)
        return f"\\boxed{{{guess}}}"