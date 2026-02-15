import random
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PrefixuffixEnv(Env):
    """Environment for the 'Prefix-Suffix Equivalence under Cyclic Shift' problem - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 1000,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
            N: Optional fixed length of the string S. If None, a random length in [min_N, max_N] is chosen per reset.
            min_N: Minimum allowed length for S (must be >= 4).
            max_N: Maximum allowed length for S.
        """
        super().__init__()
        assert min_N >= 4, "min_N should be greater than or equal to 4"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"
        if N is not None:
            assert N >= 4, "N should be greater than or equal to 4"

        self.N = N
        self.min_N = min_N
        self.max_N = max_N

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_N: Optional[int] = None
        self.current_S: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Determine the largest integer L (with 2 × L ≤ N) such that the L-prefix and the L-suffix of "
            "the given string are equivalent under a cyclic shift (moving a suffix to the front).\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Choose N
        N = self.N if self.N is not None else random.randint(self.min_N, self.max_N)
        assert N >= 4, "N should be greater than or equal to 4"
        self.current_N = N

        # Generate S ensuring there exists at least one valid L
        a_probability = random.random()

        def generate_string(length: int) -> str:
            return "".join("a" if random.random() < a_probability else "b" for _ in range(length))

        L = random.randint(1, N // 2)
        L1 = random.randint(0, L)
        L2 = L - L1
        S1, S2 = generate_string(L1), generate_string(L2)
        S = (S1 + S2) + generate_string(N - 2 * L) + (S2 + S1)
        self.current_S = S

        # Compute the reference answer using the original algorithm
        ans = self._compute_answer(S)
        assert L <= ans <= N // 2, "Computed answer is not within the expected range"
        self.reference_answer = ans

        # Build problem prompt
        self.current_problem = (
            'Define two strings S1 and S2 to be equivalent if one can be obtained from the other by moving a suffix '
            'to the front (i.e., performing a cyclic shift). For example, the strings "ababba" and "abbaab" are '
            'equivalent because "ababba" = "ab" + "abba" and "abbaab" = "abba" + "ab".\n\n'
            f'You are given a string S of length {N}: {S}\n'
            f'Please output the largest integer L such that 2 × L ≤ {N}, and the L-prefix (i.e., the first L characters of S) '
            f'and the L-suffix (i.e., the last L characters of S) are equivalent (see the definition above).\n\n'
            'Output Format: Your final answer should be a single integer in \\boxed{...}.'
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step: parse and verify the submitted answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check correctness
        assert self.reference_answer is not None, "Environment not properly initialized. Call reset() first."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_N,
            "S": self.current_S
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a boxed integer guess."""
        n = self.current_N if self.current_N is not None else (self.N if self.N is not None else self.max_N)
        guess = random.randint(0, max(0, n // 2))
        return f"\\boxed{{{guess}}}"

    def _compute_answer(self, S: str) -> int:
        """Compute the largest L such that the L-prefix and L-suffix are equivalent under cyclic shift."""
        N = len(S)

        # Build interleaved string t[1..N], with t[0] a sentinel
        t = ['#'] * (N + 1)
        # fill odd positions with S[0], S[1], ...
        j = 1
        for i in range(N):
            if j <= N:
                t[j] = S[i]
            j += 2
        # fill even positions with S[N-1], S[N-2], ...
        j = 2
        for i in range(N - 1, -1, -1):
            if j <= N:
                t[j] = S[i]
            j += 2

        # p[i]: radius of the even-length palindrome centered between t[i] and t[i+1]
        p = [0] * (N + 1)
        # vis[k] = 1 iff there is a palindrome of radius exactly i at center i such that it touches t[0]
        vis = [0] * (N + 2)

        mr = 0       # rightmost reach of any palindrome seen so far
        mid2 = 0     # twice the center index of that palindrome

        # Manacher's algorithm for even-length palindromes on t
        for i in range(1, N):
            # mirror optimization
            if mid2 - i - 1 > 0 and mr - i - 1 > 0:
                p[i] = min(p[mid2 - i - 1], mr - i - 1)
            else:
                p[i] = 0
            # expand around center between i and i+1
            while i - p[i] >= 0 and i + 1 + p[i] <= N and t[i - p[i]] == t[i + 1 + p[i]]:
                p[i] += 1
            # update rightmost palindrome
            if i + 1 + p[i] > mr:
                mr = i + 1 + p[i]
                mid2 = 2 * i + 1
            # if it reaches the sentinel at t[0], mark vis
            if i == p[i]:
                vis[i + p[i]] = 1

        # Union-find to compute, for each starting point j, the max center i covering it
        f = list(range(N + 2))
        res = [0] * (N + 2)

        def find(x: int) -> int:
            while f[x] != x:
                f[x] = f[f[x]]
                x = f[x]
            return x

        # Populate res[j] = max i such that [j..i] is inside some palindrome
        for i in range(N - 1, 0, -1):
            start = i - p[i] + 1
            j = find(start)
            while j <= i:
                res[j] = i
                f[j] = find(j + 1)
                j = f[j]

        # Compute answer as the largest L ≤ N//2 where prefix and suffix are cyclically equivalent
        ans = 0
        # Case 1: using two-part palindromes
        for i in range(1, N + 1):
            if vis[i] and res[i + 1] != 0:
                val = (2 * res[i + 1] + 1 - (i + 1)) // 2
                if val > ans:
                    ans = val
        # Case 2: trivial rotations within the first part
        for i in range(1, N + 1):
            if vis[i]:
                val = i // 2
                if val > ans:
                    ans = val

        return ans