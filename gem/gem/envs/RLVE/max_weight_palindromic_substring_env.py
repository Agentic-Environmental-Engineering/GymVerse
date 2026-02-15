import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxWeightPalindromicSubstringEnv(Env):
    """Environment for the Max-Weight Palindromic Substring problem (single-turn Q&A)."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 100,
        a_probability_min: float = 0.3,
        a_probability_max: float = 0.7,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
            N: If provided, use this fixed length for the string S. Must be >= 3.
            min_N: Minimum length for random generation of S. Must be >= 3.
            max_N: Maximum length for random generation of S.
            a_probability_min: Minimum probability for character 'a' in S.
            a_probability_max: Maximum probability for character 'a' in S.
        """
        super().__init__()
        assert min_N >= 3, "N should be greater than or equal to 3"
        assert 0.0 <= a_probability_min <= 1.0, "a_probability_min must be in [0, 1]"
        assert 0.0 <= a_probability_max <= 1.0, "a_probability_max must be in [0, 1]"
        assert a_probability_min <= a_probability_max, "a_probability_min must be <= a_probability_max"

        if N is not None:
            assert N >= 3, "N should be greater than or equal to 3"

        self.N = N
        self.min_N = min_N
        self.max_N = max_N
        self.a_probability_min = a_probability_min
        self.a_probability_max = a_probability_max

        self.current_problem: Optional[str] = None
        self.string_S: Optional[str] = None
        self.reference_weight: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a string problem about palindromic substrings.\n"
            "Given a binary string S consisting of characters 'a' and 'b', "
            "find a palindromic string T that maximizes length(T) × occurrences of T in S (counted with overlaps).\n"
            "Output Format: Provide your palindromic string T in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem, and return the observation and info."""
        super().reset(seed)

        N = self.N if self.N is not None else random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"

        a_probability = random.uniform(self.a_probability_min, self.a_probability_max)
        S = "".join("a" if random.random() < a_probability else "b" for _ in range(N))
        self.string_S = S

        self.reference_weight = self._max_palindrome_existence_value(S)

        self.current_problem = (
            f"You are given a string S: {S}\n"
            f"Please find a palindromic string T such that the product of T's length and the number of times T occurs in S is maximized.\n"
            f"Output Format: Output a single line containing the string T in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer."""
        answer_text = self._parse_answer(action)

        if answer_text is None:
            # Format error: cannot parse boxed answer
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        t = answer_text.strip()

        # Validate palindrome
        is_palindrome = (t == t[::-1])

        # Compute user's weight (length * overlapping occurrences in S)
        if not is_palindrome or self.string_S is None:
            user_weight = 0
        else:
            user_weight = len(t) * self._count_overlapping_occurrences_kmp(self.string_S, t)

        # Compare with reference
        is_correct = (self.reference_weight is not None and user_weight == self.reference_weight)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_weight": self.reference_weight,
            "user_weight": user_weight,
            "user_string": t,
            "string_S": self.string_S,
            "is_palindrome": is_palindrome,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer string from \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
        return None

    def sample_random_action(self) -> str:
        """Sample a random palindromic action."""
        # Simple random palindromic guess
        candidate = random.choice(["a", "b", "aa", "bb"])
        return f"\\boxed{{{candidate}}}"

    def _count_overlapping_occurrences_kmp(self, text: str, pattern: str) -> int:
        """Count overlapping occurrences of pattern in text using KMP."""
        if not pattern or not text:
            return 0

        def build_failure_function(p: str) -> list[int]:
            m = len(p)
            failure = [0] * m
            j = 0
            for i in range(1, m):
                while j > 0 and p[i] != p[j]:
                    j = failure[j - 1]
                if p[i] == p[j]:
                    j += 1
                failure[i] = j
            return failure

        failure = build_failure_function(pattern)
        count = 0
        j = 0
        for i in range(len(text)):
            while j > 0 and text[i] != pattern[j]:
                j = failure[j - 1]
            if text[i] == pattern[j]:
                j += 1
            if j == len(pattern):
                count += 1
                j = failure[j - 1]
        return count

    def _max_palindrome_existence_value(self, S: str) -> int:
        """
        Build a palindromic tree (Eertree) for S and compute the maximum
        existence value among all palindromic substrings: length * frequency.
        """
        N = len(S)
        # Arrays sized for at most N distinct palindromes plus 2 roots
        size = 1  # last-used node index
        length = [0] * (N + 3)  # length of palindrome at each node
        fail = [0] * (N + 3)    # failure link for each node
        count = [0] * (N + 3)   # occurrence counts
        trans = [dict() for _ in range(N + 3)]  # transitions per node

        # Two roots:
        # node 1: imaginary palindrome of length -1
        # node 0: empty palindrome of length 0
        length[1] = -1
        fail[0] = 1
        fail[1] = 1

        last = 0  # node for the longest palindromic suffix of processed prefix

        for i, c in enumerate(S):
            cur = last
            # Find the largest suffix-palindrome that can be extended by c
            while True:
                if i - length[cur] - 1 >= 0 and S[i - length[cur] - 1] == c:
                    break
                cur = fail[cur]

            # Create new node if needed
            if c not in trans[cur]:
                size += 1
                length[size] = length[cur] + 2

                # Compute failure link for the new node
                f = fail[cur]
                while True:
                    if i - length[f] - 1 >= 0 and S[i - length[f] - 1] == c:
                        break
                    f = fail[f]
                fail[size] = trans[f].get(c, 0)

                # Link cur --c--> size
                trans[cur][c] = size

            # Move to the extended node and count occurrence
            last = trans[cur][c]
            count[last] += 1

        # Propagate counts from longer palindromes to their suffix-palindromes
        ans = 0
        for u in range(size, 1, -1):
            ans = max(ans, length[u] * count[u])
            count[fail[u]] += count[u]

        assert ans > 0
        return ans