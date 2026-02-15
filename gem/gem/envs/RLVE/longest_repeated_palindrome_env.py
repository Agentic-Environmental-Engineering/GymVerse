from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Longest_RepeatedPalindromeEnv(Env):
    """
    Environment for finding the longest substring T in a binary string S such that:
    T = A + reverse(A) + A + reverse(A), maximizing the length of T.

    Single-turn QA:
    - reset(seed) returns the problem statement
    - step(action) validates the boxed answer and terminates
    """

    def __init__(self, N: int = 100, **kwargs):
        """
        Initialize the environment.

        Args:
            N: Total length of the binary string S (must be >= 4).
        """
        super().__init__()
        assert N >= 4, "N should be greater than or equal to 4"
        self.N: int = N

        # Current instance variables
        self.S: Optional[str] = None
        self.reference_answer_length: Optional[int] = None
        self._inserted_A: Optional[str] = None
        self._inserted_T: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task description and the required output format."""
        return (
            "You are given a binary string S. Your task is to find a substring T of S such that:\n"
            "- T = A + reverse(A) + A + reverse(A), where A is some non-empty string and reverse(A) denotes its reverse.\n"
            "- The goal is to maximize the length of T.\n\n"
            "Output Format:\n"
            "- Provide your final answer as the substring T inside \\boxed{...}.\n"
            "- Output only one \\boxed{...} containing T.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Random generation parameters
        N = self.N
        one_probability = random.uniform(0.1, 0.9)

        # Construct an embedded valid double palindrome substring
        A_length = random.randint(1, N // 4)
        A = "".join("1" if random.random() < one_probability else "0" for _ in range(A_length))
        A_reverse = A[::-1]
        T_inserted = A + A_reverse + A + A_reverse

        # Place T_inserted somewhere inside S with random prefix/suffix
        prefix_length = random.randint(0, N - 4 * A_length)
        prefix_part = "".join("1" if random.random() < one_probability else "0" for _ in range(prefix_length))
        suffix_length = N - prefix_length - 4 * A_length
        suffix_part = "".join("1" if random.random() < one_probability else "0" for _ in range(suffix_length))
        S = prefix_part + T_inserted + suffix_part
        assert len(S) == N, "Generated S should have length N"

        # Compute the reference answer (maximum length)
        gold_length = self._compute_longest_double_palindrome_length(S)
        assert gold_length >= 4 * A_length, "Reference answer length should be at least the inserted T length"

        # Store state
        self.S = S
        self._inserted_A = A
        self._inserted_T = T_inserted
        self.reference_answer_length = gold_length

        # Build prompt
        self.current_problem = (
            f"You are given a string S: {S}\n\n"
            "Please find a substring T of S such that:\n"
            "- T = A + reverse(A) + A + reverse(A), where + denotes concatenation.\n"
            "- Try your best to maximize the length of T.\n\n"
            "Output Format: Your final answer should be the substring T in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided answer and terminate."""
        # Parse boxed content
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate environment state
        if self.S is None or self.reference_answer_length is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        T = parsed

        # Validate substring and structure
        if T not in self.S:
            info = {"correct": False, "reason": "not_substring", "user_answer_length": len(T)}
            return TERMINAL_STATE, 0.0, True, False, info

        if len(T) == 0 or (len(T) % 4 != 0):
            info = {"correct": False, "reason": "invalid_length", "user_answer_length": len(T)}
            return TERMINAL_STATE, 0.0, True, False, info

        quarter = len(T) // 4
        A = T[:quarter]
        if T != A + A[::-1] + A + A[::-1]:
            info = {"correct": False, "reason": "invalid_structure", "user_answer_length": len(T)}
            return TERMINAL_STATE, 0.0, True, False, info

        user_length = len(T)
        gold_length = self.reference_answer_length
        is_correct = (user_length == gold_length)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer_length": gold_length,
            "user_answer_length": user_length,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_longest_double_palindrome_length(self, S: str) -> int:
        """
        Compute the maximum length of a substring T = A + reverse(A) + A + reverse(A)
        using a palindromic tree (Eertree) with auxiliary links for double palindrome detection.
        """
        n = len(S)
        size = n + 3
        ch = [[0] * 2 for _ in range(size)]  # transitions
        fail = [0] * size                    # failure links
        f = [0] * size                       # auxiliary links for double palindrome
        length = [0] * size                  # palindrome lengths

        tot = 1               # total nodes (we have nodes 0 and 1)
        fail[0] = 1           # fail of even root -> odd root
        length[1] = -1        # length of odd root
        las = 0               # last added node (start at even root)

        # Shift to 1-indexed
        S1 = ' ' + S

        for i in range(1, n + 1):
            cur = las
            while S1[i] != S1[i - length[cur] - 1]:
                cur = fail[cur]
            c = int(S1[i])

            if ch[cur][c] == 0:
                tot += 1
                length[tot] = length[cur] + 2

                x = fail[cur]
                while S1[i] != S1[i - length[x] - 1]:
                    x = fail[x]
                fail[tot] = ch[x][c]
                ch[cur][c] = tot

                if length[fail[tot]] <= length[tot] // 2:
                    f[tot] = fail[tot]
                else:
                    p = f[cur]
                    while (length[p] + 2 > length[tot] // 2) or (S1[i] != S1[i - length[p] - 1]):
                        p = fail[p]
                    f[tot] = ch[p][c]

            las = ch[cur][c]

        ans = 0
        for i in range(2, tot + 1):
            if length[i] % 4 == 0 and length[f[i]] == length[i] // 2:
                ans = max(ans, length[i])
        return ans

    def sample_random_action(self) -> str:
        """Sample a random action by boxing a random substring or the embedded valid T if available."""
        if self._inserted_T is not None and random.random() < 0.5:
            return f"\\boxed{{{self._inserted_T}}}"
        # Otherwise, sample a random binary substring possibly invalid
        length = random.randint(1, max(1, self.N // 4) * 4)
        # Ensure length is multiple of 4
        length = length - (length % 4)
        if length == 0:
            length = 4
        random_str = "".join(random.choice("01") for _ in range(length))
        return f"\\boxed{{{random_str}}}"