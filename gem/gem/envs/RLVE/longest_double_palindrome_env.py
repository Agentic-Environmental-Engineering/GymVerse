import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Longest_DoublePalindromeEnv(Env):
    """Environment for finding two adjacent palindromic substrings with maximum combined length."""

    def __init__(
        self,
        N: int = 20,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: Length of the binary string S to generate. Must be >= 3.
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N

        # Internal state placeholders
        self.S: Optional[str] = None
        self.gold_answer: Optional[int] = None  # The optimal C - A
        self.current_problem: Optional[str] = None

        # Fixed rewards as per conversion requirements
        self.reward_correct: float = 1.0
        self.reward_incorrect: float = 0.0
        self.reward_format_error: float = -0.1

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given a binary string S. Your task is to find two non-empty adjacent substrings that are both palindromes.\n"
            "You must provide your final answer as three integers A, B, and C in \\boxed{A B C} format, where 0 <= A < B < C <= N.\n"
            "The goal is to maximize the value C - A.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        # Randomly choose the proportion of '1' characters
        one_probability = random.uniform(0.1, 0.9)

        # Choose three endpoints to guarantee there exists a solution with two adjacent palindromes
        endpoints = random.sample(range(N + 1), 3)
        endpoints.sort()

        def generate_random(length: int) -> str:
            assert length >= 0, "length should be non-negative"
            return "".join("1" if random.random() < one_probability else "0" for _ in range(length))

        def generate_palindrome(length: int) -> str:
            assert length >= 1, "length should be at least 1"
            half = length // 2
            first_half = "".join("1" if random.random() < one_probability else "0" for _ in range(half))
            if length % 2 == 0:
                return first_half + first_half[::-1]
            else:
                middle = "1" if random.random() < one_probability else "0"
                return first_half + middle + first_half[::-1]

        # Construct S to include two adjacent palindromic substrings
        S = (
            generate_random(endpoints[0]) +
            generate_palindrome(max(1, endpoints[1] - endpoints[0])) +
            generate_palindrome(max(1, endpoints[2] - endpoints[1])) +
            generate_random(max(0, N - endpoints[2]))
        )
        S = S[:N]  # Ensure length is exactly N (in case of clamping)
        assert len(S) == N, "S should have length N"

        # Compute the optimal answer (maximum C - A such that S[A:B] and S[B:C] are palindromes)
        gold_answer = self._compute_gold_answer(S)

        # Sanity check that the constructed palindromic intervals offer at least some length
        assert gold_answer >= max(0, endpoints[2] - endpoints[0]), "Gold answer should be at least the palindromic block length"

        self.S = S
        self.gold_answer = gold_answer

        self.current_problem = (
            f"You are given a string S of length {N} (0-indexed): {S}\n\n"
            "Please find two non-empty intervals [A, B) and [B, C) (obviously, 0 <= A < B < C <= N) such that:\n"
            "- S[A : B] and S[B : C] are both palindromes (S[a : b] refers to the substring starting at index a and ending at index b - 1, i.e., S[a] + S[a+1] + ... + S[b-1]).\n"
            "- Try your best to maximize C - A.\n\n"
            "Output Format: Your final answer should be three integers A, B, and C, separated by spaces, in \\boxed{A B C}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the submitted answer and return the result."""
        # Parse the boxed answer content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, self.reward_format_error, True, False, {"error": "format_error"}

        # Try to parse three integers A, B, C
        try:
            parts = boxed_content.strip().split()
            if len(parts) != 3:
                return TERMINAL_STATE, self.reward_format_error, True, False, {"error": "format_error"}
            A, B, C = map(int, parts)
        except Exception:
            return TERMINAL_STATE, self.reward_format_error, True, False, {"error": "format_error"}

        # Validate indices
        N = self.N
        if not (0 <= A < B < C <= N):
            info = {
                "error": "invalid_solution",
                "reason": "indices_out_of_range_or_order",
                "A": A, "B": B, "C": C,
                "N": N
            }
            return TERMINAL_STATE, self.reward_incorrect, True, False, info

        # Validate palindromic substrings
        assert self.S is not None and self.gold_answer is not None, "Environment not properly initialized"
        S = self.S

        def is_palindrome(s: str) -> bool:
            return s == s[::-1]

        left = S[A:B]
        right = S[B:C]
        if len(left) == 0 or len(right) == 0 or not (is_palindrome(left) and is_palindrome(right)):
            info = {
                "error": "invalid_solution",
                "reason": "non_palindromic_substrings",
                "A": A, "B": B, "C": C,
                "substring_left": left,
                "substring_right": right
            }
            return TERMINAL_STATE, self.reward_incorrect, True, False, info

        # Check if the provided length is optimal
        user_answer_len = C - A
        is_correct = (user_answer_len == self.gold_answer)

        info = {
            "correct": is_correct,
            "user_answer": (A, B, C),
            "user_length": user_answer_len,
            "reference_best_length": self.gold_answer,
            "N": self.N,
            "S": self.S
        }

        reward = self.reward_correct if is_correct else self.reward_incorrect
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within \\boxed{...}. Returns None if not found."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_gold_answer(self, S: str) -> int:
        """
        Compute the maximum sum of lengths l + r where the split is between two palindromic substrings.
        This uses a Manacher-like approach on a modified string to capture palindromic spans.
        """
        # Build modified string with separators
        modified: List[str] = ['@', '#']
        for ch in S:
            modified.append(ch)
            modified.append('#')
        modified.append('$')
        M = len(modified)

        # Arrays for palindromic radii
        p = [0] * M
        # Arrays to record max palindromic radii ending/starting at positions
        l = [0] * M
        r = [0] * M

        center = 0
        right = 0

        # Manacher's algorithm
        for i in range(1, M - 1):
            mirror = 2 * center - i
            if i < right:
                p[i] = min(right - i, p[mirror])
            # Expand around center i
            while modified[i + 1 + p[i]] == modified[i - 1 - p[i]]:
                p[i] += 1
            # Update center and right boundary
            if i + p[i] > right:
                center = i
                right = i + p[i]
            # Record palindromic spans
            if p[i] > 0:
                l[i + p[i]] = max(l[i + p[i]], p[i])
                r[i - p[i]] = max(r[i - p[i]], p[i])

        # Propagate best spans outward
        for i in range(M - 4, 0, -2):
            l[i] = max(l[i], l[i + 2] - 2)
        for i in range(3, M, 2):
            r[i] = max(r[i], r[i - 2] - 2)

        # Compute the answer by checking split points at separator positions
        ans = 0
        for i in range(1, M, 2):  # only consider '#' positions
            if l[i] > 0 and r[i] > 0:
                ans = max(ans, l[i] + r[i])
        return ans

    def sample_random_action(self) -> str:
        """Sample a random action; not guaranteed to be valid."""
        A = random.randint(0, max(0, self.N - 3))
        B = random.randint(A + 1, max(A + 1, self.N - 1))
        C = random.randint(B + 1, self.N)
        return f"\\boxed{{{A} {B} {C}}}"