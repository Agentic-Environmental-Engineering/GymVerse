from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SecretCowCodeEnv(Env):
    """Secret Cow Code environment - single turn Q&A."""

    def __init__(
        self,
        max_n: int = 10,
        max_k: int = 100,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: Maximum length for the initial string S (must be >= 2).
            max_k: Maximum index K to query in the infinite expansion (must be > max_n).
        """
        super().__init__()
        if max_n < 2:
            raise ValueError("max_n should be greater than or equal to 2")
        if max_k <= max_n:
            raise ValueError("max_k should be greater than max_n")
        self.max_n = max_n
        self.max_k = max_k

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.S: Optional[str] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving the Secret Cow Code problem.\n"
            "Given a lowercase string S, define right_shift(s) as moving the last character to the beginning.\n"
            "Define F(s) = s + right_shift(s). Repeatedly applying F builds an infinite string F^{∞}(S).\n"
            "Your task: output the K-th character (1-based, left to right) of F^{∞}(S).\n"
            "Output Format: Provide a single lowercase character in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate S with length in [2, max_n]
        self.S = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(2, self.max_n)))

        # Generate K in [len(S) + 1, max_k]
        self.K = random.randint(len(self.S) + 1, self.max_k)

        # Build problem statement
        self.current_problem = (
            f"You are given a string S consisting of lowercase English letters: {self.S}\n"
            f"Define F(s) as the string obtained by concatenating s with right_shift(s) (s + right_shift(s)), "
            f"where right_shift(s) means moving the last character of s to the beginning. "
            f"Let F^{chr(8734)}(S) denote the result of applying F infinitely many times to S: "
            f"F^{chr(8734)}(S) = F(F(F(...(S)...))). "
            f"Please output the {self.K}-th character (1-based index, from left to right) of the infinite string F^{chr(8734)}(S).\n\n"
            f"Output Format: Your final answer should be a single lowercase character in \\boxed{{...}}."
        )

        # Compute reference answer using the original backward-mapping logic
        self.reference_answer = self._compute_reference(self.S, self.K)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference(self, s: str, k: int) -> str:
        """Compute the K-th character of F^{∞}(S) using backward mapping."""
        n = k
        lengths = [len(s)]
        while lengths[-1] < n:
            lengths.append(lengths[-1] * 2)

        while len(lengths) > 1:
            lengths.pop()
            half = lengths[-1]
            if n > half:
                if n == half + 1:
                    n = half
                else:
                    n = n - (half + 1)
            # if n <= half: stays the same

        return s[n - 1]

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Enforce original format validations:
        # - must be a single character
        # - the character must be present in S
        assert self.S is not None and self.reference_answer is not None and self.K is not None

        if len(parsed) != 1:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if parsed not in self.S:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        is_correct = (parsed == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": parsed,
            "S": self.S,
            "K": self.K,
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

    def sample_random_action(self) -> str:
        """Sample a random action (random character from S if available, otherwise from alphabet)."""
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        if self.S and len(self.S) > 0:
            ch = random.choice(self.S)
        else:
            ch = random.choice(alphabet)
        return f"\\boxed{{{ch}}}"