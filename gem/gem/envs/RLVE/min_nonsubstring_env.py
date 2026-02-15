from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinNonsubstringEnv(Env):
    """Environment for finding the lexicographically smallest shortest binary (a/b) string
    that is not a substring of a given binary string A.
    Single-turn question-answer environment.
    """

    def __init__(self, N: int = 10, **kwargs):
        """
        Initialize the environment.

        Args:
            N: Length of the source string A. Must be >= 1.
        """
        super().__init__()
        assert N >= 1, "N should be greater than or equal to 1"
        self.N = N

        # Internal state
        self.current_problem: Optional[str] = None
        self.A: Optional[str] = None
        self.reference_answer: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a minimal non-substring problem over alphabet {a, b}.\n"
            "Provide your final answer strictly in \\boxed{...} format containing only 'a' and 'b'.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Args:
            seed: Optional random seed.

        Returns:
            observation: The instruction and problem description string.
            info: An empty dictionary or additional metadata.
        """
        super().reset(seed)

        # Generate string A with a bias determined by a random probability
        a_probability = random.random()
        # Note: Following the original logic exactly:
        # "ab"[random.random() < a_probability] selects 'b' with probability a_probability and 'a' otherwise.
        A = "".join("ab"[random.random() < a_probability] for _ in range(self.N))
        self.A = A

        # Find the lexicographically smallest shortest string B over {a,b} that is NOT a substring of A
        length = 1
        reference_answer: Optional[str] = None
        while True:
            found = False
            for B_mask in range(1 << length):
                B = "".join("ab"[(B_mask >> i) & 1] for i in range(length - 1, -1, -1))
                if B not in A:
                    reference_answer = B
                    found = True
                    break
            if found:
                break
            length += 1

        assert reference_answer is not None
        self.reference_answer = reference_answer

        # Build problem prompt
        self.current_problem = (
            f"You are given a string A = `{A}`\n\n"
            "Your task is to find a string B such that:\n"
            "(1) B consists only of the characters 'a' and 'b'.\n"
            "(2) B is NOT a contiguous substring of A.\n"
            "(3) Among all strings satisfying (1) and (2), B has the minimum possible length.\n"
            "(4) Among all strings satisfying (1), (2), and (3), B is lexicographically smallest. There is exactly one such string B.\n\n"
            "Output Format: Your final answer should be a single string B in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the answer and terminate.

        Args:
            action: The agent's answer text, expected to contain \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE.
            reward: 1.0 if correct; 0.0 if wrong or invalid answer content; -0.1 if format error.
            terminated: True (single-turn).
            truncated: False (no truncation logic).
            info: Additional metadata including correctness and the reference answer.
        """
        # Parse boxed answer
        answer = self._parse_answer(action)
        if answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        user_answer = answer.strip()

        # Validate content: must be only 'a' and 'b'
        if not all(c in "ab" for c in user_answer):
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "invalid_answer",
                "user_answer": user_answer,
                "reference_answer": self.reference_answer,
            }

        # Compare with reference answer
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "A": self.A,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within \\boxed{...} from the text (last match if multiple)."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random 'a'/'b' string wrapped in \\boxed{...}."""
        length = random.randint(1, max(1, min(6, self.N + 1)))
        s = "".join(random.choice("ab") for _ in range(length))
        return f"\\boxed{{{s}}}"