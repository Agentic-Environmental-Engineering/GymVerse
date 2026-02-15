from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class EmperorWorriesEnv(Env):
    """Emperor Worries problem environment - single-turn Q&A."""

    def __init__(
        self,
        K: int = 1,
        A_range: int = 2,
        wrong_format: float = -0.1,
        correct_answer: float = 1.0,
        wrong_answer: float = 0.0,
        **kwargs
    ):
        """
        Initialize the EmperorWorriesEnv instance.

        Parameters:
            K: Base parameter controlling N (N is either 2*K or 2*K+1).
            A_range: Controls the magnitude of medal requirements.
            wrong_format: Reward when the answer format is incorrect (must be in \\boxed{...}).
            correct_answer: Reward when the answer is correct.
            wrong_answer: Reward when the answer is incorrect.
        """
        super().__init__()
        self.K = K
        self.A_range = A_range
        self.rewards = {
            "wrong_format": wrong_format,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
        }
        self.N: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving the Emperor Worries medal assignment problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n"
            "Do not include any explanations or additional text.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Validate and determine N
        if self.K < 1:
            raise ValueError("K should be greater than or equal to 1")
        self.N = random.choice([2 * self.K, 2 * self.K + 1])

        # Generate medal requirements A[i] for each general
        self.A = [random.randint(1, self.N * self.A_range) for _ in range(self.N)]

        # Compute the reference answer based on the original algorithm
        S = sum(self.A)
        candidates: List[int] = []
        for i in range(self.N - 1):
            candidates.append(self.A[i] + self.A[i + 1])
        candidates.append(self.A[0] + self.A[-1])

        K_denom = self.N // 2
        candidates.append((S + K_denom - 1) // K_denom)  # ceil(S / floor(N/2))
        self.reference_answer = max(candidates)

        # Build the problem prompt
        requirements_str = "; ".join(
            f"General {i} needs {Ai} medals of distinct types" for i, Ai in enumerate(self.A)
        )
        self.current_problem = (
            f"There are {self.N} generals numbered from 0 to {self.N - 1}. "
            f"The medal requirements are: {requirements_str}\n"
            "Assign medals of various types to the generals so that:\n"
            "(1) The medals given to the same general are all of distinct types (no duplicate type for one general);\n"
            f"(2) Adjacent generals (i and (i+1) mod {self.N}) share no common medal type.\n"
            "What is the minimum number of medal types required to satisfy all constraints?\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.N,
            "A": self.A,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the submitted answer."""
        # Extract boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, self.rewards["wrong_format"], True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(boxed.strip())
        except ValueError:
            return TERMINAL_STATE, self.rewards["wrong_answer"], True, False, {"error": "invalid_answer"}

        # Check correctness
        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = self.rewards["correct_answer"] if is_correct else self.rewards["wrong_answer"]

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "A": self.A,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        # Use a reasonable range if reference_answer is not yet available
        upper = self.reference_answer if self.reference_answer is not None else 100
        upper = max(1, int(upper))
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"