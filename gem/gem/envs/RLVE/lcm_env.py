import math
import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LCMEnv(Env):
    """Least Common Multiple (LCM) single-turn question-answer environment."""

    prompt_templates = (
        "Please calculate the least common multiple (LCM) of {} and {}.",
        "What is the least common multiple (LCM) of {} and {}?",
        "Find the least common multiple (LCM) of {} and {}.",
        "Calculate the LCM of {} and {}.",
        "Determine the least common multiple (LCM) of {} and {}.",
        "What is the smallest positive integer that is a multiple of both {} and {}? (This is the LCM.)",
        "What is the least common multiple (LCM) of the numbers {} and {}?",
        "Compute the least common multiple (LCM) of {} and {}.",
        "Find the smallest number that is a multiple of both {} and {}. (This is the LCM.)",
        "What is the least common multiple (LCM) of these two numbers: {} and {}?",
    )

    def __init__(
        self,
        max_a_b: int = 100,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        format_error_reward: float = -0.1,
        **kwargs: Any
    ):
        """
        Initialize the LCM environment.

        Args:
            max_a_b: Upper bound (inclusive) for randomly sampling a and b. Must be >= 2.
            correct_reward: Reward for a correct answer.
            wrong_reward: Reward for an incorrect answer.
            format_error_reward: Reward for an output format error.
        """
        super().__init__()
        assert max_a_b >= 2, "max_a_b should be greater than or equal to 2"

        self.max_a_b = max_a_b
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.format_error_reward = format_error_reward

        self.a: Optional[int] = None
        self.b: Optional[int] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None
        self._template_index: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving least common multiple (LCM) problems.\n"
            "Please provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: The full instruction plus the problem statement.
            info: Additional information dict.
        """
        super().reset(seed)

        # Generate parameters
        self.a = random.randint(2, self.max_a_b)
        self.b = random.randint(2, self.max_a_b)

        # Compute reference answer
        self.reference_answer = math.lcm(self.a, self.b)

        # Select prompt template and build problem statement
        self._template_index = random.randrange(len(self.prompt_templates))
        problem_text = self.prompt_templates[self._template_index].format(self.a, self.b)
        self.current_problem = (
            f"{problem_text}\n\n"
            f"Output Format: Provide a single integer inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted answer.

        Args:
            action: The model's textual answer, expected to contain \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE for single-turn environment.
            reward: Reward based on correctness and format.
            terminated: True (single-turn).
            truncated: False for this environment.
            info: Dictionary with evaluation details.
        """
        # Parse answer from boxed format
        parsed = self._parse_answer(action)
        if parsed is None:
            return (
                TERMINAL_STATE,
                self.format_error_reward,
                True,
                False,
                {"error": "format_error", "message": "Missing or invalid \\boxed{...} format."},
            )

        # Convert to integer
        try:
            user_answer = int(parsed)
        except ValueError:
            # Non-integer content inside the box is considered a format error per original logic
            return (
                TERMINAL_STATE,
                self.format_error_reward,
                True,
                False,
                {"error": "format_error", "message": "Non-integer content inside \\boxed{...}."},
            )

        assert self.reference_answer is not None
        is_correct = (user_answer == self.reference_answer)
        reward = self.correct_reward if is_correct else self.wrong_reward

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "a": self.a,
            "b": self.b,
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
        """Sample a random action in the required \\boxed{...} format."""
        # Random guess in a reasonable range based on max_a_b
        # LCM can be up to lcm(max_a_b, max_a_b) = max_a_b
        # For randomness, we broaden a bit
        random_answer = random.randint(1, max(2, self.max_a_b * 2))
        return f"\\boxed{{{random_answer}}}"