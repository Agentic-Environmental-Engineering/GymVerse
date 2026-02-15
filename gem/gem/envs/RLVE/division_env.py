from typing import Any, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DivisionEnv(Env):
    """Integer division environment - single-turn Q&A."""

    prompt_templates = (
        "What is the result of {} divided by {}? Round down to the nearest integer.",
        "Compute {} divided by {}, rounding down to the nearest whole number.",
        "Find the integer part of {} divided by {}.",
        "Compute {} divided by {}, discarding the remainder.",
        "What is the quotient when {} is divided by {}, using integer division?",
        "If you divide {} by {}, what is the whole number result?",
        "Give me the result of {} divided by {} (rounded down).",
        "How many full times does {} fit into {}?",
        "What do you get when you divide {} by {} and round down?",
        "Determine the integer result of {} divided by {}.",
    )

    def __init__(
        self,
        divisor_digit_num: int = 1,
        answer_digit_num: int = 1,
        wrong_format_reward: float = -0.1,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        **kwargs,
    ):
        """
        Initialize DivisionEnv.

        Parameters:
            divisor_digit_num: Number of digits for the divisor (>= 1).
            answer_digit_num: Maximum number of digits for the quotient (>= 1).
            wrong_format_reward: Reward for format errors.
            correct_reward: Reward for correct answers.
            wrong_reward: Reward for wrong answers.
        """
        super().__init__()
        assert divisor_digit_num >= 1, "divisor_digit_num should be greater than or equal to 1"
        assert answer_digit_num >= 1, "answer_digit_num should be greater than or equal to 1"

        self.divisor_digit_num = divisor_digit_num
        self.answer_digit_num = answer_digit_num

        self.wrong_format_reward = wrong_format_reward
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward

        self.a: Optional[int] = None
        self.b: Optional[int] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.prompt_index: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving integer division problems.\n"
            "Compute the quotient using floor division (round down, discard remainder).\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem."""
        super().reset(seed)

        # Validate parameters again to preserve original logic
        assert self.divisor_digit_num >= 1, "divisor_digit_num should be greater than or equal to 1"
        assert self.answer_digit_num >= 1, "answer_digit_num should be greater than or equal to 1"

        # Generate divisor b and dividend a
        self.b = random.randint(1, 10 ** self.divisor_digit_num - 1)
        self.a = self.b * random.randint(0, 10 ** self.answer_digit_num - 1) + random.randint(0, self.b - 1)

        # Compute reference answer using integer division
        self.reference_answer = self.a // self.b

        # Choose prompt template
        self.prompt_index = random.randrange(len(self.prompt_templates))

        # Build problem text
        problem_text = self.prompt_templates[self.prompt_index].format(self.a, self.b)
        self.current_problem = (
            f"{problem_text}\n\n"
            f"Output Format: Provide a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "a": self.a,
            "b": self.b,
            "prompt_index": self.prompt_index,
        }

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the submitted answer."""
        # Parse answer from boxed format
        parsed = self._parse_answer(action)

        if parsed is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Validate and score
        try:
            user_answer = int(parsed)
            is_correct = (user_answer == self.reference_answer)
            reward = self.correct_reward if is_correct else self.wrong_reward
        except ValueError:
            # Not a valid integer
            return TERMINAL_STATE, self.wrong_reward, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "a": self.a,
            "b": self.b,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract answer from \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random integer answer in boxed format."""
        # Sample around plausible range: [0, 10^answer_digit_num - 1]
        max_ans = max(1, 10 ** self.answer_digit_num - 1)
        random_answer = random.randint(0, max_ans)
        return f"\\boxed{{{random_answer}}}"