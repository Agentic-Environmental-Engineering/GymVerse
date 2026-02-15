import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BucketSortingEnv(Env):
    """Bucket Sorting environment - single-turn Q&A.

    Task: Given an array of integers, find the number that appears most frequently.
    If multiple numbers share the highest frequency, output any one of them.
    The answer must be provided in \\boxed{...} format.
    """

    def __init__(
        self,
        N: int = 10,
        MAX: int = 10,
        **kwargs
    ):
        """Initialize the environment with parameters.

        Args:
            N: The length of the array to generate. Must be >= 3.
            MAX: The maximum value (inclusive) for the integers in the array. Must be >= 1.
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        assert MAX >= 1, "MAX should be greater than or equal to 1"

        self.N = N
        self.MAX = MAX

        self.array: List[int] = []
        self.value2count: Dict[int, int] = {}
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.max_count: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a frequency counting problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate the array
        self.array = [random.randint(0, self.MAX) for _ in range(self.N)]

        # Count frequencies
        self.value2count = {}
        for value in self.array:
            if value not in self.value2count:
                self.value2count[value] = 0
            self.value2count[value] += 1

        # Determine the reference answer (any value with highest frequency)
        self.reference_answer = max(self.value2count.items(), key=lambda x: x[1])[0]
        self.max_count = max(self.value2count.values())

        # Build problem prompt
        array_str = " ".join(map(str, self.array))
        self.current_problem = (
            f"You are given the following array: {array_str}\n\n"
            "Please find the number that appears most frequently in the array. "
            "If there are multiple numbers with the same highest frequency, you may output any one of them.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        answer_str = self._parse_answer(action)

        if answer_str is None:
            # Format error: no \\boxed{...} found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to parse integer from boxed content
        try:
            user_answer = int(answer_str)
        except ValueError:
            # Non-integer inside boxed content is considered an invalid answer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check if the answer exists in the array
        if user_answer not in self.value2count:
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "array": self.array,
                "error": "not_in_array",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compare frequency with the maximum count
        is_correct = (self.value2count[user_answer] == self.max_count)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "array": self.array,
            "value2count": self.value2count,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        if self.array:
            random_answer = random.choice(self.array)
        else:
            random_answer = random.randint(0, self.MAX)
        return f"\\boxed{{{random_answer}}}"