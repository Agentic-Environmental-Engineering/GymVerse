import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SortingEnv(Env):
    """Sorting environment - single-turn Q&A.

    The task is to sort a given list of integers in ascending order.
    The answer must be provided in \\boxed{...} format, with numbers separated by spaces.
    For example: \\boxed{1 2 3 4 5}
    """

    def __init__(
        self,
        N: int,
        weight_multiple: int = 5,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "mean([gold=answer])^beta",
        rewarding_beta: float = 10.0,
        rewarding_weight: float = +1.0,
        **kwargs
    ):
        super().__init__()
        assert N >= 1, "N should be greater than or equal to 1"

        # Core parameters controlling problem generation
        self.N = N
        self.weight_multiple = weight_multiple

        # Preserve original reward-related parameters for compatibility, though not used in GEM scoring
        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_beta": rewarding_beta,
            "rewarding_weight": rewarding_weight,
        }

        # Runtime state
        self.array: List[int] = []
        self.gold_answer: List[int] = []
        self.reference_answer: str = ""
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a list of integers and must sort them in ascending order.\n"
            "Please provide your final answer as the sorted numbers separated by spaces, enclosed in \\boxed{...}.\n"
            "For example: \\boxed{1 2 3 4 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate the problem array
        self.array = [random.randint(0, self.N * self.weight_multiple) for _ in range(self.N)]
        assert len(self.array) == self.N, "array should have the same length as N"

        # Compute the gold answer and reference answer
        self.gold_answer = sorted(self.array)
        assert len(self.gold_answer) == self.N, "gold_answer should have the same length as N"
        self.reference_answer = " ".join(map(str, self.gold_answer))

        # Build the problem prompt
        self.current_problem = (
            "You are given the following list of numbers:\n"
            f"{' '.join(map(str, self.array))}\n"
            "Please sort them in ascending order.\n\n"
            "Output Format: Your final answer should be the sorted numbers separated by spaces, enclosed in \\boxed{...}.\n"
            "For example: \\boxed{1 2 3 4 5}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the provided answer.

        Rewards:
        - Correct answer: 1.0
        - Wrong answer: 0.0
        - Format error (missing or invalid \\boxed{...}): -0.1
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: no boxed content
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Try to parse the boxed content as a list of integers
        try:
            user_answer_list = list(map(int, boxed_content.split()))
        except ValueError:
            # Boxed content exists but cannot be parsed into integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length
        if len(user_answer_list) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "invalid_length",
                "expected_length": self.N,
                "received_length": len(user_answer_list),
            }

        # Check correctness
        is_correct = (user_answer_list == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer_list,
            "array": self.array,
            "N": self.N,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content from \\boxed{...} in the provided text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: randomly shuffle the sorted list and return in boxed format."""
        # Create a random permutation of the gold answer (may be correct by chance)
        candidate = self.gold_answer[:]
        random.shuffle(candidate)
        return f"\\boxed{{{' '.join(map(str, candidate))}}}"