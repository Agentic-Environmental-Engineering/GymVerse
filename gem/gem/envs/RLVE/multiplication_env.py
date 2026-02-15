from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MultiplicationEnv(Env):
    """
    Single-turn multiplication environment in GEM format.

    The agent is asked to compute the product of two non-negative integers.
    The integers are sampled based on the specified digit count.
    The answer must be provided in \\boxed{...} format.
    """

    prompt_templates = (
        "Give me the answer of the following equation: {} * {} = ",
        "What is the result of {} times {}?",
        "Calculate the product of {} and {}.",
        "What do you get when you multiply {} by {}?",
        "If you multiply {} and {}, what is the answer?",
        "What is {} multiplied by {}?",
        "Find the result of {} times {}.",
        "What is the multiplication of {} and {}?",
        "Compute the product of {} and {}.",
        "What is the answer to {} times {}?",
    )

    def __init__(self, digit_num: int = 1, **kwargs) -> None:
        """
        Initialize the environment.

        Parameters:
            digit_num: The number of digits for each operand (a and b).
                       Each operand is sampled uniformly from [0, 10^digit_num - 1].
        """
        super().__init__()
        assert isinstance(digit_num, int), "digit_num must be an integer"
        assert digit_num >= 1, "digit_num should be greater than or equal to 1"

        self.digit_num = digit_num
        self.a: Optional[int] = None
        self.b: Optional[int] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None
        self._prompt_index: Optional[int] = None

    def _get_instructions(self) -> str:
        """
        Return task instructions.
        """
        return (
            "You are solving basic multiplication problems.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem.

        Returns:
            observation: A string containing instructions and the problem description.
            info: An empty info dictionary.
        """
        super().reset(seed)

        # Generate operands based on digit count
        upper = 10 ** self.digit_num - 1
        self.a = random.randint(0, upper)
        self.b = random.randint(0, upper)
        self.reference_answer = self.a * self.b

        # Select a prompt template
        self._prompt_index = random.randrange(len(self.prompt_templates))

        # Build the problem description
        prompt = self.prompt_templates[self._prompt_index].format(self.a, self.b)
        self.current_problem = (
            f"{prompt}\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Verify the answer and terminate immediately.

        Parameters:
            action: The agent's response string, expected to contain \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE
            reward: 1.0 if correct, 0.0 if wrong, -0.1 for format errors
            terminated: True
            truncated: False
            info: Dictionary with verification details
        """
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(parsed.strip())
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Verify correctness
        assert self.reference_answer is not None, "Environment not properly initialized; call reset() first."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "a": self.a,
            "b": self.b,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside \\boxed{...} from the text.

        Returns:
            The extracted string if found, otherwise None.
        """
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random action in the required \\boxed{...} format.
        """
        # The product of two digit_num-digit numbers can be up to (10^digit_num - 1)^2
        max_product = (10 ** self.digit_num - 1) ** 2
        random_answer = random.randint(0, max_product)
        return f"\\boxed{{{random_answer}}}"