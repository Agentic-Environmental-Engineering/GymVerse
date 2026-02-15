import math
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BinaryTreeLeafNumExpectationEnv(Env):
    """
    Environment for computing the expected number of leaf nodes in a uniformly random binary tree
    with exactly N nodes. The answer must be provided as a reduced fraction A/B inside \\boxed{...}.
    Single-turn question-answer format.
    """

    prompt_template = (
        "We uniformly at random generate a binary tree with exactly {N} nodes "
        "(all distinct binary trees with {N} nodes are equally likely). Two binary trees are considered identical if and only if:\n"
        "- both are empty, OR\n"
        "- both are non-empty, and their left subtrees are identical and their right subtrees are identical.\n\n"
        "What is the expected number of leaf nodes (nodes whose left and right children are both empty) "
        "in the generated binary tree?"
    )

    def __init__(
        self,
        max_n: int = 100,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        format_error_reward: float = -0.1,
        **kwargs: Any,
    ):
        """
        Initialize the environment.

        Args:
            max_n: Maximum N to sample from (inclusive). Must be >= 5.
            correct_reward: Reward for a correct answer.
            wrong_reward: Reward for an incorrect but well-formed answer.
            format_error_reward: Reward for a formatting error (e.g., not in \\boxed{...} or invalid fraction).
        """
        super().__init__()
        if max_n < 5:
            raise ValueError("max_n should be greater than or equal to 5")
        self.max_n = max_n
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.format_error_reward = format_error_reward

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_A: Optional[int] = None
        self.gold_B: Optional[int] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the general task instructions."""
        return (
            "You are solving a combinatorics/probability problem about uniformly random binary trees.\n"
            "Please provide your final answer as a reduced positive fraction A/B inside \\boxed{...}.\n"
            "For example: \\boxed{3/5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample problem parameter N
        self.N = random.randint(1, self.max_n)

        # Compute the reduced fraction A/B according to the original logic
        A = self.N * (self.N + 1)
        B = 2 * (2 * self.N - 1)
        gcd_ab = math.gcd(A, B)
        A //= gcd_ab
        B //= gcd_ab

        self.gold_A = A
        self.gold_B = B
        self.reference_answer = f"{A}/{B}"

        # Build the problem description
        problem_text = self.prompt_template.format(N=self.N)
        problem_text += (
            "\n\nOutput Format: Your final answer must be a positive reduced fraction A/B and must be placed inside "
            "\\boxed{...}. Do not include any extra text."
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and return the result."""
        # Extract the content within \boxed{...}
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        # Parse fraction A/B
        try:
            parts = list(map(str.strip, boxed_content.split("/")))
            if len(parts) != 2:
                return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "invalid_fraction_format"}

            A_user = int(parts[0])
            B_user = int(parts[1])

            # Validate positivity
            if A_user <= 0 or B_user <= 0:
                return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "non_positive_fraction"}

            # Reduce user's fraction
            g = math.gcd(A_user, B_user)
            A_user //= g
            B_user //= g
        except Exception:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "invalid_fraction_parse"}

        # Compare with reference answer
        assert self.gold_A is not None and self.gold_B is not None
        is_correct = (A_user == self.gold_A and B_user == self.gold_B)
        reward = self.correct_reward if is_correct else self.wrong_reward

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": f"{A_user}/{B_user}",
            "N": self.N,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} from the given text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{A/B} format."""
        # Random positive integers A, B (not necessarily reduced)
        A = random.randint(1, 10)
        B = random.randint(1, 10)
        return f"\\boxed{{{A}/{B}}}"