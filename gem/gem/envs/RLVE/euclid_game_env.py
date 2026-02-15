from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class EuclidGameEnv(Env):
    """Euclid's Game environment - single-turn question answering."""

    def __init__(
        self,
        max_x_y: int = 100,
        **kwargs
    ):
        """
        Initialize the EuclidGameEnv instance.

        Parameters:
        - max_x_y: The maximum value for X and Y (must be >= 1).
        """
        super().__init__()
        assert max_x_y >= 1, "max_x_y should be greater than or equal to 1"
        self.max_x_y = max_x_y

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.X: Optional[int] = None
        self.Y: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving Euclid's game optimal play problems.\n"
            "Please provide your final answer in \\boxed{...} format.\n"
            "Your answer must be exactly one word: Stan or Ollie.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Randomly decide which player should be the winner to balance data
        self.reference_answer = "Stan" if random.random() < 0.5 else "Ollie"

        # Generate X and Y until the computed winner matches the chosen reference answer
        while True:
            self.X = random.randint(1, self.max_x_y)
            self.Y = random.randint(1, self.max_x_y)
            winner = "Stan" if self._first_player_wins(max(self.X, self.Y), min(self.X, self.Y)) else "Ollie"
            if winner == self.reference_answer:
                break

        # Build the problem statement
        self.current_problem = (
            f"Stan and Ollie are playing a game starting with two integers {self.X} and {self.Y}. Stan goes first.\n\n"
            "On each turn, a player may subtract any positive multiple of one integer from the other, as long as the result is non-negative. "
            "The player who makes one of the numbers become zero wins the game.\n\n"
            "If both players always play optimally, who will win — Stan or Ollie?\n\n"
            "Output Format: Your final answer should be a single word in \\boxed{...}, either Stan or Ollie."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error (no valid \boxed{...})
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        answer = boxed_content.strip()

        # Validate answer
        if answer not in ("Stan", "Ollie"):
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "invalid_answer",
                "user_answer": answer,
                "reference_answer": self.reference_answer,
                "X": self.X,
                "Y": self.Y,
            }

        # Check correctness
        is_correct = (answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": answer,
            "X": self.X,
            "Y": self.Y
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

    def _first_player_wins(self, x: int, y: int) -> bool:
        """
        Determine if the first player (current turn) wins with optimal play
        given state (x, y) with x >= y > 0.

        This follows the logic from the original environment:
        - If y == 0: current player loses.
        - If x // y != 1: current player can win immediately.
        - Otherwise, the game proceeds to (y, x - y) and the turn switches.
        """
        def check(a: int, b: int) -> bool:
            if b == 0:
                return False
            if a // b != 1:
                return True
            return not check(b, a - b)

        return check(x, y)

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        return f"\\boxed{{{random.choice(['Stan', 'Ollie'])}}}"