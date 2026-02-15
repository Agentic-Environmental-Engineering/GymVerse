from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class QuantumLockPuzzleEnv(Env):
    """
    Quantum Lock Puzzle environment - single-turn Q&A.

    Task:
      - There is a binary variable X, initially 0.
      - There is an integer variable Y, starting at Y_start.
      - There are N buttons. Pressing any button toggles X (X becomes 1 - X).
      - When X is 0 and you press button i, Y is updated by a rule specific to (X=0, i).
      - When X is 1 and you press button i, Y is updated by a rule specific to (X=1, i).
      - Find a sequence of button presses that makes Y equal to the target Y_target.

    Answer format:
      - Provide the sequence of button indices separated by spaces, enclosed in \\boxed{...}.
      - Example: \\boxed{0 1 0 2}
      - For the empty sequence, submit \\boxed{}.
    """

    def __init__(
        self,
        N: int = 4,
        steps: int = 3,
        operation_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize the QuantumLockPuzzleEnv.

        Args:
            N: Number of buttons (must be >= 2).
            steps: Base number of random steps used to generate a solvable target (must be >= 2).
            operation_weights: Probabilities for operations ["+", "-", "*"] when generating rules.
                               Defaults to [0.4, 0.4, 0.2].
        """
        super().__init__()
        if operation_weights is None:
            operation_weights = [0.4, 0.4, 0.2]

        # Parameter validation
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")
        if steps < 2:
            raise ValueError("steps should be greater than or equal to 2")
        if len(operation_weights) != 3 or abs(sum(operation_weights) - 1.0) > 1e-8:
            raise ValueError("operation_weights must be a list of three probabilities that sum to 1")

        self.N = N
        self.steps = steps
        self.operation_weights = operation_weights

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.buttons: Optional[List[List[List]]] = None  # shape: [N][2][operation, value]
        self.Y_start: Optional[int] = None
        self.Y_target: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a quantum lock puzzle with a toggling variable X and an integer Y.\n"
            "Provide your final sequence of button indices inside \\boxed{...}.\n"
            "Do not include backticks or quotes.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new puzzle instance."""
        super().reset(seed)

        # Generate initial Y and button rules
        Y = random.randint(-self.N, +self.N)
        self.Y_start = Y
        buttons: List[List[List]] = []

        # Rule generator for each button and X state
        def rule_generator() -> List:
            operation = random.choices(["+", "-", "*"], weights=self.operation_weights, k=1)[0]
            if operation in ("+", "-"):
                value = random.randint(1, self.N)
            elif operation == "*":
                value = random.randint(2, 3)
            else:
                raise NotImplementedError(f"Unknown operation: {operation}")
            return [operation, value]

        # Each button has 2 rules: one for X=0 and one for X=1
        for _ in range(self.N):
            buttons.append([rule_generator() for _ in range(2)])

        self.buttons = buttons

        # Decide number of steps to simulate to create a reachable target
        steps_total = self.steps + random.randint(0, 1)

        # Simulate random presses to get a solvable target Y_target
        X = 0
        pressed_buttons: List[int] = []
        existing_Y = {Y}
        reference_answer_sequence: Optional[List[int]] = None
        target: Optional[int] = None

        for _ in range(steps_total):
            button = random.randint(0, self.N - 1)
            pressed_buttons.append(button)
            Y = self._operate(Y, buttons[button][X])
            X = 1 - X
            if Y not in existing_Y:
                existing_Y.add(Y)
                reference_answer_sequence = pressed_buttons.copy()
                target = Y

        if target is None:
            # No new Y found; empty sequence is correct
            self.Y_target = self.Y_start
            self.reference_answer = ""
        else:
            self.Y_target = target
            self.reference_answer = " ".join(map(str, reference_answer_sequence))

        # Build problem prompt
        x0_rules = "\n".join(
            f"When you press button {i}, Y becomes Y {button[0][0]} {button[0][1]}"
            for i, button in enumerate(self.buttons)
        )
        x1_rules = "\n".join(
            f"When you press button {i}, Y becomes Y {button[1][0]} {button[1][1]}"
            for i, button in enumerate(self.buttons)
        )

        problem = (
            f"There is a 0/1 variable X, which is initially 0. You also have a variable Y, which starts at {self.Y_start}. "
            f"You can press the buttons in any order, and you may press the same button multiple times. There are {self.N} buttons in total. "
            f"Each time you press any button, X toggles: it becomes 1 - X.\n\n"
            f"When X is 0 and you press a button, Y changes according to the following rules:\n{x0_rules}\n\n"
            f"When X is 1 and you press a button, Y changes according to the following rules:\n{x1_rules}\n\n"
            f"Please find a sequence of button presses that will make Y equal to {self.Y_target}.\n\n"
            f"Output Format: Your final answer should be the sequence of button indices separated by spaces, enclosed in \\boxed{{...}}. "
            f"For example, \\boxed{{0 1 0 2}}. For the empty sequence, submit \\boxed{{}}."
        )

        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted sequence and return the result."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert boxed content into list of integers (button indices)
        try:
            sequence = self._parse_sequence(boxed_content)
        except ValueError:
            # Non-integer token encountered -> format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate by simulating the presses
        assert self.Y_start is not None and self.buttons is not None and self.Y_target is not None

        X, Y = 0, self.Y_start
        for button in sequence:
            if not (0 <= button < self.N):
                # Invalid button index -> wrong answer
                info = {
                    "correct": False,
                    "reference_answer": self.reference_answer,
                    "user_answer": " ".join(map(str, sequence)),
                    "error": "invalid_button_index",
                }
                return TERMINAL_STATE, 0.0, True, False, info
            Y = self._operate(Y, self.buttons[button][X])
            X = 1 - X

        is_correct = (Y == self.Y_target)
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, sequence)),
        }
        return TERMINAL_STATE, reward, True, False, info

    def _operate(self, Y: int, rule: List) -> int:
        """Apply a rule (operation, value) to Y."""
        operation, value = rule
        if operation == "+":
            return Y + value
        elif operation == "-":
            return Y - value
        elif operation == "*":
            return Y * value
        else:
            raise NotImplementedError(f"Unknown operation: {operation}")

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Supports empty content."""
        import re

        # This pattern captures even empty content between the braces.
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            # Use the last boxed content if multiple are present
            return matches[-1].strip()
        return None

    def _parse_sequence(self, content: str) -> List[int]:
        """
        Parse the boxed content into a list of integers.
        Empty content corresponds to an empty sequence.
        Raises ValueError on token-to-int conversion failure.
        """
        if content == "":
            return []
        tokens = content.split()
        return [int(tok) for tok in tokens]

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        length = random.randint(0, self.steps)
        seq = [str(random.randint(0, max(0, self.N - 1))) for _ in range(length)]
        return f"\\boxed{{{' '.join(seq)}}}"