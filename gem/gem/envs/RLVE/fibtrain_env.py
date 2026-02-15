from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FibtrainEnv(Env):
    """Train boarding/alighting Fibonacci-like environment - single-turn Q&A."""

    def __init__(
        self,
        max_n: int = 100,
        max_a_b: int = 100,
        **kwargs
    ):
        """
        Initialize the FibtrainEnv instance.

        Parameters:
        - max_n: Maximum number of stations (N). Must be >= 5.
        - max_a_b: Maximum value for initial boarding parameters A and B. Must be >= 1.
        """
        super().__init__()
        assert max_n >= 5, "max_n should be greater than or equal to 5"
        assert max_a_b >= 1, "max_a_b should be greater than or equal to 1"

        self.max_n = max_n
        self.max_a_b = max_a_b

        # Internal state variables
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Parameters for the current instance
        self.N: Optional[int] = None
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.X: Optional[int] = None
        self.M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a train passenger counting problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate parameters
        N = random.randint(5, self.max_n)
        A = random.randint(1, self.max_a_b)
        B = random.randint(1, self.max_a_b)

        # Compute boarding and total passengers per station
        # Indices represent station numbers; we use 1..N
        boarding = [0] * (N + 1)
        total = [0] * (N + 1)

        boarding[1], boarding[2] = A, B
        total[1], total[2] = A, A  # Station 2 has equal boarding and alighting; net zero change

        for i in range(3, N):
            boarding[i] = boarding[i - 1] + boarding[i - 2]
            # At station i, alighting equals boarding at previous station
            # Net change = boarding[i] - boarding[i - 1]
            total[i] = total[i - 1] + boarding[i] - boarding[i - 1]

        # At final station N, all remaining passengers get off; this equals total at station N-1
        M = total[N - 1]

        # Query station X after departure; choose X in [3, N-1]
        X = random.randint(3, N - 1)
        reference_answer = total[X]

        # Store parameters
        self.N = N
        self.A = A
        self.B = B
        self.X = X
        self.M = M
        self.reference_answer = reference_answer

        # Build problem statement
        self.current_problem = (
            f"A train departs from its starting station (Station 1) with {A} passengers onboard. "
            f"There are {N} stations in total, numbered from 1 to {N}.\n\n"
            f"At Station 2, an equal number of passengers get on and off, so the total number of passengers "
            f"onboard remains unchanged at {A}.\n\n"
            f"From Station 3 onward (including Station 3) up to Station {N - 1}, the boarding and alighting follow a specific rule:\n"
            f"- The number of boarding passengers at each station is the sum of the number of boarding passengers at the previous two stations.\n"
            f"- The number of alighting passengers at each station is equal to the number of boarding passengers at the previous station.\n\n"
            f"At the final station (Station {N}), all remaining passengers get off, and the number of passengers who get off is {M}.\n\n"
            f"Given this setup, what is the number of passengers on the train after it departs from Station {X}?\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single integer in \\boxed{{...}}, representing the number of passengers onboard after the train departs from Station {X}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step: parse and verify the user's answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check correctness
        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "A": self.A,
            "B": self.B,
            "X": self.X,
            "M": self.M,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # If reference_answer is available, randomly choose correct or a nearby random number
        if self.reference_answer is not None:
            if random.random() < 0.5:
                return f"\\boxed{{{self.reference_answer}}}"
            else:
                noise = random.randint(-5, 5)
                return f"\\boxed{{{self.reference_answer + noise}}}"
        # Fallback random integer
        random_answer = random.randint(0, max(1, self.max_a_b * self.max_n))
        return f"\\boxed{{{random_answer}}}"