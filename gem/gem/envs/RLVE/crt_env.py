import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CRTEnv(Env):
    """Chinese Remainder-like congruence environment - single-turn Q&A.

    The environment generates a system of modular congruences based on a hidden integer X
    and a list of moduli B. The task is to provide any non-negative integer x that satisfies
    all generated congruences: x % b == (X % b) for each b in B.

    Answer must be provided in \\boxed{...} format.
    """

    def __init__(
        self,
        max_x: int = 1000,
        M: int = 3,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(satisfied/all)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_x: Upper bound for the hidden solution X (inclusive lower bound is 2).
        - M: Number of modular congruences to generate (at most X-1 will be used).
        - wrong_format, rewarding_strategy, rewarding_weight, rewarding_beta:
          Preserved parameters from the original environment for compatibility,
          but not used in GEM reward computation.
        """
        super().__init__()
        self.max_x: int = max_x
        self.M: int = M

        # Preserved reward-related settings from the original environment (unused in GEM step)
        self.wrong_format: float = wrong_format
        self.rewarding_strategy: str = rewarding_strategy
        self.rewarding_weight: float = rewarding_weight
        self.rewarding_beta: float = rewarding_beta

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None  # a known solution X
        self.mods: List[int] = []
        self.remainders: List[int] = []

        # Validate parameters
        if self.max_x < 2:
            raise ValueError("MAX_X should be greater than or equal to 2")
        if self.M < 1:
            raise ValueError("M should be greater than or equal to 1")

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given a system of modular congruences.\n"
            "Your task is to find any non-negative integer x that satisfies all the equations.\n"
            "Please provide your answer in \\boxed{...} format with a single integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem."""
        super().reset(seed)

        # Generate hidden X
        X = random.randint(2, self.max_x)
        self.reference_answer = X

        # Generate B and corresponding remainders X % b
        B = random.sample(range(2, X + 1), min(self.M, X - 1))
        X_mod_B = [X % b for b in B]

        self.mods = B
        self.remainders = X_mod_B

        # Build problem statement
        equations_text = "\n".join(
            f"x ≡ {r} (mod {b})" for r, b in zip(self.remainders, self.mods)
        )
        self.current_problem = (
            f"You are given a system of {len(self.mods)} modular congruences:\n"
            f"{equations_text}\n\n"
            "Your task is to find any non-negative integer x that satisfies all of the above congruences.\n\n"
            "Output Format: Your output should be a single integer x in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer."""
        # Parse answer in \boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer
        try:
            user_answer = int(parsed.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check non-negative constraint
        if user_answer < 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "negative_answer"}

        # Verify all congruences
        satisfied = sum(int(user_answer % b == r) for r, b in zip(self.remainders, self.mods))
        is_correct = satisfied == len(self.mods)

        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "constraints_satisfied": satisfied,
            "total_constraints": len(self.mods),
            "mods": self.mods,
            "remainders": self.remainders,
            "user_answer": user_answer,
            "a_known_solution": self.reference_answer,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a boxed non-negative integer."""
        # Sample within a reasonable range
        random_answer = random.randint(0, self.max_x)
        return f"\\boxed{{{random_answer}}}"