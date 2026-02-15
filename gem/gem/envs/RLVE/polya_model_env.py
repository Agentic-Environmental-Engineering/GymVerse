from typing import Any, Optional, SupportsFloat, Tuple
import random
from fractions import Fraction
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PolyaModelEnv(Env):
    """Polya's Urn Model probability environment - single-turn Q&A.

    The agent is asked to compute the probability that certain specified events
    (colors drawn at specified steps) occur in a Polya's urn process with reinforcement.
    The answer must be provided as a reduced fraction p/q in \\boxed{p/q} format.
    """

    def __init__(
        self,
        max_t_n: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_t_n: Upper bound (inclusive) for T, N, initial counts, and D.
                     Must be >= 2.
        """
        super().__init__()
        assert max_t_n >= 2, "max_t_n should be greater than or equal to 2"
        self.max_t_n = max_t_n

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[dict[str, int]] = None

        # Parameters used to generate the current problem (for debugging/inspection)
        self.T: Optional[int] = None
        self.color2num: Optional[list[int]] = None
        self.D: Optional[int] = None
        self.N: Optional[int] = None
        self.events: Optional[list[tuple[int, int]]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a probability problem based on Polya's Urn model.\n"
            "Output Format: Provide your final answer as a single fraction p/q in \\boxed{p/q}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate parameters
        T = random.randint(2, self.max_t_n)
        color2num = [random.randint(1, self.max_t_n) for _ in range(T)]
        D = random.randint(1, self.max_t_n)

        N = random.randint(1, self.max_t_n)
        steps = sorted(random.sample(range(1, N + 1), random.randint(1, N)))
        events = [(step, random.randint(1, T)) for step in steps]

        # Compute the reference probability as a reduced fraction
        ar = color2num.copy()
        s = sum(ar)
        ans = Fraction(1, 1)
        for _, col in events:
            y = col - 1  # zero-based index
            ans *= Fraction(ar[y], s)
            ar[y] += D
            s += D

        # Save to state
        self.T = T
        self.color2num = color2num
        self.D = D
        self.N = N
        self.events = events

        self.reference_answer = str(ans)  # "p/q"
        self.gold_answer = {"numerator": int(ans.numerator), "denominator": int(ans.denominator)}

        # Build problem statement
        color_desc = ", ".join(f"{num} balls of color {idx}" for idx, num in enumerate(color2num, start=1))
        events_desc = ", ".join(f"at step {step} the drawn ball is of color {color}" for step, color in events)

        self.current_problem = (
            f"You have a bag with balls of {T} colors. The initial counts are: {color_desc}\n"
            f"Process:\n"
            f"- At each step (starting from step 1), draw one ball uniformly at random from the bag.\n"
            f"- Return the drawn ball to the bag, then add {D} additional balls of the same color to the bag.\n\n"
            f"Given the following event(s): {events_desc}\n"
            f"What's the probability that all specified events occur?\n"
            f"Output a single fraction p/q (coprime, non-negative), where if the probability is 0, output 0/1; if it is 1, output 1/1.\n"
            f"Final Answer Format: \\boxed{{p/q}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        # Extract content inside \boxed{...}
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse fraction p/q
        frac = boxed.strip()
        parts = frac.split("/")
        if len(parts) != 2:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        try:
            a = int(parts[0].strip())
            b = int(parts[1].strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        user_answer = {"numerator": a, "denominator": b}
        is_correct = bool(self.gold_answer is not None and user_answer == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "T": self.T,
            "color2num": self.color2num,
            "D": self.D,
            "N": self.N,
            "events": self.events,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Return None if not found."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{p/q} format."""
        # Random non-negative integers; ensure denominator is positive and non-zero
        numerator = random.randint(0, 10)
        denominator = random.randint(1, 10)
        return f"\\boxed{{{numerator}/{denominator}}}"