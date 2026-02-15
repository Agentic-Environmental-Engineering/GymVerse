from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LampChangingEnv(Env):
    """Lamp changing environment - single-turn Q&A.

    There are N lamps arranged in a circle. Each lamp's next state depends on its current state
    and the state of the next lamp clockwise:
    - If the two lamps have the same state, then the lamp will be OFF in the next moment.
    - If the two lamps have different states, then the lamp will be ON in the next moment.

    The task is to determine the state (ON/OFF) of a specific lamp at a given time.
    The answer must be provided as \\boxed{ON} or \\boxed{OFF}.
    """

    def __init__(self, max_n_t: int = 100, **kwargs):
        super().__init__()
        assert max_n_t >= 3, "max_n_t should be greater than or equal to 3"
        self.max_n_t: int = max_n_t

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.N: Optional[int] = None
        self.T: Optional[int] = None
        self.K: Optional[int] = None
        self.B: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a lamp state evolution problem on a circular arrangement.\n"
            "Please provide your final answer strictly in the format \\boxed{ON} or \\boxed{OFF}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Randomly target a reference answer to balance ON/OFF outcomes
        target_answer = random.choice(["ON", "OFF"])

        # Generate a valid instance where computed result matches the target answer
        while True:
            N = random.randint(3, self.max_n_t)
            on_probability = random.random()
            B = [1 if random.random() < on_probability else 0 for _ in range(N)]
            T = random.randint(2, self.max_n_t)
            K = random.randint(1, N)

            res = self._compute_lamp_state(B, N, K, T)  # 0 for OFF, 1 for ON
            result_str = ("OFF", "ON")[res]
            if result_str == target_answer:
                self.N, self.T, self.K, self.B = N, T, K, B
                self.reference_answer = result_str
                break

        situations = "; ".join(
            f"Lamp {i} is {'ON' if Bi else 'OFF'}" for i, Bi in enumerate(self.B, start=1)
        )

        self.current_problem = (
            f"There are {self.N} lamps arranged in a circle, labeled clockwise from 1 to {self.N}. "
            f"At each next moment, the state of each lamp depends on its current state and the state "
            f"of the next lamp in the clockwise direction:\n"
            f"- If the two lamps have the same state, then the lamp will be OFF in the next moment.\n"
            f"- If the two lamps have different states, then the lamp will be ON in the next moment.\n\n"
            f"The initial moment is time 0, and the initial states of all lamps are: {situations}\n"
            f"What's the state of lamp {self.K} at time {self.T} (Output either ON or OFF)?\n\n"
            f"Output Format: Provide your final answer as \\boxed{{ON}} or \\boxed{{OFF}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step by verifying the provided answer."""
        boxed = self._parse_answer(action)

        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        user_answer_raw = boxed.strip()
        user_answer = user_answer_raw.upper()

        if user_answer not in ("ON", "OFF"):
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "invalid_answer",
                "user_answer": user_answer_raw,
            }

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "T": self.T,
            "K": self.K,
            "B": self.B,
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

    def _compute_lamp_state(self, B: List[int], N: int, K: int, T: int) -> int:
        """Compute the state (0 for OFF, 1 for ON) of lamp K at time T using XOR aggregation.

        Uses the property that binomial coefficients modulo 2 correspond to submasks of T:
        After T steps, the state is the XOR of initial states at positions (K + i) for all i such that i is a submask of T.
        """
        res = 0
        for i in range(T + 1):
            if (T & i) == i:
                res ^= B[(i + K - 1) % N]
        return res

    def sample_random_action(self) -> str:
        """Sample a random valid action."""
        return f"\\boxed{{{random.choice(['ON', 'OFF'])}}}"