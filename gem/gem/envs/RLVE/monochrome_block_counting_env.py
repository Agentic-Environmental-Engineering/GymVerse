from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MonochromeBlockCountingEnv(Env):
    """Monochrome Block Counting environment - single-turn Q&A.

    Task:
      - Build a tower of layers where the i-th layer has exactly i blocks.
      - Each layer is monochrome (all black or all white).
      - Use at most A black blocks and at most B white blocks in total.
      - Build a tower with the maximum possible number of layers N under these constraints.
      - Compute the total number of distinct ways to build such a tower with the maximum number of layers.

    Answer format:
      - The final answer should be provided in \\boxed{...}.
    """

    def __init__(
        self,
        max_a_b: int = 100,
        **kwargs
    ):
        super().__init__()
        if max_a_b < 1:
            raise ValueError("max_a_b must be >= 1")
        self.max_a_b = max_a_b

        # Problem state
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.T: Optional[int] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are building a tower of blocks with the following rules:\n"
            "- The i-th layer (from top to bottom) must contain exactly i blocks (i is from 1 to N if the tower has N layers).\n"
            "- All blocks in the same layer must be of the same color: either black or white.\n"
            "- You may use at most A black blocks and at most B white blocks in total.\n"
            "- You should build a tower with the maximum possible number of layers (N) under these constraints.\n\n"
            "Please compute the total number of distinct ways to build such a tower with the maximum number of layers.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample A and B
        self.A = random.randint(1, self.max_a_b)
        self.B = random.randint(1, self.max_a_b)

        # Compute the maximum number of layers T such that 1 + 2 + ... + T <= A + B
        total_blocks_available = self.A + self.B
        T = 0
        while (T + 1) * (T + 2) // 2 <= total_blocks_available:
            T += 1
        self.T = T
        total_blocks_in_tower = T * (T + 1) // 2

        # Count number of ways to select which layers are black (subset of {1..T})
        # such that total black blocks <= A and total white blocks <= B.
        # Let j be the total number of black blocks (sum of chosen layer sizes).
        # Constraint: j <= A and total_blocks_in_tower - j <= B => j >= total_blocks_in_tower - B.
        # DP F[j] = number of subsets of {1..T} with sum j, for j in [0..A].
        F = [0] * (self.A + 1)
        F[0] = 1
        for i in range(1, T + 1):
            for j in range(self.A, i - 1, -1):
                F[j] += F[j - i]

        lower = max(total_blocks_in_tower - self.B, 0)
        upper = self.A
        self.reference_answer = sum(F[j] for j in range(lower, upper + 1))

        # Build problem statement
        self.current_problem = (
            "You are building a tower of blocks with the following rules:\n"
            f"- The i-th layer (from top to bottom) must contain exactly i blocks (i is from 1 to N if the tower has N layers).\n"
            "- All blocks in the same layer must be of the same color: either black or white.\n"
            f"- You may use at most {self.A} black blocks and at most {self.B} white blocks in total.\n"
            "- You should build a tower with the maximum possible number of layers (N) under these constraints.\n\n"
            "Please compute the total number of distinct ways to build such a tower with the maximum number of layers.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Interpret as integer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "invalid_answer"}

        # Check correctness
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "A": self.A,
            "B": self.B,
            "T": self.T,
            "total_blocks_in_tower": None if self.T is None else self.T * (self.T + 1) // 2,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...}."""
        # Randomly sample a non-negative integer as a guess
        guess = random.randint(0, 10**6)
        return f"\\boxed{{{guess}}}"