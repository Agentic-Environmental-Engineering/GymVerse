import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SubsetSumEnv(Env):
    """Subset Sum problem environment - single-turn Q&A.

    The agent is given an array A of length N with positive integers.
    The goal is to output a subset of distinct indices whose corresponding
    values sum exactly to a specified target. The answer must be provided
    inside \\boxed{...}, with indices separated by spaces.
    """

    def __init__(self, N: int = 8, **kwargs):
        """
        Initialize the SubsetSumEnv instance.

        Args:
            N: Length of the array A. Must be >= 3.

        Note:
            This environment enforces single-turn interaction.
            Rewards:
              - Correct answer: 1.0
              - Wrong answer: 0.0
              - Format error: -0.1
        """
        super().__init__()
        self.N: int = N
        self.A: Optional[List[int]] = None
        self.target: Optional[int] = None
        self.reference_answer_indices: Optional[List[int]] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Subset Sum problem over indices of an array.\n"
            "Please provide your final answer in \\boxed{...} format.\n"
            "Inside the box, list the selected indices separated by spaces (e.g., \\boxed{0 3 5}).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Parameter validation
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 3, "N should be greater than or equal to 3"

        # Generate the array A with values in [1, N]
        self.A = [random.randint(1, self.N) for _ in range(self.N)]

        # Choose a random subset of indices of size in [2, N-1]
        k = random.randint(2, self.N - 1)
        self.reference_answer_indices = random.sample(range(self.N), k=k)

        # Compute the target as the sum of A over the chosen indices
        self.target = sum(self.A[i] for i in self.reference_answer_indices)

        # Build the problem prompt string
        A_str = " ".join(f"A[{i}]={val}" for i, val in enumerate(self.A))
        self.current_problem = (
            f"You are given an array A of length {self.N}, indexed from 0 to {self.N - 1}:\n"
            f"{A_str}\n\n"
            f"Please find a subset of distinct indices i1, i2, ..., ik such that the sum "
            f"A[i1] + A[i2] + ... + A[ik] is exactly equal to {self.target}.\n\n"
            f"Output Format: Your final answer should be a single line containing the selected "
            f"indices i1 i2 ... ik, separated by spaces, wrapped in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step by verifying the provided answer."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse indices from boxed content
        indices: List[int] = []
        tokens = boxed_content.strip().split()
        try:
            indices = [int(tok) for tok in tokens] if tokens else []
        except ValueError:
            # Non-integer content inside the box
            info = {"error": "invalid_indices_format"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate indices: must be within range and distinct
        invalid = False
        if not all(0 <= i < self.N for i in indices):
            invalid = True
        if len(indices) != len(set(indices)):
            invalid = True

        # Compute correctness
        is_correct = False
        if not invalid and self.A is not None and self.target is not None:
            subset_sum = sum(self.A[i] for i in indices)
            is_correct = (subset_sum == self.target)
        else:
            subset_sum = None

        reward = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "N": self.N,
            "array": self.A,
            "target": self.target,
            "user_indices": indices,
            "user_sum": subset_sum,
        }
        if invalid:
            info["error"] = "invalid_solution"

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by choosing a random valid subset of indices."""
        if self.N < 2:
            return r"\boxed{}"
        k = random.randint(0, self.N)  # allow empty subset occasionally
        subset = sorted(random.sample(range(self.N), k=k))
        content = " ".join(str(i) for i in subset)
        return f"\\boxed{{{content}}}"