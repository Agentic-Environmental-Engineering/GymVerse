from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class WeightedLISEnv(Env):
    """Weighted Longest Nondecreasing Subsequence environment - single-turn Q&A.

    Task:
      - Given arrays A and B of length N, choose a strictly increasing sequence of indices
        i1 < i2 < ... < ik such that A[i1] <= A[i2] <= ... <= A[ik].
      - Maximize the sum B[i1] + B[i2] + ... + B[ik].
      - Submit the chosen indices separated by spaces inside \\boxed{...}.
    """

    def __init__(
        self,
        N: int = 10,
        **kwargs
    ):
        super().__init__()
        assert N >= 1, "N should be greater than or equal to 1"
        self.N: int = N
        self.arrayA: Optional[List[int]] = None
        self.arrayB: Optional[List[int]] = None
        self.reference_sum: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given two arrays A and B of length N.\n"
            "Select a strictly increasing sequence of indices i1, i2, ..., ik such that:\n"
            "- 0 ≤ i1 < i2 < ... < ik < N\n"
            "- A[i1] ≤ A[i2] ≤ ... ≤ A[ik]\n"
            "Your goal is to maximize B[i1] + B[i2] + ... + B[ik].\n\n"
            "Answer format: Provide the selected indices separated by spaces inside \\boxed{...}.\n"
            "Example: \\boxed{0 2 3}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        N = self.N
        assert N >= 1, "N should be greater than or equal to 1"

        # Generate arrays A and B
        self.arrayA = [random.randint(0, N) for _ in range(N)]
        assert len(self.arrayA) == N, "A should have the same length as N"
        self.arrayB = [random.randint(1, N) for _ in range(N)]
        assert len(self.arrayB) == N, "B should have the same length as N"

        # Compute the maximum achievable sum using dynamic programming
        dpF = [0] * N
        for i in range(N):
            dpF[i] = self.arrayB[i]
            for j in range(i):
                if self.arrayA[j] <= self.arrayA[i]:
                    dpF[i] = max(dpF[i], dpF[j] + self.arrayB[i])
        self.reference_sum = max(dpF)
        assert self.reference_sum is not None and self.reference_sum > 0, "reference_sum should be greater than 0"

        # Build the problem description
        A_str = " ".join(f"A[{idx}]={val}" for idx, val in enumerate(self.arrayA))
        B_str = " ".join(f"B[{idx}]={val}" for idx, val in enumerate(self.arrayB))
        self.current_problem = (
            f"You are given two arrays `A` and `B`, each of length {N}. Their values are (indexing starts at 0):\n"
            f"{A_str}\n"
            f"{B_str}\n\n"
            f"Your task is to select a strictly increasing sequence of indices i1, i2, ..., ik such that:\n"
            f"- 0 ≤ i1 < i2 < ... < ik < {N}\n"
            f"- A[i1] ≤ A[i2] ≤ ... ≤ A[ik]\n"
            f"- Try your best to maximize the sum: B[i1] + B[i2] + ... + B[ik].\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single line containing the selected indices i1, i2, ..., ik, "
            f"separated by spaces, wrapped in \\boxed{{...}}.\n"
            f"Example: \\boxed{{0 2 3}}\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": N,
            "arrayA": self.arrayA,
            "arrayB": self.arrayB,
            "reference_sum": self.reference_sum,
        }

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the user's answer."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error: no boxed content found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse indices from boxed content
        tokens = boxed_content.strip().split()
        indices: List[int] = []
        try:
            for tok in tokens:
                indices.append(int(tok))
        except ValueError:
            # The boxed content is not a list of integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate indices and constraints
        N = self.N
        A = self.arrayA or []
        B = self.arrayB or []

        is_valid = True
        reason = None
        if any(not (0 <= idx < N) for idx in indices):
            is_valid = False
            reason = "index_out_of_range"

        if is_valid:
            for i in range(1, len(indices)):
                if not (indices[i - 1] < indices[i]):
                    is_valid = False
                    reason = "indices_not_strictly_increasing"
                    break

        if is_valid:
            for i in range(1, len(indices)):
                if not (A[indices[i - 1]] <= A[indices[i]]):
                    is_valid = False
                    reason = "A_not_nondecreasing_along_indices"
                    break

        user_sum = 0
        if is_valid:
            for idx in indices:
                user_sum += B[idx]

        # Check correctness
        correct = is_valid and (user_sum == self.reference_sum)
        reward: float = 1.0 if correct else 0.0

        info = {
            "correct": correct,
            "indices_valid": is_valid,
            "reason": reason,
            "reference_sum": self.reference_sum,
            "user_sum": user_sum,
            "selected_indices": indices,
            "N": N,
            "arrayA": A,
            "arrayB": B,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: random subset of indices (not necessarily valid)."""
        k = random.randint(0, self.N)  # may choose empty or full length
        indices = sorted(random.sample(range(self.N), k))
        # Randomly shuffle order to avoid always valid increasing sequence
        random.shuffle(indices)
        content = " ".join(str(i) for i in indices)
        return f"\\boxed{{{content}}}"