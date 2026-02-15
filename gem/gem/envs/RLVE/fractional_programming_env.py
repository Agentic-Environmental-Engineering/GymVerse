from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FractionalProgrammingEnv(Env):
    """Fractional programming environment - single-turn Q&A.

    Task:
      - Given arrays A and B of length N, select K distinct indices to maximize
        (sum A[i]) / (sum B[i]) over the selected indices.

    Answer format:
      - Provide the selected indices separated by spaces inside \\boxed{...}.
        Example: \\boxed{0 3 5}
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 50,
        max_proportion: int = 2,
        **kwargs
    ):
        super().__init__()
        # Parameter configuration
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if max_proportion < 1:
            raise ValueError("max_proportion should be at least 1")

        self.N_fixed: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N
        self.max_proportion: int = max_proportion

        # Current problem state
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.B: Optional[List[int]] = None
        self.reference_indices: Optional[List[int]] = None
        self.gold_SumA: Optional[int] = None
        self.gold_SumB: Optional[int] = None

        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a fractional programming selection problem.\n"
            "Given two arrays A and B and an integer K, select K distinct indices to maximize\n"
            "the ratio (sum of selected A[i]) / (sum of selected B[i]).\n"
            "Output Format: Provide the selected indices separated by spaces inside \\boxed{...}.\n"
            "Example: \\boxed{0 3 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            N = self.N_fixed
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Generate K, A, B
        K = random.randint(2, N - 1)
        B = [random.randint(1, N) for _ in range(N)]
        A = [random.randint(1, self.max_proportion * b) for b in B]

        # Binary search to find optimal ratio and a corresponding set of indices
        l, r = 0.0, max(a / b for a, b in zip(A, B))
        solution: Optional[List[int]] = None
        for _ in range(256):
            mid = (l + r) / 2.0
            indices = list(range(N))
            indices.sort(key=lambda idx: A[idx] - mid * B[idx], reverse=True)
            if sum(A[idx] - mid * B[idx] for idx in indices[:K]) >= 0:
                l = mid
                solution = indices[:K].copy()
            else:
                r = mid

        if solution is None:
            # Fallback (should not happen with the above construction)
            indices = list(range(N))
            indices.sort(key=lambda idx: A[idx] / B[idx], reverse=True)
            solution = indices[:K]

        gold_SumA = sum(A[i] for i in solution)
        gold_SumB = sum(B[i] for i in solution)

        # Store state
        self.N = N
        self.K = K
        self.A = A
        self.B = B
        self.reference_indices = solution
        self.gold_SumA = gold_SumA
        self.gold_SumB = gold_SumB

        # Build problem statement
        lines = "\n".join(f"A[{i}]={A[i]} B[{i}]={B[i]}" for i in range(N))
        self.current_problem = (
            f"You are given two arrays A and B, each containing {N} integers:\n"
            f"{lines}\n\n"
            f"Please select {K} distinct indices i_1, ..., i_{K} to maximize "
            f"(A[i_1] + ... + A[i_{K}]) / (B[i_1] + ... + B[i_{K}]).\n\n"
            f"Output Format: Your final answer should be the {K} selected indices in any order, "
            f"separated by spaces, inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take one step by validating the proposed selection of indices."""
        if self.A is None or self.B is None or self.K is None or self.N is None:
            # Environment not properly reset
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_initialized"}

        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to list of integers
        try:
            selected_indices = [int(x) for x in boxed_content.strip().split()]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate solution constraints
        if len(selected_indices) != self.K:
            info = {"error": "invalid_solution", "reason": "wrong_number_of_indices"}
            return TERMINAL_STATE, 0.0, True, False, info
        if not all(0 <= idx < self.N for idx in selected_indices):
            info = {"error": "invalid_solution", "reason": "index_out_of_range"}
            return TERMINAL_STATE, 0.0, True, False, info
        if len(set(selected_indices)) != len(selected_indices):
            info = {"error": "invalid_solution", "reason": "duplicate_indices"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute sums
        answer_SumA = sum(self.A[idx] for idx in selected_indices)
        answer_SumB = sum(self.B[idx] for idx in selected_indices)

        # Compare with optimal ratio using cross multiplication to avoid precision issues
        gold_SumA = self.gold_SumA if self.gold_SumA is not None else 0
        gold_SumB = self.gold_SumB if self.gold_SumB is not None else 1  # safe fallback

        # Determine correctness: ratios equal implies optimal
        is_correct = (answer_SumA * gold_SumB) == (answer_SumB * gold_SumA)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "user_indices": selected_indices,
            "user_SumA": answer_SumA,
            "user_SumB": answer_SumB,
            "gold_SumA": gold_SumA,
            "gold_SumB": gold_SumB,
            "reference_indices": self.reference_indices,
            "N": self.N,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action in the required format."""
        if self.N is None or self.K is None:
            # If not initialized, provide a generic placeholder action
            return "\\boxed{0 1}"
        indices = random.sample(range(self.N), self.K)
        indices_str = " ".join(map(str, indices))
        return f"\\boxed{{{indices_str}}}"