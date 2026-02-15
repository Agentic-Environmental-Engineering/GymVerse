import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PipelineArrangementEnv(Env):
    """Two-machine flow shop scheduling environment (single-turn Q&A).
    
    The task is to determine an ordering (permutation) of N products, each processed
    first on machine A then on machine B. Machine A processes products in the chosen
    order, while machine B processes them in the order they complete on machine A.
    The objective is to minimize the completion time of the last product on machine B.
    
    The answer must be provided inside \\boxed{...} and should be the indices of the
    products in the chosen order, separated by spaces.
    """

    def __init__(
        self,
        N: int = 5,
        **kwargs
    ):
        """Initialize the environment with problem size N.
        
        Parameters:
            N: Number of products (must be >= 2).
        """
        super().__init__()
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")
        self.N: int = N

        # Problem state
        self.A: List[int] = []
        self.B: List[int] = []
        self.reference_order: List[int] = []
        self.reference_finishing_time: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a two-machine flow shop scheduling problem.\n"
            "Provide a permutation of product indices that minimizes the completion time on machine B.\n"
            "Answer format: place the permutation inside \\boxed{...}, with indices separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate processing times
        self.A = [random.randint(1, self.N) for _ in range(self.N)]
        self.B = [random.randint(1, self.N) for _ in range(self.N)]

        # Compute reference order using Johnson's rule
        tasks: List[Tuple[int, int, int]] = []
        for i in range(self.N):
            if self.A[i] < self.B[i]:
                tasks.append((self.A[i], 0, i))
            else:
                tasks.append((self.B[i], 1, i))

        tasks.sort(key=lambda x: x[0])

        order: List[Optional[int]] = [None] * self.N
        left, right = 0, self.N - 1
        for time, belong, idx in tasks:
            if belong == 0:
                order[left] = idx
                left += 1
            else:
                order[right] = idx
                right -= 1

        # Finalize reference order and finishing time
        self.reference_order = [idx for idx in order if idx is not None]
        self.reference_finishing_time = self._get_finishing_time(self.reference_order)

        # Build problem statement
        times_str = "\n".join(
            f"A[{i}]={self.A[i]}, B[{i}]={self.B[i]}"
            for i in range(self.N)
        )
        self.current_problem = (
            f"You need to process {self.N} products labeled from 0 to {self.N - 1}. "
            f"Each product must go through two machines, A and B, in order.\n\n"
            f"The processing times for each product on machines A and B are given as:\n"
            f"{times_str}\n\n"
            f"Please determine a permutation (ordering) of all products to minimize the time when "
            f"the last product finishes on machine B.\n\n"
            f"Output Format: Your final answer should be a single line containing the indices of the products "
            f"in the chosen order, separated by spaces, inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse permutation from boxed content
        tokens = boxed_content.replace(",", " ").split()
        try:
            order = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate permutation
        if len(order) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "wrong_length"}
        if len(set(order)) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "duplicates"}
        if not all(0 <= i < self.N for i in order):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "out_of_range"}

        # Compute finishing time for user's order
        user_time = self._get_finishing_time(order)
        is_correct = (user_time == self.reference_finishing_time)

        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_order": self.reference_order,
            "reference_finishing_time": self.reference_finishing_time,
            "user_order": order,
            "user_finishing_time": user_time,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _get_finishing_time(self, order: List[int]) -> int:
        """Compute the finishing time on machine B given a processing order."""
        tA = 0
        tB = 0
        for idx in order:
            tA += self.A[idx]
            if tB < tA:
                tB = tA
            tB += self.B[idx]
        return tB

    def sample_random_action(self) -> str:
        """Sample a random valid permutation action in \\boxed{...} format."""
        perm = list(range(self.N))
        random.shuffle(perm)
        return f"\\boxed{{{' '.join(map(str, perm))}}}"