from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BAJBytecomputerEnv(Env):
    """BAJ Bytecomputer environment - single-turn Q&A.

    Task:
    - You are given an array X of length N with elements in {-1, 0, 1}.
    - Operation: choose an index i (1 ≤ i < N), and update X[i + 1] := X[i + 1] + X[i].
    - Goal: make X non-decreasing (X[1] ≤ X[2] ≤ ... ≤ X[N]) using the minimum number of operations.
    - Answer format: return the minimum number of operations in \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 20,
        **kwargs
    ):
        super().__init__()
        # Parameter configuration
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.N_fixed: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Runtime state
        self.N: Optional[int] = None
        self.X: Optional[List[int]] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a minimum-operations array transformation problem.\n"
            "Operation allowed: choose i with 1 ≤ i < N and set X[i + 1] := X[i + 1] + X[i].\n"
            "Goal: make X non-decreasing with the minimum number of operations.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Choose N
        if self.N_fixed is not None:
            N = self.N_fixed
        else:
            N = random.randint(self.min_N, self.max_N)

        # Generate a valid instance and compute the reference answer
        while True:
            distribution = [random.randint(1, N) for _ in range(3)]
            X = [random.choices([-1, 0, 1], weights=distribution)[0] for _ in range(N)]

            ans = self._compute_min_operations(X)
            # Ensure the instance is solvable within the considered bounds
            if ans is not None:
                break

        self.N = N
        self.X = X
        self.reference_answer = ans

        # Build problem prompt
        X_str = ", ".join(f"X[{i + 1}]={v}" for i, v in enumerate(X))
        self.current_problem = (
            f"You are given an array X of length {N}, where each element is initially -1, 0, or +1: {X_str}\n"
            f"You may perform the following operation any number of times: choose an index i (1 ≤ i < {N}), "
            f"and update X[i + 1] := X[i + 1] + X[i]. "
            f"Your goal is to make the array non-decreasing, i.e., X[1] ≤ X[2] ≤ ... ≤ X[{N}]; "
            f"please output the minimum number of operations required to achieve this.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_min_operations(self, X: List[int]) -> Optional[int]:
        """Compute the minimum number of operations required using dynamic programming."""
        N = len(X)
        # At most 2 operations per transition, so upper bound is 2*N + buffer
        INF = 2 * N + 5

        # Possible values are -1, 0, 1
        val = [-1, 0, 1]

        prev = [INF] * 3
        prev[X[0] + 1] = 0

        for i in range(1, N):
            curr = [INF] * 3
            x = X[i]
            for j in range(3):
                ops_so_far = prev[j]
                if ops_so_far >= INF:
                    continue
                prev_val = val[j]

                # 0 operations on x
                new_x = x
                if new_x >= prev_val:
                    curr[new_x + 1] = min(curr[new_x + 1], ops_so_far)

                # 1 operation on x: x + prev_val
                new_x = x + prev_val
                if -1 <= new_x <= 1 and new_x >= prev_val:
                    curr[new_x + 1] = min(curr[new_x + 1], ops_so_far + 1)

                # 2 operations on x: x + 2 * prev_val
                new_x = x + 2 * prev_val
                if -1 <= new_x <= 1 and new_x >= prev_val:
                    curr[new_x + 1] = min(curr[new_x + 1], ops_so_far + 2)

            prev = curr

        ans = min(prev)
        if ans < INF:
            return ans
        return None

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the provided answer."""
        # Parse boxed answer
        content = self._parse_answer(action)
        if content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(content)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "X": self.X,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        # A rough range for possible answers; the DP used INF ~ 2*N + 5
        N = self.N if self.N is not None else (self.N_fixed or self.max_N)
        upper = 2 * max(3, N) + 5
        guess = random.randint(0, upper)
        return f"\\boxed{{{guess}}}"