from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class InversionPairEnv(Env):
    """Environment for the Inversion Pair problem - single-turn Q&A.

    Task:
    - Given two arrays A and B of length N with distinct integers within each array.
    - You may swap adjacent elements in either A or B any number of times.
    - The goal is to minimize the sum of squared differences between corresponding elements:
      (A[0] - B[0])^2 + ... + (A[N-1] - B[N-1])^2.
    - Among all ways to achieve the minimum possible sum, output the minimum number of adjacent swaps needed.

    Answer format:
    - The answer must be a single integer wrapped in \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 30,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
        - N: If provided, fixes the array length to this value (must be >= 3).
        - min_n: Minimum N when sampling randomly (inclusive, must be >= 3).
        - max_n: Maximum N when sampling randomly (inclusive, must be >= min_n).
        """
        super().__init__()
        if N is not None:
            if N < 3:
                raise AssertionError("N should be greater than or equal to 3")
        else:
            if min_n < 3:
                raise AssertionError("min_n should be greater than or equal to 3")
            if min_n > max_n:
                raise AssertionError("min_n should be less than or equal to max_n")

        self.fixed_N: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.B: Optional[List[int]] = None
        self.N_used: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return global task instructions."""
        return (
            "You are solving an array reordering optimization problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
        if N < 3:
            raise AssertionError("N should be greater than or equal to 3")
        self.N_used = N

        # Generate arrays with distinct integers within each array
        A = random.sample(range(2 * N), N)
        B = random.sample(range(2 * N), N)
        self.A = A
        self.B = B

        # Compute reference answer using the inversion count of the rank mapping
        # Step 1: indices sorted by values for A and B
        a_idx = list(range(N))
        b_idx = list(range(N))
        a_idx.sort(key=lambda i: A[i])
        b_idx.sort(key=lambda i: B[i])

        # Step 2: l[i] is the rank of A[i] in B's sorted order
        l = [0] * N
        for rank in range(N):
            l[a_idx[rank]] = b_idx[rank]

        # Step 3: count inversions in l using a BIT (Fenwick Tree)
        BIT = [0] * (N + 1)

        def add(pos: int, val: int) -> None:
            while pos <= N:
                BIT[pos] += val
                pos += pos & -pos

        def query(pos: int) -> int:
            s = 0
            while pos > 0:
                s += BIT[pos]
                pos -= pos & -pos
            return s

        ans = 0
        for i in range(N - 1, -1, -1):
            pos = l[i] + 1  # 1-indexed position in BIT
            ans += query(pos - 1)
            add(pos, 1)

        self.reference_answer = ans

        # Build the problem description
        arr_A_str = " ".join(f"A[{i}]={val}" for i, val in enumerate(A))
        arr_B_str = " ".join(f"B[{i}]={val}" for i, val in enumerate(B))

        self.current_problem = (
            f"You are given two arrays A and B, each containing {N} distinct integers:\n"
            f"{arr_A_str}\n"
            f"{arr_B_str}\n\n"
            "You may perform the following operation any number of times: "
            "Swap two adjacent elements (i.e., elements at indices i and i+1) in either A or B.\n"
            "Your goal is to minimize the sum: "
            f"(A[0] - B[0])^2 + (A[1] - B[1])^2 + ... + (A[{N-1}] - B[{N-1}])^2.\n"
            "Among all ways to achieve the minimum possible sum, please output the minimum number of adjacent swaps needed.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided answer and return the result."""
        answer_text = self._parse_answer(action)

        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        correct = (self.reference_answer == user_answer)
        reward = 1.0 if correct else 0.0

        info = {
            "correct": correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N_used,
            "A": self.A,
            "B": self.B,
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
        """Sample a random action in the required \\boxed{...} format."""
        n = self.N_used if self.N_used is not None else (self.fixed_N if self.fixed_N is not None else self.min_n)
        # Maximum number of inversions for length n
        max_inv = n * (n - 1) // 2
        random_answer = random.randint(0, max_inv if max_inv > 0 else 0)
        return f"\\boxed{{{random_answer}}}"