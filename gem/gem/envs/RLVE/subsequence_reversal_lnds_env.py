import random
from typing import Any, List, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SubsequenceReversalLNDSEnv(Env):
    """Environment for the Subsequence Reversal to Maximize LNDS problem (single-turn Q&A)."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 15,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed length of the sequence. If None, a random N in [min_N, max_N] will be used.
        - min_N: Minimum allowed length for random N (at least 4).
        - max_N: Maximum allowed length for random N (must be >= min_N).
        """
        super().__init__()
        self.fixed_N: Optional[int] = N
        self.min_N: int = max(4, min_N)
        self.max_N: int = max(max_N, self.min_N)
        if self.fixed_N is not None:
            assert self.fixed_N >= 4, "N should be greater than or equal to 4"

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_N: Optional[int] = None
        self.current_A: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a sequence optimization problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n"
            "Do not include any extra text besides the boxed answer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
        assert N >= 4, "N should be greater than or equal to 4"
        self.current_N = N

        # Generate sequence A with values in [1..N]
        A = [random.randint(1, N) for _ in range(N)]
        self.current_A = A[:]

        # Compute reference answer using the original DP algorithm
        self.reference_answer = self._compute_reference_answer(A)

        # Build problem statement
        A_str = ", ".join(f"A[{i}]={v}" for i, v in enumerate(A, start=1))
        problem = (
            f"You are given a sequence A of {N} integers: {A_str}\n"
            "You may choose a subsequence of A, defined by a strictly increasing sequence of indices i₁, ..., iₖ "
            f"(1 ≤ i₁ < ... < iₖ ≤ {N}, k >= 1), and reverse the order of the elements at those indices "
            "(i.e., A[i₁] becomes A[iₖ], ..., A[iₖ] becomes A[i₁]). "
            "Please maximize the length of the longest non-decreasing subsequence (not necessarily contiguous) in the resulting array. "
            "Output a single integer — the maximum achievable length.\n\n"
            "Output Format: Provide only the integer answer in \\boxed{...}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, arr: List[int]) -> int:
        """Compute the maximum achievable LNDS length after reversing a subsequence."""
        N = len(arr)

        # Use 1-indexed array with a leading zero
        A = [0] + arr
        M = max(A)

        # dp[l][r][L][R]: max LIS length in A[l..r] after reversing at most one subsequence,
        # considering only values in [L..R]
        dp = [[[[0] * (M + 2) for _ in range(M + 2)] for _ in range(N + 2)] for _ in range(N + 2)]

        # Base case: intervals of length 1
        for i in range(1, N + 1):
            for L in range(1, A[i] + 1):
                for R in range(A[i], M + 1):
                    dp[i][i][L][R] = 1

        # Build up for intervals of length = 2..N
        for length in range(2, N + 1):
            for l in range(1, N - length + 2):
                r = l + length - 1
                for span in range(1, M + 1):
                    for L in range(1, M - span + 2):
                        R = L + span - 1

                        # 1) shrink the allowed value range
                        val = dp[l][r][L + 1][R]
                        if dp[l][r][L][R - 1] > val:
                            val = dp[l][r][L][R - 1]

                        # 2) extend by taking A[l] at the left (if it matches L)
                        tmp = dp[l + 1][r][L][R] + (1 if A[l] == L else 0)
                        if tmp > val:
                            val = tmp

                        # 3) extend by taking A[r] at the right (if it matches R)
                        tmp = dp[l][r - 1][L][R] + (1 if A[r] == R else 0)
                        if tmp > val:
                            val = tmp

                        # 4) reverse a subsequence spanning the ends
                        tmp = dp[l + 1][r - 1][L][R]
                        if A[l] == R:
                            tmp += 1
                        if A[r] == L:
                            tmp += 1
                        if tmp > val:
                            val = tmp

                        dp[l][r][L][R] = val

        # The answer is dp[1][N][1][M]
        return dp[1][N][1][M]

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        parsed = self._parse_answer(action)

        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == user_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_N,
            "A": self.current_A,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re

        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (answer) in boxed format."""
        upper = self.current_N if self.current_N is not None else self.max_N
        guess = random.randint(0, max(1, upper))
        return f"\\boxed{{{guess}}}"