import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from bisect import bisect_left
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RangeConstrained_IncreasingSequence_CountingEnv(Env):
    """
    Environment for counting integer sequences under range constraints with strictly increasing
    non-zero elements. Single-turn Q&A in GEM format.

    Task:
    - Given arrays L and R of length N, count the number of sequences A[0..N-1] such that:
      * For each i, A[i] is either 0 or an integer in [L[i], R[i]].
      * At least one A[i] is greater than 0.
      * All non-zero A[i] form a strictly increasing sequence in order.
    - Return the count modulo MOD.
    """

    def __init__(
        self,
        min_n: int = 2,
        max_n: int = 10,
        modulo: int = 10**9 + 7,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            min_n: Minimum length of the sequence N (must be >= 2).
            max_n: Maximum length of the sequence N (must be >= min_n).
            modulo: Modulo for the answer (default 1e9+7).
        """
        super().__init__()
        if min_n < 2:
            raise ValueError("min_n should be greater than or equal to 2")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        self.min_n = min_n
        self.max_n = max_n
        self.MOD = modulo

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.L: Optional[List[int]] = None
        self.R: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics counting problem.\n"
            "Provide ONLY the final integer answer inside \\boxed{...}.\n"
            "For example: \\boxed{123}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem.

        Returns:
            observation: The problem statement as a string.
            info: Additional info dictionary (empty for this environment).
        """
        super().reset(seed)

        # Generate N and ranges L, R
        N = random.randint(self.min_n, self.max_n)
        assert N >= 2, "N should be greater than or equal to 2"
        L = [random.randint(1, N * N) for _ in range(N)]
        R = [random.randint(Li, N * N) for Li in L]

        self.N = N
        self.L = L
        self.R = R

        # Build problem statement
        L_and_R_lines = "\n".join(f"L[{i}]={Li} R[{i}]={Ri}" for i, (Li, Ri) in enumerate(zip(L, R)))
        self.current_problem = (
            f"Count the number of integer sequences A[0], A[1], ..., A[{N-1}] of length {N} such that:\n"
            f"- For each A[i], it is either 0 or an integer in [L[i], R[i]]\n"
            f"- At least one A[i] is greater than 0\n"
            f"- All non-zero A[i] form a strictly increasing sequence in order (i.e., if A[i] > 0 and A[j] > 0 with i < j, then A[i] < A[j])\n\n"
            f"The bounds L[i] and R[i] for each position are given as:\n"
            f"{L_and_R_lines}\n\n"
            f"Output the number of such sequences modulo {self.MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(N, L, R, self.MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, N: int, L: List[int], R: List[int], MOD: int) -> int:
        """
        Compute the reference answer using the original algorithm:
        Coordinate compression + DP with binomial-like coefficients per segment.
        """
        A, B = L.copy(), R.copy()
        coords: List[int] = []
        for ai, bi in zip(A, B):
            coords.append(ai)
            coords.append(bi + 1)

        # Coordinate compression
        coords = sorted(set(coords))
        tot = len(coords)
        for i in range(N):
            A[i] = bisect_left(coords, A[i])
            B[i] = bisect_left(coords, B[i] + 1)

        # Precompute modular inverses up to N
        inv = [0] * (N + 1)
        if N >= 1:
            inv[1] = 1
        for i in range(2, N + 1):
            inv[i] = (MOD - MOD // i) * inv[MOD % i] % MOD

        # DP arrays
        # C[k] will hold binomial-like coefficients for each segment length
        C = [0] * (N + 1)
        # g[k] is the number of ways ending with the k-th chosen position (k from 0 to N)
        g = [0] * (N + 1)
        g[0] = 1  # base: no position chosen yet

        # Process each compressed segment j
        for j in range(tot - 1):
            length = coords[j + 1] - coords[j]
            if length <= 0:
                continue
            # Build C array: C[k] = C(length + k - 1, k)
            C[0] = 1
            for k in range(1, N + 1):
                C[k] = C[k - 1] * (length + k - 1) % MOD * inv[k] % MOD

            # Update DP in reverse order to avoid overwriting
            for i in range(N, 0, -1):
                # If position i-1 can cover this segment
                if A[i - 1] <= j < B[i - 1]:
                    f = 0
                    m = 1
                    c_val = length % MOD
                    # Sum contributions from previous states
                    for p in range(i - 1, -1, -1):
                        f = (f + c_val * g[p]) % MOD
                        # If previous position (p-1) also covers, increase combination size
                        if p > 0 and A[p - 1] <= j < B[p - 1]:
                            m += 1
                            c_val = C[m]
                    g[i] = (g[i] + f) % MOD

        # Sum all ways where at least one position participates
        return sum(g[1:]) % MOD

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute a single step by parsing and checking the submitted answer.

        Args:
            action: The agent's response text, expected to contain \\boxed{...} with an integer.

        Returns:
            observation: TERMINAL_STATE for single-turn environment.
            reward: 1.0 if correct, 0.0 if incorrect, -0.1 if format error.
            terminated: Always True for single-turn.
            truncated: Always False for this environment.
            info: Dictionary with evaluation details.
        """
        # Parse the boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate integer
        try:
            user_answer = int(boxed.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check (must be within [0, MOD))
        if not (0 <= user_answer < self.MOD):
            is_correct = False
            reward = 0.0
            info = {
                "correct": is_correct,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "error": "out_of_range"
            }
            return TERMINAL_STATE, reward, True, False, info

        # Compare to reference
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer
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

    def sample_random_action(self) -> str:
        """Sample a random action by returning a random integer in boxed format."""
        random_answer = random.randint(0, self.MOD - 1)
        return f"\\boxed{{{random_answer}}}"