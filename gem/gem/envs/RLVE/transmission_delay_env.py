from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TransmissionDelayEnv(Env):
    """
    Transmission Delay environment (single-turn Q&A).

    Problem:
      You are given a binary (0/1) array A of length N (1-indexed).
      You can generate a new array A' by the following operation:
        1) Choose a permutation P of 1, 2, ..., N such that for every i (1 ≤ i ≤ N), |i − P[i]| ≤ D.
        2) For every i (1 ≤ i ≤ N), set A′[i] = A[P[i]].
      Count the number of distinct arrays A′ that can be obtained by such operations.

    Answer format:
      The agent must output the final integer in \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 50,
        **kwargs
    ):
        """
        Initialize the TransmissionDelayEnv.

        Args:
            N: If provided, fixes the array length to this value (must be >= 4).
            min_N: Minimum N to sample when N is not fixed.
            max_N: Maximum N to sample when N is not fixed.
        """
        super().__init__()
        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Runtime state
        self.N: Optional[int] = None
        self.D: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a combinatorics/permutation constraint problem on a binary array.\n"
            "Please read the problem carefully and provide your final numeric answer.\n"
            "Output Format: Your final answer must be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            if self.fixed_N < 4:
                raise ValueError("N should be greater than or equal to 4")
            N = self.fixed_N
        else:
            if self.min_N > self.max_N:
                raise ValueError("min_N should be <= max_N")
            N = random.randint(max(4, self.min_N), self.max_N)

        # Generate A and D following the original logic, ensuring a valid, non-trivial instance
        A, D = self._generate_instance(N)

        # Compute the reference answer using the DP logic
        ref_answer = self._compute_reference_answer(A, D)

        # Store state
        self.N = N
        self.A = A
        self.D = D
        self.reference_answer = ref_answer

        # Build problem statement
        A_kv = ";".join(f"A[{i}]={val}" for i, val in enumerate(A, start=1))
        problem = (
            f"You are given a binary (0/1) array A of length {N} (1-indexed): {A_kv}\n\n"
            f"You can generate a new array A' by the following operation:\n"
            f"1) Choose a permutation P of 1, 2, ..., {N} such that for every i (1 ≤ i ≤ {N}), |i − P[i]| ≤ {D}.\n"
            f"2) For every i (1 ≤ i ≤ {N}), set A'[i] = A[P[i]].\n\n"
            f"Can you tell me the number of distinct arrays A' that can be obtained by such operations?\n\n"
            f"Output Format: Provide a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Validate the submitted answer and end the episode."""
        # Parse the boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and score
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if correct else 0.0

        info: Dict[str, Any] = {
            "correct": correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "D": self.D,
            "A": self.A,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content as the submitted answer."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        # Heuristic random guess
        random_answer = random.randint(1, 10**6)
        return f"\\boxed{{{random_answer}}}"

    def _generate_instance(self, N: int) -> Tuple[List[int], int]:
        """
        Generate a valid binary array A and distance D using the original logic:
          - A is non-trivial: number of ones is in [2, N-2]
          - D is sampled in [1, max_D] where max_D is derived from the positions of 0s and 1s
        """
        assert N >= 4, "N should be greater than or equal to 4"

        while True:
            zero_probability = random.random()
            A = [0 if random.random() < zero_probability else 1 for _ in range(N)]
            ones = sum(A)
            if not (2 <= ones <= N - 2):
                continue

            max_D = 0
            for c in (0, 1):
                indices = [i for i, x in enumerate(A, start=1) if x == c]
                # Left and right boundaries relative to the nearest c's
                max_D = max(max_D, max(indices[0] - 2, N - 1 - indices[-1]))
                if len(indices) > 1:
                    # Check gaps between consecutive c's
                    max_D = max(max_D, max((indices[i] - indices[i - 1] - 2) // 2 for i in range(1, len(indices))))
            if max_D >= 1:
                break

        D = random.randint(1, max_D)
        return A, D

    def _compute_reference_answer(self, A: List[int], D: int) -> int:
        """
        Compute the number of distinct arrays A' obtainable under the given constraints.
        This implements the DP logic from the original environment.
        """
        N = len(A)
        S = "".join(map(str, A))
        # 1-based indexing convenience
        S = " " + S

        # Collect positions of 0s and 1s; keep a dummy 0 at index 0
        p0 = [0]
        p1 = [0]
        for i in range(1, N + 1):
            if S[i] == '0':
                p0.append(i)
            else:
                p1.append(i)
        cnt0 = len(p0) - 1
        cnt1 = len(p1) - 1

        # DP table F with Python integers to avoid overflow
        # F[i][j]: number of ways from position i with j zeros and (L - j) ones already placed among last L positions
        F: List[List[int]] = [[0] * (cnt0 + 1) for _ in range(N + 2)]

        # Base case
        F[N + 1][0] = 1

        # Fill DP from i = N down to 1
        for i in range(N, 0, -1):
            L = N - i + 1
            j_min = max(0, L - cnt1)
            j_max = min(L, cnt0)

            for j in range(j_min, j_max + 1):
                k_ones = L - j
                total = 0

                # Place a '0' at position i
                if j > 0:
                    idx0 = cnt0 - j + 1  # the next remaining zero (from the end)
                    if abs(p0[idx0] - i) <= D:
                        total += F[i + 1][j - 1]

                # Place a '1' at position i
                if k_ones > 0:
                    idx1 = cnt1 - k_ones + 1  # the next remaining one (from the end)
                    if abs(p1[idx1] - i) <= D:
                        total += F[i + 1][j]

                F[i][j] = total

        answer = F[1][cnt0]
        if answer <= 0:
            # This should not happen given generation logic
            raise RuntimeError("Reference answer should be positive")
        return answer