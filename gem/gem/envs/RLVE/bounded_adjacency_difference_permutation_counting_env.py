from typing import Any, Optional, SupportsFloat, Tuple
import random
from itertools import permutations
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BoundedAdjacencyDifference_Permutation_CountingEnv(Env):
    """
    Environment for counting permutations of 1..N such that the absolute difference
    between any two adjacent elements is at most K. Single-turn Q&A.
    """

    def __init__(
        self,
        min_n: int = 4,
        max_n: int = 50,
        k_min: int = 2,
        k_max: int = 4,
        fixed_n: Optional[int] = None,
        fixed_k: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            min_n: Minimum value of N to sample (inclusive). Must be >= 4.
            max_n: Maximum value of N to sample (inclusive). Must be >= min_n.
            k_min: Minimum value of K to sample (inclusive). Must be >= 2.
            k_max: Maximum value of K to sample (inclusive). Must be <= 4.
            fixed_n: If provided, use this fixed N instead of sampling.
            fixed_k: If provided, use this fixed K (must satisfy constraints with N).
        """
        super().__init__()

        if min_n < 4:
            raise ValueError("min_n must be at least 4.")
        if max_n < min_n:
            raise ValueError("max_n must be >= min_n.")
        if k_min < 2:
            raise ValueError("k_min must be at least 2.")
        if k_max > 4:
            raise ValueError("k_max must be at most 4.")
        if k_min > k_max:
            raise ValueError("k_min must be <= k_max.")
        if fixed_n is not None and fixed_n < 4:
            raise ValueError("fixed_n must be at least 4.")

        # Ensure feasibility: for any sampled N, we need k_min <= N - 2
        if min_n < k_min + 2:
            raise ValueError("min_n must be at least k_min + 2 to ensure feasible K selection.")

        self.min_n = min_n
        self.max_n = max_n
        self.k_min = k_min
        self.k_max = k_max
        self.fixed_n = fixed_n
        self.fixed_k = fixed_k

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics counting problem.\n"
            "Task: Count the number of permutations of 1, 2, ..., N such that for every two adjacent elements, "
            "the absolute difference between them is at most K.\n"
            "Output Format: Provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new single-turn problem."""
        super().reset(seed)

        # Choose N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)

        if N < 4:
            raise ValueError("N must be at least 4.")

        # Choose K with constraints: 2 <= K <= min(4, N - 2)
        k_high = min(self.k_max, N - 2)
        if k_high < self.k_min:
            raise ValueError("No feasible K: ensure N >= k_min + 2 and k_max <= 4.")
        if self.fixed_k is not None:
            if not (self.k_min <= self.fixed_k <= k_high):
                raise ValueError(f"fixed_k must be between {self.k_min} and {k_high} for the chosen N={N}.")
            K = self.fixed_k
        else:
            K = random.randint(self.k_min, k_high)

        # Compute reference answer
        ans = self._count_permutations(N, K)
        if ans <= 0:
            raise RuntimeError("The computed answer should be positive.")

        self.N = N
        self.K = K
        self.reference_answer = ans

        # Build problem prompt
        problem_text = (
            f"What is the number of permutations of 1, 2, ..., {N} such that for every two adjacent elements "
            f"(i.e., the i-th and (i+1)-th elements for all 1 <= i < {N}), "
            f"the absolute difference between them is at most {K}?\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        # Parse answer from boxed format
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if correct else 0.0

        info = {
            "correct": correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        # Random guess in a broad range
        random_answer = random.randint(0, 10**9)
        return f"\\boxed{{{random_answer}}}"

    def _count_permutations(self, N: int, K: int) -> int:
        """
        Count permutations of 1..N where any two adjacent elements differ by at most K.
        Implements the original DP algorithm specialized for small K (K <= 4).
        """
        # Precompute factorials up to K (K <= 4)
        FACT = [1] * (K + 1)
        for i in range(1, K + 1):
            FACT[i] = FACT[i - 1] * i
        FK = FACT[K]  # K!

        # All permutations of length K in lexicographic order
        PERMS = [list(p) for p in permutations(range(K))]
        TM = (1 << (K + 1)) - 1  # mask with (K+1) ones

        # DP over i (size), ip (permutation index), ic (mask)
        prev = [[0] * (TM + 1) for _ in range(FK)]
        for ip in range(FK):
            prev[ip][TM] = 1  # base: i = K

        for i in range(K + 1, N + 1):
            cur = [[0] * (TM + 1) for _ in range(FK)]
            for ip in range(FK):
                tp = PERMS[ip]  # current permutation of size K
                for ic in range(TM + 1):
                    val = prev[ip][ic]
                    if not val:
                        continue
                    # Try to insert the new maximum at each available slot j
                    for j in range(K + 1):
                        if ((ic >> j) & 1) == 0:
                            continue

                        # Insert into permutation representation
                        ttp_ins = tp[:j] + [K] + tp[j:]           # length K+1, values in {0..K}
                        l0 = ttp_ins.index(0)                      # first position of '0'
                        ttp_trim = ttp_ins[:l0] + ttp_ins[l0 + 1:] # remove that '0'
                        ttp = [x - 1 for x in ttp_trim]            # now a perm of {0..K-1}

                        # Update slot mask
                        tc_bits = [ (ic >> l) & 1 for l in range(K + 1) ]
                        ttc2 = tc_bits[:j] + [1] + tc_bits[j:]     # insert a '1' at j
                        # remove index l0+1 and then clear index l0
                        ttc_removed = ttc2[:l0 + 1] + ttc2[l0 + 2:]
                        ttc_removed[l0] = 0
                        icc = 0
                        for l in range(K + 1):
                            if ttc_removed[l]:
                                icc |= (1 << l)

                        # Lehmer code -> permutation index 'ipp'
                        ipp = 0
                        seen = [0] * K
                        for pos in range(K):
                            v = ttp[pos]
                            ch = 0
                            for z in range(v):
                                if seen[z] == 0:
                                    ch += 1
                            seen[v] = 1
                            ipp += ch * FACT[K - 1 - pos]

                        cur[ipp][icc] += val
            prev = cur

        ans = 0
        for ip in range(FK):
            for ic in range(TM + 1):
                ans += prev[ip][ic]
        return ans