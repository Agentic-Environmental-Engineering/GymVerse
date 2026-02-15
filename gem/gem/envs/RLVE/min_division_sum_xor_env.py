from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinDivisionSumXorEnv(Env):
    """
    Environment for the "Minimize OR of Batch Sums" problem.

    Task:
    - You are given N numbers A[1..N].
    - Divide the sequence into k consecutive batches (1 <= k <= K).
    - The cost of a division is the bitwise OR of the sums of all batches.
    - Find a division that minimizes this cost.

    Answer format:
    - Provide the batch endpoints as a space-separated list inside \\boxed{...}, e.g., \\boxed{1 3 N}.
    - The last endpoint must be N.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        K: Optional[int] = None,
        A: Optional[List[int]] = None,
        min_N: int = 2,
        max_N: int = 50,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed sequence length. If None, N is sampled in [min_N, max_N].
        - K: Optional maximum number of batches. If None, sampled uniformly from [2, N].
        - A: Optional fixed array of integers. If provided, it defines N if N is None.
             Elements must be non-negative integers.
        - min_N: Minimum N when sampling.
        - max_N: Maximum N when sampling.
        """
        super().__init__()
        if min_N < 2:
            raise ValueError("min_N should be at least 2")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N = N
        self.fixed_K = K
        self.fixed_A = A
        self.min_N = min_N
        self.max_N = max_N

        # Current instance variables for a sampled problem
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.A: Optional[List[int]] = None

        self.current_problem: Optional[str] = None
        self.reference_cost: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a partitioning problem on an integer sequence.\n"
            "You must split the sequence into consecutive batches (1 <= k <= K).\n"
            "The cost is the bitwise OR of the sum of each batch. Minimize this cost.\n"
            "Please provide your answer in \\boxed{...} format, containing the endpoints (end indices) of each batch.\n"
            "Example: \\boxed{1 2 N} means there are 3 batches ending at 1, 2, and N.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N and A
        if self.fixed_A is not None:
            # Validate provided A
            if not isinstance(self.fixed_A, list) or len(self.fixed_A) == 0:
                raise ValueError("A must be a non-empty list of non-negative integers")
            if any((not isinstance(x, int) or x < 0) for x in self.fixed_A):
                raise ValueError("All elements in A must be non-negative integers")
            self.A = list(self.fixed_A)
            self.N = len(self.A) if self.fixed_N is None else self.fixed_N
            if self.N != len(self.A):
                raise ValueError("Provided N does not match the length of A")
        else:
            # Determine or sample N
            if self.fixed_N is not None:
                if self.fixed_N < 2:
                    raise ValueError("N should be greater than or equal to 2")
                self.N = self.fixed_N
            else:
                self.N = random.randint(self.min_N, self.max_N)

            # Generate A uniformly in [0, N*N]
            self.A = [random.randint(0, self.N * self.N) for _ in range(self.N)]

        # Determine or sample K
        if self.fixed_K is not None:
            if not (1 <= self.fixed_K <= self.N):
                raise ValueError("K must satisfy 1 <= K <= N")
            self.K = self.fixed_K
        else:
            # Match the original behavior: sample K in [2, N]
            low_k = 2 if self.N >= 2 else 1
            self.K = random.randint(low_k, self.N)

        # Build problem prompt
        A_lines = "\n".join(f"A[{i + 1}]={val}" for i, val in enumerate(self.A))
        problem_text = (
            f"You are given {self.N} numbers A[1], A[2], ..., A[{self.N}]. The values are given as:\n"
            f"{A_lines}\n\n"
            f"You may divide these numbers (in order) into some consecutive batches. "
            f"Let the total number of batches be k (we must have 1 ≤ k ≤ {self.K}), and let end[1], end[2], ..., end[k] "
            f"(1 ≤ end[1] < end[2] < ... < end[k] = {self.N}) denote the last index in each batch. This means:\n"
            f"- Batch 1 contains A[1] to A[end[1]]\n"
            f"- Batch 2 contains A[end[1] + 1] to A[end[2]]\n"
            f"- ...\n"
            f"- Batch k contains A[end[k−1] + 1] to A[end[k]] (with end[k] = {self.N})\n\n"
            f"Define the cost of one such division as follows:\n"
            f"- First compute the sum of values in each batch.\n"
            f"- Then take the bitwise OR of all batch sums. That is the cost.\n\n"
            f"Please find a batch division (with 1 ≤ k ≤ {self.K}) that minimizes the total cost.\n\n"
            f"Output Format:\n"
            f"A single line containing end[1] end[2] ... end[k], separated by spaces (with end[k] always equal to {self.N}).\n"
            f"Example: \\boxed{{1 2 {self.N}}} — this means:\n"
            f"- There are 3 batches,\n"
            f"- First batch ends at index 1,\n"
            f"- Second ends at index 2,\n"
            f"- Third ends at index {self.N} and includes the remaining numbers.\n\n"
            f"Your answer must be provided in \\boxed{{...}} format."
        )
        self.current_problem = problem_text

        # Compute the minimal possible cost (reference_cost)
        self.reference_cost = self._compute_min_or_cost(self.A, self.K)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the user's answer and return the outcome."""
        boxed = self._parse_answer(action)
        if boxed is None:
            # Format error: \\boxed{...} not found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices
        ends: List[int]
        try:
            parts = boxed.strip().split()
            if not parts:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            ends = list(map(int, parts))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate indices
        N = self.N if self.N is not None else 0
        K = self.K if self.K is not None else 0
        if not (1 <= len(ends) <= K):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "k_out_of_range"}
        for i, e in enumerate(ends):
            if not (1 <= e <= N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "endpoint_out_of_range"}
            if i > 0 and not (ends[i - 1] < e):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "not_strictly_increasing"}
        if ends[-1] != N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "last_endpoint_not_N"}

        # Compute user's cost
        user_cost = self._compute_or_cost_from_partition(self.A, ends)

        # Compare with reference cost
        is_correct = (user_cost == self.reference_cost)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_cost": self.reference_cost,
            "user_cost": user_cost,
            "ends": ends,
            "N": self.N,
            "K": self.K,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action (random partition) in boxed format."""
        if self.N is None or self.K is None:
            # Fallback random
            return "\\boxed{1}"
        N = self.N
        K = self.K
        # Choose a random number of batches k between 1 and K
        k = random.randint(1, K)
        # Choose k-1 unique endpoints from [1, N-1] and add N
        if k == 1:
            ends = [N]
        else:
            choices = list(range(1, N))
            random.shuffle(choices)
            selected = sorted(choices[: k - 1])
            ends = selected + [N]
        return "\\boxed{" + " ".join(map(str, ends)) + "}"

    @staticmethod
    def _compute_or_cost_from_partition(A: List[int], ends: List[int]) -> int:
        """Compute the bitwise OR of batch sums given a partition endpoints list."""
        total_or = 0
        last = 0  # 0-based start index in A list
        # Convert 1-based ends to segments
        for e in ends:
            # e is 1-based end index
            batch_sum = sum(A[last:e])  # sum over A[last:e], since e is 1-based, this works
            total_or |= batch_sum
            last = e
        return total_or

    @staticmethod
    def _compute_min_or_cost(A: List[int], K: int) -> int:
        """
        Compute the minimal possible OR of batch sums with at most K batches
        using a bitwise DP approach.
        """
        N = len(A)
        # Prefix sums for quick segment sum
        prefix = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix[i] = prefix[i - 1] + A[i - 1]

        def check(idx: int, mask: int) -> bool:
            # DP f[i]: minimum groups to cover first i items
            INF = N + 1
            f = [INF] * (N + 1)
            f[0] = 0
            for i in range(1, N + 1):
                # try last segment [j, i)
                for j in range(i - 1, -1, -1):
                    seg_sum = prefix[i] - prefix[j]
                    # If bit idx in seg_sum is 1, we cannot keep this bit zero
                    if ((seg_sum >> idx) & 1) != 0:
                        continue
                    # If seg_sum introduces any lower bit not in mask, it is invalid
                    if (((seg_sum >> idx) << idx) | mask) != mask:
                        continue
                    if f[j] + 1 < f[i]:
                        f[i] = f[j] + 1
            return f[N] <= K

        ans = 0
        total_sum = prefix[N]
        # Iterate from the highest relevant bit down to 0
        for idx in range(total_sum.bit_length() + 1, -1, -1):
            if not check(idx, ans):
                ans |= (1 << idx)
        return ans