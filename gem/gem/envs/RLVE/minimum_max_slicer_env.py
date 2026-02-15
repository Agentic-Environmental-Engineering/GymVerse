from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Minimum_MaxSlicerEnv(Env):
    """Environment for the Minimum Max Slicer problem - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 50,
        M_range_coefficient: int = 2,
        **kwargs
    ):
        """
        Initialize the Minimum_MaxSlicerEnv instance.

        Parameters:
        - N: If provided, use a fixed array length N (must be >= 4).
        - min_N: Minimum possible N if N is not provided (default 4).
        - max_N: Maximum possible N if N is not provided (default 50).
        - M_range_coefficient: Controls the range of M; M is uniformly sampled
          from [3, max(3, N // M_range_coefficient)] (default 2).
        """
        super().__init__()
        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N
        self.M_range_coefficient: int = M_range_coefficient

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.reference_gold: Optional[int] = None
        self.reference_ends: Optional[List[int]] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the general task instructions."""
        return (
            "You are solving a partitioning problem on arrays.\n"
            "Task: Given an array A of length N and an integer M, you must divide A (in order)\n"
            "into M consecutive batches to minimize the maximum batch sum.\n"
            "Output Format: Provide the space-separated indices end[1], end[2], ..., end[M]\n"
            "representing the last index of each batch (0-indexed), enclosed in \\boxed{...}.\n"
            "Constraints: 0 <= end[1] < end[2] < ... < end[M] = N - 1.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)

        assert N >= 4, "N must be at least 4"
        self.N = N

        # Determine M
        M_low = 3
        M_high = max(3, N // self.M_range_coefficient)
        M = random.randint(M_low, M_high)
        assert M < N, "M must be less than N"
        self.M = M

        # Generate array A with values in [1, N]
        A = [random.randint(1, N) for _ in range(N)]
        self.A = A

        # Compute minimal possible maximal batch sum (gold) via binary search
        left, right = min(A), sum(A)

        def can_partition(d: int) -> bool:
            now_sum, index, counting = 0, 0, 1
            while True:
                if now_sum + A[index] <= d:
                    now_sum += A[index]
                else:
                    counting += 1
                    if A[index] <= d:
                        now_sum = A[index]
                    else:
                        return False
                index += 1
                if index == N:
                    break
            return counting <= M

        while left < right:
            mid = (left + right) // 2
            if can_partition(mid):
                right = mid
            else:
                left = mid + 1

        gold = left
        assert gold > 0, "gold must be greater than 0"
        self.reference_gold = gold

        # Construct a valid set of ends achieving the gold (greedy)
        ends: List[int] = []

        def get_ends(d: int) -> None:
            now_sum, index = 0, 0
            while True:
                if now_sum + A[index] <= d:
                    now_sum += A[index]
                else:
                    ends.append(index - 1)
                    now_sum = A[index]
                index += 1
                if index == N:
                    ends.append(index - 1)
                    break

        get_ends(gold)
        if len(ends) < M:
            missing = sorted(set(range(N)) - set(ends))
            ends += missing[: M - len(ends)]
            ends.sort()
        assert len(ends) == M
        assert ends[-1] == N - 1
        self.reference_ends = ends

        # Build problem text
        problem_text = (
            f"You are given an array A of length {N}. The values are as follows (indexing starts at 0):\n"
            + "\n".join(f"A[{i}]={A[i]}" for i in range(N))
            + "\n\n"
            f"You may divide these items (in order) into {M} consecutive batches. Let end[1], end[2], ..., end[{M}] "
            f"(0 <= end[1] < end[2] < ... < end[{M}] = {N - 1}) represent the last index of each batch. This means:\n"
            f"- Batch 1 contains items from index 0 to end[1]\n"
            f"- Batch 2 contains items from index end[1] + 1 to end[2]\n"
            f"- ...\n"
            f"- Batch {M} contains items from index end[{M - 1}] + 1 to end[{M}] (which is {N - 1})\n\n"
            f"Try your best to minimize the maximum sum among all batches. In other words, minimize: "
            f"max(S[1], S[2], ..., S[{M}]), where each S[i] is the sum of A values in batch i.\n\n"
            f"Output Format: Your final answer should be the indices end[1], end[2], ..., end[{M}] (with end[{M}] always equal to {N - 1}), "
            f"separated by spaces, and enclosed in \\boxed{{...}}.\n"
            f"Example: \\boxed{{{' '.join(map(str, range(M - 1)))} {N - 1}}}\n"
        )

        self.current_problem = self._get_instructions() + problem_text
        return self.current_problem, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the reward."""
        if self.N is None or self.M is None or self.A is None or self.reference_gold is None:
            # Environment has not been properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices from boxed content
        try:
            ends = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        N = self.N
        M = self.M
        A = self.A

        # Validate ends format
        if len(ends) != M:
            info = {"error": "invalid_solution", "expected_M": M, "received_len": len(ends)}
            return TERMINAL_STATE, 0.0, True, False, info
        for i in range(len(ends)):
            if not (0 <= ends[i] < N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "index_out_of_range"}
            if i and not (ends[i - 1] < ends[i]):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "not_strictly_increasing"}
        if ends[-1] != N - 1:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "last_index_must_be_N_minus_1"}

        # Compute maximal batch sum for the proposed partition
        def segment_sum(l: int, r: int) -> int:
            return sum(A[idx] for idx in range(l, r + 1))

        max_sum = segment_sum(0, ends[0])
        for i in range(1, len(ends)):
            max_sum = max(max_sum, segment_sum(ends[i - 1] + 1, ends[i]))

        is_correct = (max_sum == self.reference_gold)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_gold": self.reference_gold,
            "user_max_sum": max_sum,
            "reference_ends": self.reference_ends,
            "user_ends": ends,
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
        """Sample a random valid-format action (not necessarily optimal)."""
        if self.N is None or self.M is None:
            # Fallback: produce a minimal placeholder
            return r"\boxed{0}"

        N = self.N
        M = self.M
        if M == 1:
            ends = [N - 1]
        else:
            # Choose M-1 unique indices from [0, N-2], sort them, and append N-1
            candidates = list(range(0, N - 1))
            sampled = sorted(random.sample(candidates, M - 1))
            ends = sampled + [N - 1]

        return "\\boxed{" + " ".join(map(str, ends)) + "}"