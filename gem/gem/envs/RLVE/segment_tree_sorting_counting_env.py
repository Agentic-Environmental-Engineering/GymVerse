import random
from typing import Any, Optional, SupportsFloat, Tuple, Dict, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SegmentTreeSortingCountingEnv(Env):
    """Environment for counting segment-swap operation sequences that sort a permutation."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 12,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: If provided, use this fixed N for all episodes. Otherwise sample N in [min_N, max_N].
            min_N: Minimum N to sample (inclusive) if N is not fixed.
            max_N: Maximum N to sample (inclusive) if N is not fixed.
        """
        super().__init__()
        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_N: Optional[int] = None
        self.current_A: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given a permutation A of integers from 1 to 2^N.\n"
            "There are N operation types. Each type i (1 <= i <= N) can be applied at most once, in any order:\n"
            "- Divide the array into 2^(N - i + 1) segments, each of length 2^(i - 1).\n"
            "- You may swap any two segments.\n\n"
            "Task: Count the number of distinct operation sequences that sort the array into increasing order.\n"
            "Two sequences are different if they have different lengths, or they perform different operations at any same step.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Choose N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 2, "N should be greater than or equal to 2"

        # Generate array A by applying random valid operations to the sorted array
        A = list(range(1, 2 ** N + 1))

        operation_count = random.randint(1, N)
        operation_types = random.sample(range(1, N + 1), operation_count)
        for operation_type in operation_types:
            seg_num = 2 ** (N - operation_type + 1)
            seg_size = 2 ** (operation_type - 1)
            i, j = random.sample(range(seg_num), 2)
            i_start, j_start = i * seg_size, j * seg_size
            for k in range(seg_size):
                A[i_start + k], A[j_start + k] = A[j_start + k], A[i_start + k]

        # Compute the reference answer using the original DFS/check logic
        po = [1] * (N + 1)
        for i in range(1, N + 1):
            po[i] = po[i - 1] * i

        ans = 0

        def check(k: int) -> bool:
            seg_size = 1 << k
            half = 1 << (k - 1)
            cnt = 1 << (N - k)
            for i in range(cnt):
                start = i * seg_size
                if A[start] + half != A[start + half]:
                    return False
            return True

        def swap(i: int, j: int, length: int) -> None:
            for m in range(length):
                A[i + m], A[j + m] = A[j + m], A[i + m]

        def dfs(now: int, num: int) -> None:
            nonlocal ans
            if now > 0 and not check(now):
                return
            if now == N:
                ans += po[num]
                return

            # Option 1: skip operation type now+1
            dfs(now + 1, num)

            # Option 2: apply operation of this type by swapping two segments
            seg_size = 1 << now
            total_segments = 1 << (N - now)
            tmp: List[int] = []
            for i in range(1, total_segments, 2):
                s1 = (i - 1) * seg_size
                s2 = i * seg_size
                if A[s2] != A[s1] + seg_size:
                    tmp.append(i)
                    tmp.append(i + 1)
                    if len(tmp) > 4:
                        return
            if not tmp:
                return
            for p in range(len(tmp)):
                for q in range(p + 1, len(tmp)):
                    i_seg = tmp[p] - 1
                    j_seg = tmp[q] - 1
                    i_start = i_seg * seg_size
                    j_start = j_seg * seg_size
                    swap(i_start, j_start, seg_size)
                    dfs(now + 1, num + 1)
                    swap(i_start, j_start, seg_size)

        dfs(0, 0)
        assert ans > 0

        # Store state
        self.current_N = N
        self.current_A = A[:]
        self.reference_answer = ans

        # Build the problem statement
        A_str = " ".join(f"A[{i + 1}]={val}" for i, val in enumerate(A))
        problem = (
            f"You are given a permutation of integers from 1 to 2^{N} (A[1], A[2], ..., A[{2**N}]). "
            f"The array is: {A_str}\n\n"
            f"There are {N} types of operations. You may apply each type at most once, and you may choose to apply them in any order. "
            f"The i-th type of operation (1 ≤ i ≤ {N}) is defined as follows:\n"
            f"- Divide the array into 2^({N} - i + 1) segments, each of length 2^(i - 1). (Each element belongs to exactly one segment.)\n"
            f"- You may swap any two segments (freely chosen by you).\n\n"
            f"Please count the number of distinct sequences of operations that can sort the array into increasing order. Two sequences are considered different if:\n"
            f"- They have different lengths, OR\n"
            f"- They perform different operations at any same position in the sequence (i.e., the type or the pair of segments swapped differs at that step).\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        self.current_problem = problem
        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the provided answer and terminate."""
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text.strip())
        except ValueError:
            # Invalid number format counts as wrong answer with zero reward
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not initialized. Call reset() before step()."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_N,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        from math import factorial
        N = self.current_N if self.current_N is not None else (self.fixed_N if self.fixed_N is not None else max(self.min_N, 2))
        try:
            upper = max(1, factorial(N))
        except OverflowError:
            upper = 10**6
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"