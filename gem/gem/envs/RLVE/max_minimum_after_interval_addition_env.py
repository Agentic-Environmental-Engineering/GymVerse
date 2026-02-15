import heapq
import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxMinimum_AfterIntervalAdditionEnv(Env):
    """Environment for selecting K intervals to maximize the minimum array value after additions.

    The task:
      - Given an array of length N and M intervals [l, r] (1-based).
      - Choose exactly K distinct intervals.
      - For each chosen interval, add a value A to every element covered by the interval.
      - Additions are cumulative.
      - The goal is to maximize the minimum value in the array after applying all K additions.

    The agent should output exactly K interval indices (1-based), space-separated, inside \\boxed{...}.
    """

    def __init__(
        self,
        max_n_m: int = 50,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_n_m: Upper bound for N and M. Must be >= 3.
        """
        super().__init__()
        assert max_n_m >= 3, "max_n_m should be greater than or equal to 3"
        self.max_n_m: int = max_n_m

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None
        self.A: Optional[int] = None
        self.ARRAY: Optional[List[int]] = None
        self.intervals: Optional[List[Tuple[int, int]]] = None

        # Derived data
        self.current_problem: Optional[str] = None
        self.gold_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Select exactly K distinct intervals to add A to array elements covered by those intervals. "
            "The goal is to maximize the minimum value in the array after the additions.\n"
            "Indices are 1-based.\n"
            "Output Format: Provide exactly K interval indices separated by single spaces inside \\boxed{...}. "
            "Example: \\boxed{1 3 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(3, self.max_n_m)
        M = random.randint(3, self.max_n_m)
        K = random.randint(2, M - 1)
        A = random.randint(1, self.max_n_m)

        ARRAY = [random.randint(1, self.max_n_m * random.randint(1, K)) for _ in range(N)]

        intervals: List[Tuple[int, int]] = []
        for _ in range(M):
            length = random.randint(1, N)
            start = random.randint(1, N - length + 1)
            intervals.append((start, start + length - 1))

        # Store state
        self.N = N
        self.M = M
        self.K = K
        self.A = A
        self.ARRAY = ARRAY
        self.intervals = intervals

        # Compute gold answer (maximum achievable minimum value)
        self.gold_answer = self._compute_gold_answer(N, M, K, A, ARRAY, intervals)
        assert self.gold_answer is not None and self.gold_answer > 0, "The gold answer should be positive"

        # Build problem text
        array_str = ", ".join(f"ARRAY[{i}]={val}" for i, val in enumerate(ARRAY, start=1))
        intervals_str = "\n".join(f"Interval {i}: [{l}, {r}]" for i, (l, r) in enumerate(intervals, start=1))
        self.current_problem = (
            f"You are given an array ARRAY of length {N}: {array_str}\n\n"
            f"You are also given {M} intervals (numbered 1 to {M}):\n{intervals_str}\n\n"
            f"Select exactly {K} distinct intervals; for each selected interval [l, r], add the value {A} to every element "
            f"of ARRAY from index l to r (inclusive). Additions are cumulative. Your goal is to maximize the minimum value "
            f"in ARRAY after applying all {K} additions.\n\n"
            f"Output exactly {K} integers — the selected interval indices (in any order), separated by spaces.\n"
            f"Output Format: \\boxed{{i1 i2 ... i{K}}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step: parse and evaluate the submitted answer."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compute result
        info: dict[str, Any] = {}
        try:
            # Normalize separators and split
            normalized = boxed_content.replace(",", " ").strip()
            parts = normalized.split()
            indices = list(map(int, parts))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Basic checks
        if self.K is None or self.M is None or self.A is None or self.ARRAY is None or self.intervals is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        if len(indices) != self.K:
            info["error"] = "invalid_selection_length"
            info["expected_length"] = self.K
            info["provided_length"] = len(indices)
            return TERMINAL_STATE, 0.0, True, False, info

        if len(set(indices)) != self.K:
            info["error"] = "duplicate_indices"
            info["provided_indices"] = indices
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(1 <= idx <= self.M for idx in indices):
            info["error"] = "index_out_of_range"
            info["provided_indices"] = indices
            info["valid_range"] = [1, self.M]
            return TERMINAL_STATE, 0.0, True, False, info

        # Apply chosen intervals and compute minimum
        array_copy = self.ARRAY.copy()
        for idx in indices:
            l, r = self.intervals[idx - 1]
            # Convert to 0-based indices
            for i in range(l - 1, r):
                array_copy[i] += self.A

        achieved_min = min(array_copy)
        gold_min = self.gold_answer

        # Determine reward
        is_correct = (achieved_min == gold_min)
        reward: float = 1.0 if is_correct else 0.0

        info.update({
            "correct": is_correct,
            "reference_answer": gold_min,
            "achieved_min": achieved_min,
            "user_indices": indices
        })

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_gold_answer(
        self,
        N: int,
        M: int,
        K: int,
        A: int,
        ARRAY: List[int],
        intervals: List[Tuple[int, int]]
    ) -> int:
        """Compute the maximum achievable minimum using at most K interval additions of A."""
        # Build operations list
        # Each op: (pos, tp, val)
        # tp: 0 = left endpoint, 1 = sequence point, 2 = right endpoint
        OPS: List[Tuple[int, int, int]] = []

        # Sequence points (positions 1..N)
        for i in range(1, N + 1):
            t = ARRAY[i - 1]
            OPS.append((i, 1, t))

        # Intervals and right endpoints
        R = [0] * (M + 1)  # R[i] stores right endpoint (1-based) for interval i
        for i, (L_i, R_i) in enumerate(intervals, start=1):
            OPS.append((L_i, 0, i))   # left endpoint
            OPS.append((R_i, 2, i))   # right endpoint
            R[i] = R_i

        # Sort by position, then type: left(0) < sequence(1) < right(2)
        OPS.sort(key=lambda x: (x[0], x[1]))

        lf = min(ARRAY)          # lower bound of minimum
        ri = lf + M * A          # upper bound (loose but valid)

        def jud(mid: int) -> bool:
            """Check if we can achieve min >= mid using at most K intervals."""
            flow = 0  # current accumulated +A from chosen intervals covering current position
            tot = 0   # total intervals chosen so far
            pq: List[Tuple[int, int]] = []  # max-heap by R[v] using negative values
            book = [0] * (M + 1)  # book[v] == 1 means interval v has been selected

            for pos, tp, val in OPS:
                if tp == 0:
                    # left endpoint: push interval into candidate heap
                    v = val
                    heapq.heappush(pq, (-R[v], v))
                elif tp == 1:
                    # position event: ensure ARRAY[pos] + applied additions >= mid
                    ned = mid - val - flow
                    if ned < 0:
                        continue
                    ch = (ned + A - 1) // A  # ceil division
                    if tot + ch > K:
                        return False
                    while pq and ch:
                        _, v = heapq.heappop(pq)
                        if R[v] < pos:
                            return False
                        book[v] = 1
                        flow += A
                        ch -= 1
                        tot += 1
                    if ch > 0:
                        return False
                else:
                    # right endpoint: if chosen, remove its contribution as we pass beyond its range
                    v = val
                    if book[v]:
                        flow -= A
            return True

        # Binary search for the best achievable minimum
        while lf != ri:
            mid = (lf + ri + 1) // 2
            if jud(mid):
                lf = mid
            else:
                ri = mid - 1

        return lf

    def sample_random_action(self) -> str:
        """Sample a random action: pick K distinct random intervals and return in boxed format."""
        if self.M is None or self.K is None:
            # Fallback boxed content if environment not initialized
            return "\\boxed{1}"
        indices = random.sample(range(1, self.M + 1), k=self.K)
        content = " ".join(map(str, indices))
        return f"\\boxed{{{content}}}"