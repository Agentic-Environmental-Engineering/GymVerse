from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from collections import deque
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class WILEnv(Env):
    """WIL environment converted to GEM format - single-turn Q&A.

    Task:
    - You are given an array A of length N (1-indexed).
    - First, choose an interval [l1, r1] with length at most D, and set all A[i] = 0 for l1 ≤ i ≤ r1.
    - Then, find an interval [l2, r2] such that the sum of A[i] over l2 ≤ i ≤ r2 is at most P,
      and the length of this interval is as long as possible.

    Output format:
    - Provide l1, r1, l2, r2 separated by single spaces, wrapped in \\boxed{...}.
    - Example: \\boxed{1 3 2 10}

    Reward:
    - 1.0 if the submitted intervals form a valid solution whose length equals the optimal (gold) length.
    - 0.0 otherwise.
    - -0.1 if the answer format is invalid (cannot parse four integers from \\boxed{...}).
    """

    def __init__(self, N: int, **kwargs):
        super().__init__()
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        self.N: int = N

        # Problem state
        self.A: List[int] = []
        self.D: int = 0
        self.P: int = 0
        self.gold_length: int = 0
        self.current_problem: Optional[str] = None

        # Rewards
        self.reward_correct: float = 1.0
        self.reward_incorrect: float = 0.0
        self.reward_format_error: float = -0.1

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an interval optimization problem.\n"
            "Please read the problem carefully and output your answer as four integers l1 r1 l2 r2, "
            "wrapped in \\boxed{...} format.\n"
            "Example: \\boxed{1 3 2 10}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment by generating a new problem instance."""
        super().reset(seed)

        N = self.N
        # Generate array A with values in [1, N]
        A = [random.randint(1, N) for _ in range(N)]

        # D is a random integer in [1, N-1]
        D = random.randint(1, N - 1)

        # P is a random integer in [1, sum(A) - sum(top D elements of A)]
        # This ensures feasibility and avoids degenerate upper bound
        sum_A = sum(A)
        top_D_sum = sum(sorted(A, reverse=True)[:D])
        P = random.randint(1, sum_A - top_D_sum)

        # Compute optimal (gold) length using the original algorithm
        gold = self._compute_gold_length(A, D, P)

        # Build problem prompt
        array_repr = ", ".join(f"A[{i}]={v}" for i, v in enumerate(A, start=1))
        self.current_problem = (
            f"You are given an array A of length {N}, indexed from 1 to {N}. The array is: {array_repr}\n\n"
            f"Your task is as follows:\n"
            f"1. First, choose an interval [l1, r1] (such that r1 - l1 + 1 <= {D}) and set all A[i] = 0 for l1 ≤ i ≤ r1.\n"
            f"2. Then, find an interval [l2, r2] such that the sum of A[i] over l2 ≤ i ≤ r2 is at most {P}, "
            f"and the length of this interval is as long as possible.\n\n"
            f"Output Format: Provide l1, r1, l2, and r2 (in order) — separated by spaces in a single line, "
            f"wrapped in \\boxed{{...}}. For example: \\boxed{{1 3 2 10}}."
        )

        # Store state
        self.A = A
        self.D = D
        self.P = P
        self.gold_length = gold

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_gold_length(self, A: List[int], D: int, P: int) -> int:
        """Compute the optimal (gold) interval length given A, D, and P."""
        N = len(A)
        # Build prefix sums S where S[i] = sum of A[0..i-1]
        S = [0] * (N + 1)
        for i in range(1, N + 1):
            S[i] = S[i - 1] + A[i - 1]

        # Deque to maintain candidate segment endpoints (indices in [D..N])
        # sorted so that the front q[0] has the segment of length D with the largest sum
        q: deque[int] = deque([D])

        ans = D  # zero out a segment of length D, giving at least length D
        l = 1    # current window left endpoint (1-based for S)

        # Slide right endpoint i from D+1 to N (1-based)
        for i in range(D + 1, N + 1):
            # Add the new segment [i-D+1..i], with sum = S[i] - S[i-D].
            # Maintain deque in decreasing order of segment-sums.
            curr_seg_sum = S[i] - S[i - D]
            while q and curr_seg_sum > (S[q[-1]] - S[q[-1] - D]):
                q.pop()
            q.append(i)

            # Move l forward while the best window [l..i] (minus best segment) exceeds P
            # Best segment to zero is the one at q[0]
            while q and S[i] - S[l - 1] - (S[q[0]] - S[q[0] - D]) > P:
                l += 1
                # Drop any segments that no longer fit entirely in [l..i]
                while q and (q[0] - D + 1) < l:
                    q.popleft()

            # Update answer: window length is i - l + 1
            ans = max(ans, i - l + 1)

        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the provided action (answer) and return the reward."""
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, self.reward_format_error, True, False, {"error": "format_error"}

        l1, r1, l2, r2 = parsed
        N, D, P, A = self.N, self.D, self.P, self.A

        # Validate indices
        if not (1 <= l1 <= r1 <= N and 1 <= l2 <= r2 <= N):
            info = {
                "error": "invalid_solution",
                "reason": "index_out_of_range",
                "N": N,
                "D": D,
                "P": P,
                "submitted": (l1, r1, l2, r2),
            }
            return TERMINAL_STATE, self.reward_incorrect, True, False, info

        # Check zeroing length constraint
        if (r1 - l1 + 1) > D:
            info = {
                "error": "invalid_solution",
                "reason": "zeroing_length_exceeds_D",
                "N": N,
                "D": D,
                "P": P,
                "submitted": (l1, r1, l2, r2),
            }
            return TERMINAL_STATE, self.reward_incorrect, True, False, info

        # Apply zeroing and check sum constraint for the second interval
        A_prime = A.copy()
        for i in range(l1, r1 + 1):
            A_prime[i - 1] = 0

        sub_sum = sum(A_prime[l2 - 1 : r2])
        if sub_sum > P:
            info = {
                "error": "invalid_solution",
                "reason": "sum_exceeds_P",
                "subarray_sum": sub_sum,
                "P": P,
                "N": N,
                "D": D,
                "submitted": (l1, r1, l2, r2),
            }
            return TERMINAL_STATE, self.reward_incorrect, True, False, info

        user_length = r2 - l2 + 1
        gold = self.gold_length
        is_optimal = (user_length == gold)

        reward = self.reward_correct if is_optimal else self.reward_incorrect
        info = {
            "correct": is_optimal,
            "reference_gold_length": gold,
            "user_length": user_length,
            "N": N,
            "D": D,
            "P": P,
            "submitted": (l1, r1, l2, r2),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[Tuple[int, int, int, int]]:
        """Extract l1, r1, l2, r2 from \\boxed{...} in the text."""
        import re

        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if not matches:
            return None

        content = matches[-1].strip()
        parts = content.split()
        if len(parts) != 4:
            return None

        try:
            l1, r1, l2, r2 = map(int, parts)
            return l1, r1, l2, r2
        except ValueError:
            return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format (not guaranteed to be valid)."""
        if self.N <= 0:
            return "\\boxed{1 1 1 1}"
        # Randomly generate a zeroing interval within [1..N] of length at most D
        l1 = random.randint(1, self.N)
        r1 = random.randint(l1, min(self.N, l1 + max(0, self.D - 1)))
        # Randomly generate a query interval within [1..N]
        l2 = random.randint(1, self.N)
        r2 = random.randint(l2, self.N)
        return f"\\boxed{{{l1} {r1} {l2} {r2}}}"