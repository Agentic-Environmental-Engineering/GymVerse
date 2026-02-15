import random
from typing import Any, Optional, SupportsFloat, Tuple
from functools import cmp_to_key
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LASLaserEnv(Env):
    """
    LASLaser environment (converted to GEM format).

    Task:
    There are N segments in the 2D plane. You may shoot at most K rays from the origin (0, 0) in any directions.
    Each segment is allowed to intersect with at most one of these rays.
    Please output the maximum number of segments that can be intersected by a single ray.

    Answer format: The final answer must be a single integer wrapped in \\boxed{...}.
    """

    def __init__(self, N: int = 10, **kwargs):
        """
        Initialize the LASLaserEnv instance.

        Parameters:
        - N: Number of segments (must be >= 2).
        """
        super().__init__()
        assert N >= 2, "N should be greater than or equal to 2"
        self.N: int = N

        # Internal state
        self.segments: list[tuple[int, int, int, int]] = []
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a geometry problem involving rays and line segments.\n"
            "You may shoot rays from the origin and count how many segments can be intersected.\n"
            "Please provide your answer in \\boxed{...} format containing a single integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        N = self.N

        # Generate random segments with coordinates in [1, 2N]
        self.segments = [
            (
                random.randint(1, 2 * N),
                random.randint(1, 2 * N),
                random.randint(1, 2 * N),
                random.randint(1, 2 * N),
            )
            for _ in range(N)
        ]

        # Build all 2N endpoint vectors
        p0: list[tuple[int, int]] = [None] * (2 * N)  # type: ignore
        for i, (x1, y1, x2, y2) in enumerate(self.segments):
            p0[i] = (x1, y1)
            p0[N + i] = (x2, y2)

        # Comparator for sorting by angle via cross-product
        def cmp(i: int, j: int) -> int:
            x1, y1 = p0[i]
            x2, y2 = p0[j]
            c = x1 * y2 - y1 * x2
            if c > 0:
                return -1   # i comes before j
            elif c < 0:
                return 1    # i comes after j
            else:
                return 0    # same direction

        # Sort all endpoint indices by their angle from the origin
        p = list(range(2 * N))
        p.sort(key=cmp_to_key(cmp))

        # Discretize unique directions into 1..top
        w = [0] * (2 * N)
        top = 1
        now = p[0]
        w[now] = 1
        for idx in p[1:]:
            # If this direction is not collinear with 'now', it's a new bucket
            if p0[idx][0] * p0[now][1] - p0[idx][1] * p0[now][0] != 0:
                top += 1
                now = idx
            w[idx] = top

        # Prepare interval data structures
        size = top + 2
        INF = top + 1
        left = [INF] * size
        num = [0] * size

        # Build intervals [a, b] on the angle-index line for each segment
        for i in range(N):
            a = w[i]
            b = w[N + i]
            if a > b:
                a, b = b, a
            # Record the leftmost start for any interval ending at b
            if a < left[b]:
                left[b] = a
            # Difference array to count how many intervals cover each point
            num[a] += 1
            num[b + 1] -= 1

        # Prefix-sum to get coverage count at each discrete angle
        for i in range(1, top + 1):
            num[i] += num[i - 1]

        # Make left[i] = min(left[i..top])
        for i in range(top - 1, 0, -1):
            if left[i] > left[i + 1]:
                left[i] = left[i + 1]

        # DP: f[i] = max covered with last ray chosen at or before i
        f = [0] * size
        Ks: list[int] = []
        Answers: list[int] = []
        for K in range(1, N + 1):
            # Try placing one more ray at each i, in descending order
            for i in range(top, 0, -1):
                cand = f[left[i] - 1] + num[i]
                if cand > f[i]:
                    f[i] = cand
            # Allow skipping placing at i (carry forward max)
            for i in range(1, top + 1):
                if f[i - 1] > f[i]:
                    f[i] = f[i - 1]

            if len(Answers) == 0 or f[top] > Answers[-1]:
                Ks.append(K)
                Answers.append(f[top])
            if Answers[-1] == N:
                break

        # Randomly select a K among improving steps and set reference answer
        index = random.randint(0, len(Answers) - 1)
        self.K = Ks[index]
        self.reference_answer = Answers[index]

        # Build problem description
        segments_str = "\n".join(
            f"({x1}, {y1})-({x2}, {y2})" for (x1, y1, x2, y2) in self.segments
        )
        self.current_problem = (
            f"There are {N} segments in the 2D plane, given as:\n"
            f"{segments_str}\n\n"
            f"You may shoot at most {self.K} rays from the origin (0, 0) in any directions. "
            f"Each segment is allowed to intersect with at most one of these rays. "
            f"Please output the maximum number of segments that can be intersected by a single ray.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": N,
            "K": self.K,
            "segments": self.segments,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the answer."""
        # Parse the boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} format."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer answer in boxed format)."""
        random_answer = random.randint(0, self.N)
        return f"\\boxed{{{random_answer}}}"