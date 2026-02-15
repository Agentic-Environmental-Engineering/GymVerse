from typing import Any, Optional, SupportsFloat, Tuple, List, Set
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SquareUndamagedPointCountingEnv(Env):
    """Environment for counting undamaged-vertex squares on a grid - single-turn Q&A."""

    def __init__(
        self,
        max_n_m: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n_m: Upper bound for both N and M (inclusive). Must be >= 2.
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        self.max_n_m = max_n_m

        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.damaged_points: Optional[List[tuple[int, int]]] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a combinatorial geometry counting problem.\n"
            "Task: Count the number of distinct squares (not necessarily axis-aligned) in a grid with integer coordinates,\n"
            "subject to the constraint that none of the vertices of the square is a damaged point.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate grid dimensions
        N = random.randint(1, self.max_n_m)
        M = random.randint(1, self.max_n_m)

        # Generate damaged points
        all_points = [(x, y) for x in range(N + 1) for y in range(M + 1)]
        sample_size = random.randint(1, min(N * M, self.max_n_m))
        damaged_points = random.sample(all_points, sample_size)

        # Compute reference answer using the original algorithm
        reference_answer = self._compute_reference_answer(N, M, damaged_points)

        # Build problem prompt
        damaged_str = ", ".join(f"({x}, {y})" for x, y in damaged_points)
        problem = (
            "Please count the number of distinct squares (not necessarily axis-aligned) such that:\n"
            f"- All four vertices are integer coordinate points with 0 ≤ x ≤ {N} and 0 ≤ y ≤ {M}.\n"
            f"- None of the four vertices is among the damaged points. The list of damaged points is given as follows: {damaged_str}\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Store state
        self.N = N
        self.M = M
        self.damaged_points = damaged_points
        self.reference_answer = reference_answer
        self.current_problem = problem

        obs = self._get_instructions() + problem
        info = {
            "N": N,
            "M": M,
            "damaged_points": damaged_points,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the user's answer."""
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not initialized. Call reset() before step()."
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format (for testing)."""
        random_answer = random.randint(0, 1000)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, N: int, M: int, damaged_points: List[tuple[int, int]]) -> int:
        """Compute the number of valid squares using the original algorithm."""
        # Copy and sort exactly as original code
        pts = damaged_points.copy()
        pts.sort()

        # Compress each (x, y) to a single integer id = x*(M+1)+y for O(1) lookup
        deleted: Set[int] = {x * (M + 1) + y for (x, y) in pts}
        get_id = lambda x, y: x * (M + 1) + y

        # cnt0: total number of squares in a complete grid
        limit = min(N, M)
        cnt0 = 0
        for s in range(1, limit + 1):
            cnt0 += (N - s + 1) * (M - s + 1) * s

        # Auxiliary function matching lgh in the original
        def add_lgh(lim: int, len1: int, len2: int) -> int:
            res = lim * (lim + 3) // 2
            if lim > len1:
                d = lim - len1
                res -= d * (d + 1) // 2
            if lim > len2:
                d = lim - len2
                res -= d * (d + 1) // 2
            return res

        # cnt1: squares counted by at least one deleted vertex
        cnt1 = 0
        for x, y in pts:
            u, d_ = x, N - x  # up and down available steps
            l, r = y, M - y   # left and right available steps
            cnt1 += add_lgh(min(M, u), l, r)
            cnt1 += add_lgh(min(M, d_), l, r)
            cnt1 += add_lgh(min(N, l), u, d_)
            cnt1 += add_lgh(min(N, r), u, d_)
            cnt1 -= min(l, u)
            cnt1 -= min(u, r)
            cnt1 -= min(r, d_)
            cnt1 -= min(d_, l)

        # cnt2, cnt3, cnt4: inclusion-exclusion on pairs
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        Klen = len(pts)

        def inside(x: int, y: int) -> bool:
            return 0 <= x <= N and 0 <= y <= M

        def process(x3: int, y3: int, x4: int, y4: int) -> None:
            nonlocal cnt2, cnt3, cnt4
            if not (inside(x3, y3) and inside(x4, y4)):
                return
            t1 = get_id(x3, y3) in deleted
            t2 = get_id(x4, y4) in deleted
            cnt2 += 1
            if t1:
                cnt3 += 1
            if t2:
                cnt3 += 1
            if t1 and t2:
                cnt4 += 1

        for i in range(Klen):
            x1, y1 = pts[i]
            for j in range(i + 1, Klen):
                x2, y2 = pts[j]

                # The two orientations where (x1,y1)-(x2,y2) is a side
                process(x1 - (y2 - y1), y1 + (x2 - x1),
                        x2 - (y2 - y1), y2 + (x2 - x1))
                process(x1 + (y2 - y1), y1 - (x2 - x1),
                        x2 + (y2 - y1), y2 - (x2 - x1))

                # Orientation where they are the diagonal
                a = (x2 - x1) + (y2 - y1)
                b = (x2 - x1) - (y2 - y1)
                if (a & 1) or (b & 1):
                    continue
                a //= 2
                b //= 2
                process(x1 + b, y1 + a, x2 - b, y2 - a)

        # Correct over-counting
        cnt3 //= 3
        cnt4 //= 6

        # Final inclusion-exclusion
        return cnt0 - cnt1 + cnt2 - cnt3 + cnt4