import random
import functools
from typing import Any, List, Optional, Sequence, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumTriangleAreaEnv(Env):
    """Environment for computing the sum of areas of all triangles (times 2) formed by a given set of points."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 30,
        seed: Optional[int] = None,
    ):
        """
        Initialize the SumTriangleAreaEnv.

        Args:
            N: Number of points. If None, a random N in [min_N, max_N] will be used.
            min_N: Minimum number of points (inclusive) when sampling N randomly. Must be >= 3.
            max_N: Maximum number of points (inclusive) when sampling N randomly. Must be >= min_N.
            seed: Optional random seed for reproducibility.
        """
        super().__init__()
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")

        self.N = N
        self.min_N = min_N
        self.max_N = max_N

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.points: Optional[List[Tuple[int, int]]] = None

        # Optionally seed the RNG at creation time
        if seed is not None:
            random.seed(seed)

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given N distinct points on a 2D integer grid.\n"
            "Task: Compute the sum of the areas of all triangles that can be formed by any three distinct points.\n"
            "For degenerate triangles (collinear points), the area is 0.\n"
            "Output the total area multiplied by 2 (twice the sum). The result is guaranteed to be an integer.\n"
            "Answer format: Put your final integer answer inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N is None:
            N = random.randint(self.min_N, self.max_N)
        else:
            N = self.N

        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Generate N distinct points sampled from the grid [0, N] x [0, N]
        grid_points = [(x, y) for x in range(0, N + 1) for y in range(0, N + 1)]
        self.points = random.sample(grid_points, N)

        # Compute the reference answer (twice the sum of triangle areas)
        self.reference_answer = self._compute_twice_sum_triangle_areas(self.points)

        # Build the problem prompt
        points_str = "\n".join(f"({x}, {y})" for x, y in self.points)
        self.current_problem = (
            f"There are {N} points in a 2D plane, each represented by its coordinates (x, y). "
            f"The points are given as follows:\n{points_str}\n\n"
            "Please compute the sum of the areas of all triangles that can be formed by any three distinct points in this set. "
            "If a triangle is degenerate (i.e., the three points are collinear), its area is considered 0. "
            "Output the total area multiplied by 2 (i.e., twice the sum of all triangle areas), which will always be an integer.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": N,
            "points": self.points[:],
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_str)
        except ValueError:
            # Not a valid integer; treat as wrong answer (not a format error)
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": answer_str,
                "error": "invalid_answer",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re

        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_twice_sum_triangle_areas(self, points: Sequence[Tuple[int, int]]) -> int:
        """
        Compute twice the sum of areas of all triangles formed by the given points.
        Uses an O(N^2 log N) algorithm based on sorting by polar angle and suffix sums.
        """
        A = sorted(points, key=lambda p: (p[0], p[1]))
        N = len(A)
        ans = 0

        for i in range(N):
            xi, yi = A[i]
            # Build vectors from A[i] to all later points
            s = [(x - xi, y - yi) for x, y in A[i + 1 :]]

            # Sort by polar angle around the origin using cross-product comparator
            s.sort(
                key=functools.cmp_to_key(
                    lambda a, b: -1
                    if a[1] * b[0] < a[0] * b[1]
                    else (1 if a[1] * b[0] > a[0] * b[1] else 0)
                )
            )

            m = len(s)
            # Build suffix sums of x- and y-components
            sx = [0] * (m + 1)
            sy = [0] * (m + 1)
            for j in range(m - 1, -1, -1):
                sx[j] = sx[j + 1] + s[j][0]
                sy[j] = sy[j + 1] + s[j][1]
                # Accumulate cross-products to sum triangle areas (twice the area)
                ans += s[j][0] * sy[j + 1] - s[j][1] * sx[j + 1]

        return ans

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        # Heuristic random guess; not guaranteed to be in range
        random_answer = random.randint(0, 10**6)
        return f"\\boxed{{{random_answer}}}"