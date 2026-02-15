import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LargestRectangle_AmongPointsEnv(Env):
    """
    Environment: Largest Rectangle Among Points (single-turn QA)

    Task:
    - Given N points with integer coordinates in 2D plane.
    - Find four distinct points that form a rectangle (not necessarily axis-aligned).
    - Among all rectangles formed by these points, choose one with the maximum possible area.
    - Output the 0-based indices of the four selected points, separated by spaces, in \\boxed{...} format.

    Reward:
    - Correct (maximum-area rectangle): 1.0
    - Wrong answer (including invalid rectangle or non-maximum area): 0.0
    - Format error (no valid \\boxed{...} found): -0.1
    """

    def __init__(
        self,
        N: Optional[int] = None,
        N_min: int = 5,
        N_max: int = 30,
        **kwargs: Any,
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, use this fixed number of points (must be >= 5).
        - N_min: Minimum number of points sampled if N is None (must be >= 5).
        - N_max: Maximum number of points sampled if N is None (must be >= N_min).
        """
        super().__init__()
        if N is not None and N < 5:
            raise ValueError("N should be greater than or equal to 5")
        if N is None:
            if N_min < 5:
                raise ValueError("N_min should be greater than or equal to 5")
            if N_max < N_min:
                raise ValueError("N_max should be greater than or equal to N_min")

        self.fixed_N: Optional[int] = N
        self.N_min: int = N_min
        self.N_max: int = N_max

        # State variables for the current episode
        self.N: int = 0
        self.points: List[Tuple[int, int]] = []
        self.reference_area: int = 0
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a set of points on a 2D plane.\n"
            "Your task is to select four distinct points that form a rectangle (not necessarily axis-aligned).\n"
            "Among all such rectangles, you must select one with the maximum possible area.\n"
            "Output Format: Provide the 0-based indices of the four points separated by spaces, "
            "and wrap them in \\boxed{...}, e.g., \\boxed{0 2 5 7}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem instance, and return the observation."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            self.N = self.fixed_N
        else:
            self.N = random.randint(self.N_min, self.N_max)
        assert self.N >= 5, "N should be greater than or equal to 5"

        # Generate points
        self.points = []
        # Ensure the first 4 points form a rectangle via perpendicular vectors
        self.points.append(
            (random.randint(-self.N // 2, +self.N // 2), random.randint(-self.N // 2, +self.N // 2))
        )

        while True:
            dx, dy = random.randint(-self.N // 2, +self.N // 2), random.randint(-self.N // 2, +self.N // 2)
            if dx == 0 and dy == 0:
                continue
            x, y = self.points[0]
            # Construct a rectangle A, B = A + v, D = A + w, C = A + v + w with w perpendicular to v
            self.points.append((x + dx, y + dy))              # B = A + v
            self.points.append((x - dy, y + dx))              # D = A + w (perpendicular)
            self.points.append((x + dx - dy, y + dy + dx))    # C = A + v + w
            break

        for _ in range(4, self.N):
            self.points.append((random.randint(-self.N, +self.N), random.randint(-self.N, +self.N)))

        random.shuffle(self.points)

        # Compute the maximum rectangle area using diagonals grouping method
        self.reference_area = self._compute_max_rectangle_area(self.points)
        assert self.reference_area > 0, "The maximum area should be greater than 0"

        # Build problem description
        points_str = "\n".join(f"Point {i}: ({x}, {y})" for i, (x, y) in enumerate(self.points))
        self.current_problem = (
            f"You are given a set of {self.N} points in a 2D plane, each represented by its coordinates (x, y):\n"
            f"{points_str}\n\n"
            "Your task is to find four distinct points such that they form a rectangle (NOT necessarily axis-aligned). "
            "Among all such rectangles, choose one with the maximum possible area.\n\n"
            "Output Format: Output one line containing the indices (0-based) of the four selected points, separated by spaces, "
            "in \\boxed{...} format. For example: \\boxed{0 1 2 3}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Parse and evaluate the user's answer, then terminate the episode."""
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        tokens = boxed.strip().split()
        if len(tokens) != 4:
            # Invalid answer (wrong number of indices)
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Parse indices
        try:
            indices = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check bounds and distinctness
        if not all(0 <= idx < self.N for idx in indices):
            return TERMINAL_STATE, 0.0, True, False, {"error": "out_of_bounds"}
        if len(set(indices)) != 4:
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_distinct"}

        # Validate rectangle and compute its area
        selected_points = [self.points[idx] for idx in indices]
        area = self._rectangle_area_if_valid(selected_points)

        if area is None:
            info = {
                "correct": False,
                "valid_rectangle": False,
                "user_area": None,
                "reference_area": self.reference_area,
                "indices": indices,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (area == self.reference_area)
        info = {
            "correct": is_correct,
            "valid_rectangle": True,
            "user_area": area,
            "reference_area": self.reference_area,
            "indices": indices,
        }
        reward = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _compute_max_rectangle_area(points: List[Tuple[int, int]]) -> int:
        """
        Compute the maximum rectangle area that can be formed by the given points.

        Approach:
        - Enumerate all point pairs as potential diagonals.
        - Group by equal squared length and equal midpoint (represented by doubled midpoint).
        - For each pair of diagonals in the same group, compute rectangle area using cross product trick.
        """
        n = len(points)
        lines: List[Tuple[int, int, int, int, int]] = []

        # Build list of all point pairs (diagonals): (squared_length, sum_x, sum_y, idx1, idx2)
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dx = xi - xj
                dy = yi - yj
                s = dx * dx + dy * dy
                sx = xi + xj  # midpoint * 2
                sy = yi + yj
                lines.append((s, sx, sy, i, j))

        # Sort by (length, midpoint_x, midpoint_y)
        lines.sort(key=lambda t: (t[0], t[1], t[2]))

        ans = 0
        m = len(lines)
        i = 0
        while i < m:
            s0, sx0, sy0, idx1, idx2 = lines[i]
            j = i + 1
            # Process other diagonals with the same (s, sx, sy)
            while j < m and lines[j][0] == s0 and lines[j][1] == sx0 and lines[j][2] == sy0:
                _, _, _, idx3, _ = lines[j]
                # Compute area = |(C − A) × (B − A)| where A=points[idx1], C=points[idx2], B=points[idx3]
                x1, y1 = points[idx1]  # A
                x2, y2 = points[idx2]  # C (opposite)
                x3, y3 = points[idx3]  # B (one endpoint of the other diagonal)
                tmp = abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)
                if tmp > ans:
                    ans = tmp
                j += 1
            i += 1

        return ans

    @staticmethod
    def _rectangle_area_if_valid(P: List[Tuple[int, int]]) -> Optional[int]:
        """
        Check whether the given 4 points form a rectangle and return its area if valid.
        The order of points is arbitrary.

        Validation steps:
        - Pick A = P[0], sort other three points by squared distance to A.
        - Let B and D be the two nearest, C the farthest.
        - Check for non-zero side lengths.
        - Check perpendicularity of AB and AD.
        - Check the parallelogram property: expected_C == C.
        - Area = |(B - A) × (D - A)|.
        """
        A = P[0]
        others = P[1:]

        d2 = []
        for X in others:
            dx, dy = X[0] - A[0], X[1] - A[1]
            d2.append((dx * dx + dy * dy, X, dx, dy))
        d2.sort(key=lambda t: t[0])

        d1, B, dx1, dy1 = d2[0]
        d2_val, D, dx2, dy2 = d2[1]
        C = d2[2][1]

        # Zero-length side check
        if d1 == 0 or d2_val == 0:
            return None

        # Perpendicularity check
        if dx1 * dx2 + dy1 * dy2 != 0:
            return None

        # Parallelogram property: A + (B - A) + (D - A) == C
        expected_C = (B[0] + D[0] - A[0], B[1] + D[1] - A[1])
        if expected_C != C:
            return None

        # Area of rectangle
        area = abs(dx1 * dy2 - dy1 * dx2)
        return area

    def sample_random_action(self) -> str:
        """Sample a random action in boxed format: four distinct indices."""
        if self.N <= 0:
            # Fallback if called before reset
            return "\\boxed{0 1 2 3}"
        indices = random.sample(range(self.N), 4)
        return f"\\boxed{{{indices[0]} {indices[1]} {indices[2]} {indices[3]}}}"