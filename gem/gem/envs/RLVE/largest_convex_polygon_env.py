import random
from functools import cmp_to_key
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LargestConvexPolygonEnv(Env):
    """Environment for finding the largest convex polygon from a set of points.

    The agent is given N points in the plane with constraints:
    - No two points share the same coordinates
    - No three points are collinear

    Task:
    Find a subset of points that forms the vertices of a convex polygon,
    maximizing the number of points in this subset. Output the labels of the
    selected points (1-based indexing) in any order.

    Answer format:
    The answer must be provided inside \\boxed{...} with labels separated by spaces.
    Example: \\boxed{1 5 3 2}
    """

    def __init__(
        self,
        N: int = 10,
        **kwargs
    ):
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N
        self.points: List[Tuple[int, int]] = []
        self.gold_answer_size: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given N points in the 2D plane, labeled from 1 to N. "
            "No two points share the same coordinates, and no three points are collinear.\n"
            "Find a subset of distinct points that forms the vertices of a convex polygon, "
            "and maximize the number of points in this subset.\n"
            "Output Format: Provide the labels of the selected points (separated by spaces) inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)
        N = self.N

        # Generate points with constraints:
        # - No duplicate coordinates
        # - No three points collinear
        points_set = set()
        lines = set()

        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a

        for _ in range(N):
            while True:
                x = random.randint(0, N)
                y = random.randint(0, N)
                if (x, y) in points_set:
                    continue

                collinear = False
                new_lines = set()
                for (px, py) in points_set:
                    if px == x:
                        a, b, c = 1, 0, -x
                    else:
                        a, b = py - y, x - px
                        c = -(a * x + b * y)

                    g = gcd(abs(a), gcd(abs(b), abs(c)))
                    a, b, c = a // g, b // g, c // g

                    if a < 0:
                        a, b, c = -a, -b, -c
                    elif a == 0 and b < 0:
                        b, c = -b, -c

                    if (a, b, c) in lines:
                        collinear = True
                        break

                    new_lines.add((a, b, c))

                if collinear:
                    continue

                points_set.add((x, y))
                lines.update(new_lines)
                break

        self.points = list(points_set)

        # Compute the size of the largest convex polygon using the original DP approach
        P = self.points

        def octant(dx: int, dy: int) -> int:
            if dx == 0 and dy > 0:   # up
                return 1
            elif dx > 0 and dy > 0:  # NE
                return 2
            elif dx > 0 and dy == 0: # right
                return 3
            elif dx > 0 and dy < 0:  # SE
                return 4
            elif dx == 0 and dy < 0: # down
                return 5
            elif dx < 0 and dy < 0:  # SW
                return 6
            elif dx < 0 and dy == 0: # left
                return 7
            else:                    # dx < 0 and dy > 0 -> NW
                return 8

        edges = []
        for u in range(N):
            xu, yu = P[u]
            for v in range(N):
                if u == v:
                    continue
                xv, yv = P[v]
                dx = xv - xu
                dy = yv - yu
                edges.append((u, v, dx, dy, octant(dx, dy)))

        def cmp_edges(e1, e2):
            # sort by octant first (clockwise starting from up),
            # then by slope via cross product (dy1*dx2 ? dy2*dx1)
            if e1[4] != e2[4]:
                return -1 if e1[4] < e2[4] else 1
            cross = e1[3] * e2[2] - e2[3] * e1[2]  # dy1*dx2 - dy2*dx1
            if cross > 0:
                return -1
            elif cross < 0:
                return 1
            else:
                return 0

        edges.sort(key=cmp_to_key(cmp_edges))
        EV = [(u, v) for (u, v, _, _, _) in edges]

        ans = 0
        for i in range(N):
            mx: List[Optional[int]] = [None] * N
            mx[i] = 0
            for u, v in EV:
                val = mx[u]
                if val is not None:
                    cand = val + 1
                    if mx[v] is None or cand > mx[v]:
                        mx[v] = cand
            if mx[i] is not None and mx[i] > ans:
                ans = mx[i]

        assert ans >= 3, "The answer should be greater than or equal to 3"
        self.gold_answer_size = ans

        # Build prompt
        points_str = "\n".join(f"Point {i}: ({x}, {y})" for i, (x, y) in enumerate(self.points, start=1))
        self.current_problem = (
            f"You are given {N} points in the 2D plane, labeled from 1 to {N}. "
            f"No two points share the same coordinates, and no three points are collinear:\n{points_str}\n\n"
            "Find a subset of distinct points that forms the vertices of a convex polygon, and "
            "maximize the number of points in this subset; please output the labels of the selected "
            "points in one line, separated by spaces (in any order); if multiple answers exist, output any one.\n\n"
            "Output Format: Provide your final answer as labels separated by spaces inside \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {
            "N": N,
            "points": self.points,
            "gold_answer_size": self.gold_answer_size
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step: parse and validate the user's answer."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices from boxed content
        try:
            indices = list(map(int, boxed_content.split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.N
        # Validate indices range
        if not all(1 <= i <= N for i in indices):
            return TERMINAL_STATE, 0.0, True, False, {"error": "out_of_range"}

        # Validate uniqueness
        if len(indices) != len(set(indices)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "duplicate_indices"}

        # Validate convex polygon formation
        selected_points = [self.points[i - 1] for i in indices]
        if not self._can_form_convex_polygon(selected_points):
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_convex_polygon"}

        # Check if maximal
        user_size = len(indices)
        gold_size = self.gold_answer_size if self.gold_answer_size is not None else 0
        is_correct = (user_size == gold_size)
        reward = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "user_indices": indices,
            "user_size": user_size,
            "gold_size": gold_size
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

    @staticmethod
    def _cross(o: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Cross product of OA and OB vectors."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _can_form_convex_polygon(self, points: List[Tuple[int, int]]) -> bool:
        """Check if given points form a convex polygon with all points on the hull."""
        pts = sorted(set(points))
        n = len(pts)
        if n < 3:
            return False

        lower: List[Tuple[int, int]] = []
        for p in pts:
            while len(lower) >= 2 and self._cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: List[Tuple[int, int]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and self._cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        return len(hull) == n

    def sample_random_action(self) -> str:
        """Sample a random action: randomly select a subset of labels."""
        k = random.randint(3, self.N)
        indices = random.sample(range(1, self.N + 1), k)
        indices_str = " ".join(map(str, indices))
        return f"\\boxed{{{indices_str}}}"