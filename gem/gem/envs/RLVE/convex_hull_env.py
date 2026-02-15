from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from math import gcd as math_gcd
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ConvexHullEnv(Env):
    """Convex hull area (2x) computation environment - single-turn Q&A."""

    def __init__(
        self,
        N: int = 8,
        **kwargs
    ):
        """
        Initialize the ConvexHullEnv.

        Args:
            N: Number of points to generate (must be >= 3).
        """
        super().__init__()
        self.N: int = N
        self.points: List[Tuple[int, int]] = []
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a computational geometry problem about convex hulls.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Returns:
            observation: The full problem description string.
            info: Auxiliary information dictionary.
        """
        super().reset(seed)
        assert self.N >= 3, "N should be greater than or equal to 3"

        # Generate points with no duplicate coordinates and no three colinear
        points_set = set()
        lines = set()  # normalized line representations ax + by + c = 0

        def gcd3(a: int, b: int, c: int) -> int:
            return math_gcd(abs(a), math_gcd(abs(b), abs(c)))

        for _ in range(self.N):
            while True:
                x = random.randint(0, self.N)
                y = random.randint(0, self.N)
                if (x, y) in points_set:
                    continue

                colinear = False
                new_lines = set()

                for (px, py) in points_set:
                    if px == x:
                        a, b, c = 1, 0, -x
                    else:
                        a = py - y
                        b = x - px
                        c = -(a * x + b * y)

                    g = gcd3(a, b, c)
                    if g != 0:
                        a //= g
                        b //= g
                        c //= g

                    if a < 0:
                        a, b, c = -a, -b, -c
                    elif a == 0 and b < 0:
                        b, c = -b, -c

                    if (a, b, c) in lines:
                        colinear = True
                        break

                    new_lines.add((a, b, c))

                if colinear:
                    continue

                points_set.add((x, y))
                lines.update(new_lines)
                break

        self.points = list(points_set)

        # Compute convex hull using Andrew's algorithm (monotone chain)
        def cross(o: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        pts = sorted(self.points, key=lambda p: (p[0], p[1]))
        lower: List[Tuple[int, int]] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: List[Tuple[int, int]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]

        # Compute 2 * area via shoelace formula (sum over edges x1*y2 - x2*y1)
        area2 = 0
        for i in range(len(hull)):
            j = (i + 1) % len(hull)
            x1, y1 = hull[i]
            x2, y2 = hull[j]
            area2 += x1 * y2 - x2 * y1

        self.reference_answer = abs(area2)

        # Build the problem prompt
        points_str = "\n".join(f"({x}, {y})" for x, y in self.points)
        self.current_problem = (
            f"You are given a set of {self.N} points on a 2D plane labeled from 0 to {self.N - 1}.\n"
            f"It is guaranteed that:\n"
            f"(1) all the coordinates are integers;\n"
            f"(2) no two points have the same coordinates;\n"
            f"(3) no three points are on the same line.\n"
            f"Below is the set of points:\n"
            f"{points_str}\n\n"
            f"Your task is to find the convex hull of these points, which is the smallest convex polygon that contains all the points.\n\n"
            f"Output Format: Your output should be one single integer, representing the value of 2 times the area of the convex hull.\n"
            f"Please put your final answer in \\boxed{{...}}."
        )

        observation = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": self.N,
            "points": self.points[:],
            "hull_area_times_2": self.reference_answer,
        }
        return observation, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer.

        Args:
            action: The model's answer text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE since this is single-turn.
            reward: 1.0 if correct, 0.0 if incorrect, -0.1 if format error.
            terminated: Always True after one step.
            truncated: Always False.
            info: Additional info including correctness and reference answer.
        """
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate integer answer
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
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the expected \\boxed{...} format."""
        # Heuristic random guess
        guess = random.randint(1, max(1, 2 * self.N * self.N))
        return f"\\boxed{{{guess}}}"