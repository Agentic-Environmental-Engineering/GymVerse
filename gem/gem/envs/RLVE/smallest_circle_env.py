import random
import re
from math import sqrt, isfinite
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def circle_from_two_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
    center_x = (p1[0] + p2[0]) / 2.0
    center_y = (p1[1] + p2[1]) / 2.0
    radius = distance(p1, p2) / 2.0
    return (center_x, center_y), radius


def circle_from_three_points(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = (a1 * a1 + b1 * b1) / 2.0
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = (a2 * a2 + b2 * b2) / 2.0
    d = a1 * b2 - a2 * b1
    center_x = p1[0] + (c1 * b2 - c2 * b1) / d
    center_y = p1[1] + (a1 * c2 - a2 * c1) / d
    radius = distance((center_x, center_y), p1)
    return (center_x, center_y), radius


class SmallestCircleEnv(Env):
    """Environment for the Smallest Enclosing Circle problem - single-turn Q&A."""

    epsilon: float = 1e-3

    def __init__(
        self,
        N: int = 5,
        wrong_format_reward: float = -0.1,
        wrong_answer_reward: float = 0.0,
        correct_answer_reward: float = 1.0,
        **kwargs
    ):
        """
        Initialize the Smallest Circle problem environment.

        Args:
            N: Number of points to generate (must be >= 2).
            wrong_format_reward: Reward for format errors.
            wrong_answer_reward: Reward for incorrect answers.
            correct_answer_reward: Reward for correct answers.
        """
        super().__init__()
        self.N = N
        self.wrong_format_reward = wrong_format_reward
        self.wrong_answer_reward = wrong_answer_reward
        self.correct_answer_reward = correct_answer_reward

        # Runtime state
        self.points: List[Tuple[int, int]] = []
        self.opt_center: Optional[Tuple[float, float]] = None
        self.opt_radius: Optional[float] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a set of N points on a 2D plane.\n"
            "It is guaranteed that:\n"
            "(1) all the coordinates are integers;\n"
            "(2) no two points have the same coordinates;\n"
            "(3) no three points are on the same line.\n\n"
            "Your task is to find the smallest circle covering these points, measured by the radius of the circle.\n"
            "Scoring is based on feasibility and optimality of the radius.\n"
            f"The precision tolerance is {self.epsilon}.\n\n"
            "Output Format: Your output should be three floats in a single line, x, y, and r, separated by spaces.\n"
            "x and y represent the center of the circle, and r represents the radius of the circle.\n"
            "Please provide your final answer in \\boxed{x y r} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)
        assert self.N >= 2, "N should be greater than or equal to 2"

        # Generate points satisfying constraints: unique and no three colinear
        points_set = set()
        lines = set()

        def _gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a

        for _ in range(self.N):
            while True:
                x = random.randint(0, 2 * self.N)
                y = random.randint(0, 2 * self.N)
                if (x, y) in points_set:
                    continue

                colinear = False
                new_lines = set()
                for (px, py) in points_set:
                    if px == x:
                        a, b, c = 1, 0, -x
                    else:
                        a, b = py - y, x - px
                        c = -(a * x + b * y)

                    g = _gcd(abs(a), _gcd(abs(b), abs(c)))
                    if g != 0:
                        a, b, c = a // g, b // g, c // g

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

        # Compute smallest enclosing circle using randomized incremental algorithm
        pts = self.points[:]
        random.shuffle(pts)
        c = (float(pts[0][0]), float(pts[0][1]))
        r = 0.0
        for i in range(1, self.N):
            if distance((float(pts[i][0]), float(pts[i][1])), c) < r + self.epsilon:
                continue

            c = (float(pts[i][0]), float(pts[i][1]))
            r = 0.0
            for j in range(i):
                pj = (float(pts[j][0]), float(pts[j][1]))
                if distance(pj, c) < r + self.epsilon:
                    continue

                c, r = circle_from_two_points((float(pts[i][0]), float(pts[i][1])), pj)
                for k in range(j):
                    pk = (float(pts[k][0]), float(pts[k][1]))
                    if distance(pk, c) < r + self.epsilon:
                        continue

                    c, r = circle_from_three_points((float(pts[i][0]), float(pts[i][1])), pj, pk)

        self.opt_center = c
        self.opt_radius = r
        self.reference_answer = f"{c[0]} {c[1]} {r}"

        # Build problem statement
        points_str = "\n".join(f"({x}, {y})" for x, y in self.points)
        self.current_problem = (
            f"You are given a set of {self.N} points on a 2D plane.\n"
            "It is guaranteed that:\n"
            "(1) all the coordinates are integers;\n"
            "(2) no two points have the same coordinates;\n"
            "(3) no three points are on the same line.\n"
            "Below is the set of points:\n"
            f"{points_str}\n\n"
            "Your task is to find the smallest circle covering these points, measured by the radius of the circle.\n"
            "The precision tolerance is 0.001.\n\n"
            "Output Format: Your output should be three floats in a single line, x, y, and r, separated by spaces.\n"
            "x and y represent the center of the circle, and r represents the radius of the circle.\n"
            "Provide your final answer in \\boxed{x y r} format."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer."""
        content = self._parse_answer(action)
        if content is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Parse three floats x, y, r separated by spaces
        tokens = content.strip().replace(",", " ").split()
        if len(tokens) != 3:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        try:
            x, y, r = float(tokens[0]), float(tokens[1]), float(tokens[2])
        except ValueError:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Check finiteness and positivity of r
        if not (isfinite(x) and isfinite(y) and isfinite(r)):
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}
        if r <= 0:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Feasibility: all points must be covered within tolerance
        feasible = all(distance((x, y), (px, py)) <= r + self.epsilon for (px, py) in self.points)

        # Optimality: radius should match the optimal radius within tolerance
        opt_r = self.opt_radius if self.opt_radius is not None else None
        within_tolerance = False
        if opt_r is not None:
            within_tolerance = abs(r - opt_r) <= self.epsilon

        is_correct = bool(feasible and within_tolerance)

        reward = self.correct_answer_reward if is_correct else self.wrong_answer_reward

        info = {
            "correct": is_correct,
            "feasible": feasible,
            "within_tolerance": within_tolerance,
            "optimal_radius": self.opt_radius,
            "reference_answer": self.reference_answer,
            "user_answer": (x, y, r),
            "points": self.points,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random (likely incorrect) action in \\boxed{x y r} format."""
        if not self.points:
            # Fallback random values
            x = random.uniform(0.0, float(2 * max(self.N, 2)))
            y = random.uniform(0.0, float(2 * max(self.N, 2)))
            r = random.uniform(0.1, float(2 * max(self.N, 2)))
        else:
            # Use bounding circle around the bounding box as a heuristic
            min_x = min(p[0] for p in self.points)
            max_x = max(p[0] for p in self.points)
            min_y = min(p[1] for p in self.points)
            max_y = max(p[1] for p in self.points)
            x = (min_x + max_x) / 2.0
            y = (min_y + max_y) / 2.0
            r = max(distance((x, y), (px, py)) for (px, py) in self.points) + random.uniform(0.0, 1.0)

        return f"\\boxed{{{x} {y} {r}}}"