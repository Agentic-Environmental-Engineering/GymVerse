import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SkaRockGardenEnv(Env):
    """Ska Rock Garden environment - single-turn Q&A.

    Problem:
    - There are N points in a 2D plane. Point i is (X[i], Y[i]).
    - You may optionally swap coordinates of some points: (x, y) -> (y, x), each swap costs M[i].
    - Goal:
      1) Minimize the perimeter of the smallest axis-aligned rectangle enclosing all points after swapping.
      2) If multiple strategies yield the same minimum perimeter, choose the one with minimal total swap cost.

    Answer format:
    - A single line of N characters in \\boxed{...}, where:
      '0' means do NOT swap point i, and '1' means DO swap point i.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 50,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
        - N: Optional fixed problem size. If provided, must be >= 3.
        - min_N: Minimum N used when sampling randomly. Must be >= 3.
        - max_N: Maximum N used when sampling randomly. Must be >= min_N.

        Notes:
        - If N is None, a random N in [min_N, max_N] is used at reset().
        """
        super().__init__()
        if N is not None:
            assert N >= 3, "N should be greater than or equal to 3"
        assert min_N >= 3, "min_N should be greater than or equal to 3"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"

        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N

        # Problem data and answers
        self.N: Optional[int] = None
        self.X: List[int] = []
        self.Y: List[int] = []
        self.M: List[int] = []

        self.current_problem: Optional[str] = None
        self.reference_assignment: Optional[str] = None
        self.gold_perimeter: Optional[int] = None
        self.gold_cost: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a 2D coordinate swapping and bounding rectangle minimization problem.\n"
            "Please provide your answer as a single N-character 0/1 string wrapped in \\boxed{...}.\n"
            "- '0' means do NOT swap the point.\n"
            "- '1' means DO swap the point.\n"
            "No spaces or separators are allowed inside the box.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N

        # Generate X, Y, M
        self.X = [random.randint(0, 2 * N) for _ in range(N)]
        self.Y = [random.randint(0, 2 * N) for _ in range(N)]
        self.M = [random.randint(1, N) for _ in range(N)]

        # Compute gold perimeter and minimal cost assignment
        self._compute_reference()

        # Build problem prompt
        lines = "\n".join(
            f"X[{i}]={Xi} Y[{i}]={Yi} M[{i}]={Mi}"
            for i, (Xi, Yi, Mi) in enumerate(zip(self.X, self.Y, self.M))
        )
        self.current_problem = (
            f"There are {N} points in a 2D plane, where the i-th point is (X[i], Y[i]) for 0 ≤ i < {N}. "
            f"Each point has a cost M[i] to swap its coordinates (i.e., swapping (x, y) becomes (y, x)). "
            f"Your goal is as follows:\n"
            f"- First, minimize the total perimeter of the smallest axis-aligned rectangle that can enclose all points after some of them are optionally swapped. "
            f"The perimeter is 2 × ((max_x - min_x) + (max_y - min_y)), where max_x and min_x are the maximum and minimum x-coordinates after your swaps (similarly for y).\n"
            f"- If multiple swap strategies result in the same minimum perimeter, choose the one with the smallest total swap cost (i.e., sum of M[i] for all swapped points).\n\n"
            f"X, Y, and M are given as follows:\n{lines}\n\n"
            f"Output Format: Output a single line of {N} characters (no spaces or any other kinds of separators) wrapped in \\boxed{{...}}. "
            f"The i-th character should be:\n"
            f"- '0' if you do NOT swap point i,\n"
            f"- '1' if you DO swap point i."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference(self) -> None:
        """Compute the minimal perimeter and the minimal swap cost assignment."""
        assert self.N is not None
        N = self.N
        X, Y, M = self.X, self.Y, self.M

        INF = (max(max(X), max(Y)) + 1) * 2
        lx = INF
        rx = -INF
        ly = INF
        ry = -INF

        # Determine the minimal enclosing rectangle assuming a particular rule:
        # if x <= y: treat as not swapped; else: treat as swapped.
        for i in range(N):
            x, y = X[i], Y[i]
            if x <= y:
                if x < lx:
                    lx = x
                if x > rx:
                    rx = x
                if y < ly:
                    ly = y
                if y > ry:
                    ry = y
            else:
                # effectively swapped
                if y < lx:
                    lx = y
                if y > rx:
                    rx = y
                if x < ly:
                    ly = x
                if x > ry:
                    ry = x

        fence_length = 2 * ((rx - lx) + (ry - ly))

        best_weight = sum(M)  # initialize with worst-case total cost
        best_assign: Optional[List[int]] = None

        def try_bounds(lx0: int, rx0: int, ly0: int, ry0: int) -> Tuple[Optional[int], Optional[List[int]]]:
            """Try using bounds [lx0, rx0] × [ly0, ry0], returning (weight, assignment)
            or (None, None) if impossible."""
            total = 0
            assign = [0] * N
            for i in range(N):
                x, y = X[i], Y[i]
                if lx0 <= x <= rx0 and ly0 <= y <= ry0:
                    assign[i] = 0
                elif lx0 <= y <= rx0 and ly0 <= x <= ry0:
                    assign[i] = 1
                    total += M[i]
                else:
                    return None, None
            return total, assign

        # Try the 4 possible ways of interpreting the bounding box
        for (a, b, c, d) in (
            (lx, rx, ly, ry),
            (lx, ry, ly, rx),
            (ly, rx, lx, ry),
            (ly, ry, lx, rx),
        ):
            w, assn = try_bounds(a, b, c, d)
            if w is not None and w < best_weight:
                best_weight = w
                best_assign = assn

        assert best_assign is not None, "Failed to compute a valid minimal-cost assignment."

        # Store gold answers
        self.gold_perimeter = fence_length
        self.gold_cost = best_weight
        self.reference_assignment = "".join(map(str, best_assign))

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse, verify, and score the provided answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate format: length and characters
        assert self.N is not None
        if len(parsed) != self.N or any(c not in "01" for c in parsed):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Apply swaps according to user's assignment
        X = self.X[:]
        Y = self.Y[:]
        user_cost = 0
        for i, c in enumerate(parsed):
            if c == "1":
                X[i], Y[i] = Y[i], X[i]
                user_cost += self.M[i]

        # Compute user's perimeter
        user_perimeter = 2 * ((max(X) - min(X)) + (max(Y) - min(Y)))

        # Compare with gold
        assert self.gold_perimeter is not None and self.gold_cost is not None and self.reference_assignment is not None
        is_correct = (user_perimeter == self.gold_perimeter and user_cost == self.gold_cost)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "user_assignment": parsed,
            "reference_assignment": self.reference_assignment,
            "user_perimeter": user_perimeter,
            "user_cost": user_cost,
            "gold_perimeter": self.gold_perimeter,
            "gold_cost": self.gold_cost,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action in \\boxed{...} format."""
        assert self.N is not None
        random_bits = "".join(str(random.randint(0, 1)) for _ in range(self.N))
        return f"\\boxed{{{random_bits}}}"