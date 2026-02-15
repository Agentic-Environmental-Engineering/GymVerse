from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinSumDistanceSquareEnv(Env):
    """Environment for the 'Minimum Sum of Squared Distances to Groups on the X-axis' problem - single-turn Q&A.

    The task:
    - There are N groups of points on the x-axis.
    - Choose a point X on the x-axis.
    - For each group i, cost[i] is the square of the minimum distance from X to any point in that group.
    - Minimize the total cost: sum(cost[i]).
    - It can be shown an optimal solution exists with X = X' / N, where X' is an integer. Output this integer X'.
    """

    def __init__(
        self,
        M: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - M: controls both the number of generated coordinates and the coordinate range [-M, M].
             Must be >= 2.
        """
        super().__init__()
        assert M >= 2, "M should be greater than or equal to 2"
        self.M: int = M

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.minimal_total_cost: Optional[int] = None

        # Generated data
        self.N: Optional[int] = None
        self.points: Optional[List[List[int]]] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving an optimization problem on the x-axis.\n"
            "Please provide your final answer as a single integer X' in \\boxed{...} format.\n"
            "No additional text is required.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        M = self.M
        # Generate coordinates and group assignments
        coordinates = [random.randint(-M, +M) for _ in range(M)]

        N = random.randint(2, M)
        self.N = N

        # Ensure each group has at least one point
        belongings = list(range(N)) + [random.randint(0, N - 1) for _ in range(M - N)]
        random.shuffle(belongings)

        points: List[List[int]] = [[] for _ in range(N)]
        for coordinate, belonging in zip(coordinates, belongings):
            points[belonging].append(coordinate)
        self.points = points

        # Prepare per-group sorted lists
        F: List[List[int]] = [[] for _ in range(N)]
        for p, xs in enumerate(points):
            assert len(xs) > 0, "Each group must have at least one point"
            for x in xs:
                F[p].append(x)
            F[p].sort()

        # Build consecutive-pair events and initialize accumulators
        events: List[Tuple[int, int]] = []
        O = 0  # sum of squares of chosen representatives
        E = 0  # sum of chosen representatives

        for lst in F:
            lst.sort()
            O += lst[0] * lst[0]
            E += lst[0]
            for j in range(1, len(lst)):
                events.append((lst[j - 1], lst[j]))

        # Sort events by midpoint (a + b)
        events.sort(key=lambda ab: ab[0] + ab[1])

        # Find best E that minimizes N*O - E^2
        best_value = N * O - E * E
        best_E = E

        for a, b in events:
            O += b * b - a * a
            E += b - a
            value = N * O - E * E
            if value < best_value:
                best_value = value
                best_E = E

        # Store answers
        self.reference_answer = best_E
        self.minimal_total_cost = self._compute_total_cost(best_E)

        # Build problem text
        points_description = "\n".join(
            f"Group {i}: " + " ".join(map(str, xs)) for i, xs in enumerate(points)
        )
        self.current_problem = (
            f"There are {N} groups of points located on the x-axis. The coordinates of each group are given as follows:\n"
            f"{points_description}\n\n"
            f"Your task is to choose a point X on the x-axis. For each group i (0 ≤ i < {N}), define cost[i] as the square of the minimum distance "
            f"from X to any point in that group: cost[i] = (min(abs(X - x_i[j])))^2, where x_i[j] is the j-th point in group i.\n"
            f"Please find the value of X that minimizes the total cost, i.e., the sum of all cost[i].\n\n"
            f"It can be shown that there exists an optimal solution X = X' / {N}, where X' is an integer.\n"
            f"Output Format: Your final answer should be the integer X' in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_total_cost(self, X_prime: int) -> int:
        """Compute the total cost given X' (where X = X'/N)."""
        assert self.N is not None and self.points is not None, "Problem not initialized"
        N = self.N
        points = self.points
        # (X_prime / N - x)^2 = (X_prime - N * x)^2 / N^2
        # For comparison of minimality, summing (X_prime - N * x)^2 over groups suffices (constant 1/N^2 factor).
        return sum(min((X_prime - N * x) ** 2 for x in xs) for xs in points)

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare answer
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "minimal_total_cost": self.minimal_total_cost,
            "N": self.N,
            "points": self.points,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer for X')."""
        # Use a plausible range for X' based on N and M
        if self.N is not None:
            bound = max(1, self.N * self.M)
        else:
            bound = max(1, self.M * self.M)
        random_answer = random.randint(-bound, bound)
        return f"\\boxed{{{random_answer}}}"