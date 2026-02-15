import random
from typing import Any, List, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinSumChebyshevDistanceEnv(Env):
    """Environment for minimizing the weighted sum of Chebyshev distances."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 2,
        max_n: int = 50,
        wrong_format_reward: float = -0.1,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed size of arrays X, Y, T. If None, N will be sampled in reset().
        - min_n: Minimum N when sampling (must be >= 2).
        - max_n: Maximum N when sampling (must be >= min_n).
        - wrong_format_reward: Reward for format error (default: -0.1).
        """
        super().__init__()
        assert min_n >= 2, "min_n should be greater than or equal to 2"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"
        if N is not None:
            assert N >= 2, "N should be greater than or equal to 2"

        self.fixed_N: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        self.wrong_format_reward: float = wrong_format_reward

        self.current_problem: Optional[str] = None
        self.X: List[int] = []
        self.Y: List[int] = []
        self.T: List[int] = []
        self.N: int = min_n

        self.optimal_point: Tuple[int, int] = (0, 0)
        self.minimal_cost: int = 0

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a weighted minimum sum of Chebyshev distances problem.\n"
            "Please provide your answer in \\boxed{...} format, where the content is two integers: x y.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        self.N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_n, self.max_n)
        assert self.N >= 2, "N should be greater than or equal to 2"

        # Generate arrays X, Y, T
        self.X = [random.randint(1, 2 * self.N) for _ in range(self.N)]
        self.Y = [random.randint(1, 2 * self.N) for _ in range(self.N)]
        self.T = [random.randint(1, self.N) for _ in range(self.N)]

        # Prepare rotated coordinates and original points with weights
        A: List[List[int]] = []  # [x_rot, weight]
        B: List[List[int]] = []  # [y_rot, weight]
        C: List[Tuple[int, int, int]] = []  # (u, v, t)

        for u, v, t in zip(self.X, self.Y, self.T):
            x_rot = u + v
            y_rot = u - v
            A.append([x_rot, t])
            B.append([y_rot, t])
            C.append((u, v, t))

        # Sort by rotated coordinates
        A.sort(key=lambda item: item[0])
        B.sort(key=lambda item: item[0])

        def weighted_median(arr: List[List[int]]) -> int:
            """
            Find weighted median of sorted array arr where each element is [coord, weight].
            Uses two-pointer elimination to find a coordinate where cumulative weight is balanced.
            Note: This function mutates the weight values in arr.
            """
            l, r = 0, len(arr) - 1
            while l < r:
                if arr[l][1] < arr[r][1]:
                    arr[r][1] -= arr[l][1]
                    l += 1
                elif arr[l][1] > arr[r][1]:
                    arr[l][1] -= arr[r][1]
                    r -= 1
                else:
                    l += 1
                    r -= 1
            return arr[l][0]

        # Compute medians in rotated space
        posx = weighted_median(A)
        posy = weighted_median(B)

        # Convert back to original coordinates (truncate towards zero)
        xx = int((posx + posy) / 2)
        yy = int((posx - posy) / 2)

        # Check the four nearest integer points
        candidates: List[Tuple[int, int]] = [
            (xx, yy),
            (xx + 1, yy),
            (xx, yy + 1),
            (xx + 1, yy + 1),
        ]

        best_cost: Optional[int] = None
        best_point: Tuple[int, int] = (xx, yy)

        for x, y in candidates:
            cost = 0
            for u, v, t in C:
                cost += max(abs(x - u), abs(y - v)) * t
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_point = (x, y)

        assert best_cost is not None
        self.optimal_point = best_point
        self.minimal_cost = best_cost

        # Build problem prompt
        lines = []
        for i, (Xi, Yi, Ti) in enumerate(zip(self.X, self.Y, self.T)):
            lines.append(f"X[{i}]={Xi} Y[{i}]={Yi} T[{i}]={Ti}")
        arrays_block = "\n".join(lines)

        self.current_problem = (
            f"You are given three arrays X, Y, and T, each of length {self.N}:\n"
            f"{arrays_block}\n\n"
            f"Please find an integer point (x, y) such that the following sum is minimized: "
            f"sum over 0 <= i < {self.N} of max(|x - X[i]|, |y - Y[i]|) * T[i].\n"
            f"Output Format: Your final answer should be two integers 'x y' inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process a single action (answer), verify, and terminate."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Parse two integers x and y from boxed content
        tokens = boxed_content.replace(",", " ").split()
        if len(tokens) != 2:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        try:
            x = int(tokens[0])
            y = int(tokens[1])
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Compute user's cost
        user_cost = sum(max(abs(x - Xi), abs(y - Yi)) * Ti for Xi, Yi, Ti in zip(self.X, self.Y, self.T))
        is_correct = (user_cost == self.minimal_cost)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "user_point": (x, y),
            "user_cost": user_cost,
            "optimal_point": self.optimal_point,
            "minimal_cost": self.minimal_cost,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns None if not found."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the correct format."""
        # Sample x and y within a reasonable range based on N
        x = random.randint(0, 2 * self.N)
        y = random.randint(0, 2 * self.N)
        return f"\\boxed{{{x} {y}}}"