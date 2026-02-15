from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DoubleCrossCountingEnv(Env):
    """Double Cross Counting environment - single-turn Q&A."""

    def __init__(self, max_n_m: int = 10, **kwargs):
        """
        Initialize the DoubleCrossCountingEnv.

        Parameters:
        - max_n_m: Maximum value for N and M (inclusive). Must be >= 5.
        """
        super().__init__()
        if max_n_m < 5:
            raise ValueError("max_n_m should be greater than or equal to 5")
        self.max_n_m = max_n_m
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.zero_coordinates: Optional[List[tuple[int, int]]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving the Double Cross counting problem on a 0/1 matrix.\n"
            "A double cross consists of two horizontal and one vertical segments of 1s.\n"
            "Conditions for a valid double cross:\n"
            "- The two horizontal segments must not lie on adjacent rows.\n"
            "- The vertical segment must extend strictly above and strictly below the two horizontal segments.\n"
            "- The vertical segment must divide both horizontal segments into two equal halves.\n"
            "- The upper horizontal segment must be strictly shorter than the lower one.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(5, self.max_n_m)
        M = random.randint(5, self.max_n_m)
        self.N, self.M = N, M

        # Generate 0-cell coordinates (0-indexed)
        total_cells = N * M
        max_zeros = int(total_cells * 0.25)
        k = random.randint(1, max_zeros)
        all_coords = [(x, y) for x in range(N) for y in range(M)]
        zero_coordinates = random.sample(all_coords, k)
        self.zero_coordinates = zero_coordinates

        # Build problem prompt
        zero_coords_str = "\n".join(f"({x}, {y})" for x, y in zero_coordinates)
        self.current_problem = (
            f"A 0/1 matrix of size {N} × {M} is given. The coordinates of 0-cells "
            f"(0-indexed) are listed below (all unspecified cells are 1):\n"
            f"{zero_coords_str}\n\n"
            "Please compute how many valid double crosses exist in the matrix.\n"
            "Submit your final answer in \\boxed{...}."
        )

        # Compute reference answer
        self.reference_answer = self._count_double_crosses(N, M, zero_coordinates)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and check the submitted answer."""
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == user_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer answer in boxed format)."""
        # Heuristic range for random guesses
        max_guess = max(1, (self.N or 5) * (self.M or 5))
        random_answer = random.randint(0, max_guess)
        return f"\\boxed{{{random_answer}}}"

    def _count_double_crosses(self, N: int, M: int, zero_coordinates: List[tuple[int, int]]) -> int:
        """
        Core algorithm to count valid double crosses.

        Mirrors the original RLVE environment logic using 1-based indexing,
        precomputing arm lengths and sweeping with a Fenwick tree to count configurations.
        """
        size = N * M + 1  # 1-based indexing
        vis = [True] * size  # True => '1', False => '0'

        for x, y in zero_coordinates:
            x1 = x + 1
            y1 = y + 1
            vis[(x1 - 1) * M + y1] = False

        # Precompute arm lengths
        L = [0] * size  # horizontal half-length (min of both sides) – 1
        U = [0] * size  # vertical length upward – 1
        D = [0] * size  # vertical length downward – 1

        # Left sweep
        for r in range(1, N + 1):
            streak = 0
            base = (r - 1) * M
            for c in range(1, M + 1):
                idx = base + c
                streak = streak + 1 if vis[idx] else 0
                L[idx] = streak

        # Right sweep
        for r in range(1, N + 1):
            streak = 0
            base = (r - 1) * M
            for c in range(M, 0, -1):
                idx = base + c
                streak = streak + 1 if vis[idx] else 0
                L[idx] = min(L[idx], streak)
                if L[idx]:
                    L[idx] -= 1  # exclude the centre cell

        # Upward sweep
        for c in range(1, M + 1):
            streak = 0
            idx = c
            for r in range(1, N + 1):
                streak = streak + 1 if vis[idx] else 0
                U[idx] = streak - 1 if streak else 0
                idx += M

        # Downward sweep
        for c in range(1, M + 1):
            streak = 0
            idx = (N - 1) * M + c
            for r in range(N, 0, -1):
                streak = streak + 1 if vis[idx] else 0
                D[idx] = streak - 1 if streak else 0
                idx -= M

        # Fenwick tree arrays for quadratic weights
        A = [0] * (M + 1)
        B = [0] * (M + 1)
        C = [0] * (M + 1)
        tag = [0] * (M + 1)  # lazy versioning for O(#updates) clearing
        version = 1

        def lb(x: int) -> int:
            return x & -x

        def fenwick_add(x: int, w: int) -> None:
            i = x
            while i <= M:
                if tag[i] != version:
                    tag[i] = version
                    A[i] = B[i] = C[i] = 0
                A[i] += w
                B[i] += x * w
                C[i] += (x * x) * w
                i += lb(i)

        def range_add(l: int, r: int, w: int) -> None:
            if l > r or w == 0:
                return
            fenwick_add(l, w)
            fenwick_add(r + 1, -w)

        def prefix_query(x: int) -> int:
            if x <= 0:
                return 0
            s1 = s2 = s3 = 0
            i = x
            while i:
                if tag[i] == version:
                    s1 += A[i]
                    s2 += B[i]
                    s3 += C[i]
                i -= lb(i)
            res = ((x + 3) * x + 2)
            res = (res * s1 + s3 - (2 * x + 3) * s2)
            return res // 2

        # Sweep each column, building counts on the fly
        answer = 0

        for col in range(2, M):  # centres cannot be on the very first/last column
            version += 1  # “clear” the Fenwick tree for this column

            for row in range(3, N):  # need at least two rows above & below
                idx = (row - 1) * M + col

                if not vis[idx]:  # a ‘0’ breaks the vertical arm
                    version += 1  # (lazy clear)
                    continue

                # take current cell as lower horizontal bar
                if L[idx]:
                    answer += D[idx] * prefix_query(L[idx] - 1)

                # push the row immediately above as a candidate upper bar
                upper = idx - M
                if L[upper] and U[upper]:
                    range_add(1, L[upper], U[upper])

        return answer