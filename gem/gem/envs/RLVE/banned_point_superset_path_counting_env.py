import random
from typing import Any, List, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BannedPointSupersetPathCountingEnv(Env):
    """
    Single-turn environment for counting paths in 3D with bitwise-AND-constrained moves,
    avoiding a given set of forbidden points, and answering modulo MOD.
    """

    def __init__(
        self,
        max_n_m_r: int = 10,
        max_o: int = 10,
        max_mod: int = 10000,
        **kwargs,
    ):
        """
        Initialize the environment.

        Args:
            max_n_m_r: Upper bound for N, M, R (inclusive) when generating the problem.
                       Must be >= 1.
            max_o: Upper bound for the number of obstacles O. Must be >= 1.
            max_mod: Upper bound for the modulo MOD (exclusive upper bound for random generation + 1).
        """
        super().__init__()
        assert max_n_m_r >= 1, "max_n_m_r should be greater than or equal to 1"
        assert max_o >= 1, "max_o should be greater than or equal to 1"
        self.max_n_m_r = max_n_m_r
        self.max_o = max_o
        self.max_mod = max_mod

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.R: Optional[int] = None
        self.MOD: Optional[int] = None
        self.obstacles: Optional[List[Tuple[int, int, int]]] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial path counting problem in 3D space with bitwise-AND constraints.\n"
            "You must provide your final answer modulo MOD.\n"
            "Output Format: Return a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate N, M, R ensuring at least one non-trivial superset point is available
        while True:
            N = random.randint(0, self.max_n_m_r)
            M = random.randint(0, self.max_n_m_r)
            R = random.randint(0, self.max_n_m_r)
            total_superset_points = (2 ** N.bit_count()) * (2 ** M.bit_count()) * (2 ** R.bit_count())
            # Excluding origin and target
            if total_superset_points - 2 >= 1:
                break

        # Determine upper bound for obstacles based on available superset points
        max_o_cap = min(self.max_o, total_superset_points - 2)
        O = random.randint(1, max_o_cap)

        # Helper to get set bits as powers of two
        def convert_to_bits(x: int) -> List[int]:
            result: List[int] = []
            bit = 1
            while bit <= x:
                if x & bit:
                    result.append(bit)
                bit <<= 1
            return result

        # Helper to sample a random subset sum of the given bit list
        def random_subset(bits: List[int]) -> int:
            chosen = random.sample(bits, random.randint(0, len(bits)))
            return sum(chosen)

        N_bits, M_bits, R_bits = convert_to_bits(N), convert_to_bits(M), convert_to_bits(R)

        obstacles_set = set()
        while len(obstacles_set) < O:
            x = random_subset(N_bits)
            y = random_subset(M_bits)
            z = random_subset(R_bits)
            if (x, y, z) != (0, 0, 0) and (x, y, z) != (N, M, R) and (x, y, z) not in obstacles_set:
                obstacles_set.add((x, y, z))

        obstacles = list(obstacles_set)
        random.shuffle(obstacles)

        MOD = random.randint(2, self.max_mod)

        # Store generated parameters
        self.N = N
        self.M = M
        self.R = R
        self.MOD = MOD
        self.obstacles = obstacles

        # Compute reference answer using the original algorithm
        self.reference_answer = self._compute_reference_answer(N, M, R, obstacles, MOD)

        # Build the problem prompt
        obstacles_str = "\n".join(f"({x}, {y}, {z})" for x, y, z in obstacles)
        problem_prompt = (
            f"In a three-dimensional space, you start at point (0, 0, 0) and want to reach the point ({N}, {M}, {R}). "
            f"At each step, if you are currently at (x, y, z), you may move to a new (different from the current one) point of one of the following types:\n"
            f"1. (x', y, z) such that x AND x' = x\n"
            f"2. (x, y', z) such that y AND y' = y\n"
            f"3. (x, y, z') such that z AND z' = z\n"
            f"(AND refers to the bitwise AND operation.)\n\n"
            f"You are not allowed to visit any of the following points:\n"
            f"{obstacles_str}\n\n"
            f"Please count the number of distinct valid paths from (0, 0, 0) to ({N}, {M}, {R}) that avoid all forbidden points. "
            f"Output the result modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        self.current_problem = problem_prompt
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(
        self,
        N: int,
        M: int,
        R: int,
        obstacles: List[Tuple[int, int, int]],
        MOD: int
    ) -> int:
        """Compute the reference answer using the original DP and inclusion-exclusion logic."""
        points = [(0, 0, 0)] + obstacles
        points.sort()
        points.append((N, M, R))
        total = len(points)

        dx = N.bit_count()
        dy = M.bit_count()
        dz = R.bit_count()
        max_d = max(dx, dy, dz)

        # Precompute binomial coefficients up to max_d
        binom = [[0] * (max_d + 1) for _ in range(max_d + 1)]
        for i in range(max_d + 1):
            binom[i][0] = 1
            for j in range(1, i + 1):
                binom[i][j] = (binom[i - 1][j - 1] + binom[i - 1][j]) % MOD

        # Precompute f[x][y][z]: number of ways from (0,0,0) to a diff-vector with
        # x one-bit-flips in X, y flips in Y, z flips in Z (ignoring obstacles).
        f = [[[0] * (dz + 1) for _ in range(dy + 1)] for __ in range(dx + 1)]
        f[0][0][0] = 1
        for x in range(dx + 1):
            for y in range(dy + 1):
                for z in range(dz + 1):
                    if x == y == z == 0:
                        continue
                    val = 0
                    for i in range(x):
                        val = (val + f[i][y][z] * binom[x][i]) % MOD
                    for j in range(y):
                        val = (val + f[x][j][z] * binom[y][j]) % MOD
                    for k in range(z):
                        val = (val + f[x][y][k] * binom[z][k]) % MOD
                    f[x][y][z] = val

        # Inclusion-exclusion over the sorted points
        g = [0] * total
        g[0] = 1
        for i in range(1, total):
            xi, yi, zi = points[i]
            acc = 0
            for j in range(i):
                xj, yj, zj = points[j]
                if (xj & xi) == xj and (yj & yi) == yj and (zj & zi) == zj:
                    bx = (xi ^ xj).bit_count()
                    by = (yi ^ yj).bit_count()
                    bz = (zi ^ zj).bit_count()
                    acc = (acc + g[j] * f[bx][by][bz]) % MOD
            g[i] = (-acc) % MOD

        # The answer is -g[last] mod MOD
        return (-g[-1]) % MOD

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Parse the user's answer and return the reward and terminal state."""
        answer_str = self._parse_answer(action)

        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if self.MOD is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_ready"}

        # Optional range validation (0 <= answer < MOD)
        if not (0 <= user_answer < self.MOD):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "error": "out_of_range",
                "MOD": self.MOD,
                "N": self.N,
                "M": self.M,
                "R": self.R,
                "obstacles": self.obstacles,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "MOD": self.MOD,
            "N": self.N,
            "M": self.M,
            "R": self.R,
            "obstacles": self.obstacles,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content as the answer."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action in \\boxed{...} format."""
        if self.MOD is None:
            # Fallback if called before reset
            value = random.randint(0, max(1, self.max_mod) - 1)
        else:
            value = random.randint(0, self.MOD - 1)
        return f"\\boxed{{{value}}}"