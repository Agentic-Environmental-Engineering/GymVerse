import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GridColoringCountingEnv(Env):
    """Grid coloring counting environment - single-turn Q&A.

    You are given an N×M grid and C colors with usage counts X[i].
    You must count the number of valid colorings modulo MOD such that:
    1. No two different colors appear in the same row or the same column.
    2. Color i is used exactly X[i] times.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        max_mod: int = 10000,
        **kwargs
    ):
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        self.max_n_m = max_n_m
        self.max_mod = max_mod

        # Current problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Store parameters for info/debug
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.C: Optional[int] = None
        self.MOD: Optional[int] = None
        self.Xs: Optional[list[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Count valid colorings of a grid under given constraints.\n"
            "Output format: Provide a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate a problem until it has a positive number of solutions
        while True:
            N = random.randint(2, self.max_n_m)
            M = random.randint(2, self.max_n_m)
            total_cells = N * M

            sum_X = random.randint(1, total_cells)
            C = random.randint(1, min(N, M, sum_X))

            # Generate a random composition of sum_X into C positive integers
            if C == 1:
                Xs = [sum_X]
            else:
                deltas = random.sample(range(1, sum_X), C - 1)
                deltas.sort()
                deltas = [0] + deltas + [sum_X]
                Xs = [deltas[i + 1] - deltas[i] for i in range(C)]

            assert len(Xs) == C and all(x > 0 for x in Xs), "Xs should be a non-empty list of positive integers"

            MOD = random.randint(2, self.max_mod)

            answer = self._compute_answer(N, M, C, Xs, MOD)

            if answer > 0:
                self.N, self.M, self.C, self.MOD, self.Xs = N, M, C, MOD, Xs
                self.reference_answer = answer
                break

        # Build problem description
        xs_str = " ".join(f"X[{i}]={x}" for i, x in enumerate(self.Xs))
        problem = (
            f"You are given a grid of size {self.N} × {self.M}. You may color some cells (and leave others uncolored) "
            f"using {self.C} colors labeled from 0 to {self.C - 1}, such that:\n"
            f"1. No two different colors appear in the same row or the same column.\n"
            f"2. Color i is used exactly X[i] times. The array X is given as: {xs_str}\n\n"
            f"Please compute the number of valid colorings modulo {self.MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        self.current_problem = problem
        obs = self._get_instructions() + problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if self.MOD is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Optional range check (original environment required 0 <= answer < MOD)
        if not (0 <= user_answer < self.MOD):
            info = {
                "error": "out_of_range",
                "required_range": f"[0, {self.MOD})",
                "user_answer": user_answer,
                "reference_answer": self.reference_answer,
                "N": self.N,
                "M": self.M,
                "C": self.C,
                "Xs": self.Xs,
                "MOD": self.MOD,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "C": self.C,
            "Xs": self.Xs,
            "MOD": self.MOD,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in boxed format."""
        mod = self.MOD if self.MOD is not None else max(2, self.max_mod)
        random_answer = random.randint(0, mod - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_answer(self, N: int, M: int, C: int, Xs: list[int], MOD: int) -> int:
        """Compute the number of valid colorings using combinatorics and DP."""
        total_cells = N * M

        # Precompute binomial coefficients up to N*M modulo MOD
        comb = [[0] * (total_cells + 1) for _ in range(total_cells + 1)]
        for i in range(total_cells + 1):
            comb[i][0] = 1
            for j in range(1, i + 1):
                comb[i][j] = (comb[i - 1][j] + comb[i - 1][j - 1]) % MOD

        # f[i][j][k]: number of ways to place first k colors into an i×j subboard
        f = [[[0] * (C + 1) for _ in range(M + 1)] for __ in range(N + 1)]
        f[0][0][0] = 1

        # Process each color one by one
        for k in range(1, C + 1):
            x = Xs[k - 1]
            # g[a][b]: number of ways to place x pieces of this color into an a×b rectangle
            # so that every row and column used by it has at least one piece, by inclusion–exclusion
            g = [[0] * (M + 1) for _ in range(N + 1)]
            for a in range(1, N + 1):
                for b in range(1, M + 1):
                    if a * b < x:
                        continue
                    # total ways to choose x squares out of a*b
                    val = comb[a * b][x]
                    # subtract configurations that leave an unused border row or column
                    for la in range(1, a + 1):
                        for lb in range(1, b + 1):
                            if la < a or lb < b:
                                val -= g[la][lb] * comb[a][la] * comb[b][lb]
                    g[a][b] = val % MOD

            # Transition: add this color's placements to all previous subboards
            for i in range(1, N + 1):
                for j in range(1, M + 1):
                    # split the i×j board into an l×r part (already filled with k−1 colors)
                    # and a (i−l)×(j−r) part filled with k-th color
                    for l in range(i):
                        for r in range(j):
                            ti, tj = i - l, j - r
                            if ti * tj < x:
                                continue
                            ways = (
                                f[l][r][k - 1]
                                * g[ti][tj]
                                * comb[N - l][ti]
                                * comb[M - r][tj]
                            ) % MOD
                            f[i][j][k] = (f[i][j][k] + ways) % MOD

        # Sum over all non-empty subboards
        answer = 0
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                answer = (answer + f[i][j][C]) % MOD

        return answer