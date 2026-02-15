import random
import re
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class KiddingMeEnv(Env):
    """Single-turn environment for counting specific matrices under constraints."""

    def __init__(
        self,
        max_n_m: int = 10,
        mod_choices: Tuple[int, int] = (10**9 + 7, 998244353),
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_n_m: maximum bound for N and M (must be >= 2)
        - mod_choices: available modulus choices
        """
        super().__init__()
        if max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")
        self.max_n_m = max_n_m
        self.mod_choices = mod_choices

        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.MOD: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem about counting matrices.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate parameters
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)
        self.MOD = random.choice(self.mod_choices)

        # Build the problem description
        self.current_problem = (
            f"Please compute the number of {self.N} × {self.M} matrices X, such that:\n"
            f"- For each 1 <= i <= {self.N}, 1 <= j <= {self.M}, we have 0 <= X[i][j] <= {self.M}\n"
            f"- For each 1 <= i <= {self.N}, 1 <= j < {self.M}, we have X[i][j] < X[i][j + 1]\n"
            f"- For each 1 < i <= {self.N}, 1 <= j < {self.M}, we have X[i][j] < X[i - 1][j + 1]\n\n"
            f"Please output the result modulo {self.MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute the reference answer
        self.reference_answer = self._compute_reference_answer(self.N, self.M, self.MOD)

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": self.N,
            "M": self.M,
            "MOD": self.MOD,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the provided answer."""
        # Parse the boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare the answer
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # If MOD is known, optionally check the range as in original logic
        out_of_range = False
        if self.MOD is not None:
            if not (0 <= user_answer < self.MOD):
                out_of_range = True

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "MOD": self.MOD,
        }
        if out_of_range:
            info["error"] = "out_of_range"

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random boxed integer)."""
        if self.MOD is not None and self.MOD > 0:
            random_answer = random.randint(0, self.MOD - 1)
        else:
            random_answer = 0
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, N: int, M: int, MOD: int) -> int:
        """Compute the reference answer using the inclusion-exclusion algorithm with reflections."""
        # Pre-compute factorials and inverse factorials
        UP = max(N, M) * 3 + 5  # safe upper bound for every x + y that appears
        inv = [0] * (UP + 1)    # modular inverses of 1 … UP
        fact = [1] * (UP + 1)   # factorials
        inv_fact = [1] * (UP + 1)  # inverse factorials (1 / k!)

        inv[1] = 1
        for i in range(2, UP + 1):
            inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD

        for i in range(1, UP + 1):
            fact[i] = fact[i - 1] * i % MOD
            inv_fact[i] = inv_fact[i - 1] * inv[i] % MOD

        def comb(x: int, y: int) -> int:
            """Return C(x + y, x) under MOD (return 0 if any index is negative)."""
            if x < 0 or y < 0:
                return 0
            return fact[x + y] * inv_fact[x] % MOD * inv_fact[y] % MOD

        def flip1(x: int, y: int) -> tuple[int, int]:
            """Perform the first reflection: swap, then (x--, y++)."""
            return y - 1, x + 1

        def flip2(x: int, y: int) -> tuple[int, int]:
            """Perform the second reflection: swap, then (x += M + 2, y -= M + 2)."""
            return y + M + 2, x - (M + 2)

        # Main inclusion–exclusion
        x, y = N + M + 1, N
        ans = comb(x, y)

        while x >= 0 and y >= 0:
            x, y = flip1(x, y)
            ans = (ans - comb(x, y)) % MOD
            x, y = flip2(x, y)
            ans = (ans + comb(x, y)) % MOD

        x, y = N + M + 1, N
        while x >= 0 and y >= 0:
            x, y = flip2(x, y)
            ans = (ans - comb(x, y)) % MOD
            x, y = flip1(x, y)
            ans = (ans + comb(x, y)) % MOD

        return ans