import random
import re
from array import array
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CoinSquareGameEnv(Env):
    """Single-turn environment for the Coin Square Game (optimal play scoring)."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 5,
        max_N: int = 50,
        weight_multiple: int = 2,
        **kwargs,
    ):
        """
        Initialize the CoinSquareGameEnv instance.

        Parameters:
        - N: If provided, the number of coins in the row (must be >= 5).
        - min_N: Minimum number of coins when sampling (must be >= 5).
        - max_N: Maximum number of coins when sampling.
        - weight_multiple: Upper bound multiplier for coin values; each coin C[i] is sampled in [1, N * weight_multiple].
        """
        super().__init__()
        if N is not None and N < 5:
            raise ValueError("N should be greater than or equal to 5")
        if min_N < 5:
            raise ValueError("min_N should be greater than or equal to 5")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N
        self.weight_multiple = weight_multiple

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.coins: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given N coins in a row (1-indexed from left to right). "
            "Players Alice and Bob alternately remove a positive number of leftmost coins, "
            "adding the sum of their values to their own score. The game ends when no coins remain.\n"
            "- On Alice’s first turn, she may take either 1 or 2 coins.\n"
            "- Thereafter, if the previous player took k coins, the current player may take any number of coins "
            "from 1 to min(2k, the number of remaining coins).\n"
            "Assuming both players play optimally, compute the maximum total value Alice can obtain.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
        if N < 5:
            raise ValueError("N should be greater than or equal to 5")

        # Generate coin values
        C = [random.randint(1, N * self.weight_multiple) for _ in range(N)]
        self.coins = C

        # Compute the reference answer using dynamic programming (same logic as original code)
        A = C
        S = [0] * (N + 1)
        for i in range(1, N + 1):
            S[i] = S[i - 1] + A[N - i]

        dp_rows: List[array] = [None] * (N + 1)  # type: ignore
        dp_rows[0] = array('I', [0])

        for i in range(1, N + 1):
            max_j = (i + 1) // 2
            row = array('I', [0] * (max_j + 1))
            for j in range(1, max_j + 1):
                k = 2 * j - 1
                best = row[j - 1]

                r = i - k
                if r >= 0:
                    prev_row = dp_rows[r]
                    prev_max_j = len(prev_row) - 1
                    idx = k if k <= prev_max_j else prev_max_j
                    cand = S[i] - prev_row[idx]
                    if cand > best:
                        best = cand

                r2 = i - (k + 1)
                if r2 >= 0:
                    prev_row2 = dp_rows[r2]
                    prev2_max_j = len(prev_row2) - 1
                    idx2 = (k + 1) if (k + 1) <= prev2_max_j else prev2_max_j
                    cand2 = S[i] - prev_row2[idx2]
                    if cand2 > best:
                        best = cand2

                row[j] = best

            dp_rows[i] = row

        self.reference_answer = int(dp_rows[N][1])

        # Build problem prompt
        coins_str = " ".join(f"C[{i}]={Ci}" for i, Ci in enumerate(C, start=1))
        self.current_problem = (
            f"You are given {N} coins in a row (1-indexed from left to right). "
            f"The i-th coin has value C[i]: {coins_str}\n"
            "Alice and Bob play alternately, with Alice going first. On a turn, a player removes some positive number "
            "of leftmost coins and adds the sum of their values to their own score. The game ends when no coins remain.\n"
            "Rules:\n"
            "- On Alice’s first turn, she may take either 1 or 2 coins.\n"
            "- Thereafter, if the previous player took k coins, the current player may take any number of coins "
            "from 1 to min(2k, the number of remaining coins).\n\n"
            "Assuming both players play optimally, what is the maximum total value Alice can obtain?\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {"N": N, "coins": C}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step by verifying the provided answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate parsed answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} format."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        total = sum(self.coins) if self.coins is not None else 100
        random_answer = random.randint(0, total)
        return f"\\boxed{{{random_answer}}}"