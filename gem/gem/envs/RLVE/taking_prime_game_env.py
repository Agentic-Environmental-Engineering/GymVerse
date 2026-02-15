from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TakingPrimeGameEnv(Env):
    """
    Taking Prime Game environment (single-turn Q&A).

    There are N stones in a pile and two players: Stan and his opponent. On each turn,
    a player may remove any prime number of stones from the pile. A player who cannot
    make a move loses the game.

    Stan goes first. Both players play optimally:
    - If a player is guaranteed to win, they will try to win in the minimum number of moves.
    - If a player is guaranteed to lose, they will try to delay the loss as much as possible.

    The model must output:
    - The total number of moves (both players’) until Stan wins (if he must win), or
    - -1 (if he must lose).

    Output format: The final answer must be written in \\boxed{...}.
    """

    def __init__(
        self,
        max_n: int = 1000,
        lose_probability: float = 0.2,
        **kwargs
    ):
        super().__init__()
        assert max_n >= 1, "max_n should be greater than or equal to 1"
        assert 0.0 <= lose_probability <= 1.0, "lose_probability must be in [0, 1]"
        self.max_n = max_n
        self.lose_probability = lose_probability

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_n: Optional[int] = None

        # Cached precomputation
        self._primes: Optional[List[int]] = None
        self._win: Optional[List[bool]] = None
        self._dp_moves: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a combinatorial game problem.\n"
            "On each turn, a player may remove any prime number of stones from the pile.\n"
            "A player who cannot make a move loses the game.\n"
            "Stan moves first. Both players play optimally:\n"
            "- Winners minimize the number of moves to victory.\n"
            "- Losers delay the loss as much as possible.\n\n"
            "Output Format:\n"
            "Your final answer should be a single integer in \\boxed{...}:\n"
            "- The total number of moves until Stan wins (if he must win), or\n"
            "- -1 (if he must lose).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new single-turn problem."""
        super().reset(seed)

        # Precompute primes and DP if not cached
        if self._primes is None or self._win is None or self._dp_moves is None:
            self._primes = self._sieve_primes(self.max_n)
            self._win, self._dp_moves = self._compute_dp(self.max_n, self._primes)

        # Determine whether to pick a losing or winning position based on probability
        want_lose = random.random() < self.lose_probability

        # Sample N consistent with the desired outcome
        while True:
            N = random.randint(1, self.max_n)
            if (self._win[N] is False) == want_lose:
                break

        self.current_n = N
        if self._win[N]:
            self.reference_answer = self._dp_moves[N]
        else:
            self.reference_answer = -1

        self.current_problem = (
            f"There are {N} stones in a pile and two players: Stan and his opponent. "
            f"On each turn, a player may remove any prime number of stones from the pile. "
            f"A player who cannot make a move loses the game.\n\n"
            f"Stan goes first. Both players play optimally:\n"
            f"- If a player is guaranteed to win, they will try to win in the minimum number of moves possible.\n"
            f"- If a player is guaranteed to lose, they will try to delay the loss as much as possible.\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single integer in \\boxed{{...}}:\n"
            f"- The total number of moves (both players’) until Stan wins (if he must win), or\n"
            f"- -1 (if he must lose)."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the answer and terminate."""
        answer_text = self._parse_answer(action)

        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
            is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
            reward: float = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_n
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        # Simple heuristic: sample an integer in a reasonable range
        random_answer = random.randint(-1, self.max_n)  # could be -1 or number of moves
        return f"\\boxed{{{random_answer}}}"

    def _sieve_primes(self, limit: int) -> List[int]:
        """Sieve of Eratosthenes to generate primes up to 'limit'."""
        if limit < 2:
            return []
        is_prime = [True] * (limit + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if is_prime[i]:
                step_start = i * i
                for j in range(step_start, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]

    def _compute_dp(self, max_n: int, primes: List[int]) -> Tuple[List[bool], List[int]]:
        """
        Compute DP arrays:
        - win[i]: whether the current player has a winning strategy with i stones.
        - dp_moves[i]: if win[i] is True, the minimum number of moves to force a win;
                       otherwise, the maximum number of moves to delay the loss,
                       under optimal play.
        """
        win = [False] * (max_n + 1)
        dp_moves = [0] * (max_n + 1)

        # Base cases:
        # i = 0: losing position, dp_moves[0] = 0
        # i = 1: losing position (no prime move), dp_moves[1] = 0

        for i in range(2, max_n + 1):
            min_moves = (max_n + 1) * 100
            max_moves = 0
            has_winning_move = False

            for p in primes:
                if p > i:
                    break
                if not win[i - p]:
                    has_winning_move = True
                    # Move to a losing state for the opponent; minimize total moves to win
                    min_moves = min(min_moves, dp_moves[i - p] + 1)
                else:
                    # Move to a winning state for the opponent; maximize delay
                    max_moves = max(max_moves, dp_moves[i - p] + 1)

            if has_winning_move:
                win[i] = True
                dp_moves[i] = min_moves
            else:
                win[i] = False
                dp_moves[i] = max_moves

        return win, dp_moves