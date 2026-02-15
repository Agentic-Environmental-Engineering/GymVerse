from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class StoneGameEnv(Env):
    """Stone splitting impartial game environment - single-turn Q&A.

    Stan and Ollie play on heaps of stones. On a turn, a player must choose a heap
    with at least F stones and split it into M >= 2 heaps whose sizes differ by at most 1.
    If a player cannot move, they lose. Both players play optimally. Determine the winner.
    """

    def __init__(
        self,
        max_sum: int = 200,
        n_max: int = 100,
        **kwargs
    ):
        super().__init__()
        assert isinstance(max_sum, int) and max_sum >= 2, "max_sum should be an integer >= 2"
        assert isinstance(n_max, int) and n_max >= 1, "n_max should be an integer >= 1"
        self.max_sum = max_sum
        self.n_max = n_max

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.N: Optional[int] = None
        self.F: Optional[int] = None
        self.stones: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial game problem (Stone splitting).\n"
            "Please provide your final answer in \\boxed{...} format using exactly one word: Stan or Ollie.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new single-turn problem."""
        super().reset(seed)

        # Generate problem parameters
        total = random.randint(2, self.max_sum)
        n_upper = min(total // 2, self.n_max)
        N = random.randint(1, n_upper)

        if N == 1:
            stones = [total]
        else:
            cuts = sorted(random.sample(range(1, total), N - 1))
            stones = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, N - 1)] + [total - cuts[-1]]

        F = random.randint(1, max(stones) + 1)

        # Compute reference answer using Sprague-Grundy evaluation
        winner = "Stan" if self._check_winning(stones, F) else "Ollie"

        # Store state
        self.N = N
        self.F = F
        self.stones = stones
        self.reference_answer = winner

        # Build problem description
        problem = (
            "Stan and Ollie are playing a game. The game rules are as follows:\n"
            f"- There are {N} heaps of stones: {', '.join(map(str, stones))}.\n"
            "- Stan and Ollie take turns playing, and Stan goes first.\n"
            f"- On a player's turn, they must select a heap that contains at least {F} stones.\n"
            "- Then, they choose an integer M (at least 2) and split the selected heap into M smaller heaps\n"
            "  such that the sizes of the smaller heaps differ by at most 1 (i.e., as evenly as possible).\n"
            "- If a player cannot make a move (i.e., no heap contains at least the threshold), they lose.\n\n"
            "If both players always play optimally, who will win — Stan or Ollie?\n\n"
            "Output Format: Your final answer should be a single word in \\boxed{...}: either Stan or Ollie."
        )

        self.current_problem = problem
        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the terminal transition."""
        answer = self._parse_answer(action)
        if answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if answer not in ("Stan", "Ollie"):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": answer,
            "N": self.N,
            "F": self.F,
            "stones": self.stones,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action in \\boxed{...} format."""
        return f"\\boxed{{{random.choice(['Stan', 'Ollie'])}}}"

    def _check_winning(self, stones: List[int], f: int) -> bool:
        """Return True if the position is winning for the current player, else False."""
        max_stone = max(stones) if stones else 0
        sg = [-1] * (max_stone + 5)
        exist = [0] * (max_stone + 5)

        # Base cases: heaps smaller than threshold cannot be split => Grundy number 0
        for i in range(0, min(max_stone + 1, f)):
            sg[i] = 0

        def get_sg(x: int) -> int:
            if sg[x] != -1:
                return sg[x]
            i = 2
            while i <= x:
                k = x // (x // i)
                for j in range(i, min(i + 1, k) + 1):
                    s = 0
                    if (x % j) % 2 == 1:
                        s ^= get_sg(x // j + 1)
                    if (j - (x % j)) % 2 == 1:
                        s ^= get_sg(x // j)
                    exist[s] = x
                i = k + 1
            t = 0
            while True:
                if exist[t] != x:
                    sg[x] = t
                    return t
                t += 1

        nim_sum = 0
        for pile_size in stones:
            nim_sum ^= get_sg(pile_size)
        return nim_sum != 0