import math
import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MultipleFlippingGameEnv(Env):
    """Multiple Flipping Game environment - single-turn Q&A."""

    def __init__(self, N: int = 1000, **kwargs):
        """
        Initialize the MultipleFlippingGameEnv instance.

        Parameters:
            N (int): The length of the array. Must be >= 1.
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 1, "N should be greater than or equal to 1"

        self.N: int = N
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.white_indices: List[int] = []

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial game theory problem.\n"
            "Please provide your final answer in \\boxed{...} format, using either Yes or No.\n"
            "Example: \\boxed{Yes}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N

        # Randomly choose target answer and generate a matching configuration
        self.reference_answer = "Yes" if random.random() < 0.5 else "No"

        # Precompute Sprague-Grundy values using the original algorithm
        sn = int(math.isqrt(N))
        p: List[int] = []
        l = 1
        while l <= N:
            k = N // l
            r = N // k
            p.append(l)
            l = r + 1

        m = len(p)
        sg_small = [0] * (sn + 1)
        sg_large = [0] * (sn + 1)
        vis = [0] * (2 * sn + 5)

        for i in range(m - 1, -1, -1):
            li = p[i]
            t = N // li
            s = 0
            l2 = 2
            mark = i + 1
            while l2 <= t:
                k2 = t // l2
                r2 = t // k2
                v = l2 * li
                if v <= sn:
                    gv = sg_small[v]
                else:
                    gv = sg_large[k2]
                vis[s ^ gv] = mark
                if ((r2 - l2 + 1) & 1):
                    s ^= gv
                l2 = r2 + 1
            g = 1
            while vis[g] == mark:
                g += 1
            if li <= sn:
                sg_small[li] = g
            else:
                sg_large[t] = g

        def SG(x: int) -> int:
            if x <= sn:
                return sg_small[x]
            return sg_large[N // x]

        # Sample white indices that match the chosen reference answer
        while True:
            white_index_number = random.randint(1, N)
            white_indices = random.sample(range(1, N + 1), white_index_number)
            xo = 0
            for x in white_indices:
                xo ^= SG(x)
            if ("Yes" if xo else "No") == self.reference_answer:
                self.white_indices = sorted(white_indices)
                break

        # Build problem prompt
        self.current_problem = (
            f"You are given an array of length {N}, indexed from 1 to {N}.\n\n"
            "Two players, Alice and Bob, play the following game:\n"
            "+ Initially, some positions in the array are white, and the rest are black.\n"
            "+ The players take turns. On each turn, the current player selects a white cell with index x.\n"
            "+ Then, they choose an integer k such that 1 <= k <= n / x, and flip the color of all cells at indices x, 2×x, ..., k×x.\n"
            "+ A player loses if they have no valid move on their turn.\n\n"
            f"Initially, the cells at indices {', '.join(map(str, self.white_indices))} are white (all others are black).\n"
            "Determine whether the first player (Alice) has a winning strategy if both players play optimally.\n\n"
            "Output Format: Your final answer should be either Yes or No enclosed in \\boxed{...} (do not include quotes)."
        )

        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {
            "N": N,
            "white_indices": self.white_indices,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute a single step to validate the provided answer."""
        parsed = self._parse_answer(action)

        if parsed is None:
            # Format error: no boxed content
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        user_answer = parsed.strip()
        if user_answer not in ("Yes", "No"):
            # Invalid answer content
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "user_answer": user_answer}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "white_indices": self.white_indices,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (Yes or No) in boxed format."""
        return f"\\boxed{{{random.choice(['Yes', 'No'])}}}"