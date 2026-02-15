import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SplittingGameEnv(Env):
    """Splitting Game environment - single-turn Q&A."""

    def __init__(self, N: int = 3, **kwargs):
        """
        Initialize the SplittingGameEnv instance.

        Parameters:
        - N: number of bottles (must be >= 3)
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.P: Optional[list[int]] = None
        self.SG: Optional[list[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial game theory problem.\n"
            "Provide your final answer as either Alice or Bob.\n"
            "Output Format: Your final answer must be in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)
        N = self.N

        # Precompute Sprague-Grundy values for reversed positions 0..N-1
        def mex(s: set[int]) -> int:
            m = 0
            while m in s:
                m += 1
            return m

        SG = [0] * N
        for r in range(1, N):
            reachable = set()
            for j in range(r):
                for k in range(j + 1):
                    reachable.add(SG[j] ^ SG[k])
            SG[r] = mex(reachable)
        self.SG = SG

        # Randomly choose a reference answer and generate P to match it
        reference_answer = "Alice" if random.random() < 0.5 else "Bob"

        def compute_outcome(p: list[int]) -> str:
            ans = 0
            for i in range(N):
                if p[i] & 1:
                    r = N - 1 - i
                    ans ^= SG[r]
            if ans == 0:
                return "Bob"
            # Enumerate all valid moves i < j <= k with at least one bean at i
            for i in range(N):
                if p[i] == 0:
                    continue
                for j in range(i + 1, N):
                    for k in range(j, N):
                        r_i = N - 1 - i
                        r_j = N - 1 - j
                        r_k = N - 1 - k
                        if ans ^ SG[r_i] ^ SG[r_j] ^ SG[r_k] == 0:
                            return "Alice"
            return "Bob"

        # Generate P until the outcome matches the chosen reference answer
        while True:
            p = [random.randint(0, 2 * N) for _ in range(N)]
            if compute_outcome(p) == reference_answer:
                self.P = p
                break

        self.reference_answer = reference_answer

        # Build problem prompt
        P_str = " ".join(f"P[{i}]={Pi}" for i, Pi in enumerate(self.P))
        self.current_problem = (
            f"There are {N} bottles of beans, indexed from 0 to {N - 1}. "
            f"Initially, the i-th bottle contains P[i] beans. The array P is given as:\n"
            f"{P_str}\n\n"
            "Alice and Bob play a game with the following rules:\n"
            "- Alice goes first. They take turns alternately.\n"
            "- On each turn, a player must choose three indices i, j, k (0 ≤ i < j ≤ k < {N}) such that the i-th bottle contains at least one bean. "
            "The player then removes one bean from bottle i, adds one bean to bottle j, and adds one bean to bottle k. "
            "(If j = k, it means adding two beans to bottle j.)\n"
            "- The game ends when a player cannot make a move. The player who cannot move loses the game.\n\n"
            "Assuming both players play optimally, who will win the game?\n"
            "Output Format: Your final answer should be either Alice or Bob, and must be in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer."""
        answer = self._parse_answer(action)

        if answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if answer not in ("Alice", "Bob"):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": answer,
            "P": self.P,
            "N": self.N,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} format."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action."""
        rand_answer = "Alice" if random.random() < 0.5 else "Bob"
        return f"\\boxed{{{rand_answer}}}"