from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class KosDicingEnv(Env):
    """Kos Dicing environment - single-turn Q&A.

    Task:
    Given N players and M rounds, each round lists two distinct players (a, b).
    Determine the winner of each round (one of a or b) so that the maximum number
    of wins by any player equals exactly K. The answer should be M integers
    (winners) separated by spaces, wrapped in \\boxed{...}.
    """

    def __init__(self, N: int = 5, M: int = 10, **kwargs) -> None:
        super().__init__()
        # Parameter validation
        assert isinstance(N, int), "N must be an integer"
        assert isinstance(M, int), "M must be an integer"
        assert N >= 2, "N should be greater than or equal to 2"
        assert M >= 1, "M should be greater than or equal to 1"

        self.N: int = N
        self.M: int = M

        # Problem state
        self.rounds: List[Tuple[int, int]] = []
        self.K: Optional[int] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a scheduling and counting problem.\n"
            "Given N players and M rounds, each round lists two distinct players (a, b).\n"
            "You must decide a winner (either a or b) for each round such that the maximum\n"
            "number of wins achieved by any player equals exactly K.\n"
            "Output Format: Provide M integers separated by spaces (the i-th is the winner of the i-th round),\n"
            "and put your answer inside \\boxed{...}. Do not include any extra text.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment: generate a new problem instance."""
        super().reset(seed)

        # Generate rounds and a reference solution
        N = self.N
        M = self.M

        rounds: List[Tuple[int, int]] = []
        reference_answer_list: List[int] = []
        win_counts = [0] * N

        for _ in range(M):
            a, b = random.sample(range(N), 2)
            rounds.append((a, b))
            winner = random.choice((a, b))
            win_counts[winner] += 1
            reference_answer_list.append(winner)

        K = max(win_counts)
        reference_answer_str = " ".join(map(str, reference_answer_list))

        # Save state
        self.rounds = rounds
        self.K = K
        self.reference_answer = reference_answer_str

        # Build problem prompt
        rounds_str = "\n".join(f"({a}, {b})" for a, b in rounds)
        self.current_problem = (
            f"There are {N} players (labeled from 0 to {N - 1}) participating in a game consisting of {M} rounds. "
            f"Each round (a, b) involves two distinct players a and b, given as:\n{rounds_str}\n\n"
            f"In each round, exactly one of the two players wins. Please determine the outcome of all rounds such that "
            f"the maximum number of total wins by any player is exactly {K}.\n\n"
            f"Output Format: Output {M} integers, separated by spaces. The i-th integer represents the winner of the "
            f"i-th round, either a or b. Your final response must be in \\boxed{{...}} containing only these {M} integers."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse a list of integers (space-separated)
        try:
            tokens = boxed_content.strip().split()
            user_answer_list = [int(t) for t in tokens]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate solution length
        if len(user_answer_list) != self.M:
            info = {
                "error": "invalid_solution",
                "reason": "length_mismatch",
                "expected_length": self.M,
                "received_length": len(user_answer_list),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate winners and compute win counts
        counts = [0] * self.N
        for (a, b), winner in zip(self.rounds, user_answer_list):
            if winner not in (a, b):
                info = {
                    "error": "invalid_solution",
                    "reason": "winner_not_in_pair",
                    "pair": (a, b),
                    "winner": winner,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            counts[winner] += 1

        # Check correctness
        is_correct = (max(counts) == self.K)
        reward = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "rounds": self.rounds,
            "user_answer": user_answer_list,
            "reference_answer": self.reference_answer,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a (typically correct) action for debugging or testing."""
        # If a reference answer exists, return it; otherwise, produce a placeholder.
        if self.reference_answer is not None:
            return f"\\boxed{{{self.reference_answer}}}"
        # Fallback: produce a random sequence of M winners choosing one from each pair (may be invalid if not set)
        if self.rounds:
            winners = [random.choice(pair) for pair in self.rounds]
            return f"\\boxed{{{' '.join(map(str, winners))}}}"
        # Default empty
        return "\\boxed{}"