from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GraMinimaGameEnv(Env):
    """GraMinima Game environment - single-turn question answering.

    Alice and Bob play a game on a multiset of integers. On each turn, a player
    selects any non-empty subset of remaining numbers, adds the minimum of that subset
    to their score, and removes the entire subset from the game. Both play optimally
    to maximize (their score - opponent's score). The task is to compute the final
    value of (Alice's score - Bob's score).
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 1,
        max_n: int = 50,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            N: If provided, the number of integers in the game. Must be >= 1.
               If None, N will be sampled uniformly from [min_n, max_n] at reset.
            min_n: Minimum N when sampling (inclusive). Must be >= 1.
            max_n: Maximum N when sampling (inclusive). Must be >= min_n.

        Notes:
            - Rewards are fixed:
              correct answer: +1.0
              wrong answer: 0.0
              format error: -0.1
        """
        super().__init__()
        if N is not None and N < 1:
            raise ValueError("N should be greater than or equal to 1.")
        if min_n < 1:
            raise ValueError("min_n should be greater than or equal to 1.")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n.")

        self.N_fixed: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        # Current problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_N: Optional[int] = None
        self.current_A: Optional[List[int]] = None

        # Rewards
        self.reward_correct: float = 1.0
        self.reward_wrong: float = 0.0
        self.reward_format_error: float = -0.1

    def _get_instructions(self) -> str:
        """Return task description and output format instructions."""
        return (
            "You are solving the GraMinima game problem.\n"
            "On each turn, a player selects any non-empty subset of remaining numbers, "
            "adds the minimum of that subset to their score, then removes the subset from the game. "
            "Both players play optimally to maximize (their score - opponent's score).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: A string containing instructions and the problem statement.
            info: An empty dictionary or additional metadata (optional).
        """
        super().reset(seed)

        # Determine N
        N = self.N_fixed if self.N_fixed is not None else random.randint(self.min_n, self.max_n)
        if N < 1:
            raise ValueError("N should be greater than or equal to 1.")
        self.current_N = N

        # Generate A with values in [1, 2N]
        A = [random.randint(1, N * 2) for _ in range(N)]
        self.current_A = A[:]

        # Compute the reference answer
        self.reference_answer = self._compute_reference_answer(A)

        # Build the problem string
        numbers_str = " ".join(map(str, A))
        problem = (
            f"There are {N} numbers: {numbers_str}\n"
            "Alice and Bob are playing a game with these numbers. Alice goes first, and they take turns. "
            "On each turn, a player may choose any non-empty subset of the remaining numbers, add the minimum "
            "of that subset to their score, and then remove the entire subset from the game. The game ends when "
            "there are no numbers left. Each player plays optimally to maximize their score minus their opponent's score. "
            "Please compute the final value of (Alice's score - Bob's score).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, A: List[int]) -> int:
        """Compute the optimal (Alice's score - Bob's score) for the given list A."""
        B = sorted(A)
        ans = 0
        for a in B:
            ans = max(ans, a - ans)
        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer.

        Args:
            action: The agent's answer text, expected in \\boxed{...} format.

        Returns:
            observation: TERMINAL_STATE since this is a single-turn environment.
            reward: 1.0 for correct, 0.0 for wrong, -0.1 for format error.
            terminated: True (single-turn).
            truncated: False.
            info: Additional information including correctness and reference answer.
        """
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, self.reward_format_error, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, self.reward_wrong, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = self.reward_correct if is_correct else self.reward_wrong

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_N,
            "A": self.current_A,
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
        """Sample a random action in \\boxed{...} format."""
        # Heuristic range for random guess
        guess = random.randint(0, (self.current_N or self.max_n) * 2)
        return f"\\boxed{{{guess}}}"