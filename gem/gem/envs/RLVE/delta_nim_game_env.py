from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DeltaNimGameEnv(Env):
    """Delta Nim game environment - single-turn Q&A.

    Players play on a non-decreasing array A of N piles. On a turn, a player
    picks a pile i and removes any number of stones (at least 1 and at most A[i]),
    and the array must remain non-decreasing after the move. Alice moves first.
    The player who cannot move loses.

    The environment generates an instance (A) such that the optimal winner is known.
    The agent must answer with \\boxed{Alice} or \\boxed{Bob}.
    """

    def __init__(
        self,
        N: int = 2,
        **kwargs
    ):
        """Initialize the Delta Nim environment.

        Args:
            N: Number of piles. Must be >= 2.

        Raises:
            AssertionError: If N < 2.
        """
        super().__init__()
        assert N >= 2, "N should be greater than or equal to 2"
        self.N: int = N

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.A: Optional[List[int]] = None
        self.C: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a combinatorial game (Delta Nim) winner determination task.\n"
            "Please provide your final answer in \\boxed{...} format, with content being exactly Alice or Bob.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Returns:
            observation: The problem description string.
            info: An empty info dict.
        """
        super().reset(seed)

        N = self.N

        # Randomly choose the reference winner first
        reference_answer = "Alice" if random.random() < 0.5 else "Bob"

        # Generate C such that A is non-decreasing and the chosen winner is correct
        C: List[int] = [None] * N  # type: ignore
        ans = 0
        for i in range(N):
            if i != N - 1:
                # Ensure A is non-decreasing and A[0] >= 1
                C[i] = random.randint(1 if i == 0 else 0, N)
            else:
                if reference_answer == "Alice":
                    # Ensure the XOR outcome is non-zero
                    while True:
                        candidate = random.randint(0, N)
                        if (ans ^ candidate) != 0:
                            C[i] = candidate
                            break
                elif reference_answer == "Bob":
                    # Ensure the XOR outcome is zero
                    C[i] = ans
                else:
                    raise AssertionError("Invalid reference answer")
            # XOR over indices with the same parity as N-1
            if (i & 1) == ((N - 1) & 1):
                ans ^= C[i]  # type: ignore

        assert (ans == 0) == (reference_answer == "Bob"), "Reference answer does not match computed outcome"

        # Build A as cumulative sums of C to ensure non-decreasing
        A: List[int] = [0] * N
        for i in range(N):
            A[i] = (A[i - 1] if i - 1 >= 0 else 0) + C[i]  # type: ignore
            if i >= 1:
                assert A[i] >= A[i - 1], "A should be non-decreasing"
        assert A[0] >= 1, "A[0] must be at least 1"

        self.A = A
        self.C = C  # type: ignore
        self.reference_answer = reference_answer

        # Build the problem description
        A_str = " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A))
        problem_text = (
            f"Alice and Bob are playing a game with {N} piles of stones. "
            f"The number of stones in the i-th pile is A[i], for 0 <= i < {N}. "
            f"The initial array A is: {A_str}\n\n"
            "Game rules:\n"
            "- Players alternate turns, with Alice going first.\n"
            "- On a turn, a player chooses a pile i (0 <= i < N) and removes any number of stones "
            "(at least 1 and at most A[i]). After the move, the array A must still satisfy the condition: "
            "A[i] <= A[i + 1] for all 0 <= i < N - 1.\n"
            "- A player who cannot make a valid move loses.\n\n"
            "Assuming both players play optimally, determine who will win.\n"
            "Output Format: Answer with a single word in \\boxed{...}: either \\boxed{Alice} or \\boxed{Bob}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer.

        Args:
            action: The user's response text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE.
            reward: 1.0 if correct, 0.0 if wrong, -0.1 if format error.
            terminated: Always True (single-turn).
            truncated: Always False.
            info: Additional information including correctness and answers.
        """
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        user_answer = parsed.strip()
        if user_answer not in ("Alice", "Bob"):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "user_answer": user_answer}

        assert self.reference_answer is not None, "Environment not properly reset before step."
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "A": self.A,
            "N": self.N,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action."""
        choice = random.choice(["Alice", "Bob"])
        return f"\\boxed{{{choice}}}"