import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, Dict

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RoyalLockCountingEnv(Env):
    """Royal Lock Counting environment - single-turn Q&A.

    Task: On an N × N chessboard, count the number of ways to place K kings so that no two kings attack each other.
    The answer must be provided in \\boxed{...} format with a single integer.
    """

    prompt_template = (
        "On a {N} × {N} chessboard, you are to place {K} kings such that no two kings attack each other. "
        "How many different valid placement configurations are there? (The internal order of the kings does NOT matter.)\n\n"
        "A king can attack up to 8 surrounding squares: the squares directly above, below, left, right, "
        "and all 4 diagonals (top-left, top-right, bottom-left, bottom-right).\n\n"
        "Output Format: Your final answer should be a single integer in \\boxed{{...}}."
    )

    def __init__(
        self,
        min_N: int = 3,
        max_N: int = 10,
        fixed_N: Optional[int] = None,
        fixed_K: Optional[int] = None,
        **kwargs: Any
    ):
        """Initialize the RoyalLockCountingEnv instance.

        Parameters:
        - min_N: Minimum board size N (must be >= 3).
        - max_N: Maximum board size N (must be >= min_N).
        - fixed_N: If provided, use this fixed N (must be >= 3).
        - fixed_K: If provided, use this fixed K (must be >= 0 and <= N*N).
        """
        super().__init__()
        if fixed_N is not None:
            if fixed_N < 3:
                raise ValueError("fixed_N should be greater than or equal to 3")
        else:
            if min_N < 3:
                raise ValueError("min_N should be greater than or equal to 3")
            if max_N < min_N:
                raise ValueError("max_N should be greater than or equal to min_N")

        self.min_N = min_N
        self.max_N = max_N
        self.fixed_N = fixed_N
        self.fixed_K = fixed_K

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics counting problem about placing kings on a chessboard.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and generate a new problem.

        Returns:
        - observation: The task instruction concatenated with the generated problem description.
        - info: Additional information dictionary (empty in this environment).
        """
        super().reset(seed)

        # Choose N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 3:
            raise AssertionError("N should be greater than or equal to 3")
        self.N = N

        # Choose K
        if self.fixed_K is not None:
            K = self.fixed_K
            if K < 0 or K > N * N:
                raise ValueError("fixed_K must be between 0 and N*N inclusive")
        else:
            K = random.randint(1, max(1, N * N // 4))
        self.K = K

        # Compute reference answer using DP over valid row states
        self.reference_answer = self._count_configurations(N, K)

        # Build problem prompt
        self.current_problem = self.prompt_template.format(N=N, K=K)
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _count_configurations(self, N: int, K: int) -> int:
        """Count configurations of placing K kings on an N x N board with no attacks."""
        num_states = 1 << N

        valid_states = []
        king_count = [0] * num_states

        # Generate all valid row states (no adjacent kings horizontally)
        for s in range(num_states):
            if s & (s << 1):
                continue
            valid_states.append(s)
            king_count[s] = s.bit_count()

        # Compatibility between adjacent rows to avoid vertical and diagonal attacks
        compat: Dict[int, list[int]] = {s: [] for s in valid_states}
        for s in valid_states:
            for t in valid_states:
                if s & t:
                    continue  # vertical attack
                if (s << 1) & t:
                    continue  # diagonal attack (left)
                if (s >> 1) & t:
                    continue  # diagonal attack (right)
                compat[s].append(t)

        # DP arrays: F_prev[k][state] = ways for previous rows with k kings ending in 'state'
        F_prev = [[0] * num_states for _ in range(K + 1)]
        F_cur = [[0] * num_states for _ in range(K + 1)]

        # Base case: zero rows, zero kings, empty last row (state 0)
        F_prev[0][0] = 1

        # Process each row
        for _row in range(1, N + 1):
            # Reset current DP
            for k in range(K + 1):
                for s in valid_states:
                    F_cur[k][s] = 0

            # Transition for each valid state s of the current row
            for s in valid_states:
                c = king_count[s]
                for k in range(c, K + 1):
                    prev_k = k - c
                    tot = 0
                    for t in compat[s]:
                        tot += F_prev[prev_k][t]
                    F_cur[k][s] = tot

            F_prev, F_cur = F_cur, F_prev

        # Sum over ending states for exactly K kings
        return sum(F_prev[K][s] for s in valid_states)

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute a step and verify the provided answer.

        Returns:
        - observation: TERMINAL_STATE to indicate the end of the episode.
        - reward: 1.0 if correct, 0.0 if wrong, -0.1 if format error.
        - terminated: Always True (single-turn environment).
        - truncated: Always False.
        - info: Dictionary with details like correctness and reference answer.
        """
        answer_text = self._parse_answer(action)

        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == user_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
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
        # Use a simple heuristic for sampling a plausible integer answer
        random_answer = random.randint(0, max(100, (self.N or 3) * (self.K or 1)))
        return f"\\boxed{{{random_answer}}}"