from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Maximum_SubsequenceNumEnv(Env):
    """Environment for maximizing the number of essentially different subsequences.

    Single-turn Q&A environment:
    - Given an initial sequence of length M, append N integers (each in [0, K))
      to maximize the number of essentially different subsequences of the final sequence.
    - The answer must be submitted inside \\boxed{...} with N integers separated by spaces.
    """

    def __init__(
        self,
        M: int = 5,
        N: int = 5,
        K: int = 3,
        **kwargs
    ):
        super().__init__()

        # Parameter validation
        assert M >= 1, "M should be greater than or equal to 1"
        assert N >= 1, "N should be greater than or equal to 1"
        assert K >= 2, "K should be greater than or equal to 2"

        self.M = M
        self.N = N
        self.K = K

        # State holders
        self.initial_sequence: List[int] = []
        self.reference_appended_sequence: List[int] = []
        self.reference_subsequence_count: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task description."""
        return (
            "You are solving a sequence construction problem to maximize the number of essentially different subsequences.\n"
            "Definitions:\n"
            "- Subsequence: pick some (>= 1) integers from the sequence in order, not necessarily contiguous.\n"
            "- Essentially different: only the sequence of values matters — same values in the same relative order are considered the same.\n\n"
            "Output Format: Provide the N integers you append, separated by spaces, inside \\boxed{...}.\n"
            "Example: \\boxed{0 1 2}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: The problem description string.
            info: Additional information dictionary (empty).
        """
        super().reset(seed)

        M, N, K = self.M, self.N, self.K

        # Generate initial sequence of length M with values in [0, K)
        self.initial_sequence = [random.randint(0, K - 1) for _ in range(M)]

        # Create working array A with a sentinel at position 0
        A = [-1] + self.initial_sequence[:]  # A[1..M] are initial values

        # Prepare last occurrence array for values in [0, K)
        last = [0] * K
        for i in range(1, M + 1):
            last[A[i]] = i

        # Greedy construction of reference appended sequence to maximize distinct subsequences
        self.reference_appended_sequence = []
        for i in range(M + 1, M + N + 1):
            k_choice = min(range(K), key=lambda k: last[k])
            A.append(k_choice)
            self.reference_appended_sequence.append(k_choice)
            last[k_choice] = i

        # Compute reference (gold) subsequence count using DP
        self.reference_subsequence_count = self.subsequence_num(A)

        # Build problem prompt
        m_plus_n = M + N
        initial_str = " ".join(map(str, self.initial_sequence))
        self.current_problem = (
            f"We want to obtain a sequence of length {M} + {N} = {m_plus_n} from an initial sequence of length {M} "
            f"by appending {N} integers, each in [0, {K}). "
            f"The initial sequence of length {M}: {initial_str}\n\n"
            f"Try your best to maximize the number of essentially different subsequences of the final sequence.\n"
            f"Your final answer should be a single line containing the {N} integers you appended to the initial sequence, "
            f"separated by spaces, each in [0, {K}).\n\n"
            f"Output Format: Put your N integers inside \\boxed{{...}}. For example: \\boxed{{0 1 2}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def subsequence_num(self, A: List[int]) -> int:
        """Compute the number of essentially different subsequences (non-empty) of sequence A[1..M+N].

        A is expected to be of length M + N + 1 with a sentinel at A[0] = -1, and values in [0, K) for A[1..].
        """
        M, N, K = self.M, self.N, self.K
        assert len(A) == M + N + 1, "A must include sentinel and have length M + N + 1"

        F = [0] * (M + N + 1)
        F[0] = 1
        last = [0] * K

        for i in range(1, M + N + 1):
            ai = A[i]
            if last[ai] == 0:
                F[i] = F[i - 1] * 2
            else:
                F[i] = F[i - 1] * 2 - F[last[ai] - 1]
            last[ai] = i

        return F[M + N] - 1

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's appended sequence.

        Returns:
            observation: TERMINAL_STATE (single-turn environment).
            reward: 1.0 if correct, 0.0 if wrong, -0.1 if format error.
            terminated: True (single-turn).
            truncated: False.
            info: Dictionary with verification details.
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from boxed content
        try:
            user_appended = list(map(int, boxed_content.split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate answer length and range
        if len(user_appended) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "wrong_length"}
        if not all(0 <= a < self.K for a in user_appended):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "out_of_range"}

        # Compute user's subsequence count
        A_user = [-1] + self.initial_sequence + user_appended
        user_subsequence_count = self.subsequence_num(A_user)

        # Compare with reference (gold)
        is_correct = (user_subsequence_count == self.reference_subsequence_count)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "user_appended_sequence": user_appended,
            "user_subsequence_count": user_subsequence_count,
            "reference_appended_sequence": self.reference_appended_sequence,
            "reference_subsequence_count": self.reference_subsequence_count,
            "initial_sequence": self.initial_sequence,
            "M": self.M,
            "N": self.N,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: N integers in [0, K) inside \\boxed{...}."""
        random_sequence = [random.randint(0, self.K - 1) for _ in range(self.N)]
        return "\\boxed{" + " ".join(map(str, random_sequence)) + "}"