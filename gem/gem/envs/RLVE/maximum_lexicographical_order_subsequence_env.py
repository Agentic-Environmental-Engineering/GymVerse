from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from collections import deque
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaximumLexicographicalOrderSubsequenceEnv(Env):
    """Environment for finding a lexicographically maximal subsequence of fixed length K."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 100,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, the array length will be fixed to this value. Must be >= 3.
        - min_n: Minimum N when sampling N randomly. Must be >= 3.
        - max_n: Maximum N when sampling N randomly. Must be >= min_n.
        """
        super().__init__()
        # Validate parameters
        if N is not None:
            if N < 3:
                raise ValueError("N should be greater than or equal to 3")
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")

        self.fixed_N: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        # State variables for the current problem instance
        self.current_problem: Optional[str] = None
        self.reference_answer_str: Optional[str] = None
        self.gold_answer: Optional[List[int]] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.A: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions for the agent."""
        return (
            "You are solving problems about lexicographically maximal subsequences.\n"
            "Given an array A and a target length K, your task is to select a (not necessarily contiguous) subsequence\n"
            "A[i1], A[i2], ..., A[iK] (with indices 0 <= i1 < i2 < ... < iK < N) that is lexicographically maximal.\n"
            "Output Format: Provide exactly K integers (the selected subsequence values) separated by single spaces,\n"
            "enclosed in \\boxed{...}. For example: \\boxed{v1 v2 ... vK}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Sample K and array A
        K = random.randint(2, N - 1)
        A = [random.randint(1, N) for _ in range(N)]

        # Compute the lexicographically maximal subsequence using a deque strategy
        gold_answer: List[int] = []
        q: deque[int] = deque()
        for i in range(N):
            # Maintain a decreasing deque where smaller elements are removed from the back
            while q and q[-1] < A[i]:
                q.pop()
            # Append current element if we still have fewer than K candidates
            if len(q) < K:
                q.append(A[i])
            # Once i reaches N-K, start emitting the answer from the front
            if i >= N - K:
                gold_answer.append(q[0])
                q.popleft()

        reference_answer_str = " ".join(map(str, gold_answer))

        # Store state
        self.N = N
        self.K = K
        self.A = A
        self.gold_answer = gold_answer
        self.reference_answer_str = reference_answer_str

        # Build problem description
        array_repr = " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A))
        self.current_problem = (
            f"Given an array A of length {N}: {array_repr}\n\n"
            f"Please find a (not necessarily contiguous) subsequence of length {K} "
            f"(i.e., select {K} elements with increasing indices: 0 <= i1 < ... < i{K} < {N}) "
            f"such that the resulting subsequence A[i1], ..., A[i{K}] is lexicographically maximal.\n"
            f"Your answer should be the selected subsequence values separated by single spaces, "
            f"enclosed in \\boxed{{...}}."
        )

        observation = self._get_instructions() + self.current_problem
        return observation, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Ensure a problem has been generated
        if self.gold_answer is None or self.A is None or self.K is None or self.N is None:
            # No active problem; treat as terminal with format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "no_active_problem"}

        # Parse the boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert content into a list of integers
        try:
            tokens = boxed_content.split()
            user_answer_list = [int(tok) for tok in tokens]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length
        if len(user_answer_list) != self.K:
            info = {
                "correct": False,
                "error": "invalid_length",
                "expected_length": self.K,
                "user_length": len(user_answer_list),
                "reference_answer": self.gold_answer,
                "user_answer": user_answer_list,
                "N": self.N,
                "K": self.K,
                "A": self.A,
                "reference_answer_str": self.reference_answer_str,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate that the answer is a subsequence by value in order (as per original logic)
        i = 0
        valid_subsequence = True
        for val in user_answer_list:
            found = False
            while i < self.N:
                if self.A[i] == val:
                    found = True
                i += 1
                if found:
                    break
            if not found:
                valid_subsequence = False
                break

        if not valid_subsequence:
            info = {
                "correct": False,
                "error": "not_a_subsequence",
                "reference_answer": self.gold_answer,
                "user_answer": user_answer_list,
                "N": self.N,
                "K": self.K,
                "A": self.A,
                "reference_answer_str": self.reference_answer_str,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check correctness (exact match with the gold answer)
        is_correct = (user_answer_list == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.gold_answer,
            "user_answer": user_answer_list,
            "N": self.N,
            "K": self.K,
            "A": self.A,
            "reference_answer_str": self.reference_answer_str,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by selecting K elements from A (if available) or random integers."""
        if self.A is not None and self.K is not None:
            # Randomly sample a valid subsequence by value (not necessarily correct)
            # We will pick K positions with increasing indices
            indices = sorted(random.sample(range(len(self.A)), self.K))
            values = [self.A[idx] for idx in indices]
        else:
            # Fallback: produce K random integers if no active problem
            K = random.randint(2, max(2, self.max_n - 1))
            values = [random.randint(1, max(3, self.max_n)) for _ in range(K)]
        return f"\\boxed{{{' '.join(map(str, values))}}}"