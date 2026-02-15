from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Longest_MatchingSubsequenceEnv(Env):
    """
    Environment for the Longest Matching Subsequence problem.

    Task:
    - You are given an array A of length N indexed from 0 to N-1.
    - Select a strictly increasing sequence of indices i_1 < i_2 < ... < i_k.
    - Let B[1] = A[i_1], B[2] = A[i_2], ..., B[k] = A[i_k] (B is 1-based).
    - Maximize the number of positions j (1 ≤ j ≤ k) such that B[j] = j.

    Answer format:
    - Provide the selected indices separated by single spaces inside \\boxed{...}.
      Example: \\boxed{0 2}
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 100,
        **kwargs: Any
    ) -> None:
        """
        Initialize the environment.

        Args:
            N: If provided, fixes the array length N (must be >= 3).
            min_N: Minimum N (inclusive) used when N is not provided (must be >= 3).
            max_N: Maximum N (inclusive) used when N is not provided (must be >= min_N).

        Notes:
            - The environment generates A as:
              A = [randint(1, N) for _ in range(N - 1)] + [1], then shuffled.
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        self.N: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.gold_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a longest matching subsequence problem.\n"
            "Task:\n"
            "- You are given an array A of length N, indexed from 0 to N-1.\n"
            "- Select a strictly increasing sequence of indices i_1 < i_2 < ... < i_k.\n"
            "- Let B[1] = A[i_1], B[2] = A[i_2], ..., B[k] = A[i_k] (B is 1-based).\n"
            "- Maximize the number of positions j (1 ≤ j ≤ k) such that B[j] = j.\n\n"
            "Output Format:\n"
            "- Your final answer should be the selected indices separated by spaces, "
            "wrapped in \\boxed{...}. Example: \\boxed{0 2}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N

        # Generate array A
        A = [random.randint(1, N) for _ in range(N - 1)] + [1]
        random.shuffle(A)
        self.A = A

        # Compute gold_answer using the original DP logic
        answer = 0
        F: List[Optional[int]] = [None] * N
        for i in range(N):
            if A[i] <= i + 1:
                F[i] = 1
            for j in range(i):
                if A[i] - A[j] <= i - j and A[i] > A[j]:
                    if F[j] is not None:
                        val = F[j] + 1
                        if F[i] is None or val > F[i]:
                            F[i] = val
            if F[i] is not None:
                answer = max(answer, F[i])
        assert answer > 0
        self.gold_answer = answer

        # Build problem description
        A_listing = "\n".join(f"A[{index}]={value}" for index, value in enumerate(A))
        self.current_problem = (
            f"You are given an array `A` of length {N}, indexed from 0 to {N - 1}. The array is as follows:\n"
            f"{A_listing}\n\n"
            "Your task is to select a strictly increasing sequence of indices i_1, i_2, ..., i_k (0 ≤ i_1 < i_2 < ... < i_k < N) such that:\n"
            "- Let B[1] = A[i_1], B[2] = A[i_2], ..., B[k] = A[i_k] (B's indices are 1-based, while A's indices are 0-based).\n"
            "- Try your best to maximize the number of positions j (1 ≤ j ≤ k) such that B[j] = j.\n\n"
            "Output Format: Your final answer should be a single line containing the selected indices i_1, i_2, ..., i_k, "
            "separated by spaces, wrapped in \\boxed{...}. Example: \\boxed{0 2}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted answer."""
        # Parse the boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert boxed content into a list of indices
        parsed_indices = self._parse_indices(boxed_content)
        if parsed_indices is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate against environment state
        if self.N is None or self.A is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Check index validity and strict increase; build B
        B: List[int] = [-1]
        last_idx = -1
        for idx in parsed_indices:
            if not (0 <= idx < self.N):
                info = {
                    "correct": False,
                    "reason": "index_out_of_bounds",
                    "selected_indices": parsed_indices,
                    "reference_best": self.gold_answer,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            if idx <= last_idx:
                info = {
                    "correct": False,
                    "reason": "not_strictly_increasing",
                    "selected_indices": parsed_indices,
                    "reference_best": self.gold_answer,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            last_idx = idx
            B.append(self.A[idx])

        # Compute the number of matches: positions j such that B[j] == j
        user_matches = sum(int(i == bi) for i, bi in enumerate(B))
        is_correct = (user_matches == self.gold_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_best": self.gold_answer,
            "user_matches": user_matches,
            "selected_indices": parsed_indices,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_indices(self, content: str) -> Optional[List[int]]:
        """
        Parse a space-separated list of integers from the boxed content.
        Returns:
            - List[int] if parsing succeeds (empty list allowed)
            - None if format is invalid (non-integer tokens)
        """
        if content == "":
            return []
        tokens = content.split()
        indices: List[int] = []
        try:
            for tok in tokens:
                indices.append(int(tok))
        except ValueError:
            return None
        return indices

    def sample_random_action(self) -> str:
        """Sample a random action: a random valid or empty increasing index sequence wrapped in \\boxed{...}."""
        if self.N is None:
            # If not initialized, return an empty selection
            return "\\boxed{}"
        # Randomly choose a subset of indices in increasing order
        k = random.randint(0, self.N)  # may be zero
        indices = sorted(random.sample(range(self.N), k)) if k > 0 else []
        content = " ".join(map(str, indices))
        return f"\\boxed{{{content}}}"