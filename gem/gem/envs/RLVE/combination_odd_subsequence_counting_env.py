from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CombinationOddSubsequenceCountingEnv(Env):
    """Environment for counting subsequences where the product of consecutive binomial coefficients is odd.

    Task:
      - Given a sequence of distinct integers A, count the number of subsequences (a[1], ..., a[k]) with k >= 2
        such that the product C(a[1], a[2]) × C(a[2], a[3]) × ... × C(a[k−1], a[k]) is odd.
      - C(x, y) denotes the binomial coefficient "x choose y".
      - It is known (by Lucas' theorem) that C(x, y) is odd if and only if, in binary, each bit set in y is also set in x,
        i.e., y is a bitwise submask of x.

    This is a single-turn question-answer environment. The answer must be provided in \\boxed{...} format.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 100,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
          - N: If provided, fixes the sequence length to N (must be >= 2).
          - min_N: Minimum sequence length used when N is not provided (must be >= 2).
          - max_N: Maximum sequence length used when N is not provided (must be >= min_N).

        Notes:
          - All integers in the sequence will be distinct and sampled from range [1, 2*N - 1].
        """
        super().__init__()
        if N is not None and N < 2:
            raise ValueError("N should be greater than or equal to 2")
        if min_N < 2:
            raise ValueError("min_N should be greater than or equal to 2")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N

        self.array: List[int] = []
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial subsequence counting problem.\n"
            "Provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)

        # Generate a sequence of distinct integers
        # Sample from [1, 2*N - 1] to keep values reasonably small for submask enumeration
        self.array = random.sample(range(1, 2 * N), N)
        random.shuffle(self.array)

        # Build problem string
        array_str = " ".join(map(str, self.array))
        self.current_problem = (
            f"You are given a sequence of distinct integers: {array_str}\n\n"
            "Please count the number of subsequences (not necessarily contiguous, but the order must be preserved) "
            "a[1], ..., a[k] such that:\n"
            "1. k ≥ 2 (the subsequence must have at least two elements);\n"
            "2. C(a[1], a[2]) × C(a[2], a[3]) × ... × C(a[k−1], a[k]) is odd, where C(x, y) denotes the binomial coefficient \"x choose y\".\n\n"
            "Output Format: A single integer in \\boxed{...} — the number of such valid subsequences."
        )

        # Compute the reference answer
        self.reference_answer = self._compute_reference_answer(self.array)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, A: List[int]) -> int:
        """Compute the number of valid subsequences of length >= 2 using DP over submask relations."""
        N = len(A)
        max_val = max(A)
        # T[v] = index in A if v exists, else -1
        T = [-1] * (max_val + 1)
        for i, v in enumerate(A):
            T[v] = i

        # f[i] = number of submask-respecting subsequences starting at position i (including the length-1 subsequence [A[i]])
        f = [0] * N
        ans = 0

        # DP from right to left
        for i in range(N - 1, -1, -1):
            mask = A[i]
            cnt = 1  # count the subsequence [A[i]] itself
            # enumerate all non-zero proper submasks j of mask
            j = mask & (mask - 1)
            while j:
                if j <= max_val:
                    idx = T[j]
                    # extend subsequences only if that submask value appears later in the sequence
                    if idx > i:
                        cnt += f[idx]
                # move to next submask
                j = mask & (j - 1)
            f[i] = cnt
            ans += cnt

        # subtract the single-element subsequences to count only those of length >= 2
        ans -= N
        return ans

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse the answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and score
        try:
            user_answer = int(answer_text)
            is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
            reward: float = 1.0 if is_correct else 0.0
        except ValueError:
            # The boxed content is not an integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "array": self.array,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        N = len(self.array) if self.array else self.max_N
        # A loose upper bound estimate for the count; use a random guess
        random_answer = random.randint(0, max(1, N * (N - 1)))
        return f"\\boxed{{{random_answer}}}"