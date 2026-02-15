from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class KloBlocksEnv(Env):
    """KloBlocks environment - Single-turn Q&A.

    The task:
    Given an array A of N integers and an integer K, you may perform any number of actions.
    One action is to pick one item that is strictly greater than K, subtract 1 from it,
    and add 1 to an adjacent item (either to the left or right, if such an item exists).
    Maximize the length of the longest contiguous subarray where each item is greater than or equal to K.
    Output that maximum length.
    """

    def __init__(self, N: int, **kwargs) -> None:
        """Initialize the environment.

        Args:
            N: The length of the array A. Must be >= 3.
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving KloBlocks redistribution problems.\n"
            "You can perform any number of actions. One action is to pick one item that is strictly greater than K, "
            "subtract 1 from it, and add 1 to an adjacent item (either to the left or right, if such an item exists).\n"
            "Your goal is to maximize the length of the longest contiguous subarray where each item is greater than or equal to K.\n"
            "Please provide your final answer as a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        while True:
            A = [random.randint(1, 2 * N) for _ in range(N)]
            min_A, max_A = min(A), max(A)

            # Ensure there is at least one valid K strictly between min_A and max_A
            if not (min_A + 1 <= max_A - 1):
                continue

            K = random.randint(min_A + 1, max_A - 1)
            ans = self._compute_reference_answer(A, K)

            # Avoid trivial cases as in the original environment
            if ans != 1 and ans != N:
                self.A = A
                self.K = K
                self.reference_answer = ans
                break

        # Build the problem prompt
        A_str = " ".join(map(str, self.A))
        self.current_problem = (
            f"You have an array A of {N} integers, initially it is: {A_str}\n"
            f"You can perform any number of actions. One action is to pick one item that is greater than {self.K}, "
            f"subtract 1 from it, and add 1 to an adjacent item (either to the left or right, if such an item exists). "
            f"Please maximize the length of the longest contiguous subarray where each item is greater than or equal to {self.K}; "
            f"output its length.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, A: List[int], K: int) -> int:
        """Compute the reference answer using the prefix-sum and monotonic stack method."""
        N = len(A)
        b = [0] * (N + 1)
        stack: List[int] = []
        ans = 0

        # Forward pass: build b[], track any prefix with non-negative sum and maintain strictly decreasing stack
        for i in range(1, N + 1):
            b[i] = b[i - 1] + A[i - 1] - K
            if b[i] >= 0:
                ans = i
            if not stack or b[i] < b[stack[-1]]:
                stack.append(i)

        # Backward pass: match later indices with earlier minima in the stack
        for i in range(N, 0, -1):
            while stack and b[i] - b[stack[-1]] >= 0:
                ans = max(ans, i - stack[-1])
                stack.pop()

        return ans

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer and terminate immediately."""
        answer_text = self._parse_answer(action)

        if answer_text is None:
            # Format error (no \\boxed{...})
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            # Content inside \\boxed{...} is not a valid integer
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
            "A": self.A,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random integer length) in \\boxed{...} format."""
        # The answer is a length in [1, N], but we allow 0 as a random guess as well
        random_answer = random.randint(0, self.N)
        return f"\\boxed{{{random_answer}}}"