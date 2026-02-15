import random
import re
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class JosephusEnv(Env):
    """Josephus problem environment - single-turn Q&A.

    N people stand in a circle labeled from 1 to N. Starting from label 1,
    they count off in order. The person who counts to M is eliminated, and
    the next person resumes counting from 1. This continues until everyone
    is eliminated. The task is to determine the order of elimination.

    Answer format requirement: the final answer should be a single line
    containing the labels in the elimination order, separated by spaces,
    wrapped in \\boxed{...}.
    """

    def __init__(self, max_n: int = 1000, **kwargs: Any) -> None:
        super().__init__()
        if max_n < 3:
            raise ValueError("max_n should be greater than or equal to 3")
        self.max_n: int = max_n

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.reference_order: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving Josephus elimination order problems.\n"
            "Please provide your final answer wrapped in \\boxed{...} format.\n"
            "The boxed content must be a single line with labels separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment by generating a new Josephus problem instance."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(3, self.max_n)
        M = random.randint(2, N)

        self.N = N
        self.M = M

        # Build problem statement
        self.current_problem = (
            f"{N} people are standing in a circle (labeled from 1 to {N}). "
            f"Starting from the person labeled 1, they count off in order. "
            f"The person who counts to {M} is eliminated, and the next person resumes counting from 1. "
            f"This process continues until everyone is eliminated. "
            f"Please determine the order in which people are eliminated.\n\n"
            f"Output Format: Your final answer should be a single line containing the labels of the people "
            f"in the order they are eliminated, separated by spaces, wrapped in \\boxed{{...}}."
        )

        # Compute reference answer using a Binary Indexed Tree (Fenwick Tree) for k-th selection
        reference_order = self._compute_josephus_order(N, M)
        assert len(reference_order) == N, "The length of the result should be equal to N"
        self.reference_order = reference_order
        self.reference_answer_str = " ".join(map(str, reference_order))

        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {
            "N": N,
            "M": M,
        }
        return obs, info

    def _compute_josephus_order(self, N: int, M: int) -> List[int]:
        """Compute Josephus elimination order using a Fenwick Tree (BIT) with k-th selection."""
        bit = [0] * (N + 1)

        def lowbit(x: int) -> int:
            return x & -x

        def add(pos: int, val: int) -> None:
            while pos <= N:
                bit[pos] += val
                pos += lowbit(pos)

        def prefix_sum(pos: int) -> int:
            s = 0
            while pos > 0:
                s += bit[pos]
                pos -= lowbit(pos)
            return s

        def find_kth(k: int) -> int:
            """Find smallest index idx such that prefix_sum(idx) >= k."""
            idx = 0
            curr = 0
            # Use bit length to navigate the Fenwick tree
            max_bit = N.bit_length()
            for i in range(max_bit, -1, -1):
                next_idx = idx + (1 << i)
                if next_idx <= N and curr + bit[next_idx] < k:
                    idx = next_idx
                    curr += bit[next_idx]
            return idx + 1

        # Initialize BIT with 1 for all positions (all people alive)
        for i in range(1, N + 1):
            add(i, 1)

        result: List[int] = []
        remaining = N
        cur = 1
        for _ in range(N):
            cur = (cur - 1 + M - 1) % remaining + 1
            person = find_kth(cur)
            result.append(person)
            add(person, -1)
            remaining -= 1

        return result

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step by parsing and verifying the user's answer."""
        # Ensure a problem has been generated
        if self.reference_order is None or self.N is None or self.M is None:
            # No active problem; treat as truncated interaction
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: missing or malformed \\boxed{...}
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse list of integers from the boxed content
        tokens = boxed_content.strip().split()
        try:
            user_list = list(map(int, tokens))
        except ValueError:
            # Content not parseable as integers
            info = {
                "error": "invalid_answer",
                "N": self.N,
                "M": self.M,
                "reference_answer": self.reference_order,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate length, uniqueness, and range
        if len(user_list) != self.N:
            info = {
                "error": "invalid_answer",
                "N": self.N,
                "M": self.M,
                "reference_answer": self.reference_order,
                "user_answer": user_list,
                "reason": "length_mismatch",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if len(set(user_list)) != self.N or not all(1 <= x <= self.N for x in user_list):
            info = {
                "error": "invalid_answer",
                "N": self.N,
                "M": self.M,
                "reference_answer": self.reference_order,
                "user_answer": user_list,
                "reason": "duplicate_or_out_of_range",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compare with reference
        is_correct = (user_list == self.reference_order)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "N": self.N,
            "M": self.M,
            "reference_answer": self.reference_order,
            "user_answer": user_list,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required boxed format."""
        if self.N is None:
            # If no current problem, return a trivial boxed content
            return "\\boxed{1}"
        # Generate a random permutation as a random guess
        perm = list(range(1, self.N + 1))
        random.shuffle(perm)
        return f"\\boxed{{{' '.join(map(str, perm))}}}"