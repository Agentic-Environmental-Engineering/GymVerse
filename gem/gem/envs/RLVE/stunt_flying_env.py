import random
from typing import Any, List, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class StuntFlyingEnv(Env):
    """Stunt Flying optimization environment - single-turn Q&A.

    Task:
    Given K labeled elements (0..K-1) with values C[x], construct an array A of length N (each A[i] in [0, K-1]).
    Define T[i] as the distance to the last occurrence of A[i], or 0 if it has not appeared before.
    The objective is to maximize the sum over i of C[A[i]] * T[i].

    Answer format:
    Provide the space-separated array A inside \\boxed{...}.
    """

    def __init__(
        self,
        min_n: int = 4,
        max_n: int = 100,
        **kwargs
    ):
        super().__init__()
        if min_n < 4:
            raise ValueError("min_n should be greater than or equal to 4")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        self.min_n = min_n
        self.max_n = max_n

        # Problem state
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.C: Optional[List[int]] = None
        self.current_problem: Optional[str] = None
        self.reference_max_sum: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial optimization problem.\n"
            "Construct an array A that maximizes the sum of C[A[i]] × T[i], where T[i] is the distance to the previous occurrence of A[i] (or 0 if none).\n"
            "Please provide your final array inside \\boxed{...}, with elements separated by single spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment, generate a new problem."""
        super().reset(seed)

        # Generate parameters
        self.N = random.randint(self.min_n, self.max_n)
        self.K = random.randint(2, self.N)
        self.C = [random.randint(1, self.K) for _ in range(self.K)]

        # Compute the reference (maximum) sum according to the original algorithm
        self.reference_max_sum = self._compute_max_sum(self.C, self.N)

        # Build problem prompt
        C_str = "; ".join(f"C[{x}] = {cx}" for x, cx in enumerate(self.C))
        self.current_problem = (
            f"There are {self.K} elements labeled from 0 to {self.K - 1}, and each element x has an associated value C[x]. "
            f"C is: {C_str}\n"
            f"You need to build an array A of length {self.N}, where each A[i] is one of these elements "
            f"(i.e., 0 ≤ A[i] < {self.K} for all 1 ≤ i ≤ {self.N}). Each position i in A has a value defined as C[A[i]] × T[i], "
            f"where T[i] is determined as follows:\n"
            f"- If there is no previous index j (0 ≤ j < i) such that A[j] = A[i], then T[i] = 0.\n"
            f"- Otherwise, let j be the largest index (closest to i) such that A[j] = A[i] (0 ≤ j < i), and set T[i] = i - j.\n"
            f"Can you maximize the sum of all values C[A[i]] × T[i]? "
            f"Output Format: Provide A[1], A[2], ..., A[{self.N}] in order, separated by spaces, inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": self.N,
            "K": self.K,
            "C": self.C,
            "reference_max_sum": self.reference_max_sum
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step, verify the answer."""
        # Parse \\boxed{...}
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure we have a problem loaded
        if self.N is None or self.K is None or self.C is None or self.reference_max_sum is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem"}

        # Parse the array A
        try:
            tokens = boxed_content.strip().split()
            answer_array = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate array length and value range
        if len(answer_array) != self.N:
            info = {
                "error": "invalid_solution_length",
                "expected_length": self.N,
                "received_length": len(answer_array)
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(0 <= ai < self.K for ai in answer_array):
            info = {
                "error": "invalid_element_range",
                "K": self.K
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute the sum according to the given rule
        computed_sum = self._evaluate_sequence(answer_array, self.C)

        # Verify against the reference maximum sum
        is_correct = (computed_sum == self.reference_max_sum)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_max_sum": self.reference_max_sum,
            "computed_sum": computed_sum,
            "user_answer_array": answer_array,
            "N": self.N,
            "K": self.K,
            "C": self.C
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _compute_max_sum(C: List[int], N: int) -> int:
        """Compute the reference maximum sum based on the original generation logic."""
        A_sorted = sorted(C, reverse=True)
        ans = 0
        remaining = N - 1
        i = 0
        while remaining > 0 and i < len(A_sorted):
            ans += remaining * A_sorted[i]
            i += 1
            remaining -= 2
        if ans <= 0:
            raise ValueError("Reference maximum sum should be greater than 0")
        return ans

    @staticmethod
    def _evaluate_sequence(sequence: List[int], C: List[int]) -> int:
        """Evaluate the sum C[A[i]] × T[i] for the provided sequence."""
        K = len(C)
        last_indices: List[Optional[int]] = [None] * K
        total = 0
        for i, ai in enumerate(sequence):
            t = 0 if last_indices[ai] is None else i - last_indices[ai]
            total += C[ai] * t
            last_indices[ai] = i
        return total

    def sample_random_action(self) -> str:
        """Sample a random action: a random valid array inside \\boxed{...}."""
        if self.N is None or self.K is None:
            # If no problem has been generated yet, create one
            self.reset()
        assert self.N is not None and self.K is not None
        random_array = [random.randint(0, self.K - 1) for _ in range(self.N)]
        content = " ".join(str(x) for x in random_array)
        return f"\\boxed{{{content}}}"