import math
import random
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BezoutIdentityEnv(Env):
    """Bezout Identity environment - single-turn Q&A.
    
    Task:
    Given an array A of length N, find integers X[1..N] such that S = sum(A[i] * X[i]) > 0,
    and try to minimize S. The theoretically minimal positive S is gcd of all A[i].
    
    Output must be provided in \\boxed{...} where the content is N integers separated by spaces.
    """

    def __init__(
        self,
        N: int = 4,
        MAX_A: int = 50,
        **kwargs,
    ):
        """Initialize the environment with parameter configuration.
        
        Parameters:
        - N: The length of array A. Must be >= 2.
        - MAX_A: The maximum absolute value for elements in A (before random sign). Must be >= 2.
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert isinstance(MAX_A, int), "MAX_A must be an integer"
        assert N >= 2, "N should be greater than or equal to 2"
        assert MAX_A >= 2, "MAX_A should be greater than or equal to 2"

        self.N: int = N
        self.MAX_A: int = MAX_A

        self.current_problem: Optional[str] = None
        self.A: Optional[List[int]] = None
        self.reference_coefficients: Optional[List[int]] = None
        self.gold_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an array A of length N. Find integers X[1..N] such that S = A[1]*X[1] + ... + A[N]*X[N] > 0.\n"
            "Try your best to minimize the value of S while meeting the condition S > 0.\n"
            "The output must be a single line of N integers separated by spaces, enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate array A with a heuristic to encourage shared factors among elements
        A: List[int] = []
        for _ in range(self.N):
            picked_a: Optional[int] = None
            best_counting: int = -1
            for _try in range(1024):
                current_a = random.randint(2, self.MAX_A)
                counting = sum(int(math.gcd(current_a, _a) > 1) for _a in A)
                if counting > best_counting:
                    best_counting, picked_a = counting, current_a
                if best_counting == len(A):
                    break
            assert picked_a is not None, "Failed to pick a valid element for A"
            if random.random() < 0.5:
                picked_a = -picked_a
            A.append(picked_a)
        random.shuffle(A)
        assert len(A) == self.N, "The length of A should be equal to N"

        # Compute reference coefficients via iterative extended GCD to achieve minimal positive S = gcd(A)
        def exgcd(a: int, b: int) -> Tuple[int, int, int]:
            """Return (g, x, y) such that g = gcd(a, b) and a*x + b*y = g, with g >= 0."""
            if b == 0:
                return (abs(a), 1 if a >= 0 else -1, 0)
            g, x1, y1 = exgcd(b, a % b)
            x = y1
            y = x1 - (a // b) * y1
            return (g, x, y)

        g = abs(A[0])
        X = [0] * self.N
        X[0] = 1 if A[0] >= 0 else -1

        for i in range(1, self.N):
            ai = A[i]
            g2, u, v = exgcd(g, ai)
            for j in range(i):
                X[j] *= u
            X[i] = v
            g = g2

        S = sum(x * a for x, a in zip(X, A))
        assert S == g
        assert S > 0, "The sum S must be greater than 0"

        # Store problem and answers
        self.A = A
        self.reference_coefficients = X
        self.gold_answer = S

        problem_text = (
            f"You are given an array of length {self.N}, denoted as A[1], ..., A[{self.N}]. "
            f"Please find integers X[1], ..., X[{self.N}] such that the value of S = "
            f"A[1] * X[1] + ... + A[{self.N}] * X[{self.N}] satisfies the condition: S > 0. "
            f"Try your best to minimize the value of S while meeting this condition.\n\n"
            f"A: {', '.join(map(str, self.A))}\n\n"
            f"Output Format: Output a single line containing X[1], ..., X[{self.N}], separated by spaces, "
            f"enclosed in \\boxed{{...}}."
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": self.N,
            "A": self.A,
        }

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the answer."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert boxed content to a list of integers
        try:
            user_coeffs = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate length
        if len(user_coeffs) != self.N:
            info = {
                "correct": False,
                "reason": "length_mismatch",
                "expected_length": self.N,
                "received_length": len(user_coeffs),
                "A": self.A,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        assert self.A is not None and self.gold_answer is not None, "Environment not properly initialized."

        # Compute S for user's coefficients
        S_user = sum(x * a for x, a in zip(user_coeffs, self.A))

        # Check validity and correctness:
        # Valid if S_user > 0. Correct if S_user equals minimal positive sum (gold_answer).
        is_valid = S_user > 0
        is_correct = is_valid and (S_user == self.gold_answer)

        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "valid": is_valid,
            "gold_answer": self.gold_answer,
            "user_sum": S_user,
            "A": self.A,
            "reference_coefficients": self.reference_coefficients,
            "user_coefficients": user_coeffs,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random list of integers enclosed in \\boxed{...}."""
        # Generate small random integers to form a candidate coefficient vector
        coeffs = [str(random.randint(-3, 3)) for _ in range(self.N)]
        return f"\\boxed{{{' '.join(coeffs)}}}"