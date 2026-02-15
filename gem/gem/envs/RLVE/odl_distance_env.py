from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ODLDistanceEnv(Env):
    """Environment for the minimum prime operation distance problem (single-turn Q&A)."""

    def __init__(
        self,
        weight_multiple: int = 4,
        min_n: int = 3,
        max_n: int = 100,
        fixed_n: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the ODLDistanceEnv instance.

        Parameters:
            weight_multiple: Controls the range of values in A (values sampled from [1, N * weight_multiple]).
            min_n: Minimum length of the array A when N is sampled randomly.
            max_n: Maximum length of the array A when N is sampled randomly.
            fixed_n: If provided, the array length N will be set to this value (must be >= 3).
        """
        super().__init__()
        self.weight_multiple = weight_multiple
        self.min_n = min_n
        self.max_n = max_n
        self.fixed_n = fixed_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[List[int]] = None
        self.N: Optional[int] = None
        self.A: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task description and output requirements."""
        return (
            "You are given an integer array A and an operation that either multiplies by a prime number "
            "or divides by a prime number (only if divisible). Define D(a, b) as the minimum number of such operations "
            "needed to transform a into b.\n"
            "For each index i (0 <= i < N), find the index j (j ≠ i) such that D(A[i], A[j]) is minimized; "
            "if multiple such j exist, choose the smallest one.\n\n"
            "Output Format: Provide N integers (the j values for each i in order, separated by spaces) in \\boxed{...}."
            "\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem instance, and compute the reference answer."""
        super().reset(seed)

        # Determine N
        if self.fixed_n is not None:
            assert self.fixed_n >= 3, "N should be greater than or equal to 3"
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
            assert N >= 3, "N should be greater than or equal to 3"

        # Generate array A with unique values
        A = random.sample(range(1, N * self.weight_multiple + 1), N)

        # Compute gold answer
        gold = self._compute_gold_answer(A)

        # Build the problem prompt
        problem = (
            "Define an operation on an integer as either multiplying it by a prime number, "
            "or dividing it by a prime number (only if it is divisible by that prime). "
            "Define D(a, b) as the minimum number of such operations needed to transform a into b; "
            "for example, D(69, 42) = 3 because 69 → 3 → 6 → 42 (i.e., divide by 23, multiply by 2, multiply by 7).\n\n"
            f"Given an array A of length {N}: " +
            " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A)) +
            "\nFor each index i (0 <= i < {N}), find the index j (j ≠ i) such that D(A[i], A[j]) is minimized; "
            "if multiple such j exist, choose the smallest one.\n\n"
            "Output Format: Your final answer should be the N integers (space-separated) in \\boxed{...}."
        )

        # Store for step
        self.N = N
        self.A = A
        self.gold_answer = gold
        self.reference_answer = " ".join(map(str, gold))
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": N,
            "A": A
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and validate the user answer."""
        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the sequence of integers
        try:
            answer_array = list(map(int, boxed_content.split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate structural constraints
        assert self.N is not None and self.gold_answer is not None and self.reference_answer is not None
        if len(answer_array) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "wrong_length"}

        if not all(0 <= j < self.N and j != i for i, j in enumerate(answer_array)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "index_out_of_range_or_self_ref"}

        # Check correctness
        is_correct = (answer_array == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, answer_array)),
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
        """Sample a random valid-looking action in boxed format."""
        if self.N is None:
            # Fallback if called before reset
            n = max(self.min_n, 3)
        else:
            n = self.N
        # For each index i, choose a random j != i
        arr = []
        for i in range(n):
            # Choose j from [0, n) excluding i
            choices = list(range(0, i)) + list(range(i + 1, n))
            j = random.choice(choices) if choices else 0
            arr.append(j)
        return f"\\boxed{{{' '.join(map(str, arr))}}}"

    def _compute_gold_answer(self, A: List[int]) -> List[int]:
        """
        Compute the gold answer array: for each i, find j (j != i) minimizing D(A[i], A[j]),
        breaking ties in favor of the smallest j.
        """
        U = max(A)

        # compute Omega(n): number of prime factors of n with multiplicity
        num = [0] * (U + 1)
        primes: List[int] = []
        for i in range(2, U + 1):
            if num[i] == 0:
                primes.append(i)
                num[i] = 1
            for p in primes:
                x = p * i
                if x > U:
                    break
                num[x] = num[i] + 1
                if i % p == 0:
                    break

        # build linked lists of positions for each value
        N = len(A)
        t = [-1] * (U + 1)
        next_idx = [-1] * N
        for i, v in enumerate(A):
            next_idx[i] = t[v]
            t[v] = i

        # initialize answers
        INF = U + 1
        ans = [INF] * N
        ansj = [-1] * N

        # for each possible divisor x
        for x in range(1, U + 1):
            # collect all indices i with A[i] divisible by x
            q: List[int] = []
            for m in range(x, U + 1, x):
                j = t[m]
                while j != -1:
                    q.append(j)
                    j = next_idx[j]
            if not q:
                continue

            # find index b in q with minimal num[A[b]] (tie-break on smaller index)
            b = q[0]
            for i in range(1, len(q)):
                qi = q[i]
                if num[A[qi]] < num[A[b]] or (num[A[qi]] == num[A[b]] and qi < b):
                    q[i], b = b, qi

            # update distances using this common divisor x
            common = num[x] << 1
            for i in range(1, len(q)):
                a_i = q[i]
                d = num[A[a_i]] + num[A[b]] - common

                # update for a_i
                if d < ans[a_i] or (d == ans[a_i] and b < ansj[a_i]):
                    ans[a_i] = d
                    ansj[a_i] = b

                # update for b
                if d < ans[b] or (d == ans[b] and a_i < ansj[b]):
                    ans[b] = d
                    ansj[b] = a_i

        return ansj