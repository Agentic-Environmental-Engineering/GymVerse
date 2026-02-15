import random
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class QuadraticFunctionSegmentationEnv(Env):
    """Quadratic function segmentation problem environment - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 100,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, the problem size will be fixed to this value. Must be >= 4.
        - min_N: Minimum N when sampling randomly in reset() if N is not provided.
        - max_N: Maximum N when sampling randomly in reset() if N is not provided.
        """
        super().__init__()
        if N is not None and N < 4:
            raise ValueError("N should be greater than or equal to 4")
        if min_N < 4:
            raise ValueError("min_N should be greater than or equal to 4")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Problem state
        self.N: Optional[int] = None
        self.xs: Optional[List[int]] = None
        self.A_coef: Optional[int] = None
        self.B_coef: Optional[int] = None
        self.C_coef: Optional[int] = None
        self.gold_value: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Segment a sequence into consecutive batches to maximize the sum of quadratic batch values.\n"
            "Output Format: Provide the end indices of each batch (space-separated) inside \\boxed{...}.\n"
            "Example: For N=7, an answer could be \\boxed{2 5 7}, meaning batches [1..2], [3..5], [6..7].\n"
            "Note: The last end index must be N.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Choose N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 4, "N should be greater than or equal to 4"

        # Generate until the instance satisfies the non-trivial optimality condition
        while True:
            xs = [random.randint(1, N) for _ in range(N)]
            A = -random.randint(1, N)
            B = random.randint(1, random.randint(1, N) * random.randint(1, N) * random.randint(1, N))
            bound_pos = (
                random.randint(1, N)
                * random.randint(1, N)
                * random.randint(1, N)
                * random.randint(1, N)
                * random.randint(1, N)
            )
            bound_neg = (
                random.randint(1, N)
                * random.randint(1, N)
                * random.randint(1, N)
                * random.randint(1, N)
                * random.randint(1, N)
            )
            C = random.randint(-bound_neg, +bound_pos)

            # Compute prefix sums
            s = [0] * (N + 1)
            for i in range(1, N + 1):
                s[i] = s[i - 1] + xs[i - 1]

            # DP array for maximum total value up to i
            d = [0] * (N + 1)

            # Deque for convex hull (indices of candidate break points)
            q = [0] * (N + 1)
            head = tail = 0
            q[0] = 0

            # Helper functions
            def K(i: int) -> int:
                return 2 * A * s[i]

            def X(i: int) -> int:
                return s[i]

            def Y(i: int) -> int:
                # y(i) = d[i] + A*s[i]^2 - B*s[i]
                return d[i] + A * s[i] * s[i] - B * s[i]

            def slope(i: int, j: int) -> float:
                # (Y(i)-Y(j)) / (X(i)-X(j))
                return (Y(i) - Y(j)) / (X(i) - X(j))

            # Convex hull trick DP
            for i in range(1, N + 1):
                while head < tail and slope(q[head], q[head + 1]) > K(i):
                    head += 1

                j = q[head]
                d[i] = -(K(i) * X(j) - Y(j) - A * s[i] * s[i] - B * s[i] - C)

                while head < tail and slope(q[tail - 1], q[tail]) <= slope(q[tail], i):
                    tail -= 1

                tail += 1
                q[tail] = i

            gold_answer = d[N]

            def compute_value(X: int) -> int:
                return A * (X ** 2) + B * X + C

            # Trivial best comparisons
            trivial_best = max(sum(compute_value(x) for x in xs), compute_value(sum(xs)))
            prefix_sum, suffix_sum = 0, sum(xs)
            for x in xs:
                prefix_sum += x
                suffix_sum -= x
                if prefix_sum > 0 and suffix_sum > 0:
                    trivial_best = max(trivial_best, compute_value(prefix_sum) + compute_value(suffix_sum))
            if gold_answer > trivial_best:
                if gold_answer > 0:
                    # Store the generated instance
                    self.N = N
                    self.xs = xs
                    self.A_coef = A
                    self.B_coef = B
                    self.C_coef = C
                    self.gold_value = gold_answer
                    break
            else:
                assert gold_answer == trivial_best, "Gold answer should be greater than trivial best"

        # Build problem prompt
        A_lines = "\n".join(f"A[{i}]={Ai}" for i, Ai in enumerate(self.xs, start=1))
        problem = (
            f"You are given {self.N} numbers A[1], A[2], ..., A[{self.N}]. The values are given as:\n{A_lines}\n\n"
            f"You may divide these numbers (in order) into some consecutive batches. Let the total number of batches be k (1 ≤ k ≤ {self.N}), "
            f"and let end[1], end[2], ..., end[k] (1 ≤ end[1] < end[2] < ... < end[k] = {self.N}) denote the last index in each batch.\n"
            f"- Batch 1 contains elements A[1] to A[end[1]]\n"
            f"- Batch 2 contains elements A[end[1] + 1] to A[end[2]]\n"
            f"- ...\n"
            f"- Batch k contains elements A[end[k−1] + 1] to A[end[k]] (with end[k] = {self.N}).\n\n"
            f"Define the value of a batch with sum X as: {self.A_coef} × X^2 + {self.B_coef} × X + {self.C_coef}.\n"
            f"The total value of the division is the sum of values of all batches. Find a batch division that maximizes this total value.\n\n"
            f"Output Format: Provide end[1] end[2] ... end[k] inside \\boxed{{...}}, with end[k] always equal to {self.N}.\n"
            f"Example: \\boxed{{1 2 {self.N}}} means there are 3 batches; first ends at 1, second at 2, third at {self.N}."
        )

        self.current_problem = problem
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def compute_value(self, X: int) -> int:
        """Compute the batch value given sum X."""
        assert self.A_coef is not None and self.B_coef is not None and self.C_coef is not None
        return self.A_coef * (X ** 2) + self.B_coef * X + self.C_coef

    def _evaluate_partition(self, ends: List[int]) -> int:
        """Evaluate the total value of a given partition defined by end indices."""
        assert self.xs is not None and self.N is not None
        total = 0
        last = 0  # 0-based index of last end
        for end in ends:
            # end is 1-based; slice xs[last:end] covers elements last..end-1
            batch_sum = sum(self.xs[last:end])
            total += self.compute_value(batch_sum)
            last = end
        return total

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer."""
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse end indices
        try:
            tokens = boxed.split()
            ends = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.N is None or self.gold_value is None or self.xs is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Validate ends
        if not (1 <= len(ends) <= self.N):
            info = {"error": "invalid_solution", "reason": "invalid_length"}
            return TERMINAL_STATE, 0.0, True, False, info
        for i in range(len(ends)):
            if not (1 <= ends[i] <= self.N):
                info = {"error": "invalid_solution", "reason": "index_out_of_range"}
                return TERMINAL_STATE, 0.0, True, False, info
            if i and not (ends[i - 1] < ends[i]):
                info = {"error": "invalid_solution", "reason": "not_strictly_increasing"}
                return TERMINAL_STATE, 0.0, True, False, info
        if ends[-1] != self.N:
            info = {"error": "invalid_solution", "reason": "last_end_not_N"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Evaluate given partition
        user_value = self._evaluate_partition(ends)
        gold_value = self.gold_value
        # Sanity check: user's value should not exceed the computed optimum
        assert user_value <= gold_value, "User value should not be greater than the gold value"

        is_correct = (user_value == gold_value)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "gold_value": gold_value,
            "user_value": user_value,
            "N": self.N,
            "A_coef": self.A_coef,
            "B_coef": self.B_coef,
            "C_coef": self.C_coef,
            "xs": self.xs,
        }

        return TERMINAL_STATE, reward, True, False, info

    def sample_random_action(self) -> str:
        """Sample a random valid action (random segmentation) in boxed format."""
        if self.N is None:
            # Default to a simple boxed empty to trigger format error if called too early
            return "\\boxed{}"

        # Randomly choose number of batches k in [1, N]
        k = random.randint(1, self.N)
        # Choose k-1 cut positions from [1, N-1], sort them, and append N
        if k == 1:
            ends = [self.N]
        else:
            cuts = sorted(random.sample(range(1, self.N), k - 1))
            ends = cuts + [self.N]
        return "\\boxed{" + " ".join(map(str, ends)) + "}"