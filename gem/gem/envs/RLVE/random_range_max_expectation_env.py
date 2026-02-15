import random
from typing import Any, List, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RandomRangeMaxExpectationEnv(Env):
    """Environment for the Random Range Max Expectation problem - single turn Q&A.

    In each operation, a uniformly random subarray is selected and all elements
    in that subarray are changed to the maximum value within it. The task is to
    compute the expected value at each position after Q operations, scaled by
    (N × (N + 1) / 2)^Q modulo a given number.
    """

    def __init__(
        self,
        min_n: int = 2,
        max_n: int = 50,
        modulo: int = 10000,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
            min_n: Minimum length of the array N (must be >= 2).
            max_n: Maximum length of the array N.
            modulo: Modulo to be used in the computation (default 10000).
        """
        super().__init__()
        assert min_n >= 2, "min_n should be greater than or equal to 2"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"

        self.min_n = min_n
        self.max_n = max_n
        self.modulo = modulo

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[List[int]] = None
        self.N: Optional[int] = None
        self.Q: Optional[int] = None
        self.array: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an expectation problem on arrays with random range-max operations.\n"
            "Please provide your answer in \\boxed{...} format, containing N integers separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(self.min_n, self.max_n)
        Q = random.randint(1, N)
        A = [random.randint(0, N) for _ in range(N)]

        self.N = N
        self.Q = Q
        self.array = A

        # Build problem prompt
        self.current_problem = (
            f"You are given an array of {N} integers: {' '.join(map(str, A))}\n\n"
            f"You will perform {Q} operations in order. In each operation, you uniformly select a subarray "
            f"(a contiguous segment of the array) at random from all {N} × ({N} + 1) / 2 possible subarrays. "
            f"Then, all elements in that subarray are changed to the maximum value within it.\n\n"
            f"Please compute the expected value of each position in the array after all {Q} operations. "
            f"Since the expected value is a rational number with denominator ({N} × ({N} + 1) / 2)^{Q}, "
            f"output the numerator (i.e., the expected value multiplied by ({N} × ({N} + 1) / 2)^{Q}), modulo {self.modulo}.\n\n"
            f"Output Format: A single line containing {N} integers — the scaled expected values (modulo {self.modulo}) "
            f"for each position, separated by spaces, wrapped in \\boxed{{...}}."
        )

        # Compute reference answer using the original algorithm
        result = self._compute_gold_answer(A, N, Q, self.modulo)
        self.gold_answer = result
        self.reference_answer = " ".join(map(str, result))

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_gold_answer(self, A: List[int], N: int, Q: int, MOD: int) -> List[int]:
        """Compute the gold answer using the core algorithm from the original environment."""
        def calc(x: int) -> int:
            return x * (x + 1) // 2 % MOD

        # Sentinel INF just above any value in A
        INF = max(A) + 1

        # Prepare DP tables: f[0] for previous round, f[1] for current
        f = [
            [[0] * N for _ in range(N)],
            [[0] * N for _ in range(N)]
        ]

        # g[l][r] is the weight factor
        g = [[0] * N for _ in range(N)]

        # Precompute g
        for l in range(N):
            for r in range(l, N):
                length = r - l + 1
                left = l
                right = N - 1 - r
                g[l][r] = (calc(length) + calc(left) + calc(right)) % MOD

        # Base case f[0]
        for l in range(N):
            maxx = 0
            for r in range(l, N):
                if A[r] > maxx:
                    maxx = A[r]
                if l == 0 and r == N - 1:
                    f[0][l][r] = maxx % MOD
                else:
                    left_val = INF if l == 0 else A[l - 1]
                    right_val = INF if r == N - 1 else A[r + 1]
                    if left_val > maxx and right_val > maxx:
                        f[0][l][r] = (maxx - min(left_val, right_val)) % MOD

        # Perform Q random-interval operations in expectation
        for i in range(1, Q + 1):
            now = i & 1
            pre = 1 - now

            # Prefix sums s1 and suffix sums s2
            s1 = [[0] * N for _ in range(N)]
            s2 = [[0] * N for _ in range(N)]

            # Build s1: for each r, accumulate over l=0..r of f[pre][l][r] * l
            for r in range(N):
                acc = 0
                for l in range(0, r + 1):
                    acc = (acc + f[pre][l][r] * l) % MOD
                    s1[l][r] = acc

            # Build s2: for each l, accumulate over r=N-1..l of f[pre][l][r] * (N-1-r)
            for l in range(N):
                acc = 0
                for r in range(N - 1, l - 1, -1):
                    acc = (acc + f[pre][l][r] * (N - 1 - r)) % MOD
                    s2[l][r] = acc

            # Update f[now] using precomputed g, s1, s2
            for l in range(N):
                for r in range(l, N):
                    left_contrib = s1[l - 1][r] if l - 1 >= 0 else 0
                    right_contrib = s2[l][r + 1] if r + 1 < N else 0
                    f[now][l][r] = (
                        f[pre][l][r] * g[l][r]
                        + left_contrib
                        + right_contrib
                    ) % MOD

        # Collect final answers
        result: List[int] = []
        final_dp = f[Q & 1]
        for i in range(N):
            ans = 0
            for l in range(0, i + 1):
                for r in range(i, N):
                    ans = (ans + final_dp[l][r]) % MOD
            result.append(ans)

        return result

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the submitted answer."""
        # Parse boxed answer content
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure environment has a problem loaded
        if self.reference_answer is None or self.gold_answer is None or self.N is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem"}

        # Parse the list of integers
        try:
            parts = boxed_content.strip().split()
            user_answer_list = list(map(int, parts))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length
        if len(user_answer_list) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_length", "expected_length": self.N}

        # Check correctness
        is_correct = (user_answer_list == self.gold_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, user_answer_list)),
            "N": self.N,
            "Q": self.Q,
            "array": self.array,
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

    def sample_random_action(self) -> str:
        """Sample a random action: a boxed string with N random integers."""
        if self.N is None:
            # Fallback: assume a reasonable N if reset not yet called
            n = self.min_n
        else:
            n = self.N
        random_values = [str(random.randint(0, self.modulo - 1)) for _ in range(n)]
        return f"\\boxed{{{' '.join(random_values)}}}"