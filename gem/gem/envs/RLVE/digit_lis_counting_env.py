from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DigitLISCountingEnv(Env):
    """Digit LIS Counting environment - single-turn Q&A.

    Task:
    Consider all integers N in the inclusive range [L, R]. Interpret each N as a string of decimal digits.
    The power of N is defined as the length of the longest strictly increasing subsequence of its digits.
    Count how many integers N within the range [L, R] have a power value exactly equal to K.

    Answer format:
    Return a single integer wrapped in \\boxed{...}.
    """

    def __init__(self, N: int, **kwargs) -> None:
        """
        Initialize the DigitLISCountingEnv instance.

        Args:
            N: The number of digits for the upper bound R (R will have exactly N digits, no leading zeros).
               Must be >= 2.

        Raises:
            AssertionError: If N < 2.
        """
        super().__init__()
        assert N >= 2, "N should be greater than or equal to 2"
        self.N: int = N

        # Problem state
        self.L: Optional[int] = None
        self.R: Optional[int] = None
        self.K: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a digit dynamic programming problem.\n"
            "For integers N in [L, R], treat N as a decimal string. The power of N is the length of the\n"
            "longest strictly increasing subsequence of its digits. Count how many N in [L, R] have power exactly K.\n"
            "Please provide your final answer as a single integer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new single-turn problem.

        Args:
            seed: Optional random seed.

        Returns:
            A tuple of (observation, info), where observation is the full prompt string and info is an empty dict.
        """
        super().reset(seed)

        # Generate problem parameters
        R = random.randint(10 ** (self.N - 1), 10 ** self.N - 1)
        L = random.randint(0, R)
        K = random.randint(1, min(self.N, 10))

        self.L = L
        self.R = R
        self.K = K

        # Build problem prompt
        self.current_problem = (
            f"Consider all integers N in the inclusive range [{L}, {R}]. "
            f"Interpret each N as a string of decimal digits. The power of N is defined as the length of the "
            f"longest strictly increasing subsequence of its digits.\n\n"
            f"Please count how many integers N within the range [{L}, {R}] have a power value exactly equal to {K}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._count_with_power_exactly_k(R, self.N, K) - self._count_with_power_exactly_k(L - 1, self.N, K)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer.

        Args:
            action: The agent's response text, expected to contain \\boxed{...} with the final integer.

        Returns:
            TERMINAL_STATE as observation,
            reward (1.0 if correct, 0.0 if incorrect, -0.1 if format error),
            terminated=True,
            truncated=False,
            info dict containing correctness and reference information.
        """
        # Parse the boxed answer
        answer_str = self._parse_answer(action)

        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare
        try:
            user_answer = int(answer_str)
            is_correct = (user_answer == self.reference_answer)
            reward: float = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "L": self.L,
            "R": self.R,
            "K": self.K,
            "N": self.N,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within the last \\boxed{...} in the given text.

        Args:
            text: The agent's full response.

        Returns:
            The extracted string inside the last \\boxed{...}, or None if not found.
        """
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by guessing a plausible count within range length."""
        if self.L is None or self.R is None:
            # Fallback if called before reset
            guess = random.randint(0, 10 ** self.N)
        else:
            guess = random.randint(0, max(0, self.R - self.L + 1))
        return f"\\boxed{{{guess}}}"

    def _count_with_power_exactly_k(self, x: int, N: int, K: int) -> int:
        """Count numbers in [0, x] whose digit LIS length is exactly K, using digit DP.

        Args:
            x: Upper bound (inclusive).
            N: Number of digits capacity for R (defines DP max depth).
            K: Target LIS length on digits.

        Returns:
            The count of numbers in [0, x] with LIS length exactly K.
        """
        if x < 0:
            # No numbers to count when x < 0
            return 0

        # Prepare digit array (little-endian by position for DP)
        a = [0] * N

        # dp[pos][sta][K] caches when limit=False and lead=False for a fixed K.
        # pos ranges 0..N-1 in recursion; allocate N+1 for safety as in original code.
        dp = [[[-1 for _ in range(K + 1)] for _ in range(1025)] for _ in range(N + 1)]

        def new_state(sta: int, n: int) -> int:
            """Update the bitmask state for patience sorting-like DP."""
            for i in range(n, 10):
                if (sta >> i) & 1:
                    return (sta ^ (1 << i)) | (1 << n)
            return sta | (1 << n)

        def bit_count(sta: int) -> int:
            """Count set bits in the state."""
            return bin(sta).count("1")

        def dfs(pos: int, sta: int, limit: bool, lead: bool) -> int:
            """Digit DP recursion."""
            if pos == -1:
                return 1 if bit_count(sta) == K else 0
            if not limit and not lead and dp[pos][sta][K] != -1:
                return dp[pos][sta][K]
            up = a[pos] if limit else 9
            ans = 0
            for d in range(0, up + 1):
                next_sta = 0 if (lead and d == 0) else new_state(sta, d)
                ans += dfs(pos - 1, next_sta, limit and (d == up), lead and (d == 0))
            if not limit and not lead:
                dp[pos][sta][K] = ans
            return ans

        # Decompose x into digits in a[]
        pos = -1
        while x > 0:
            pos += 1
            a[pos] = x % 10
            x //= 10

        return dfs(pos, 0, True, True)