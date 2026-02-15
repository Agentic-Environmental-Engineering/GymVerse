import random
from typing import Any, Optional, SupportsFloat, Tuple, Dict

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumGCDEnv(Env):
    """Environment for computing the sum of GCD(i, j)^K over 1 ≤ i ≤ N and 1 ≤ j ≤ M.

    This is a single-turn environment. The agent must provide the final answer
    in \\boxed{...} format.
    """

    def __init__(
        self,
        max_n_m: int = 1000,
        max_K: int = 5,
        **kwargs
    ):
        """Initialize the environment with difficulty parameters.

        Args:
            max_n_m: The maximum possible value for N and M (both are in [2, max_n_m]).
            max_K: The maximum possible exponent K (K is in [1, max_K]).
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        assert max_K >= 1, "max_K should be greater than or equal to 1"

        self.max_n_m = max_n_m
        self.max_K = max_K

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Store generated parameters for info/debugging
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions in English."""
        return (
            "You are solving a number theory problem involving greatest common divisors (GCD).\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Args:
            seed: Optional random seed.

        Returns:
            observation: The problem description and instructions.
            info: An empty dict.
        """
        super().reset(seed)

        # Generate problem parameters
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)
        self.K = random.randint(1, self.max_K)

        # Build problem prompt
        self.current_problem = (
            f"Please compute sum(GCD(i, j)^{{{self.K}}}) for all pairs (i, j) such that "
            f"1 ≤ i ≤ {self.N} and 1 ≤ j ≤ {self.M}. Here, GCD(i, j) denotes the greatest "
            f"common divisor of integers i and j, and x^{self.K} denotes x raised to the power of {self.K}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute the reference answer using multiplicative function and grouping
        self.reference_answer = self._compute_sum_gcd_power(self.N, self.M, self.K)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_sum_gcd_power(self, N: int, M: int, K: int) -> int:
        """Compute sum_{i=1..N, j=1..M} gcd(i, j)^K using a multiplicative function and grouping.

        The computation uses the identity:
        sum_{i=1..N, j=1..M} gcd(i, j)^K = sum_{d=1..min(N, M)} f_K(d) * floor(N/d) * floor(M/d),
        where f_K is multiplicative with
        f_K(p^a) = p^{Ka} - p^{K(a-1)}.

        Args:
            N: Upper bound for i.
            M: Upper bound for j.
            K: Exponent for GCD power.

        Returns:
            The computed sum as an integer.
        """
        limit = min(N, M)
        if limit <= 0:
            return 0

        # Linear sieve to compute multiplicative function f_K at all integers up to limit
        is_composite = [False] * (limit + 1)
        f = [0] * (limit + 1)  # f[i] holds f_K(i)
        primes: list[int] = []
        g: list[int] = []  # g[j] = (primes[j])^K

        # Base initialization
        f[1] = 1
        for i in range(2, limit + 1):
            if not is_composite[i]:
                primes.append(i)
                gi = i ** K
                g.append(gi)
                # f_K(p) = p^K - 1
                f[i] = gi - 1

            for j, p_j in enumerate(primes):
                ip = i * p_j
                if ip > limit:
                    break
                is_composite[ip] = True
                if i % p_j == 0:
                    # For prime power: f_K(p^{a+1}) = f_K(p^a) * p^K
                    f[ip] = f[i] * g[j]
                    break
                else:
                    # Multiplicativity for coprime factors: f_K(i * p) = f_K(i) * f_K(p)
                    f[ip] = f[i] * (g[j] - 1)

        # Prefix sum of f to allow range sum queries in grouping
        for i in range(1, limit + 1):
            f[i] = f[i] + f[i - 1]

        # Grouping by equal floor divisions
        ans = 0
        i = 1
        while i <= limit:
            ni = N // i
            mi = M // i
            nxt = min(N // ni, M // mi)
            s = f[nxt] - f[i - 1]
            ans += s * ni * mi
            i = nxt + 1

        return ans

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step by verifying the submitted answer.

        Args:
            action: The agent's response text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE for single-turn environment.
            reward: 1.0 if correct, 0.0 if incorrect, -0.1 for format error.
            terminated: Always True for single-turn.
            truncated: Always False.
            info: Additional information including correctness and reference answer.
        """
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
            "K": self.K,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text.

        Args:
            text: The agent's output.

        Returns:
            The extracted answer string, or None if not found.
        """
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format.

        Returns:
            A random integer wrapped in \\boxed{...}.
        """
        # If reference answer is known, sample around it; otherwise fallback to a small random number.
        if self.reference_answer is not None and self.reference_answer > 0:
            # To avoid extremely large random sampling, cap the range reasonably.
            upper = min(self.reference_answer * 2, 10**12)
            random_answer = random.randint(0, int(upper))
        else:
            random_answer = random.randint(0, 1000)
        return f"\\boxed{{{random_answer}}}"