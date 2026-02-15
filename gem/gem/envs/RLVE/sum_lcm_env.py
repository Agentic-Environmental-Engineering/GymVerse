import random
from typing import Any, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumLCMEnv(Env):
    """Environment for computing the sum of LCM(i, j) over 1 ≤ i ≤ N and 1 ≤ j ≤ M."""

    def __init__(
        self,
        max_n_m: int = 100000,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        """
        Initialize the SumLCMEnv environment.

        Args:
            max_n_m: Upper bound for both N and M. Must be >= 2.
            wrong_format: Preserved from original environment (unused in GEM scoring).
            rewarding_strategy: Preserved from original environment (unused in GEM scoring).
            rewarding_weight: Preserved from original environment (unused in GEM scoring).
            rewarding_beta: Preserved from original environment (unused in GEM scoring).
            **kwargs: Additional keyword arguments for extensibility.
        """
        super().__init__()
        if max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")

        self.max_n_m = max_n_m
        # Preserve original reward-related parameters (not used in GEM reward logic)
        self.rewards_config = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an arithmetic problem involving least common multiples (LCM).\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters N and M
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Build the problem prompt
        self.current_problem = (
            f"Please compute sum(LCM(i, j)) for all pairs (i, j) such that 1 ≤ i ≤ {self.N} and 1 ≤ j ≤ {self.M}. "
            f"Here, LCM(i, j) denotes the least common multiple of integers i and j.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute the reference answer using the Mobius inversion approach
        self.reference_answer = self._compute_sum_lcm(self.N, self.M)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Extract the boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            # Format error: no boxed content found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate that the parsed content is an integer
        try:
            user_answer = int(parsed)
        except ValueError:
            # Not a valid integer inside the box
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check correctness
        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        import re

        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by generating a random integer answer in boxed format."""
        random_answer = random.randint(0, 10**18)
        return f"\\boxed{{{random_answer}}}"

    @staticmethod
    def _compute_sum_lcm(N: int, M: int) -> int:
        """
        Compute the sum of LCM(i, j) for 1 ≤ i ≤ N, 1 ≤ j ≤ M using Mobius inversion and grouping.
        """
        max_rep = max(N, M)

        # Compute the Mobius function up to max_rep
        mu = [0] * (max_rep + 1)
        mu[1] = 1
        primes = []
        vis = bytearray(max_rep + 1)

        for i in range(2, max_rep + 1):
            if not vis[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                ip = i * p
                if ip > max_rep:
                    break
                vis[ip] = 1
                if i % p == 0:
                    mu[ip] = 0
                    break
                mu[ip] = -mu[i]

        # Prefix of mu[i] * i^2
        pref = [0] * (max_rep + 1)
        for i in range(1, max_rep + 1):
            pref[i] = pref[i - 1] + mu[i] * i * i

        def tri(t: int) -> int:
            # Triangular number sum: 1 + 2 + ... + t
            return (1 + t) * t // 2

        ans = 0
        for d in range(1, max_rep + 1):
            nx, ny = N // d, M // d
            limit = nx if nx < ny else ny
            l = 1
            subtotal = 0
            while l <= limit:
                r = min(nx // (nx // l), ny // (ny // l))
                mu_segment = pref[r] - pref[l - 1]
                sx = tri(nx // l)
                sy = tri(ny // l)
                subtotal += mu_segment * sx * sy
                l = r + 1
            ans += subtotal * d

        return ans