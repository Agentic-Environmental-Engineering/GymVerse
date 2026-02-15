from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FactorialTrailingZeroCountEnv(Env):
    """Environment for computing the number of trailing zeros of N! in base K."""

    def __init__(self, max_n_k: int = 10, **kwargs) -> None:
        """
        Initialize the environment.

        Parameters:
            max_n_k (int): Maximum value for both N and K. Must be at least 10.

        Raises:
            ValueError: If max_n_k is less than 10.
        """
        super().__init__()
        if max_n_k < 10:
            raise ValueError("max_n_k should be greater than or equal to 10")
        self.max_n_k: int = max_n_k

        # Problem state
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving trailing zero count problems for factorials expressed in different bases.\n"
            "Task: Compute N! (N is given in base 10), express the result in base K, "
            "and find the number of trailing zeros in that base-K representation.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        self.N = random.randint(3, self.max_n_k)
        self.K = random.randint(2, self.max_n_k)

        # Build the problem prompt
        self.current_problem = (
            f"Compute {self.N}! (the factorial of {self.N}; {self.N} is in base 10) "
            f"and express the result in base {self.K}. "
            f"What's the number of trailing zeros in this base-{self.K} representation?\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute the reference answer
        self.reference_answer = self._compute_trailing_zeros(self.N, self.K)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer == user_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
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

    def _compute_trailing_zeros(self, n: int, k: int) -> int:
        """
        Compute the number of trailing zeros of n! in base k.

        The number of trailing zeros in base k is determined by the minimum over all prime
        factors p_i of k of floor(v_{p_i}(n!) / c_i), where k = prod p_i^{c_i}.
        """
        # Factorize k into primes: k = product p_i^{c_i}
        primes: list[int] = []
        counts: list[int] = []
        t = k
        i = 2
        while i * i <= t:
            if t % i == 0:
                cnt = 0
                while t % i == 0:
                    t //= i
                    cnt += 1
                primes.append(i)
                counts.append(cnt)
            i += 1
        if t > 1:
            primes.append(t)
            counts.append(1)

        # Compute min_i floor(v_p_i(n!) / c_i)
        ans: Optional[int] = None
        for idx, p in enumerate(primes):
            exp = 0
            now = n
            while now:
                now //= p
                exp += now
            t_val = exp // counts[idx]
            if ans is None or t_val < ans:
                ans = t_val

        return ans if ans is not None else 0

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # If N is known, the trailing zeros cannot exceed N
        upper = self.N if self.N is not None else self.max_n_k
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"