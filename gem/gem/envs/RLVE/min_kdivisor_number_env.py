from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinKDivisorNumberEnv(Env):
    """
    Environment for the problem:
    Find the smallest positive integer M such that it has exactly K distinct positive divisors.
    Single-turn Q&A style. The agent must respond in \\boxed{...} format.
    """

    def __init__(self, max_k: int = 100000, **kwargs) -> None:
        """
        Initialize the environment.

        Args:
            max_k: The maximum possible value for K. K will be sampled uniformly from [1, max_k].
        """
        super().__init__()
        assert isinstance(max_k, int), "max_k must be an integer"
        assert max_k >= 1, "max_k should be greater than or equal to 1"
        self.max_k: int = max_k

        self.current_k: Optional[int] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """
        Return task instructions for the agent.
        """
        return (
            "You are solving a number theory problem.\n"
            "Task: Find the smallest positive integer M such that it has exactly K distinct positive divisors.\n"
            "Answer Format: Provide a single integer in \\boxed{...} (do not include quotes or backticks).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: The instruction and the generated problem statement.
            info: An auxiliary information dict (empty for this environment).
        """
        super().reset(seed)

        # Sample K
        K = random.randint(1, self.max_k)
        self.current_k = K

        # Build problem prompt
        self.current_problem = (
            f"Find the smallest positive integer M such that it has exactly {K} distinct positive divisors.\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute reference answer using the same algorithm as in the original environment
        sum_e = sum(e for _, e in self._prime_factorization_internal(K, float("inf")))
        all_primes = self._generate_first_n_primes(sum_e)

        dp_cache: dict[Tuple[int, int], int] = {}

        def dp(p: int, n: int) -> int:
            if n == 1:
                return 1
            key = (p, n)
            if key in dp_cache:
                return dp_cache[key]
            ans = all_primes[p] ** (n - 1)
            if p + 1 < len(all_primes):
                factors: List[int] = []
                f = 1
                # Enumerate factors of n
                while f * f <= n:
                    if n % f == 0:
                        factors.append(f)
                        if n // f > f:
                            factors.append(n // f)
                    f += 1
                for factor in factors:
                    if factor > 1:
                        candidate = (all_primes[p] ** (factor - 1)) * dp(p + 1, n // factor)
                        if candidate < ans:
                            ans = candidate
            dp_cache[key] = ans
            return ans

        self.reference_answer = dp(0, K)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step by validating the agent's answer.

        Args:
            action: The agent's textual answer, expected in \\boxed{...} format.

        Returns:
            observation: TERMINAL_STATE since this is a single-turn environment.
            reward: 1.0 if correct; 0.0 if wrong; -0.1 if format error or non-positive number.
            terminated: Always True for single-turn tasks.
            truncated: Always False for this environment.
            info: Dictionary with evaluation details.
        """
        # Parse answer
        extracted = self._parse_answer(action)
        if extracted is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Check if it is an integer
        try:
            user_answer = int(extracted)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Non-positive numbers are invalid for this task
        if user_answer <= 0:
            return TERMINAL_STATE, -0.1, True, False, {"error": "non_positive"}

        assert self.reference_answer is not None, "Environment not properly reset before step."
        assert self.current_k is not None, "Environment not properly reset before step."

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        # Auxiliary checks for user answer (divisor count)
        divisor_count: Optional[int] = None
        has_k_divisors: Optional[bool] = None
        try:
            factorization_result = self._prime_factorization_internal(user_answer, int(1e7))
            if factorization_result is not None:
                divisor_count = 1
                for _, e in factorization_result:
                    divisor_count *= (e + 1)
                has_k_divisors = (divisor_count == self.current_k)
        except Exception:
            pass

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "k": self.current_k,
            "divisor_count": divisor_count,
            "has_k_divisors": has_k_divisors,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside \\boxed{...}.
        If multiple boxed answers are present, return the last one.
        """
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random action for testing, returning an answer in \\boxed{...} format.
        """
        if self.reference_answer is not None:
            upper = max(10, self.reference_answer * 2)
        else:
            upper = 100
        random_answer = random.randint(1, upper)
        return f"\\boxed{{{random_answer}}}"

    def _is_prime(self, n: int) -> bool:
        """
        Check if n is a prime number.
        """
        if n == 2 or n == 3:
            return True
        if n < 2 or n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    def _generate_first_n_primes(self, n: int) -> List[int]:
        """
        Generate the first n prime numbers.
        """
        if n <= 0:
            return []
        primes = [2]
        while len(primes) < n:
            candidate = primes[-1] + 1
            while not self._is_prime(candidate):
                candidate += 1
            primes.append(candidate)
        return primes

    def _prime_factorization_internal(self, num: int, limit: float) -> Optional[List[Tuple[int, int]]]:
        """
        Prime factorization with an upper bound on the trial divisor.

        Args:
            num: The number to factorize.
            limit: The maximum allowed trial divisor. If exceeded, return None.

        Returns:
            A list of (prime, exponent) pairs, or None if the limit is exceeded.
        """
        n = num
        factors: List[Tuple[int, int]] = []
        d = 2
        while d * d <= n:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            if e > 0:
                factors.append((d, e))
            d += 1
            if d > limit:
                return None
        if n > 1:
            factors.append((n, 1))
        return factors