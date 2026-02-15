from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class HeapCountingEnv(Env):
    """Environment for counting heap-ordered permutations modulo a prime - single-turn QA.

    The task: Compute the number of permutations P of the numbers 1..N such that
    for all 2 ≤ i ≤ N, it holds that P[i] > P[i // 2]. Output the result modulo a prime P.
    """

    def __init__(
        self,
        max_n: int = 1000,
        **kwargs: Any
    ):
        """Initialize the environment.

        Args:
            max_n: Maximum value for N (N will be sampled uniformly from [3, max_n]).
        """
        super().__init__()
        assert max_n >= 3, "max_n should be greater than or equal to 3"
        self.max_n: int = max_n

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.P: Optional[int] = None  # prime modulus

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving heap-ordered permutation counting problems (binary heap property).\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: A string containing the instructions and the generated problem.
            info: An empty dictionary or additional metadata.
        """
        super().reset(seed)

        # 1) Generate parameters N and prime P
        N = random.randint(3, self.max_n)
        is_prime = [True] * ((5 * N) + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int((5 * N) ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, (5 * N) + 1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, (5 * N) + 1) if is_prime[i]]
        P = random.choice(primes)

        self.N = N
        self.P = P

        # 2) Compute reference answer using Lucas theorem and DP on heap structure

        def mod_pow(a: int, b: int, p: int) -> int:
            """Compute a^b mod p via binary exponentiation."""
            res = 1
            a %= p
            while b:
                if b & 1:
                    res = (res * a) % p
                a = (a * a) % p
                b >>= 1
            return res

        def comb_small(n: int, k: int, p: int, fact: List[int]) -> int:
            """Compute C(n, k) mod p for 0 ≤ n, k < p (p is prime)."""
            if k < 0 or k > n:
                return 0
            denom = (fact[k] * fact[n - k]) % p
            return (fact[n] * mod_pow(denom, p - 2, p)) % p

        def lucas(n: int, k: int, p: int, fact: List[int]) -> int:
            """Compute C(n, k) mod p using Lucas theorem (p is prime)."""
            if k == 0:
                return 1
            return (lucas(n // p, k // p, p, fact) * comb_small(n % p, k % p, p, fact)) % p

        # Precompute factorials modulo P up to N
        fact = [1] * (N + 1)
        for i in range(1, N + 1):
            fact[i] = (fact[i - 1] * i) % P

        # Subtree sizes in the implicit binary heap (1-indexed)
        S = [0] * (5 * N + 2)
        for i in range(1, N + 1):
            S[i] = 1
        for i in range(N, 1, -1):
            S[i >> 1] += S[i]

        # DP for counting heap-ordered labelings
        DP = [1] * (2 * N + 2)  # beyond N treated as 1
        for i in range(N, 0, -1):
            left = i * 2
            right = left + 1
            dp_left = DP[left] if left < len(DP) else 1
            dp_right = DP[right] if right < len(DP) else 1
            choose_left = lucas(S[i] - 1, S[left] if left < len(S) else 0, P, fact)
            DP[i] = (choose_left * dp_left * dp_right) % P

        self.reference_answer = DP[1] % P

        # 3) Build problem prompt
        self.current_problem = (
            f"Compute the number of permutations P of the numbers 1 through {N} such that "
            f"for all 2 ≤ i ≤ {N}, it holds that P[i] > P[i // 2]. "
            f"Since the answer may be large, output the result modulo {P}, where {P} is a prime number.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer.

        Args:
            action: The agent's response string containing \\boxed{answer}.

        Returns:
            observation: TERMINAL_STATE since this is a single-turn environment.
            reward: 1.0 if correct; 0.0 if incorrect; -0.1 if format error.
            terminated: True (single-turn interaction).
            truncated: False.
            info: Additional information including correctness and reference answer.
        """
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check (answer should be in [0, P))
        out_of_range = False
        if self.P is not None and not (0 <= user_answer < self.P):
            out_of_range = True

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "P": self.P,
            "out_of_range": out_of_range
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the input text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        modulus = self.P if self.P is not None else 1000000007
        random_answer = random.randint(0, modulus - 1)
        return f"\\boxed{{{random_answer}}}"