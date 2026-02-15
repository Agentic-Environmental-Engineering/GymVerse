import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TwoSet_AllCoprime_CountingEnv(Env):
    """Environment for counting disjoint subset pairs (S, T) with all elements coprime across S and T."""

    def __init__(
        self,
        N: int = 50,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs,
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Upper bound for integers used to form the set (must be >= 3).
        - wrong_format: Preserved parameter from original environment (not used).
        - rewarding_strategy: Preserved parameter from original environment (not used).
        - rewarding_weight: Preserved parameter from original environment (not used).
        - rewarding_beta: Preserved parameter from original environment (not used).
        """
        super().__init__()
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        self.N = N

        # Preserved configuration parameters (not used for reward calculation in GEM)
        self.wrong_format = wrong_format
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # State variables
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.set_data: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a combinatorial number theory problem about coprime subsets.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate a random set of unique integers between 2 and N (inclusive)
        set_size = random.randint(2, self.N - 1)
        A = random.sample(range(2, self.N + 1), set_size)

        # Ensure uniqueness and store
        assert len(A) == len(set(A)) == set_size, "The set must contain unique integers"
        self.set_data = A

        # Compute reference answer using the original algorithm
        self.reference_answer = self._compute_reference(A)
        assert self.reference_answer is not None and self.reference_answer > 0

        # Build problem prompt
        problem_text = (
            f"You are given a set of integers: {' '.join(map(str, A))}\n\n"
            "Please compute the number of set pairs (S, T) such that:\n"
            "1. S and T are disjoint subsets of the given set.\n"
            "2. For every x in S and y in T, gcd(x, y) = 1 (i.e., there is no pair with gcd > 1).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute a single step by validating the submitted answer."""
        # Extract boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate the numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "set": self.set_data,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        import re

        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        random_answer = random.randint(0, 100)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference(self, A: List[int]) -> int:
        """Compute the reference answer using the original algorithm."""
        MAX = max(A)

        # Sieve to find the largest prime factor for each number up to MAX
        is_prime = [True] * (MAX + 1)
        is_prime[0] = False
        is_prime[1] = False
        max_prime_factor: List[Optional[int]] = [None] * (MAX + 1)
        for i in range(2, MAX + 1):
            if is_prime[i]:
                max_prime_factor[i] = i
                for j in range(2 * i, MAX + 1, i):
                    is_prime[j] = False
                    max_prime_factor[j] = i

        group2numbers: Dict[int, List[List[int]]] = {}
        small_primes: Dict[int, int] = {}

        # Group numbers based on their largest prime factor
        for a in A:
            prime_factors: List[int] = []
            x = a
            while x > 1:
                prime = max_prime_factor[x]
                assert prime is not None
                prime_factors.append(prime)
                x //= prime

            # Ensure the largest prime factor is the first one
            assert max(prime_factors) == prime_factors[0], "The largest prime factor must be the first one"

            if prime_factors[0] * prime_factors[0] > MAX:
                group = prime_factors[0]
                prime_factors = [p for p in prime_factors if p != group]
                if group not in group2numbers:
                    group2numbers[group] = []
                group2numbers[group].append(prime_factors)
            else:
                group2numbers[-a] = [prime_factors]

            for prime in prime_factors:
                if prime not in small_primes:
                    small_primes[prime] = len(small_primes)

        m = len(small_primes)
        F = [[0] * (1 << m) for _ in range(1 << m)]
        F[0][0] = 1

        # Dynamic programming over groups
        for _, prime_factors_list in group2numbers.items():
            G0 = [[F[S][T] for T in range(1 << m)] for S in range(1 << m)]
            G1 = [[F[S][T] for T in range(1 << m)] for S in range(1 << m)]
            for prime_factors in prime_factors_list:
                mask = 0
                for prime in prime_factors:
                    mask |= (1 << small_primes[prime])

                new_G0 = [[G0[S][T] for T in range(1 << m)] for S in range(1 << m)]
                new_G1 = [[G1[S][T] for T in range(1 << m)] for S in range(1 << m)]
                for S in range(1 << m):
                    T = (1 << m) - 1 - S
                    while True:
                        assert (T & S) == 0, "S and T must be disjoint"
                        if (mask & T) == 0:
                            new_G0[S | mask][T] += G0[S][T]
                        if (mask & S) == 0:
                            new_G1[S][T | mask] += G1[S][T]
                        if T == 0:
                            break
                        T = (T - 1) & ((1 << m) - 1 - S)
                G0 = new_G0
                G1 = new_G1

            for S in range(1 << m):
                for T in range(1 << m):
                    F[S][T] = G0[S][T] + G1[S][T] - F[S][T]

        reference_answer = sum(F[S][T] for S in range(1 << m) for T in range(1 << m))
        return reference_answer