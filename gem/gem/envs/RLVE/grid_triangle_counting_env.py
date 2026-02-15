from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GridTriangleCountingEnv(Env):
    """Grid triangle counting environment - single-turn Q&A."""

    prompt_template: str = (
        "How many non-degenerate triangles have all three vertices located at "
        "integer coordinate points (x, y) where 0 ≤ x ≤ {N} and 0 ≤ y ≤ {M}?"
    )

    def __init__(
        self,
        max_n_m: int = 100,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs,
    ):
        """
        Initialize the GridTriangleCountingEnv instance.

        Parameters:
            max_n_m: Maximum value for N and M (both are sampled in [1, max_n_m]).
            wrong_format: Preserved parameter from original environment (not used in GEM scoring).
            rewarding_strategy: Preserved parameter from original environment (not used in GEM scoring).
            rewarding_weight: Preserved parameter from original environment (not used in GEM scoring).
            rewarding_beta: Preserved parameter from original environment (not used in GEM scoring).
        """
        super().__init__()
        if max_n_m < 1:
            raise ValueError("max_n_m must be at least 1")
        self.max_n_m = max_n_m

        # Preserve original reward configuration parameters (not used in GEM scoring)
        self.wrong_format = wrong_format
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial geometry counting problem on a grid.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate N and M
        N = random.randint(1, self.max_n_m)
        M = random.randint(1, self.max_n_m)
        if N > M:
            N, M = M, N

        # Compute Euler's totient (phi) up to N using a sieve
        phi = [0] * (N + 1)
        mark = [False] * (N + 1)
        primes = []
        if N >= 1:
            phi[1] = 1
        for i in range(2, N + 1):
            if not mark[i]:
                primes.append(i)
                phi[i] = i - 1
            for p in primes:
                ip = i * p
                if ip > N:
                    break
                mark[ip] = True
                if i % p == 0:
                    phi[ip] = phi[i] * p
                    break
                else:
                    phi[ip] = phi[i] * (p - 1)

        # Combination function C(x, 3) = x*(x-1)*(x-2)/6
        def comb3(x: int) -> int:
            return x * (x - 1) * (x - 2) // 6

        # Compute the contribution from degenerate (colinear) triples
        degenerate = 0
        for d in range(2, N + 1):
            term = phi[d]
            term *= (N - d + N % d + 2) * (N // d)
            term *= (M - d + M % d + 2) * (M // d)
            degenerate += term // 2

        # Total number of triples of points minus colinear ones
        total_points = (N + 1) * (M + 1)
        total_triples = comb3(total_points)
        subtract_N_lines = (M + 1) * comb3(N + 1)
        subtract_M_lines = (N + 1) * comb3(M + 1)

        reference_answer = total_triples - subtract_N_lines - subtract_M_lines - degenerate
        assert reference_answer > 0

        # Store state
        self.N, self.M = N, M
        self.reference_answer = reference_answer

        # Build problem prompt
        self.current_problem = self.prompt_template.format(N=N, M=M)
        obs = self._get_instructions() + self.current_problem + "\n\n" + \
              "Output Format: Your final answer should be a single integer in \\boxed{...}."
        return obs, {"N": N, "M": M}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Perform one step: parse and verify the answer."""
        # Ensure a problem has been generated
        if self.reference_answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "no_problem"}

        # Parse answer from \\boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate integer parsing
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "invalid_answer"}

        # Check correctness
        is_correct = (user_answer == self.reference_answer)
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
        """Extract the answer inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        random_answer = random.randint(0, 10**6)
        return f"\\boxed{{{random_answer}}}"