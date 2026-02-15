from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SelfPowerSequenceMODEnv(Env):
    """Environment for the 'self power sequence modulo' problem - single-turn Q&A.

    Problem:
    - Define a[0] = 1, and for n >= 1, a[n] = 2^(a[n-1]).
    - Let b[n] = a[n] mod MOD.
    - It can be proven that b[n] becomes constant after some point.
    - Given MOD, find this eventual constant value.
    """

    def __init__(
        self,
        max_mod: int = 1_000_000,
        **kwargs
    ):
        super().__init__()
        if max_mod < 3:
            raise ValueError("max_mod should be greater than or equal to 3")
        self.max_mod = max_mod
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.mod: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a modular arithmetic problem about a rapidly growing sequence.\n"
            "Define a[0] = 1 and a[n] = 2^(a[n-1]) for n >= 1. Let b[n] = a[n] mod MOD.\n"
            "It can be proven that b[n] becomes constant after some point. Your task is to find this eventual constant value.\n"
            "Please provide your final answer in \\boxed{...} format with a single integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate random MOD within the valid range
        MOD = random.randint(3, self.max_mod)
        self.mod = MOD

        # Define helper functions (Euler's totient, modular exponentiation, and recursive solver)
        def phi(n: int) -> int:
            result = n
            i = 2
            while i * i <= n:
                if n % i == 0:
                    while n % i == 0:
                        n //= i
                    result = result // i * (i - 1)
                i += 1
            if n > 1:
                result = result // n * (n - 1)
            return result

        def pow_mod(x: int, p: int, mod: int) -> int:
            result = 1
            x %= mod
            while p:
                if p & 1:
                    result = (result * x) % mod
                x = (x * x) % mod
                p >>= 1
            return result

        def solve(p: int) -> int:
            if p == 1:
                return 0
            t = phi(p)
            return pow_mod(2, solve(t) + t, p)

        # Compute the reference answer using the original algorithm
        self.reference_answer = solve(MOD)

        # Build the problem description
        self.current_problem = (
            f"Define a sequence a where a[0] = 1 and a[n] = 2^(a[n-1]) for n >= 1. "
            f"Let b[n] = a[n] mod {MOD}. It can be proven that b[n] becomes constant after some point.\n"
            f"Task: Find this eventual constant value of b[n].\n\n"
            f"Output Format: Provide a single integer in \\boxed{{...}} representing the eventual constant value modulo {MOD}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted answer.

        Rewards:
        - Correct answer: 1.0
        - Wrong answer (including out-of-range): 0.0
        - Format error (no \\boxed{...}): -0.1
        """
        # Parse the boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate integer conversion
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate range (must be 0 <= answer < MOD)
        out_of_range = False
        if self.mod is not None:
            if not (0 <= user_answer < self.mod):
                out_of_range = True

        # Compare with reference answer
        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "MOD": self.mod,
        }
        if out_of_range:
            info["error"] = "out_of_range"

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the expected \\boxed{...} format."""
        if self.mod is None:
            # Fallback if reset has not been called
            random_answer = random.randint(0, max(3, self.max_mod) - 1)
        else:
            random_answer = random.randint(0, self.mod - 1)
        return f"\\boxed{{{random_answer}}}"