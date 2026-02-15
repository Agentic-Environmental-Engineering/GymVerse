from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DerangementExtensionEnv(Env):
    """
    Single-turn environment for counting permutations with exactly M fixed points.

    Task:
      Given N and M, compute the number of permutations p of {1, 2, ..., N}
      such that exactly M indices i satisfy p[i] = i. Return the result modulo MOD.

    Answer format:
      The agent must output the final answer wrapped in \\boxed{...}.
    """

    MODS = (666623333, 998244353, 10**9 + 7)

    def __init__(
        self,
        fixed_n: Optional[int] = None,
        min_n: int = 4,
        max_n: int = 200000,
        mods: Optional[Tuple[int, ...]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the environment.

        Args:
            fixed_n: If provided, use this fixed N (must be >= 4).
            min_n: Minimum N when sampling randomly. Must be >= 4.
            max_n: Maximum N when sampling randomly. Must be >= min_n.
            mods: Optional tuple of moduli to choose from. Defaults to MODS.
            **kwargs: Extra arguments (ignored).
        """
        super().__init__()
        self.fixed_n = fixed_n
        self.min_n = max(4, min_n)
        self.max_n = max(self.min_n, max_n)
        self.mods = mods if mods is not None else self.MODS

        # State variables for the current episode
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_N: Optional[int] = None
        self.current_M: Optional[int] = None
        self.current_MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task description and output format requirements."""
        return (
            "You are solving a combinatorics problem about permutations with fixed points.\n"
            "Task: For given N and M, compute the number of permutations p of {1, 2, ..., N}\n"
            "such that exactly M indices i satisfy p[i] = i (1-indexed), and report the result modulo MOD.\n"
            "Output Format: Provide your final answer as a single integer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: The problem statement string.
            info: Auxiliary information dictionary (empty for this env).
        """
        super().reset(seed)

        # Determine N
        if self.fixed_n is not None:
            if self.fixed_n < 4:
                raise ValueError("fixed_n must be >= 4")
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)

        # Generate M and MOD
        M = random.randint(0, N)
        MOD = random.choice(self.mods)

        # Precompute factorial products, inverse factorials (via Fermat), and derangements
        prod, inv, derangements = self._init_precompute(N, MOD)

        # Compute reference answer
        reference_answer = self._compute_answer(N, M, MOD, prod, inv, derangements)

        # Build the problem description
        problem = (
            f"What's the number of permutations p of 1, 2, ..., {N} such that exactly {M} indices i "
            f"satisfy p[i] = i (1-indexed)? Let me know the result modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Store current state
        self.current_problem = problem
        self.reference_answer = reference_answer
        self.current_N = N
        self.current_M = M
        self.current_MOD = MOD

        obs = self._get_instructions() + problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Validate the submitted answer.

        Args:
            action: The agent's response text containing \\boxed{...} with an integer.

        Returns:
            observation: TERMINAL_STATE
            reward: 1.0 if correct, 0.0 if wrong or invalid, -0.1 if format error (\boxed missing).
            terminated: Always True for single-turn environment.
            truncated: Always False.
            info: Details including correctness and reference answer.
        """
        # Parse \boxed{...}
        extracted = self._parse_answer(action)
        if extracted is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate integer
        try:
            user_answer = int(extracted)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check with respect to MOD (treated as wrong if out of range)
        mod = self.current_MOD if self.current_MOD is not None else None
        if mod is None or self.reference_answer is None:
            # Safety fallback; environment should always have these set.
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_state_error"}

        if not (0 <= user_answer < mod):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "N": self.current_N,
                "M": self.current_M,
                "MOD": self.current_MOD,
                "error": "out_of_range",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_N,
            "M": self.current_M,
            "MOD": self.current_MOD,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the agent's response."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _init_precompute(self, max_n: int, mod: int) -> Tuple[list[int], list[int], list[int]]:
        """
        Precompute factorial products (prod), inverse factorials (inv) via Fermat,
        and derangements numbers up to max_n under modulo mod.

        Note: inv[i] stores the modular inverse of prod[i] = i! (consistent with the original logic).
        """
        prod = [1] * (max_n + 1)
        inv = [0] * (max_n + 1)
        for i in range(1, max_n + 1):
            prod[i] = (prod[i - 1] * i) % mod
            inv[i] = pow(prod[i], mod - 2, mod)  # modular inverse of i! via Fermat

        # Derangements array D(n), with D(0)=1, D(1)=0, D(2)=1, and recurrence:
        # D(n) = (n - 1) * (D(n - 1) + D(n - 2))
        a = [0] * (max_n + 1)
        if max_n >= 0:
            a[0] = 1
        if max_n >= 1:
            a[1] = 0
        if max_n >= 2:
            a[2] = 1
        for i in range(3, max_n + 1):
            a[i] = (i - 1) * ((a[i - 1] + a[i - 2]) % mod) % mod

        return prod, inv, a

    def _compute_answer(
        self,
        N: int,
        M: int,
        MOD: int,
        prod: list[int],
        inv: list[int],
        derangements: list[int],
    ) -> int:
        """
        Compute the answer using:
          - Special cases:
              M == 0 => D(N)
              N == M => 1
              N - 1 == M => 0
          - Otherwise:
              C(N, M) * D(N - M) mod MOD
        """
        if M == 0:
            return derangements[N] % MOD
        if N == M:
            return 1
        if N - 1 == M:
            return 0

        # C(N, M) = N! / (M! * (N-M)!)
        comb = (prod[N] * inv[M] % MOD) * inv[N - M] % MOD
        ans = (comb * derangements[N - M]) % MOD
        return ans

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...} within the current MOD range if available."""
        if self.current_MOD is not None:
            random_answer = random.randint(0, self.current_MOD - 1)
        else:
            # Fallback if called before reset
            random_answer = random.randint(0, 100)
        return f"\\boxed{{{random_answer}}}"