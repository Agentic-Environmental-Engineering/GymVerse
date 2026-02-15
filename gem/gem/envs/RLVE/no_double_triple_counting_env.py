from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NoDoubleTripleCountingEnv(Env):
    """Environment for counting subsets of {1, 2, ..., N} with the constraint:
    If x is in the subset, then neither 2*x nor 3*x is in the subset.
    Single-turn Q&A environment.
    """

    def __init__(
        self,
        max_n: int = 1000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: Upper bound for N (inclusive). Must be >= 3.
        """
        super().__init__()
        assert max_n >= 3, "max_n should be greater than or equal to 3"
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics counting problem.\n"
            "Task: Count the number of subsets of {1, 2, ..., N} such that if x is in the subset, "
            "then neither 2*x nor 3*x is in the subset.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameter N
        N = random.randint(3, self.max_n)
        self.N = N

        # Build problem prompt
        self.current_problem = (
            f"How many subsets of 1, 2, ..., {N} satisfy that if x is in the subset, "
            f"then neither 2 × x nor 3 × x is in the subset?\n\n"
            f"Output Format: Provide a single non-negative integer in \\boxed{{...}}."
        )

        # Compute the reference answer using the component DP approach
        self.reference_answer = self._compute_answer(N)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_answer(self, N: int) -> int:
        """Compute the number of valid subsets for the given N."""
        S = list(range(1, N + 1))
        assert len(S) == N, "S should contain exactly N elements"

        # visited[i] indicates whether value (i+1) has already been included in some component
        visited = [False] * N

        def dp(root: int) -> int:
            # Build the 2-chain: root, 2*root, 4*root, ... <= N
            pow2_chain = []
            v = root
            while v <= N:
                pow2_chain.append(v)
                v *= 2
            L = len(pow2_chain)

            # For each in the 2-chain, build its 3-chain: v, 3*v, 9*v, ... <= N
            pow3_chains = []
            for v2 in pow2_chain:
                chain = []
                u = v2
                while u <= N:
                    chain.append(u)
                    u *= 3
                pow3_chains.append(chain)

            # Mark all nodes in this component as visited
            for chain in pow3_chains:
                for u in chain:
                    visited[u - 1] = True

            # lmt0[i] = maximum mask value at level i (0..i=L)
            # Level 0 has only mask 0
            lmt0 = [0] + [(1 << len(chain)) - 1 for chain in pow3_chains]

            # f[i][mask] = number of ways up to level i with configuration 'mask' at level i
            f = [[0] * (l + 1) for l in lmt0]
            f[0][0] = 1

            # Transition from level i to i+1
            for i in range(L):
                for mask_j, ways in enumerate(f[i]):
                    if not ways:
                        continue
                    # Try every subset mask_k on next 3-chain
                    for mask_k in range(lmt0[i + 1] + 1):
                        # No conflict with previous level, and no adjacent picks in this level
                        if (mask_j & mask_k) == 0 and (mask_k & (mask_k << 1)) == 0:
                            f[i + 1][mask_k] += ways

            # Sum over all mask states at the last real level
            return sum(f[L])

        ans = 1
        for x in S:
            if not visited[x - 1]:
                ans *= dp(x)

        return ans

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer."""
        # Parse the boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare
        try:
            user_answer = int(answer_str)
            is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
            reward: float = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}."""
        random_answer = random.randint(0, 10**6)
        return f"\\boxed{{{random_answer}}}"