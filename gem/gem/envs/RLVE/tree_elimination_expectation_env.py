from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TreeElimination_ExpectationEnv(Env):
    """Tree Elimination Expectation Problem Environment - Single-turn Q&A"""

    def __init__(
        self,
        N: int,
        modulo: int = 10**9 + 7,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N (int): Number of vertices in the tree (must be >= 2).
        - modulo (int): Modulo for the expected value calculation (default: 1e9+7).
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 2, "N should be greater than or equal to 2"
        self.N = N
        self.modulo = modulo

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.parents: list[tuple[int, int]] = []

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a tree coloring expectation problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        MOD = self.modulo

        # Generate a random tree with N vertices using the original logic
        P = list(range(2, N + 1))
        random.shuffle(P)
        P = [1] + P

        parents: list[tuple[int, int]] = []
        for i in range(1, N):
            parent, u = P[random.randint(0, i - 1)], P[i]
            parents.append((parent, u))
        self.parents = parents

        # Compute reference answer using the original algorithm
        def mod_inverse(a: int) -> int:
            return pow(a, MOD - 2, MOD)

        def dfs(u: int, children: list[list[int]], size: list[int], fac: list[int], inv: list[int]) -> int:
            total = 0
            size[u] = 1
            for v in children[u]:
                total += dfs(v, children, size, fac, inv)
                size[u] += size[v]
            total += fac[size[u] - 1] * inv[size[u]] % MOD
            return total % MOD

        children: list[list[int]] = [[] for _ in range(N + 1)]
        for parent, u in parents:
            children[parent].append(u)

        fac = [1] * (N + 1)
        for i in range(1, N + 1):
            fac[i] = fac[i - 1] * i % MOD
        inv = [1] * (N + 1)
        inv[N] = mod_inverse(fac[N])
        for i in range(N, 0, -1):
            inv[i - 1] = inv[i] * i % MOD

        size = [0] * (N + 1)
        reference_answer = dfs(1, children, size, fac, inv)
        assert size[1] == N, "size[1] should be equal to N"

        self.reference_answer = reference_answer

        # Build problem prompt
        parents_str = "\n".join(f"parent[{u}]={parent}" for parent, u in parents)
        self.current_problem = (
            f"You are given a tree with {N} vertices labeled from 1 to {N}, where vertex 1 is the root of the tree. "
            f"Each vertex (except the root 1) has a parent, specified as follows:\n"
            f"{parents_str}\n\n"
            "Initially, all vertices are uncolored. In each step, you randomly select an uncolored vertex (with equal probability) "
            "and color all vertices on the entire path from the selected vertex to the root.\n\n"
            f"Please compute the expected number of steps required until all vertices are colored. Please give the expectation modulo {MOD}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the answer."""
        answer = self._parse_answer(action)

        if answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check: answer must be within [0, modulo)
        if not (0 <= user_answer < self.modulo):
            return TERMINAL_STATE, 0.0, True, False, {"error": "out_of_range"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "parents": self.parents,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} format."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action."""
        random_answer = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{random_answer}}}"