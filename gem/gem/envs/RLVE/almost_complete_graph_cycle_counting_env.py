from typing import Any, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class AlmostCompleteGraphCycleCountingEnv(Env):
    """Single-turn environment: Count simple cycles in an almost complete graph.

    Graph description:
    - N vertices labeled from 1 to N.
    - The graph is complete except the edge (1, N) is missing.
    - Count the number of simple cycles modulo MOD.
    - A simple cycle:
        * Has at least 3 vertices,
        * Contains no repeated vertices or edges,
        * Two cycles are the same if they have the same set of edges (order or starting point does not matter).

    Answer format:
    - The agent must output the answer in \\boxed{...} format.
    """

    def __init__(
        self,
        max_N: int = 1000,
        max_MOD: int = 1000000,
        **kwargs
    ):
        super().__init__()
        if max_N < 4:
            raise ValueError("max_N should be greater than or equal to 4")
        if max_MOD < 3:
            raise ValueError("max_MOD should be greater than or equal to 3")
        self.max_N = max_N
        self.max_MOD = max_MOD

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem about counting simple cycles in an almost complete graph.\n"
            "You must output your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters
        N = random.randint(4, self.max_N)
        # Ensure MOD is a random odd number
        r = random.randint(1, self.max_MOD // 2)
        MOD = 2 * r + 1

        self.N = N
        self.MOD = MOD

        # Build problem statement
        self.current_problem = (
            f"Consider a graph with {N} vertices labeled from 1 to {N}. "
            f"Every pair of vertices is connected by an undirected edge, except for the edge between vertices 1 and {N} "
            f"(so the graph has {N} × ({N} - 1) / 2 - 1 edges).\n\n"
            "What's the number of simple cycles in this graph? A simple cycle must:\n"
            "- Have at least 3 vertices,\n"
            "- Contain no repeated vertices or edges,\n"
            "- Be considered the same as any cycle with the same set of edges (regardless of order or starting point); "
            "for example, (1, 2, 3, 4) and (2, 1, 4, 3) are the same, but (1, 2, 3, 4) and (2, 1, 3, 4) are different.\n\n"
            f"Output the answer modulo {MOD}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(N, MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": N,
            "MOD": MOD
        }

    def _compute_reference_answer(self, N: int, MOD: int) -> int:
        """Compute the number of simple cycles modulo MOD based on the original algorithm."""
        if N <= 3:
            return 0 % MOD

        INV2 = (MOD + 1) // 2

        def calc(x: int, y: int, s: int, target_N: int) -> int:
            # x: current count of cycles for K_s
            # y: current count of paths of length 1 (one edge) in K_s
            # s: starting i value (we've precomputed up to K_s)
            # target_N: target N
            for i in range(s, target_N):
                half = ((i - 1) % MOD) * ((i - 2) % MOD) % MOD * INV2 % MOD
                x = (x + y * half) % MOD
                y = (y * ((i - 2) % MOD) + 1) % MOD
            half_n = ((target_N - 2) % MOD) * ((target_N - 3) % MOD) % MOD * INV2 % MOD
            return (x + y * half_n) % MOD

        return calc(1, 2, 4, N)

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the agent's answer and terminate."""
        # Parse boxed answer
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse integer
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None and self.MOD is not None and self.N is not None

        # Optional range check: answer should be in [0, MOD)
        if not (0 <= user_answer < self.MOD):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "N": self.N,
                "MOD": self.MOD,
                "error": "range_error"
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "MOD": self.MOD
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action consistent with the required format."""
        mod = self.MOD if self.MOD is not None else max(3, self.max_MOD | 1)
        random_answer = random.randint(0, mod - 1)
        return f"\\boxed{{{random_answer}}}"