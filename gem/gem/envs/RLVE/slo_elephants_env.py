import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SLOElephantsEnv(Env):
    """SLO Elephants environment - single-turn Q&A.

    Task:
    - There are N items (elephants) labeled from 0 to N-1.
    - Each item i has a cost C[i].
    - Initial arrangement is A (positions contain item labels).
    - Target arrangement is B.
    - You may swap any two items labeled i and j at cost C[i] + C[j].
    - Minimize the total cost to transform A into B.
    - Output the minimal total cost as a single integer in \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 12,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            N: If provided, fixes the problem size. Must be >= 3.
            min_n: Minimum N (inclusive) used when N is not provided. Must be >= 3.
            max_n: Maximum N (inclusive) used when N is not provided.
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")

        self.fixed_N: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        # State for the current problem
        self.N: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.B: Optional[List[int]] = None
        self.C: Optional[List[int]] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a minimum-cost swap problem on labeled items.\n"
            "Return only the minimal total swap cost as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Choose N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        self.N = N

        # Generate A, B (random permutations, ensure A != B)
        A = list(range(N))
        B = list(range(N))
        while True:
            random.shuffle(A)
            random.shuffle(B)
            if A != B:
                break

        # Generate costs C[i] in [1, N]
        C = [random.randint(1, N) for _ in range(N)]

        # Compute the minimal cost using cycle decomposition
        gold_answer = self._compute_min_cost(A, B, C)

        if gold_answer <= 0:
            # Extremely unlikely given A != B and positive costs, but keep consistent with original assert.
            raise RuntimeError("The minimal cost answer should be greater than 0")

        # Build and store the problem text
        c_str = " ".join(f"C[{i}]={Ci}" for i, Ci in enumerate(C))
        a_str = " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A))
        b_str = " ".join(f"B[{i}]={Bi}" for i, Bi in enumerate(B))

        problem_text = (
            f"There are {N} items labeled from 0 to {N - 1}. Each item labeled `i` has an associated cost C[i]. "
            f"The array C is: {c_str}\n"
            f"Initially, the items are arranged in the order A (this means the item at position 0 has label A[0], at position 1 has label A[1], etc): {a_str}\n"
            f"You are required to rearrange the items into the target order B: {b_str}\n\n"
            f"You may perform any number of swaps. Swapping the items labeled `i` and `j` incurs a cost of C[i] + C[j]. "
            f"Please minimize the total cost of all swaps.\n\n"
            f"Output Format: Return only the minimal total cost as a single integer in \\boxed{{...}}."
        )

        self.A = A
        self.B = B
        self.C = C
        self.current_problem = problem_text
        self.reference_answer = gold_answer

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the user's answer."""
        # Parse the boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric and compare with reference
        try:
            user_answer = int(boxed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

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

    @staticmethod
    def _compute_min_cost(A: List[int], B: List[int], C: List[int]) -> int:
        """Compute the minimal total cost to transform permutation A into B using given item costs C.

        Uses the classic cycle decomposition method with two strategies:
        - Rearranging within cycle
        - Using the globally minimal cost element to assist
        """
        N = len(A)

        # Build destination position for each label in B
        dest_pos = [0] * N
        for idx, e in enumerate(B):
            dest_pos[e] = idx

        # next_id[e] = the elephant currently occupying e's final place
        next_id = [A[dest_pos[e]] for e in range(N)]

        visited = [False] * N
        overall_min = min(C)
        total_cost = 0

        for e in range(N):
            if visited[e]:
                continue

            cycle_sum = 0
            cycle_min = 10**9
            length = 0
            x = e
            while not visited[x]:
                visited[x] = True
                m = C[x]
                cycle_sum += m
                if m < cycle_min:
                    cycle_min = m
                length += 1
                x = next_id[x]

            # Trivial cycle (length 0 or 1) requires no swaps
            if length <= 1:
                continue

            # Strategy 1: Rearrange within the cycle directly
            cost_within = cycle_sum + cycle_min * (length - 2)

            # Strategy 2: Use the globally minimal element to assist
            cost_global = cycle_sum + cycle_min + overall_min * (length + 1)

            total_cost += min(cost_within, cost_global)

        return total_cost

    def sample_random_action(self) -> str:
        """Sample a random action in boxed format."""
        # Provide a random guess around a plausible range.
        # Since costs are in [1, N], a naive upper bound can be N * (N-1).
        N = self.N if self.N is not None else max(self.min_n, 3)
        random_answer = random.randint(0, max(1, N * (N - 1)))
        return f"\\boxed{{{random_answer}}}"