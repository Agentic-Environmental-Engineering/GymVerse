import random
from bisect import bisect_left
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxSumLDSEnv(Env):
    """
    Environment for the "Maximize Sum of Longest Decreasing Subsequences (LDS) lengths"
    under a given LIS-length array A.

    Task:
    - Given N and an array A defined from a hidden permutation P as:
        A[0] = 0.
        For 1 <= i <= N:
          A[i] = max(A[j]) + 1 over j in [0, i-1], with (j == 0) or (P[j] < P[i]).
      (This is essentially the standard LIS-length DP with a dummy 0-th element.)
    - You must output a permutation P of {1, 2, ..., N} that induces exactly this A.
    - Among all such permutations, maximize the value of sum_{i=1..N} B[i], where:
        B[N+1] = 0.
        For N >= i >= 1:
          B[i] = max(B[j]) + 1 over j in [i+1, N+1], with (j == N+1) or (P[j] < P[i]).
    - Output P as space-separated integers in \\boxed{...} format.

    Scoring:
    - Reward 1.0 if your permutation induces the given A and achieves the optimal
      sum of B[i] (i.e., equals the reference best).
    - Reward 0.0 otherwise.
    - If the output format is wrong (cannot extract \\boxed{...}), reward is -0.1.

    This is a single-turn environment: step() immediately terminates.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 100,
        **kwargs: Any,
    ):
        """
        Initialize the environment.

        Args:
            N: If provided, use this fixed N (must be >= 3).
            min_N: Minimum N when sampling randomly (inclusive, must be >= 3).
            max_N: Maximum N when sampling randomly (inclusive).
        """
        super().__init__()
        self.N_fixed: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Current episode state
        self.current_N: Optional[int] = None
        self.A_with_zero: Optional[List[int]] = None  # A[0..N], with A[0] = 0
        self.gold_answer: Optional[int] = None        # Optimal sum of B[i]
        self.current_problem: Optional[str] = None

        # Validation
        if self.N_fixed is not None:
            assert self.N_fixed >= 3, "N should be greater than or equal to 3"
        assert self.min_N >= 3, "min_N should be greater than or equal to 3"
        assert self.min_N <= self.max_N, "min_N must be <= max_N"

    def _get_instructions(self) -> str:
        """Return general instructions for the task."""
        return (
            "You are given N and an array A defined from an unknown permutation P of {1..N} by:\n"
            "- A[0] = 0.\n"
            "- For 1 <= i <= N: A[i] = max(A[j]) + 1 over j in [0, i-1], where j = 0 or P[j] < P[i].\n"
            "You must find a permutation P that induces exactly this A and maximizes the sum of B[i] for i=1..N, where:\n"
            "- B[N+1] = 0.\n"
            "- For N >= i >= 1: B[i] = max(B[j]) + 1 over j in [i+1, N+1], where j = N+1 or P[j] < P[i].\n\n"
            "Output Format:\n"
            "- Return P[1], P[2], ..., P[N] as space-separated integers inside \\boxed{...}.\n"
            "  Example: \\boxed{3 1 2 5 4}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            N = self.N_fixed
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"
        self.current_N = N

        # Generate a random permutation and derive A and B from it
        P = list(range(1, N + 1))
        random.shuffle(P)
        P = [None] + P  # 1-based with a dummy None at index 0

        A_list, B_list = self.get_A_B(P)
        # Store A[0..N] (exclude trailing None)
        self.A_with_zero = A_list[:-1]  # length N+1, indices 0..N

        # Compute a reference best (gold) using the same logic as original environment
        # Use A[1..N] to build adjacency and compute the LIS-based upper bound.
        A_core = A_list[1:-1]  # A[1..N]
        assert len(A_core) == N, "A should have length N when excluding A[0] and A[N+1]"

        # Build the adjacency list (nodes 0..N, with 0 as a dummy root)
        adj: List[List[int]] = [[] for _ in range(N + 1)]
        last_pos: List[int] = [0] * (N + 1)  # last_pos[k] = last index i with LIS length k seen so far

        for i, x in enumerate(A_core, start=1):
            parent = last_pos[x - 1]
            adj[parent].append(i)
            adj[i].append(parent)
            last_pos[x] = i

        # Match C++ head-insert neighbor order by reversing adjacency lists
        for nbrs in adj:
            nbrs.reverse()

        # Iterative DFS to get preorder numbers dfn[0..N]
        dfn: List[int] = [0] * (N + 1)
        cnt = 0
        stack: List[Tuple[int, int, int]] = [(0, -1, 0)]  # (node, parent, next-neighbor-index)
        while stack:
            u, p, idx = stack.pop()
            if idx == 0:
                cnt += 1
                dfn[u] = cnt
            if idx < len(adj[u]):
                v = adj[u][idx]
                stack.append((u, p, idx + 1))
                if v != p:
                    stack.append((v, u, 0))

        # Shift dfn[1..N] down by 1 (ignore dfn[0])
        for i in range(1, N + 1):
            dfn[i] -= 1

        # Build sequence B': B'[i] = dfn[N - i] for i = 0..N-1 (equivalent to b[i]=dfn[n-i+1] in 1-based)
        B_seq = [dfn[pos] for pos in range(N, 0, -1)]

        # Compute sum of LIS lengths over B_seq (strictly increasing), using patience sorting with bisect_left
        tails: List[int] = []
        ans = 0
        for v in B_seq:
            pos = bisect_left(tails, v)
            if pos == len(tails):
                tails.append(v)
            else:
                tails[pos] = v
            ans += pos + 1

        # Sanity check: sum of B from the generated permutation should be <= ans
        B_trim = B_list[1:-1]
        assert len(B_trim) == N, "B should have length N"
        sumB = sum(B_trim)
        assert 0 < sumB <= ans, "Sum of B should be less than or equal to the optimal answer"

        self.gold_answer = ans

        # Build problem statement
        A_str = ", ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(self.A_with_zero))
        problem_text = (
            f"Given a permutation of numbers from 1 to {N}, denoted as P[1], P[2], ..., P[{N}], define:\n"
            f"- A[0] = 0. For 1 <= i <= {N}, A[i] = max(A[j]) + 1 such that: (i) 0 <= j <= i - 1, and (ii) j = 0 or P[j] < P[i].\n"
            f"- B[{N} + 1] = 0. For {N} >= i >= 1, B[i] = max(B[j]) + 1 such that: (i) i + 1 <= j <= {N} + 1, and (ii) j = {N} + 1 or P[j] < P[i].\n\n"
            f"You are given the array A: {A_str}\n"
            f"Find a permutation P such that this A is obtained, and maximize the value of B[1] + B[2] + ... + B[{N}].\n"
            f"Output P[1], P[2], ..., P[{N}] in one line, separated by spaces, in \\boxed{{...}} format."
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def get_A_B(self, P: List[Optional[int]]) -> Tuple[List[Optional[int]], List[Optional[int]]]:
        """
        Compute arrays A and B for a given permutation P (1-based, P[0] is None).
        A and B are returned as lists of length N+2:
        - A[0..N], A[N+1] = None
        - B[0] = None, B[1..N], B[N+1] = 0 used in definition (we keep structure consistent with original)
        """
        assert self.current_N is not None, "Environment not initialized. Call reset() first."
        N = self.current_N
        assert len(P) == N + 1
        assert P[0] is None, "P[0] should be None"

        A: List[Optional[int]] = [0] * (N + 2)
        for i in range(1, N + 1):
            A[i] = max(A[j] for j in range(i) if j == 0 or P[j] < P[i]) + 1  # type: ignore
        A[N + 1] = None

        B: List[Optional[int]] = [0] * (N + 2)
        for i in range(N, 0, -1):
            B[i] = max(B[j] for j in range(i + 1, N + 2) if j == N + 1 or P[j] < P[i]) + 1  # type: ignore
        B[0] = None

        return A, B

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Validate the proposed permutation. Single-turn environment.
        Returns TERMINAL_STATE and terminated=True after one step.
        """
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Must have a current problem
        if self.current_N is None or self.A_with_zero is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        N = self.current_N
        # Parse permutation entries
        try:
            # Allow spaces; tolerate accidental commas by replacing them with spaces
            tokens = boxed.replace(",", " ").split()
            perm = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate permutation structure
        perm_valid = (len(perm) == N) and (set(perm) == set(range(1, N + 1)))
        if not perm_valid:
            info = {
                "error": "invalid_permutation",
                "expected_length": N,
                "received_length": len(perm),
                "unique_elements": len(set(perm)),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute A and B from the submitted permutation
        P_user = [None] + perm
        A_user, B_user = self.get_A_B(P_user)
        A_user_with_zero = A_user[:-1]  # A[0..N]

        valid_A = (A_user_with_zero == self.A_with_zero)
        if not valid_A:
            info = {
                "correct": False,
                "valid_A": False,
                "sum_B": None,
                "gold_answer": self.gold_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # If A matches, check optimality by comparing sum of B[i] to gold
        B_trim = B_user[1:-1]
        user_sum_B = sum(int(b) for b in B_trim if b is not None)
        is_optimal = (user_sum_B == self.gold_answer)

        reward: float = 1.0 if is_optimal else 0.0
        info = {
            "correct": is_optimal,
            "valid_A": True,
            "sum_B": user_sum_B,
            "gold_answer": self.gold_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random permutation in boxed format."""
        if self.current_N is None:
            # Fallback to a small default if not initialized
            n = self.N_fixed if self.N_fixed is not None else max(self.min_N, 3)
        else:
            n = self.current_N
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        return f"\\boxed{{{' '.join(map(str, perm))}}}"