from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from collections import deque
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinConversionToCycleCostEnv(Env):
    """
    Environment for the problem: Given a directed functional graph represented by an array A (edge i -> A[i]),
    you may change A[i] to any vertex j at cost C[i]. The goal is to obtain a single directed cycle of length N
    (i.e., a permutation consisting of exactly one cycle covering all vertices) with minimal total modification cost.

    Single-turn Q&A environment:
    - reset() generates a new instance with arrays A and C and computes the minimal achievable cost.
    - step(action) expects the final A array (N integers separated by spaces) inside \\boxed{...},
      verifies it forms a single cycle and checks if its cost equals the minimal cost.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 50,
        cost_min: int = 1,
        cost_max: int = 50,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: If provided, fixes the number of vertices. Must be >= 3.
            min_n: Minimum N if N is not provided (inclusive). Must be >= 3.
            max_n: Maximum N if N is not provided (inclusive). Must be >= min_n.
            cost_min: Minimum cost value for C[i] (inclusive).
            cost_max: Maximum cost value for C[i] (inclusive).
            **kwargs: Extra unused arguments for compatibility.
        """
        super().__init__()
        if N is not None:
            if N < 3:
                raise ValueError("N should be greater than or equal to 3")
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        if cost_min < 0 or cost_max < cost_min:
            raise ValueError("Invalid cost range")

        self.fixed_N = N
        self.min_n = min_n
        self.max_n = max_n
        self.cost_min = cost_min
        self.cost_max = cost_max

        # Problem instance state
        self.N: Optional[int] = None
        self.A_initial: Optional[List[int]] = None
        self.C: Optional[List[int]] = None
        self.minimal_cost: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a directed functional graph represented by an array A of length N, where each node i has\n"
            "a single outgoing edge to A[i] (i -> A[i]), with 0-based indexing.\n"
            "You may modify A[i] to any integer j in [0, N-1] at cost C[i]. Your goal is to produce a new array A\n"
            "so that the graph forms a single directed cycle of length N (i.e., a single cycle that visits all vertices exactly once),\n"
            "while minimizing the total modification cost sum of C[i] over indices where A[i] changed.\n\n"
            "Output Format: Provide the final array A[0], A[1], ..., A[N-1] as N integers separated by single spaces inside \\boxed{...}.\n"
            "Example: \\boxed{1 2 0} for N=3.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Generate initial A with no self-loops (i -> A[i], A[i] != i)
        A = []
        for i in range(N):
            while True:
                a = random.randint(0, N - 1)
                if a != i:
                    A.append(a)
                    break
        assert len(A) == N

        # Generate cost array C
        C = [random.randint(self.cost_min, self.cost_max) for _ in range(N)]

        # Compute minimal achievable cost using the original algorithm
        minimal_cost = self._compute_minimal_cost(A, C)

        # Build problem prompt
        A_str = " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A))
        C_str = " ".join(f"C[{i}]={Ci}" for i, Ci in enumerate(C))
        prompt = (
            f"You are given a directed graph with {N} vertices labeled from 0 to {N-1}.\n"
            f"The graph is represented by an array A where each node i has a single outgoing edge to A[i].\n"
            f"The initial array A is: {A_str}\n"
            f"You are allowed to modify A[i] to any other vertex j (0 ≤ j < {N}) at a cost of C[i].\n"
            f"The cost array is: {C_str}\n\n"
            f"Your goal is to make the entire graph a single directed cycle of length {N} while minimizing the total modification cost.\n"
            f"Output Format: Provide the final A[0], A[1], ..., A[{N-1}] as N integers separated by spaces inside \\boxed{{...}}."
        )

        self.N = N
        self.A_initial = A
        self.C = C
        self.minimal_cost = minimal_cost
        self.current_problem = prompt

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted solution."""
        # Extract boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse array of integers
        try:
            parts = boxed.strip().split()
            proposal = list(map(int, parts))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate dimensions and range
        N = self.N
        A0 = self.A_initial
        C = self.C
        min_cost = self.minimal_cost

        assert N is not None and A0 is not None and C is not None and min_cost is not None

        if len(proposal) != N:
            info = {"error": "invalid_length", "expected_length": N, "got": len(proposal)}
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(0 <= x < N for x in proposal):
            info = {"error": "out_of_range", "N": N}
            return TERMINAL_STATE, 0.0, True, False, info

        # Check if proposal forms a single directed cycle of length N
        visited = [False] * N
        x = 0
        steps = 0
        while True:
            if visited[x]:
                break
            visited[x] = True
            x = proposal[x]
            steps += 1
            if steps > N + 1:
                # Safety guard; should not happen if values in range
                break

        is_single_cycle = (x == 0 and all(visited))

        # Compute user's total modification cost
        user_cost = sum(ci if ai != bi else 0 for ai, bi, ci in zip(A0, proposal, C))

        # Determine correctness: must be single cycle and minimal cost
        is_correct = is_single_cycle and (user_cost == min_cost)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "is_single_cycle": is_single_cycle,
            "reference_minimal_cost": min_cost,
            "user_cost": user_cost,
            "N": N,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_minimal_cost(self, A: List[int], C: List[int]) -> int:
        """
        Compute the minimal modification cost to transform the functional graph defined by A (i -> A[i])
        into a single directed cycle of length N, with modification cost C[i] if A[i] is changed.

        This function implements the logic from the original environment.
        """
        N = len(A)

        # Compute indegree for each node in the functional graph
        h = [0] * N
        for v in A:
            h[v] += 1

        # Queue of nodes with indegree 0 (leaves in the reverse graph)
        q = deque(i for i in range(N) if h[i] == 0)

        # f[v] tracks the best incoming cost seen so far for v
        f = [0] * N
        ans = 0

        # Special case: if there are no leaves, the graph is pure cycles
        # Check if it's exactly one big cycle already
        vis = [False] * N
        if not q:
            count = 0
            j = 0
            while not vis[j]:
                vis[j] = True
                count += 1
                j = A[j]
            if count == N:
                return ans  # already one cycle, cost 0

        # Peel off trees attached to cycles
        while q:
            x = q.popleft()
            y = A[x]
            if f[y]:
                # We already have one candidate edge into y; choose the cheaper
                ans += min(f[y], C[x])
                # Keep the more expensive as the "best so far" for future comparisons
                f[y] = max(f[y], C[x])
            else:
                # First edge into y
                f[y] = C[x]
            h[y] -= 1
            if h[y] == 0:
                q.append(y)

        # Now only the cycles remain (h[i] > 0 for nodes in cycles)
        for i in range(N):
            if h[i] > 0:
                diffs = []
                j = i
                # Walk the cycle, breaking h[] as we go
                while h[A[j]] > 0:
                    v = A[j]
                    h[v] = 0
                    ans += f[v]  # pay the best incoming from the attached tree (or 0)
                    diffs.append(f[v] - C[j])
                    j = v
                # To make this component a single cycle, we must drop one edge (the max diff)
                diffs.sort()
                ans -= diffs[-1]
                # And for any other positive diffs, we can save more by replacing edges
                for d in diffs[:-1]:
                    if d > 0:
                        ans -= d
        return ans

    def sample_random_action(self) -> str:
        """
        Sample a random action: produce a random single cycle on N nodes in \\boxed{...} format.
        If no current problem, returns an empty boxed answer.
        """
        if self.N is None:
            return "\\boxed{}"
        N = self.N
        # Generate a random permutation forming a single cycle
        perm = list(range(N))
        random.shuffle(perm)
        # Convert permutation to a single cycle mapping: i -> perm[(idx_of_i + 1) % N]
        position = [0] * N
        for idx, val in enumerate(perm):
            position[val] = idx
        mapping = [0] * N
        for i in range(N):
            idx = position[i]
            nxt = perm[(idx + 1) % N]
            mapping[i] = nxt
        ans_str = " ".join(str(x) for x in mapping)
        return f"\\boxed{{{ans_str}}}"