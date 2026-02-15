import heapq
import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Max_TreeConstrainedPermutation_WeightEnv(Env):
    """Environment for maximizing weighted sum under tree-like precedence constraints.

    The task:
    - You are given an array W of length N.
    - Find a permutation P of 1..N that respects given precedence constraints:
      For each i in [1..N], if A[i] != 0 then element A[i] must come before element i.
    - Maximize sum_{i=1..N} W[P[i]] * i.

    Answer format:
    - Return the permutation as space-separated integers wrapped in \\boxed{...}.
      Example: \\boxed{3 1 2}
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 50,
        **kwargs
    ):
        super().__init__()
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        self.min_n = min_n
        self.max_n = max_n

        # Problem state
        self.N: Optional[int] = None
        self.W: Optional[List[int]] = None  # 1-based conceptual, stored 0-based
        self.A: Optional[List[int]] = None  # A[i] in [0..i-1] as parent, stored 0-based for i in [0..N-1]
        self.reference_best: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task description and required answer format."""
        return (
            "Task: You are given a weight array W and precedence constraints.\n"
            "Find a permutation P of 1..N that respects all constraints and maximizes the sum of W[P[i]] × i.\n"
            "Answer format: Provide your permutation as space-separated integers inside \\boxed{...}.\n"
            "Example: \\boxed{3 1 2}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem instance."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(self.min_n, self.max_n)
        W = [random.randint(1, N) for _ in range(N)]
        # A[i] is the parent of element (i+1), chosen from [0..i]
        # This guarantees no cycles and ensures a tree-like partial order towards 0.
        A = [random.randint(0, i) for i in range(N)]

        # Compute the optimal value using the original algorithm
        reference_best = self._compute_optimal_value(N, W, A)

        # Build the problem prompt
        conditions_lines = "\n".join(
            (
                f"- The element {i + 1} has no constraint."
                if Ai == 0
                else f"- The element {Ai} must come before element {i + 1}."
            )
            for i, Ai in enumerate(A)
        )
        problem_prompt = (
            f"You are given an array W of length {N}: "
            + " ".join(f"W[{i + 1}]={Wi}" for i, Wi in enumerate(W))
            + "\n\n"
            "Please find a permutation P of 1 to {N} such that the following conditions are satisfied:\n"
            f"{conditions_lines}\n\n"
            "Try your best to maximize the sum of W[P[i]] × i for all i from 1 to {N}.\n\n"
            "Output Format: Your final answer should be the permutation P[1], ..., P[N], "
            "separated by spaces and wrapped in \\boxed{...}."
        ).replace("{N}", str(N))

        # Store state
        self.N = N
        self.W = W
        self.A = A
        self.reference_best = reference_best
        self.current_problem = problem_prompt

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": N,
            "W": W,
            "A": A,
            "reference_best": reference_best,
        }

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted permutation and assign rewards."""
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.N is None or self.W is None or self.A is None or self.reference_best is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Parse permutation from boxed content
        try:
            tokens = boxed.strip().split()
            P = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer_format"}

        # Validate permutation
        N = self.N
        if len(P) != N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_length", "expected_length": N}

        if set(P) != set(range(1, N + 1)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_a_permutation"}

        # Check precedence constraints: A[i] must come before (i+1) if A[i] != 0
        positions = [None] * (N + 1)
        for i, Pi in enumerate(P):
            positions[Pi] = i
        for i, Ai in enumerate(self.A):  # i in [0..N-1], element index is (i+1)
            if Ai != 0:
                # Ai must come before element (i+1)
                if positions[Ai] >= positions[i + 1]:
                    return TERMINAL_STATE, 0.0, True, False, {"error": "constraint_violation", "violated": (Ai, i + 1)}

        # Compute achieved value
        achieved_value = sum(self.W[Pi - 1] * (i + 1) for i, Pi in enumerate(P))
        is_optimal = (achieved_value == self.reference_best)
        reward = 1.0 if is_optimal else 0.0

        info = {
            "correct": is_optimal,
            "reference_best": self.reference_best,
            "achieved_value": achieved_value,
            "N": self.N,
            "W": self.W,
            "A": self.A,
            "user_permutation": P,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid permutation that respects the constraints."""
        if self.N is None or self.A is None:
            # If not initialized, sample a random short permutation
            n = 5
            perm = list(range(1, n + 1))
            random.shuffle(perm)
            return f"\\boxed{{{' '.join(map(str, perm))}}}"

        N = self.N
        A = self.A

        # Build indegree and adjacency from constraints: edge A[i] -> (i+1) if A[i] != 0
        indeg = [0] * (N + 1)  # 1-based indices
        adj: List[List[int]] = [[] for _ in range(N + 1)]
        for i, Ai in enumerate(A):
            u = Ai
            v = i + 1
            if u != 0:
                adj[u].append(v)
                indeg[v] += 1

        # Kahn's algorithm for random topological order
        available = [i for i in range(1, N + 1) if indeg[i] == 0]
        result = []
        while available:
            random.shuffle(available)
            u = available.pop()
            result.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    available.append(v)

        if len(result) != N:
            # Fallback: return a random permutation
            result = list(range(1, N + 1))
            random.shuffle(result)

        return f"\\boxed{{{' '.join(map(str, result))}}}"

    def _compute_optimal_value(self, N: int, W: List[int], A: List[int]) -> int:
        """Compute the optimal value using the original heap/DSU merging strategy."""
        # Convert A to 1-based parent array where parent[i] in [0..i-1]
        parent = [0] + A[:]  # parent[1] = A[0], ..., parent[N] = A[N-1]
        weights_input = [0] + W[:]  # 1-based for convenience

        # Build children lists
        children: List[List[int]] = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            children[parent[i]].append(i)

        # DFS from 0 to ensure reachability (should hold for generated A)
        visited = [False] * (N + 1)
        stack = [0]
        visited[0] = True
        cnt = 1
        while stack:
            u = stack.pop()
            for v in children[u]:
                if visited[v]:
                    # Should not happen for our generation method
                    raise ValueError("Cycle detected in constraints")
                visited[v] = True
                cnt += 1
                stack.append(v)
        if cnt <= N:
            # Some node is not reachable from root 0
            raise ValueError("Unreachable node detected in constraints")

        # DSU structures
        dsu = list(range(N + 1))
        size = [1] * (N + 1)
        weight = [0] * (N + 1)
        for i in range(1, N + 1):
            weight[i] = weights_input[i]

        def find(u: int) -> int:
            while dsu[u] != u:
                dsu[u] = dsu[dsu[u]]
                u = dsu[u]
            return u

        class NodeData:
            __slots__ = ("u", "sz", "w")

            def __init__(self, u: int, sz: int, w: int):
                self.u = u
                self.sz = sz
                self.w = w

            def __lt__(self, other: "NodeData") -> bool:
                # Compare by average weight: pop the smallest average first
                return self.w * other.sz < other.w * self.sz

        heap: List[NodeData] = []
        for i in range(1, N + 1):
            heapq.heappush(heap, NodeData(i, 1, weight[i]))

        ans = 0
        while heap:
            s = heapq.heappop(heap)
            u = find(s.u)
            if size[u] != s.sz:
                continue
            p = find(parent[u])
            ans += weight[u] * size[p]
            weight[p] += weight[u]
            size[p] += size[u]
            dsu[u] = p
            if p != 0:
                heapq.heappush(heap, NodeData(p, size[p], weight[p]))

        if ans <= 0:
            raise ValueError("Computed optimal value is non-positive, which should not happen")
        return ans