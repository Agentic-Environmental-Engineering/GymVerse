import heapq
import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FireworkShowEnv(Env):
    """Firework Show tree equalization problem environment - single-turn Q&A.

    You are given a rooted tree with N vertices labeled from 1 to N (root is 1).
    Each non-root vertex i has a parent p and an edge length w between i and p.
    You can reduce any edge length w to any integer w' such that 0 ≤ w', paying cost |w - w'|.
    Your goal is to make all leaf-to-root path sums equal while minimizing the total cost.
    The answer is the minimum total cost.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 200,
    ):
        """
        Initialize the FireworkShowEnv.

        Args:
            N: If provided, the exact number of vertices for the tree (must be >= 2).
               If None, N will be sampled uniformly from [min_N, max_N].
            min_N: Minimum N to sample when N is None (inclusive, must be >= 2).
            max_N: Maximum N to sample when N is None (inclusive, must be >= min_N).
        """
        super().__init__()
        assert min_N >= 2, "min_N should be greater than or equal to 2"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"

        if N is not None:
            assert N >= 2, "N should be greater than or equal to 2"

        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N

        # Problem state
        self.N: Optional[int] = None
        self.parents: List[Tuple[int, int]] = []
        self.leaves: List[int] = []
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a tree edge reduction problem to equalize all leaf-to-root path sums.\n"
            "Provide the minimum total cost required to achieve this.\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 2, "N should be greater than or equal to 2"

        self.N = N

        # Generate random tree: for each node i >= 2, choose a parent p in [1, i-1] and a weight w in [1, N]
        parents: List[Tuple[int, int]] = []
        is_leaf = [None] + [True] * N  # index 0 unused; default all as leaf
        for i in range(2, N + 1):
            p = random.randint(1, i - 1)
            w = random.randint(1, N)
            parents.append((p, w))
            is_leaf[p] = False
        leaves = [i for i in range(2, N + 1) if is_leaf[i]]

        self.parents = parents
        self.leaves = leaves

        # Build problem prompt
        parents_desc = "\n".join(
            f"Vertex {i}: ({p}, {w})" for i, (p, w) in enumerate(parents, start=2)
        )
        leaves_desc = ", ".join(map(str, leaves))
        problem_text = (
            f"You are given a tree with {N} vertices labeled from 1 to {N}, where vertex 1 is the root. "
            f"Each vertex (except the root) has a parent p, and the edge connecting the vertex to its parent has length w. "
            f"The list of (parent, weight) pairs for each non-root vertex is given as:\n"
            f"{parents_desc}\n\n"
            f"Note that these vertices are leaf nodes (i.e., vertices with no children): {leaves_desc}\n"
            f"You can reduce the length of any edge. Specifically, you can change an edge's length w to any integer w' such that 0 ≤ w'; "
            f"the cost of changing an edge from w to w' is abs(w - w'). You need to make the sum of the edge lengths on the path from each "
            f"leaf node to the root equal — in other words, all leaf-to-root paths should have the same total length. "
            f"Output the minimum total cost required to achieve this.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        # Compute reference answer using the same algorithm as the original environment
        self.reference_answer = self._compute_reference_answer(N, parents, is_leaf)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(
        self,
        N: int,
        parents: List[Tuple[int, int]],
        is_leaf: List[Optional[bool]],
    ) -> int:
        """Compute the minimal cost reference answer using the original heap-based algorithm."""
        # Build adjacency and weights
        children: List[List[int]] = [[] for _ in range(N + 1)]
        w = [0] * (N + 1)
        res = 0

        for i in range(2, N + 1):
            p, c = parents[i - 2]
            children[p].append(i)
            w[i] = c
            res += c

        def dfs(x: int, depth: int = 0) -> List[int]:
            # Prevent infinite recursion with depth limit
            if depth > N:  # Maximum possible depth in a tree is N
                return []

            assert 1 <= x <= N, "Node index out of bounds"
            # Store values as negatives to use heapq as a max-heap
            heap: List[int] = []
            for y in children[x]:
                child_heap = dfs(y, depth + 1)
                # Small-to-large merge strategy
                if len(heap) < len(child_heap):
                    heap, child_heap = child_heap, heap
                for val in child_heap:
                    heapq.heappush(heap, val)

            l = r = 0
            if not is_leaf[x]:
                d = len(children[x])
                if d > 0:  # Only process if there are children
                    # Remove the d - 1 largest values
                    for _ in range(d - 1):
                        if heap:
                            heapq.heappop(heap)
                    # Pop the next two largest into r and l
                    if heap:
                        r = -heapq.heappop(heap)
                    if heap:
                        l = -heapq.heappop(heap)
            else:
                # For leaf nodes, ensure they have no children (as expected)
                pass

            # Push back with the current edge weight
            heapq.heappush(heap, -(l + w[x]))
            heapq.heappush(heap, -(r + w[x]))
            return heap

        root_heap = dfs(1)

        # Discard the single largest, then subtract every remaining value from res
        if root_heap:
            heapq.heappop(root_heap)
        while root_heap:
            res -= -heapq.heappop(root_heap)

        return res

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and terminate."""
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not initialized. Call reset() first."
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        # Heuristic: sample a non-negative integer; if reference present, use a nearby range
        if self.reference_answer is not None:
            lo = max(0, self.reference_answer - 10)
            hi = self.reference_answer + 10
            guess = random.randint(lo, hi)
        else:
            guess = random.randint(0, 100)
        return f"\\boxed{{{guess}}}"