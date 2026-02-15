import heapq
import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Max_NonAdjacent_KElementSumEnv(Env):
    """Environment for selecting exactly K non-adjacent indices to maximize the sum of array elements.

    The task:
    - Given an array A of N positive integers.
    - Select exactly K indices such that no two selected indices are adjacent.
    - Maximize the sum of the selected elements.

    The agent must output the selected indices (0-based) in \\boxed{...} format, separated by spaces.
    """

    def __init__(self, N: int = 10, **kwargs) -> None:
        """Initialize the environment.

        Args:
            N: Length of the array A. Must be >= 4.
            **kwargs: Ignored extra arguments for compatibility.
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 4, "N should be greater than or equal to 4"
        self.N: int = N

        # Problem state
        self.K: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.gold_sum: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the general task instructions."""
        return (
            "You are given an array A of N positive integers. Your goal is to select exactly K indices such that:\n"
            "- No two selected indices are adjacent (i.e., there are no i and i+1 both selected).\n"
            "- The sum of A over the selected indices is maximized.\n\n"
            "Output Format: Provide exactly K indices (0-based) separated by spaces, wrapped in \\boxed{...}.\n"
            "Example: \\boxed{0 2 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem instance.

        Args:
            seed: Optional random seed.

        Returns:
            A tuple (observation, info) where observation is the problem string with instructions.
        """
        super().reset(seed)

        N = self.N
        K = random.randint(2, N // 2)
        A = [random.randint(1, N) for _ in range(N)]

        gold = self._compute_gold_sum(A, K)

        # Save state
        self.K = K
        self.A = A
        self.gold_sum = gold

        # Build problem description
        array_str = " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A))
        problem = (
            f"You are given an array A of {N} positive integers:\n"
            f"{array_str}\n\n"
            f"Please select exactly {K} indices i1, ..., i{K}, such that:\n"
            f"- No two selected indices are adjacent (i.e., there does not exist any i and i + 1 such that both i and i + 1 are selected).\n"
            f"- The sum A[i1] + ... + A[i{K}] is maximized.\n\n"
            f"Output Format: A single line containing the {K} selected indices in any order, separated by spaces, wrapped in \\boxed{{...}}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer.

        Args:
            action: A string containing the proposed indices in \\boxed{...}.

        Returns:
            TERMINAL_STATE, reward, terminated, truncated, info
        """
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure problem has been generated
        if self.A is None or self.K is None or self.gold_sum is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Parse indices
        try:
            # Split on whitespace
            tokens = boxed_content.strip().split()
            user_indices = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        N = self.N
        K = self.K
        A = self.A
        gold = self.gold_sum

        # Validate indices
        info: dict[str, Any] = {}
        if len(user_indices) != K:
            info.update({"error": "invalid_solution", "reason": "wrong_number_of_indices"})
            return TERMINAL_STATE, 0.0, True, False, info

        if len(user_indices) != len(set(user_indices)):
            info.update({"error": "invalid_solution", "reason": "duplicate_indices"})
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(0 <= i < N for i in user_indices):
            info.update({"error": "invalid_solution", "reason": "index_out_of_range"})
            return TERMINAL_STATE, 0.0, True, False, info

        user_indices_sorted = sorted(user_indices)
        if any(user_indices_sorted[i] + 1 == user_indices_sorted[i + 1] for i in range(len(user_indices_sorted) - 1)):
            info.update({"error": "invalid_solution", "reason": "adjacent_indices"})
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute user sum and compare to gold
        user_sum = sum(A[i] for i in user_indices)
        is_optimal = (user_sum == gold)

        reward = 1.0 if is_optimal else 0.0
        info.update({
            "correct": is_optimal,
            "reference_sum": gold,
            "user_sum": user_sum,
            "K": K,
            "N": N,
            "indices": user_indices,
        })

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_gold_sum(self, A: List[int], K: int) -> int:
        """Compute the maximum sum of exactly K non-adjacent elements using the original heap-based algorithm."""
        N = len(A)
        vals = A.copy()

        # Dynamic sentinel value larger than any sum
        INF = sum(abs(v) for v in vals) + 1

        # Initialize doubly linked list arrays (0..N+1)
        L = list(range(N + 2))
        R = list(range(N + 2))
        val = [0] * (N + 2)
        vis = [False] * (N + 2)

        # Fill values and neighbors; use 1-based for internal representation
        for i, v in enumerate(vals, start=1):
            val[i] = v
            L[i] = i - 1
            R[i] = i + 1

        # Sentinels at 0 and N+1
        val[0] = val[N + 1] = -INF
        L[0] = 0
        R[0] = 1
        L[N + 1] = N
        R[N + 1] = N + 1

        # Build max-heap using negatives
        heap: List[Tuple[int, int]] = []
        for i in range(1, N + 1):
            heapq.heappush(heap, (-val[i], i))

        ans = 0
        # Perform K merges
        for _ in range(K):
            # Pop until an unvisited position is found
            while True:
                neg_x, pos = heap[0]
                if vis[pos]:
                    heapq.heappop(heap)
                else:
                    break
            x = -neg_x
            heapq.heappop(heap)

            ans += x
            l = L[pos]
            r = R[pos]

            # Bypass l and r
            L[pos] = L[l]
            R[pos] = R[r]
            R[L[pos]] = pos
            L[R[pos]] = pos

            # Mark removed neighbors
            vis[l] = True
            vis[r] = True

            # Update current value and push back
            val[pos] = val[l] + val[r] - x
            heapq.heappush(heap, (-val[pos], pos))

        assert ans > 0
        return ans

    def sample_random_action(self) -> str:
        """Sample a random valid action (indices) and return in \\boxed{...} format."""
        if self.A is None or self.K is None:
            # If no problem generated yet, create one
            self.reset()

        N = self.N
        K = self.K if self.K is not None else 2

        # Randomly construct a non-adjacent set of size K
        # Approach: shuffle candidate positions and greedily pick with non-adjacency constraint
        indices = list(range(N))
        random.shuffle(indices)
        selected: List[int] = []
        used = set()

        for idx in indices:
            if (idx not in used) and ((idx - 1) not in used) and ((idx + 1) not in used):
                selected.append(idx)
                used.add(idx)
                used.add(idx - 1)
                used.add(idx + 1)
                if len(selected) == K:
                    break

        # Fallback (should not happen if K <= N//2) - construct deterministically
        if len(selected) < K:
            selected = list(range(0, 2 * K, 2))[:K]

        content = " ".join(map(str, selected))
        return f"\\boxed{{{content}}}"