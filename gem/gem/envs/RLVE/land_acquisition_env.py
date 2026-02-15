import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LandAcquisitionEnv(Env):
    """
    Land Acquisition partition optimization environment - single-turn Q&A.

    Task:
    - There are N items, each with attributes W[i] and L[i].
    - Partition all items into an arbitrary number of disjoint non-empty sets.
    - For a set S, cost(S) = max(W[i] for i in S) * max(L[i] for i in S).
    - Minimize the total cost, which is the sum of costs over all sets.

    Answer format:
    - Return your partition inside \\boxed{...}.
    - Inside the braces, write M lines (M is the number of sets).
    - Each line contains one or more integers (1-based indices), separated by spaces, representing one set.
    """

    def __init__(self, N: int = 8, max_sampling_attempts: int = 1000, **kwargs) -> None:
        """
        Initialize the LandAcquisitionEnv instance.

        Args:
            N: Number of items (must be >= 4)
            max_sampling_attempts: Maximum attempts to sample a problem instance where the optimal
                answer is strictly better than naive heuristics. If exceeded, accept the last sample.
        """
        super().__init__()
        assert N >= 4, "N should be greater than or equal to 4"
        self.N: int = N
        self.max_sampling_attempts: int = max_sampling_attempts

        # Problem state
        self.W: Optional[List[int]] = None
        self.L: Optional[List[int]] = None
        self.reference_cost: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a partition optimization problem.\n"
            "There are N items, each item i has attributes W[i] and L[i].\n"
            "You must partition all items (1..N) into disjoint non-empty sets. For a set S, its cost is\n"
            "  cost(S) = max(W[i] for i in S) × max(L[i] for i in S)\n"
            "The total cost is the sum of costs of all sets. Minimize this total cost.\n\n"
            "Output Format:\n"
            "- Put your entire answer inside \\boxed{...}.\n"
            "- Inside the braces, write M lines (M is the number of sets in your partition).\n"
            "- Each line contains the 1-based indices of items in one set, separated by spaces.\n"
            "- Each item index from 1 to N must appear exactly once across all lines.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: The task instructions concatenated with the problem description.
            info: An empty dict.
        """
        super().reset(seed)

        N = self.N
        attempts = 0
        W: List[int]
        L: List[int]
        reference_cost: int

        while True:
            W = [random.randint(1, N * N) for _ in range(N)]
            L = [random.randint(1, N * N) for _ in range(N)]

            # Build list of pairs (w, l) with 1-based conceptual indexing
            Land_pairs = [(w, l) for (w, l) in zip(W, L)]

            # Sort by width asc, then length asc
            Land_sorted = sorted(Land_pairs, key=lambda x: (x[0], x[1]))

            # Remove dominated rectangles: keep strictly decreasing lengths
            stack: List[Tuple[int, int]] = []
            for w, l in Land_sorted:
                while stack and l > stack[-1][1]:
                    stack.pop()
                stack.append((w, l))

            cnt = len(stack)

            # 1-indexed 'needto' with a sentinel at the end so needto[i+1] is safe
            needto: List[Tuple[int, int]] = [None] + stack + [(0, 0)]  # type: ignore

            # DP with Convex Hull Trick-like optimization
            dp: List[Optional[int]] = [None] * (cnt + 1)
            dp[0] = 0

            q: List[int] = [0]
            head = 0

            for i in range(1, cnt + 1):
                # Move head forward while the next candidate is better
                while head < len(q) - 1:
                    j0 = q[head]
                    j1 = q[head + 1]
                    lhs = dp[j0] - dp[j1]  # type: ignore
                    rhs = -needto[i][0] * (needto[j0 + 1][1] - needto[j1 + 1][1])
                    if lhs >= rhs:
                        head += 1
                    else:
                        break

                j = q[head]
                dp[i] = (dp[j] if dp[j] is not None else 0) + needto[i][0] * needto[j + 1][1]  # type: ignore

                # Maintain convexity of the hull
                while head < len(q) - 1:
                    j_last = q[-1]
                    j_prev = q[-2]
                    left = (dp[j_last] - dp[j_prev]) * (needto[i + 1][1] - needto[j_prev + 1][1])  # type: ignore
                    right = (dp[i] - dp[j_prev]) * (needto[j_last + 1][1] - needto[j_prev + 1][1])  # type: ignore
                    if left <= right:
                        q.pop()
                    else:
                        break

                q.append(i)

            reference_cost = dp[cnt] if dp[cnt] is not None else 0
            assert reference_cost > 0

            # Compute a naive baseline to filter too-easy instances
            item_indices = list(range(N))
            item_indices.sort(key=lambda i: (W[i], L[i]))

            naive_answer = min(
                max(W) * max(L),  # all items in one group
                sum(Wi * Li for Wi, Li in zip(W, L))  # each item in its own group
            )
            for i in range(N - 1):
                group_1 = max(W[j] for j in item_indices[: i + 1]) * max(L[j] for j in item_indices[: i + 1])
                group_2 = max(W[j] for j in item_indices[i + 1 :]) * max(L[j] for j in item_indices[i + 1 :])
                naive_answer = min(naive_answer, group_1 + group_2)

            # Prefer instances where the optimal is strictly better than naive
            if reference_cost <= naive_answer and reference_cost < naive_answer:
                break

            attempts += 1
            if attempts >= self.max_sampling_attempts:
                # Accept the current instance to avoid infinite loops
                break

        # Save problem state
        self.W = W
        self.L = L
        self.reference_cost = reference_cost

        # Build problem statement
        W_L_lines = "\n".join(
            f"W[{i}]={Wi} L[{i}]={Li}" for i, (Wi, Li) in enumerate(zip(self.W, self.L), start=1)
        )

        self.current_problem = (
            f"There are {self.N} items, and the i-th item has two attributes W[i] and L[i]. "
            f"The arrays W and L are given as follows:\n{W_L_lines}\n\n"
            "Partition all items into an arbitrary number of disjoint non-empty sets. "
            "For each set S, its cost is defined as: cost(S) = max(W[i] for i ∈ S) × max(L[i] for i ∈ S).\n"
            "Minimize the total cost, i.e., the sum of costs over all sets.\n\n"
            "Output M lines inside a single \\boxed{...} block, where M is the number of sets in your partition; "
            "each line should contain the 1-based indices of the items in one set (separated by spaces)."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step by verifying the user's partition.

        Args:
            action: A string containing the user's answer in \\boxed{...} format.

        Returns:
            observation: TERMINAL_STATE since this is single-turn.
            reward: 1.0 if optimal partition cost is achieved, 0.0 otherwise; -0.1 for format errors.
            terminated: True
            truncated: False
            info: Additional details including correctness and costs.
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.W is None or self.L is None or self.reference_cost is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        # Parse groups from boxed content
        try:
            groups: List[List[int]] = []
            for line in boxed_content.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                group = list(map(int, line.split()))
                if len(group) == 0:
                    # Non-empty sets required
                    return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution_empty_group"}
                groups.append(group)
        except Exception:
            # Parsing integers failed -> treat as format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        N = self.N

        # Validate coverage and disjointness
        all_items = [item for group in groups for item in group]
        if len(all_items) != N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution_incorrect_count"}
        if set(all_items) != set(range(1, N + 1)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution_incorrect_items"}

        # Compute user's total cost
        user_cost = 0
        for group in groups:
            w_max = max(self.W[i - 1] for i in group)
            l_max = max(self.L[i - 1] for i in group)
            user_cost += w_max * l_max

        is_correct = (user_cost == self.reference_cost)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_cost": self.reference_cost,
            "user_cost": user_cost,
            "N": N,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([\s\S]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid partition action in \\boxed{...} format."""
        if self.W is None or self.L is None:
            # If not initialized, default to N from config with singleton groups
            N = self.N
        else:
            N = len(self.W)

        # Create a random partition by shuffling indices and randomly splitting
        indices = list(range(1, N + 1))
        random.shuffle(indices)

        groups: List[List[int]] = []
        i = 0
        while i < N:
            # Random group size between 1 and remaining
            remaining = N - i
            size = random.randint(1, remaining)
            groups.append(indices[i : i + size])
            i += size

        lines = [" ".join(map(str, g)) for g in groups]
        return "\\boxed{" + "\n".join(lines) + "}"