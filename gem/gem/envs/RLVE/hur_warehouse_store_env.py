import heapq
import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class HURWarehouseStoreEnv(Env):
    """Warehouse store scheduling environment - single-turn Q&A.

    Problem statement:
    - For N days, you receive A[i] items in the morning and a customer demands B[i] items in the evening.
    - You may satisfy the customer on day i only if you have at least B[i] items in stock at that time.
    - Your goal is to maximize the number of customers satisfied.
    - Output the indices of the days you choose to satisfy customers, separated by spaces, inside \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        max_N: int = 100,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
        - N: Optional fixed number of days. If None, it will be sampled in reset() in [3, max_N].
        - max_N: Upper bound for N when sampling (must be >= 3).
        """
        super().__init__()
        self.N: Optional[int] = N
        self.max_N: int = max_N

        # Problem state
        self.A: Optional[List[int]] = None
        self.B: Optional[List[int]] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[int] = None
        self.optimal_days: Optional[List[int]] = None  # 1-based indices

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are running a warehouse store scheduling problem.\n"
            "On day i, in the morning you receive A[i] items; in the evening a customer demands B[i] items.\n"
            "You can satisfy the customer only if you have at least B[i] items in stock.\n"
            "Your goal is to maximize the number of customers satisfied.\n\n"
            "Output Format: Provide the indices of the days (1-based) you choose to satisfy customers,\n"
            "separated by spaces, inside \\boxed{...}. For example: \\boxed{1 3 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N is None:
            N = random.randint(3, max(3, self.max_N))
        else:
            N = self.N
        assert N >= 3, "N should be greater than or equal to 3"

        # Generate A and B ensuring at least one feasible day exists
        while True:
            A = [random.randint(0, N) for _ in range(N)]
            B = [random.randint(1, N) for _ in range(N)]

            # Check that there exists at least one day that can be satisfied
            answer_not_zero, stock = False, 0
            for Ai, Bi in zip(A, B):
                stock += Ai
                if stock >= Bi:
                    answer_not_zero = True
                    break
            if answer_not_zero:
                break

        # Compute optimal selection via greedy strategy with a max-heap
        tot = 0
        count = 0
        heap: List[Tuple[int, int]] = []  # store (-B[i], i)
        vis = [False] * N

        for i in range(N):
            tot += A[i]

            # If we can't satisfy B[i], consider replacing a previously accepted larger demand
            if heap and tot < B[i]:
                largest_neg_b, idx = heap[0]
                largest_b = -largest_neg_b
                if largest_b > B[i]:
                    heapq.heappop(heap)
                    vis[idx] = False
                    tot += largest_b
                    count -= 1

            # Try to accept today
            if tot >= B[i]:
                tot -= B[i]
                heapq.heappush(heap, (-B[i], i))
                vis[i] = True
                count += 1

        assert count > 0, "There should be at least one customer satisfied"

        # Save problem state
        self.N = N
        self.A = A
        self.B = B
        self.gold_answer = count
        self.optimal_days = [i + 1 for i, v in enumerate(vis) if v]
        self.reference_answer = " ".join(str(idx) for idx in self.optimal_days)

        # Build problem prompt
        problem_lines = "\n".join(
            f"A[{i + 1}]={Ai} B[{i + 1}]={Bi}" for i, (Ai, Bi) in enumerate(zip(A, B))
        )
        problem_statement = (
            f"You are running a warehouse store for {N} days. "
            f"On the morning of day i, you receive A[i] items; in the evening of the same day, a customer arrives and demands B[i] items. "
            f"You can choose to satisfy the customer only if you have at least B[i] items in stock. The arrays A and B are given as follows:\n"
            f"{problem_lines}\n\n"
            f"Please maximize the number of customers you can satisfy. "
            f"Output a single line containing the indices of the days when you satisfy the customers, separated by spaces. "
            f"Your answer must be inside \\boxed{{...}}."
        )
        self.current_problem = problem_statement

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": self.N,
            "A": self.A,
            "B": self.B,
            "gold_answer": self.gold_answer,
            "reference_answer": self.reference_answer,
            "optimal_days": self.optimal_days,
        }

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process a single action (answer) and return the evaluation."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices from boxed content
        tokens = boxed_content.strip().split()
        day_indices: List[int] = []
        try:
            for t in tokens:
                day_indices.append(int(t))
        except ValueError:
            # Contents not all integers
            info = {
                "error": "invalid_answer",
                "raw_content": boxed_content,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate schedule
        assert self.N is not None and self.A is not None and self.B is not None
        N = self.N
        satisfy = [False] * N

        for day in day_indices:
            day0 = day - 1
            if not (0 <= day0 < N):
                info = {
                    "error": "out_of_range_day",
                    "day": day,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            if satisfy[day0]:
                info = {
                    "error": "duplicate_day",
                    "day": day,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            satisfy[day0] = True

        stock = 0
        for i in range(N):
            stock += self.A[i]
            if satisfy[i]:
                if stock < self.B[i]:
                    info = {
                        "error": "insufficient_stock",
                        "day": i + 1,
                        "stock_before": stock,
                        "required": self.B[i],
                    }
                    return TERMINAL_STATE, 0.0, True, False, info
                stock -= self.B[i]

        # If valid, check optimality: number of satisfied customers must equal gold_answer
        assert self.gold_answer is not None
        is_optimal = (len(day_indices) == self.gold_answer)
        reward = 1.0 if is_optimal else 0.0

        info = {
            "correct": is_optimal,
            "gold_answer": self.gold_answer,
            "reference_answer": self.reference_answer,
            "user_days": day_indices,
            "valid": True,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random feasible-looking action (not guaranteed optimal)."""
        if self.N is None:
            # Fallback if called before reset
            return "\\boxed{}"
        # Sample a random subset of days (unique)
        k = random.randint(0, self.N)
        days = sorted(random.sample(range(1, self.N + 1), k))
        content = " ".join(str(d) for d in days)
        return f"\\boxed{{{content}}}"