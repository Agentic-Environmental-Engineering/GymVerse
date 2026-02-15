import random
from bisect import bisect_left
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
import re


class MYJEnv(Env):
    """Environment for assigning shop prices to maximize total revenue from customers."""

    def __init__(
        self,
        N: int,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 5.0,
        **kwargs
    ):
        """
        Initialize the MYJEnv instance.

        Parameters:
            N (int): Number of shops (must be >= 1).
            wrong_format (float): Legacy parameter from RLVE (not used for reward in GEM).
            invalid_solution (float): Legacy parameter from RLVE (not used for reward in GEM).
            rewarding_strategy (str): Legacy parameter from RLVE (not used for reward in GEM).
            rewarding_weight (float): Legacy parameter from RLVE (not used for reward in GEM).
            rewarding_beta (float): Legacy parameter from RLVE (not used for reward in GEM).
        """
        super().__init__()
        assert N >= 1, "N should be greater than or equal to 1"
        self.N: int = N

        # Legacy reward parameters kept for compatibility (not used in GEM reward computation)
        self.wrong_format: float = wrong_format
        self.invalid_solution: float = invalid_solution
        self.rewarding_strategy: str = rewarding_strategy
        self.rewarding_weight: float = rewarding_weight
        self.rewarding_beta: float = rewarding_beta

        # Problem data
        self.M: Optional[int] = None
        self.customers: List[Tuple[int, int, int]] = []
        self.current_problem: Optional[str] = None

        # Reference data
        self.gold_answer: Optional[int] = None
        self.reference_assignment: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are given N shops labeled from 1 to N. Each shop i has an item price P[i].\n"
            "There are M customers, each with a triple (a, b, c). A customer will buy an item from the shop with the lowest price in the range [a, b], "
            "but only if that lowest price is at most c; otherwise, the customer does not buy anything.\n"
            "Your goal is to assign prices P[1..N] to maximize the total money earned from all customers.\n"
            "Output Format: Provide P[1], P[2], ..., P[N] separated by spaces, wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment by generating a new problem."""
        super().reset(seed)

        N = self.N
        M = random.randint(1, N * (N + 1) // 2)
        self.M = M

        # Generate customers
        customers: List[Tuple[int, int, int]] = []
        for _ in range(M):
            a = random.randint(1, N)
            b = random.randint(1, N)
            c = random.randint(1, N * (N + 1) // 2)
            customers.append((min(a, b), max(a, b), c))
        self.customers = customers

        # Prepare arrays
        A = [0] * (M + 1)
        B = [0] * (M + 1)
        C = [0] * (M + 1)
        D: List[int] = []

        for i in range(1, M + 1):
            a, b, c = customers[i - 1]
            A[i] = a
            B[i] = b
            C[i] = c
            D.append(c)

        # Sort costs and compress them to 1..M
        D_sorted = sorted(D)
        for i in range(1, M + 1):
            C[i] = bisect_left(D_sorted, C[i]) + 1

        # Allocate DP, traceback, bucket and answer arrays
        # f[l][r][i]: maximum total value in segment [l..r] using cost-levels >= i
        f = [
            [
                [0] * (M + 2)
                for _ in range(N + 2)
            ]
            for __ in range(N + 2)
        ]
        # tr[l][r][i]: (cost_index, position) choice for segment [l..r] at level i
        tr = [
            [
                [(0, 0)] * (M + 2)
                for _ in range(N + 2)
            ]
            for __ in range(N + 2)
        ]
        # buc[l][r]: number of customers whose interval [a_j..b_j] is contained in [l..r]
        #            among those with cost-index >= current i
        buc = [
            [0] * (N + 2)
            for _ in range(N + 2)
        ]
        # Final assigned prices
        ans = [0] * (N + 2)

        # Recursive reconstruction of the chosen positions/prices
        def dfs(l: int, r: int, i: int) -> None:
            if l > r:
                return
            cost_i, pos = tr[l][r][i]
            if cost_i == 0 or pos == 0:
                return
            ans[pos] = D_sorted[cost_i - 1]
            dfs(l, pos - 1, cost_i)
            dfs(pos + 1, r, cost_i)

        # Main DP: process cost-levels from high to low
        for i in range(M, 0, -1):
            # Add all intervals whose compressed cost == i into the bucket counts
            for j in range(1, M + 1):
                if C[j] == i:
                    for l in range(1, A[j] + 1):
                        for r in range(B[j], N + 1):
                            buc[l][r] += 1

            # Solve subproblems for all segments [l..r]
            for length in range(1, N + 1):
                for l in range(1, N - length + 2):
                    r = l + length - 1
                    # Option 1: skip using cost-level i
                    f[l][r][i] = f[l][r][i + 1]
                    tr[l][r][i] = tr[l][r][i + 1]

                    # Option 2: pick a position p in [l..r] with price = D_sorted[i-1]
                    for p in range(l, r + 1):
                        coef = buc[l][r]
                        coef -= buc[l][p - 1] if p - 1 >= 1 else 0
                        coef -= buc[p + 1][r] if p + 1 <= N else 0
                        v = f[l][p - 1][i] + f[p + 1][r][i] + coef * D_sorted[i - 1]
                        if v > f[l][r][i]:
                            f[l][r][i] = v
                            tr[l][r][i] = (i, p)

                    # If we never picked anything at this level, default to placing at l
                    if tr[l][r][i][0] == 0:
                        tr[l][r][i] = (i, l)

        # Output the maximum total and one valid price assignment
        self.gold_answer = f[1][N][1]
        dfs(1, N, 1)
        self.reference_assignment = " ".join(str(ans[i]) for i in range(1, N + 1))

        # Build problem string
        customers_str = "\n".join(f"({a}, {b}, {c})" for a, b, c in customers)
        self.current_problem = (
            f"There are {N} shops labeled from 1 to {N} (from left to right); every shop has a price, "
            f"and the price of an item at shop i is P[i]. There are {M} customers; each customer is represented "
            f"by a tuple (a, b, c); the customer will consider buying the item from a shop in the range [a, b] with "
            f"the lowest price, but only if that price is at most c (if the lowest price in the range is greater than c, "
            f"the customer will not buy anything):\n{customers_str}\n\n"
            f"Please assign an item price for each shop to maximize the total money earned from all customers.\n"
            f"Output P[1], P[2], ..., P[{N}] in one line, separated by spaces.\n"
            f"Your final answer should be wrapped in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {
            "N": N,
            "M": M,
            "customers": customers,
            "gold_total": self.gold_answer,
            "reference_assignment": self.reference_assignment,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step: parse and verify the price assignment."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse prices from boxed content
        try:
            parts = boxed_content.strip().split()
            prices = [int(x) for x in parts]
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.N
        if len(prices) != N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution_length", "expected_length": N}

        P = [None] + prices  # 1-based indexing
        user_total = 0
        for a, b, c in self.customers:
            min_price = min(P[a:b + 1])
            if min_price <= c:
                user_total += min_price

        gold = self.gold_answer if self.gold_answer is not None else 0
        assert user_total <= gold, "The answer should not exceed the gold answer"

        is_correct = (user_total == gold)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "gold_total": gold,
            "user_total": user_total,
            "user_assignment": " ".join(str(x) for x in prices),
            "reference_assignment": self.reference_assignment,
            "N": N,
            "M": self.M,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: random prices wrapped in \\boxed{...}."""
        N = self.N
        max_price = N * (N + 1) // 2
        random_prices = [str(random.randint(1, max_price)) for _ in range(N)]
        return f"\\boxed{{{' '.join(random_prices)}}}"