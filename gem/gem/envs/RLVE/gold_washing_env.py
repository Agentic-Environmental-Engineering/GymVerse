import heapq
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GoldWashingEnv(Env):
    """
    GoldWashing environment converted to GEM format.

    Problem:
    Define f(x) as the product of the digits of x. For example, f(123) = 1 × 2 × 3 = 6.

    Let g(a, b) be the number of pairs (x, y) such that:
    1. x, y ∈ [1, N]
    2. f(x) = a and f(y) = b

    Compute g(a, b) for all 1 ≤ a, b ≤ N, then sort all g(a, b) values in non-increasing order.
    Output the sum of the largest K values (i.e., the first K values in the sorted list).

    This is a single-turn Q&A environment. The agent should return the final answer in \\boxed{...}.
    """

    def __init__(
        self,
        max_n: int = 1_000_000,
        fixed_n: Optional[int] = None,
        fixed_k: Optional[int] = None,
        **kwargs: Any
    ):
        """
        Initialize the environment.

        Args:
            max_n: Upper bound for N (N will be sampled uniformly from [2, max_n] if fixed_n is not provided).
            fixed_n: If provided, use this fixed N instead of sampling.
            fixed_k: If provided, use this fixed K instead of sampling.
        """
        super().__init__()
        if max_n < 2:
            raise ValueError("max_n should be greater than or equal to 2.")
        self.max_n: int = max_n

        if fixed_n is not None:
            if fixed_n < 2 or fixed_n > max_n:
                raise ValueError("fixed_n must satisfy 2 ≤ fixed_n ≤ max_n.")
        self.fixed_n: Optional[int] = fixed_n

        if fixed_k is not None:
            if fixed_n is None:
                raise ValueError("fixed_k is provided but fixed_n is None. Please provide fixed_n together with fixed_k.")
            if fixed_k < 1 or fixed_k > fixed_n:
                raise ValueError("fixed_k must satisfy 1 ≤ fixed_k ≤ fixed_n.")
        self.fixed_k: Optional[int] = fixed_k

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial counting problem based on digit-products.\n"
            "Provide only the final numeric answer wrapped in \\boxed{...}.\n"
            "Do not include any intermediate steps or additional text.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters N and K
        N = self.fixed_n if self.fixed_n is not None else random.randint(2, self.max_n)
        K = self.fixed_k if self.fixed_k is not None else random.randint(1, N)
        self.N = N
        self.K = K

        # Build problem statement
        problem_text = (
            "Define f(x) as the product of the digits of x. For example, f(123) = 1 × 2 × 3 = 6.\n\n"
            f"Let g(a, b) be the number of pairs (x, y) such that:\n"
            f"1. x, y ∈ [1, {N}]\n"
            f"2. f(x) = a and f(y) = b\n\n"
            f"Compute g(a, b) for all 1 ≤ a, b ≤ {N}, then sort all g(a, b) values in non-increasing order. "
            f"Output the sum of the largest {K} values (i.e., the first {K} values in the sorted list).\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )
        self.current_problem = problem_text

        # Compute reference answer
        self.reference_answer = self._compute_reference(N, K)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Check correctness
        try:
            user_answer = int(answer_text)
            is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
            reward: float = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # Use a simple heuristic range for random guessing
        guess_upper = max(10, (self.N or 10) * (self.K or 1))
        random_answer = random.randint(0, guess_upper)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference(self, N: int, K: int) -> int:
        """
        Compute the reference answer using the algorithm adapted from the original RLVE environment.
        Counts numbers x in [1..N] with no zero digits, groups them by digit product f(x),
        then considers all pair products of the counts and sums the largest K values.
        """
        S = str(N)
        n = len(S)

        # 1) Generate all products 2^a * 3^b * 5^c * 7^d <= N
        primes = [2, 3, 5, 7]
        products: List[int] = []

        def gen(idx: int, cur: int) -> None:
            if idx == 4:
                products.append(cur)
                return
            p = primes[idx]
            x = cur
            while x <= N:
                gen(idx + 1, x)
                x *= p

        gen(0, 1)

        prod_list = sorted(products)
        M_prime = len(prod_list)
        index_of: Dict[int, int] = {v: i for i, v in enumerate(prod_list)}

        # 2) Precompute counts for all lengths < n (numbers without zeros)
        #    fLen[L][j] = number of L-digit numbers (digits 1..9) whose product = prod_list[j]
        fLen: List[Optional[List[int]]] = [None] * (n + 1)

        # length = 1
        f1 = [0] * M_prime
        for d in range(1, 10):
            if d > N:
                break
            j = index_of.get(d)
            if j is not None:
                f1[j] += 1
        fLen[1] = f1

        for L in range(2, n):
            prev = fLen[L - 1]
            curr = [0] * M_prime
            if prev is not None:
                for j_idx, cnt in enumerate(prev):
                    if cnt == 0:
                        continue
                    base = prod_list[j_idx]
                    for d in range(1, 10):
                        newp = base * d
                        if newp > N:
                            break
                        newj = index_of[newp]
                        curr[newj] += cnt
            fLen[L] = curr

        # 3) Digit-DP for length = n, counting numbers in [1..N] with no zeros
        digits = list(map(int, S))
        dp_tight = [0] * M_prime   # prefix == N so far
        dp_loose = [0] * M_prime   # prefix  < N so far
        dp_tight[index_of[1]] = 1  # product = 1 at start

        for pos in range(n):
            new_tight = [0] * M_prime
            new_loose = [0] * M_prime
            ub = digits[pos]

            # transitions from loose (already < N)
            for j_idx, cnt in enumerate(dp_loose):
                if cnt == 0:
                    continue
                base = prod_list[j_idx]
                for d in range(1, 10):
                    newp = base * d
                    if newp > N:
                        break
                    newj = index_of[newp]
                    new_loose[newj] += cnt

            # transitions from tight (== N so far)
            if ub > 0:
                for j_idx, cnt in enumerate(dp_tight):
                    if cnt == 0:
                        continue
                    base = prod_list[j_idx]
                    # choose d < ub -> becomes loose
                    for d in range(1, ub):
                        newp = base * d
                        if newp > N:
                            break
                        newj = index_of[newp]
                        new_loose[newj] += cnt
                    # choose d == ub -> stays tight
                    newp_eq = base * ub
                    if newp_eq <= N:
                        newj_eq = index_of[newp_eq]
                        new_tight[newj_eq] += cnt

            dp_tight, dp_loose = new_tight, new_loose

        # fBound[j] = count of n-digit numbers <= N, no zeros, product = prod_list[j]
        fBound = [dp_tight[i] + dp_loose[i] for i in range(M_prime)]

        # 4) Total counts A[j] = sum over lengths 1..n-1 plus fBound for length n
        A = fBound[:]  # copy
        for L in range(1, n):
            row = fLen[L]
            if row is None:
                continue
            for j_idx, cnt in enumerate(row):
                if cnt:
                    A[j_idx] += cnt

        # 5) We have sums A[j]; sort them ascending
        sums = sorted(A)

        # 6) Take the top K products from the multiset { sums[i]*sums[j] }
        if K > M_prime * M_prime:
            K = M_prime * M_prime

        heap: List[Tuple[int, int, int]] = []
        last = M_prime - 1
        for i in range(M_prime):
            heap.append((-sums[i] * sums[last], i, last))
        heapq.heapify(heap)

        ans = 0
        for _ in range(K):
            negval, i, j = heapq.heappop(heap)
            val = -negval
            ans += val
            if j > 0:
                new_pair = sums[i] * sums[j - 1]
                heapq.heappush(heap, (-new_pair, i, j - 1))

        return ans