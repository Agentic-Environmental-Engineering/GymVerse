import random
from typing import Any, Optional, SupportsFloat, Tuple, Dict, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class GradeRankingCountingEnv(Env):
    """Single-turn environment for a combinatorics counting problem about grade rankings."""

    def __init__(
        self,
        max_n_m: int = 10,
        allowed_mods: Tuple[int, int] = (10**9 + 7, 998244353),
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n_m: Maximum value for both N and M (both at least 2 in the generated problems).
            allowed_mods: A tuple of possible moduli to choose from.
        """
        super().__init__()
        assert isinstance(max_n_m, int) and max_n_m >= 2, "max_n_m must be an integer >= 2"
        assert isinstance(allowed_mods, tuple) and len(allowed_mods) >= 1, "allowed_mods must be a non-empty tuple"
        self.max_n_m = max_n_m
        self.allowed_mods = allowed_mods

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

        # Store parameters for info/debugging
        self.problem_params: Dict[str, Any] = {}

    def _get_instructions(self) -> str:
        """Return the general instructions for the task."""
        return (
            "You are solving a combinatorics counting problem on grade rankings.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate core parameters
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        MOD = random.choice(self.allowed_mods)

        # Construct a generating matrix A to induce parameters U, R, K
        A: List[List[Optional[int]]] = [[None] * M for _ in range(N)]
        losers = set(random.sample(range(1, N), k=random.randint(0, N - 1)))
        U: List[int] = [random.randint(1, N) for _ in range(M)]
        R: List[int] = [0] * M

        for j in range(M):
            A[0][j] = random.randint(1, U[j])
            for i in range(1, N):
                if i in losers:
                    # This guarantees dominated in column j
                    A[i][j] = random.randint(1, A[0][j])
                else:
                    A[i][j] = random.randint(1, U[j])
                R[j] += int(A[i][j] > A[0][j])

        K = sum(int(all(A[0][j] >= A[i][j] for j in range(M))) for i in range(1, N))
        assert K >= len(losers), "K should be at least the number of losers"

        # Build the textual problem description (show the original R list, not the transformed one)
        U_str = " ".join(f"U[{j}]={Uj}" for j, Uj in enumerate(U))
        R_str = " ".join(f"R[{j}]={Rj}" for j, Rj in enumerate(R))

        problem_text = (
            f"Count the number of matrices A of size {N} × {M} (0-indexed) that satisfy the following conditions:\n"
            f"1. Each element A[i][j] (0 ≤ i < {N}, 0 ≤ j < {M}) is an integer in the range [1, U[j]]. U is: {U_str}\n"
            f"2. For each column j (0 ≤ j < {M}), there are exactly R[j] rows i (1 ≤ i < {N}) such that A[i][j] > A[0][j]. R is: {R_str}\n"
            f"3. There are exactly {K} rows i (1 ≤ i < {N}) such that A[0][j] ≥ A[i][j] holds for all j (0 ≤ j < {M}).\n\n"
            f"Output the number of such matrices modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        # Compute the reference answer using the provided combinatorics logic
        # ---------- basic combinatorics ----------
        def prepare_factorials(limit: int):
            """Precompute factorials and inverse factorials up to 'limit' inclusive."""
            fact = [1] * (limit + 1)
            for x in range(1, limit + 1):
                fact[x] = fact[x - 1] * x % MOD
            inv_fact = [1] * (limit + 1)
            inv_fact[limit] = pow(fact[limit], MOD - 2, MOD)
            for x in range(limit, 0, -1):
                inv_fact[x - 1] = inv_fact[x] * x % MOD
            return fact, inv_fact

        def C(n: int, k: int) -> int:
            if k < 0 or k > n:
                return 0
            return FACT[n] * INV_FACT[k] % MOD * INV_FACT[n - k] % MOD

        # ---------- Σ k^p for huge k (Faulhaber via Lagrange) ----------
        def power_sum(p: int, n: int) -> int:
            """
            S_p(n) = sum_{k=1..n} k^p   (0 ≤ p ≤ 2N, n can be large)
            Uses Lagrange interpolation over equally spaced nodes 0 … p+1 in O(p).
            """
            if n == 0:
                return 0
            d = p + 1
            if n <= d:
                s = 0
                for k_ in range(1, n + 1):
                    s = (s + pow(k_, p, MOD)) % MOD
                return s

            y = [0] * (d + 1)
            partial = 0
            for i_ in range(1, d + 1):
                partial = (partial + pow(i_, p, MOD)) % MOD
                y[i_] = partial

            x_val = n % MOD

            P = 1
            for j_ in range(d + 1):
                P = P * ((x_val - j_) % MOD) % MOD

            res_ = 0
            for i_ in range(d + 1):
                num = P * pow((x_val - i_) % MOD, MOD - 2, MOD) % MOD
                sign = MOD - 1 if (d - i_) & 1 else 1
                denom_inv = sign * INV_FACT[i_] % MOD * INV_FACT[d - i_] % MOD
                res_ = (res_ + y[i_] * num % MOD * denom_inv) % MOD
            return res_

        # ---------- single course contribution ----------
        def course_contribution(U_i: int, A_i: int, N_: int) -> int:
            """
            A_i students must be strictly above the benchmark B in this course.
            B_i = N_-1-A_i students are ≤ B.
            f_i = Σ_{S=1..U_i} (U_i-S)^{A_i} · S^{B_i}
                = Σ_{j=0..A_i} (-1)^j C(A_i,j) U_i^{A_i-j} · Σ_{k=1..U_i} k^{B_i+j}
            """
            B_i = N_ - 1 - A_i
            V = U_i
            res = 0
            for j_ in range(A_i + 1):
                coeff = C(A_i, j_)
                if j_ & 1:
                    coeff = MOD - coeff
                term = coeff * pow(V, A_i - j_, MOD) % MOD
                term = term * power_sum(B_i + j_, V) % MOD
                res = (res + term) % MOD
            return res

        # ---------- inclusion–exclusion over dominated students ----------
        def pattern_count(N_: int, K_: int, A_list_: List[int]) -> int:
            """
            Count ways to pick, for every course i, a subset of size A_i
            (taken from the S = N_-1-K_ non-dominated students)
            so that every non-dominated student appears ≥1 time.
            """
            S_ = N_ - 1 - K_
            total_ = 0
            for t_ in range(S_ + 1):
                if t_:
                    ok_ = all(A_ <= S_ - t_ for A_ in A_list_)
                    if not ok_:
                        continue
                prod_ = 1
                for A_ in A_list_:
                    prod_ = prod_ * C(S_ - t_, A_) % MOD
                term_ = C(S_, t_) * prod_ % MOD
                if t_ & 1:
                    total_ = (total_ - term_) % MOD
                else:
                    total_ = (total_ + term_) % MOD
            total_ = total_ * C(N_ - 1, K_) % MOD
            return total_

        # Prepare factorials up to 2N + 2 (covers all exponents and combinations)
        MAX_F = 2 * N + 2
        FACT, INV_FACT = prepare_factorials(MAX_F)

        # Transform local R for internal use: R_internal = [r + 1 for r in R]
        R_internal = [r + 1 for r in R]

        # Compute per-course numeric factor
        F_product = 1
        A_list = []
        for i in range(M):
            A_i = R_internal[i] - 1  # equals original R[i]
            A_list.append(A_i)
            F_product = F_product * course_contribution(U[i], A_i, N) % MOD

        # Compute combinatorial patterns for the “> benchmark” sets
        PATTERNS = pattern_count(N, K, A_list)

        # Final answer
        answer = F_product * PATTERNS % MOD

        # Save state
        self.current_problem = problem_text
        self.reference_answer = answer
        self.problem_params = {
            "N": N,
            "M": M,
            "MOD": MOD,
            "U": U,
            "R": R,
            "K": K,
            "reference_answer": answer,
        }

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step: parse and validate the proposed answer."""
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        MOD = self.problem_params.get("MOD")
        if MOD is not None:
            if not (0 <= user_answer < MOD):
                info = {
                    "correct": False,
                    "reference_answer": self.reference_answer,
                    "user_answer": user_answer,
                    "error": "out_of_range",
                }
                return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer
        }
        info.update(self.problem_params)

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random integer mod the current MOD in \\boxed{...} format."""
        MOD = self.problem_params.get("MOD", 1000000007)
        random_answer = random.randint(0, MOD - 1)
        return f"\\boxed{{{random_answer}}}"