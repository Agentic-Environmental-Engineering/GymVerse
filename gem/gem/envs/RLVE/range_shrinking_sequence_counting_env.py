from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from array import array
import re

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RangeShrinkingSequenceCountingEnv(Env):
    """Range Shrinking Sequence Counting problem environment - single-turn Q&A."""

    def __init__(
        self,
        min_N: int = 3,
        max_N: int = 50,
        max_MOD: int = 1000000,
        **kwargs
    ):
        super().__init__()
        assert min_N >= 3, "min_N should be greater than or equal to 3"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"
        assert max_MOD >= 2, "max_MOD should be greater than or equal to 2"

        self.min_N = min_N
        self.max_N = max_N
        self.max_MOD = max_MOD

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_N: Optional[int] = None
        self.current_R: Optional[List[int]] = None
        self.current_MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial sequence counting problem with range-shrinking constraints.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"

        # Generate a valid shrinking sequence consistent with constraints
        shrinking_sequence = [random.randint(1, N), random.randint(1, N)]
        l, r = 1, N
        if shrinking_sequence[0] >= shrinking_sequence[1]:
            r = shrinking_sequence[1]
        if shrinking_sequence[0] <= shrinking_sequence[1]:
            l = shrinking_sequence[1]
        for i in range(2, N):
            shrinking_sequence.append(random.randint(l, r))
            if shrinking_sequence[i - 1] >= shrinking_sequence[i]:
                assert shrinking_sequence[i - 1] <= r
                r = shrinking_sequence[i]
            if shrinking_sequence[i - 1] <= shrinking_sequence[i]:
                assert shrinking_sequence[i - 1] >= l
                l = shrinking_sequence[i]
            assert 1 <= l <= r <= N

        R = [random.randint(a, N) for a in shrinking_sequence]
        MOD = random.randint(2, self.max_MOD)

        # Store current parameters
        self.current_N = N
        self.current_R = R
        self.current_MOD = MOD

        # Compute the reference answer using the DP algorithm
        self.reference_answer = self._compute_reference_answer(N, R, MOD)

        # Build problem prompt
        R_str = ", ".join(f"R[{i}]={Ri}" for i, Ri in enumerate(R, start=1))
        problem_text = (
            f"Count the number of sequences A[1], A[2], ..., A[{N}] such that:\n"
            f"- For each i (1 ≤ i ≤ {N}), 1 ≤ A[i] ≤ R[i], where R is given as: {R_str}\n"
            f"- For each i (3 ≤ i ≤ {N}):\n"
            f"  - Let r = the minimum value among A[1], ..., A[i−2] that is ≥ A[i−1] (if none exists, r = +∞).\n"
            f"  - Let l = the maximum value among A[1], ..., A[i−2] that is ≤ A[i−1] (if none exists, l = −∞).\n"
            f"  - Then A[i] must satisfy l ≤ A[i] ≤ r.\n\n"
            f"Can you let me know the number of valid sequences modulo {MOD}?\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the user's answer."""
        # Parse answer from \boxed{...}
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check range: answer must be in [0, MOD)
        if self.current_MOD is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_ready"}

        if not (0 <= user_answer < self.current_MOD):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "error": "out_of_range",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check correctness
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer within \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        if self.current_MOD is None:
            # Fallback if environment not yet reset
            rnd = random.randint(0, 9)
        else:
            rnd = random.randint(0, self.current_MOD - 1)
        return f"\\boxed{{{rnd}}}"

    def _compute_reference_answer(self, N: int, R: List[int], MOD: int) -> int:
        """Compute the number of valid sequences modulo MOD using DP."""
        MAXV = max(R) if R else 0
        SENT = MAXV + 1
        SIZE = SENT + 2
        TOT = SIZE * SIZE * SIZE

        def base_idx(L1: int, R1: int) -> int:
            return (L1 * SIZE + R1) * SIZE

        def add_at(A: array, idx: int, val: int) -> None:
            s = A[idx] + val
            if s >= MOD:
                s -= MOD
            A[idx] = s

        def sub_at(A: array, idx: int, val: int) -> None:
            cur = A[idx]
            if cur >= val:
                A[idx] = cur - val
            else:
                A[idx] = cur - val + MOD

        # DP arrays
        f = array('I', [0]) * TOT
        g = array('I', [0]) * TOT

        # Initialization: for i in 1..R[0], f[0][SENT][i] = 1
        L0 = 0
        Rinf = SENT
        L1 = L0 + 1
        R1 = Rinf + 1
        b = base_idx(L1, R1)
        for x in range(1, R[0] + 1):
            X1 = x + 1
            f[b + X1] = 1

        # Iterate positions 2..N
        for i in range(1, N):
            Ai = R[i]
            g = array('I', [0]) * TOT

            for L in range(0, SENT + 1):
                L1 = L + 1
                for RR in range(L, SENT + 1):
                    R1 = RR + 1
                    bf = base_idx(L1, R1)
                    for x in range(L, RR + 1):
                        X1 = x + 1
                        c = f[bf + X1]
                        if c == 0:
                            continue

                        # 1) choose in (L, min(x-1, Ai))
                        l = L + 1
                        r = min(x - 1, Ai)
                        if l <= r:
                            tgtL1 = L1
                            tgtR1 = X1
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (l + 1), c)
                            sub_at(g, bg + (r + 1 + 1), c)

                        # 2) choose in (x+1, min(RR-1, Ai))
                        l = x + 1
                        r = min(RR - 1, Ai)
                        if l <= r:
                            tgtL1 = X1
                            tgtR1 = R1
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (l + 1), c)
                            sub_at(g, bg + (r + 1 + 1), c)

                        # 3) choose L exactly if valid (L > 0 and L <= Ai)
                        if L != 0 and L <= Ai:
                            tgtL1 = L1
                            tgtR1 = L1
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (L + 1), c)
                            sub_at(g, bg + (L + 1 + 1), c)

                        # 4) choose RR exactly if RR is a real bound (RR <= MAXV), RR <= Ai, and L != RR
                        if RR <= Ai and RR <= MAXV and L != RR:
                            tgtL1 = R1
                            tgtR1 = R1
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (RR + 1), c)
                            sub_at(g, bg + (RR + 1 + 1), c)

                        # 5) choose x exactly if x <= Ai and it's not equal to L or RR
                        if x <= Ai and L != x and RR != x:
                            tgtL1 = X1
                            tgtR1 = X1
                            bg = base_idx(tgtL1, tgtR1)
                            add_at(g, bg + (x + 1), c)
                            sub_at(g, bg + (x + 1 + 1), c)

            # Prefix sums along the 3rd dimension: g[L][R][x] += g[L][R][x-1]
            for L in range(0, SENT + 1):
                L1 = L + 1
                for RR in range(L, SENT + 1):
                    R1 = RR + 1
                    bg = base_idx(L1, R1)
                    pref = 0
                    for x in range(L, RR + 1):
                        X1 = x + 1
                        val = g[bg + X1]
                        s = val + pref
                        if s >= MOD:
                            s -= MOD
                        g[bg + X1] = s
                        pref = s

            f = g

        # Sum all f[L][R][x] over 0<=L<=R<=SENT, L<=x<=R
        ans = 0
        for L in range(0, SENT + 1):
            L1 = L + 1
            for RR in range(L, SENT + 1):
                R1 = RR + 1
                bf = base_idx(L1, R1)
                for x in range(L, RR + 1):
                    X1 = x + 1
                    val = f[bf + X1]
                    ans += val
                    if ans >= MOD:
                        ans -= MOD

        return ans