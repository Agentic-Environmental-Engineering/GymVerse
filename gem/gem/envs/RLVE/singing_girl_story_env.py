import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from bisect import bisect_left
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SingingGirlStoryEnv(Env):
    """Singing Girl Story environment - single-turn Q&A.

    The task: Given N, A, and constraints on subarray maxima, count the number
    of arrays H[1..N] with values in [1, A] that satisfy all constraints,
    modulo MOD.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        mods: Tuple[int, int, int] = (666623333, 998244353, 10**9 + 7),
        **kwargs,
    ):
        """Initialize the environment.

        Args:
            max_n_m: Upper bound for N and M (N in [3, max_n_m], M in [1, max_n_m]).
            mods: A tuple of candidate MOD values to choose from randomly each reset.
        """
        super().__init__()
        if max_n_m < 3:
            raise ValueError("max_n_m should be greater than or equal to 3")
        self.max_n_m = max_n_m
        self.mods = mods

        # State for the current episode
        self.N: Optional[int] = None
        self.A: Optional[int] = None
        self.conditions: Optional[List[Tuple[int, int, int]]] = None
        self.mod: Optional[int] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem on arrays with maximum constraints.\n"
            "Given N, A, and several constraints of the form max(H[l : r + 1]) = v, "
            "compute the number of arrays H[1..N] with each H[i] in [1, A] that satisfy all constraints, modulo MOD.\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters
        N = random.randint(3, self.max_n_m)
        A = random.randint(2, N)
        H = [random.randint(1, A) for _ in range(N)]
        M = random.randint(1, self.max_n_m)

        # Generate constraints using a concrete random array H to ensure consistency
        conditions: List[Tuple[int, int, int]] = []
        for _ in range(M):
            length = random.randint(2, N)
            start = random.randint(1, N - length + 1)
            end = start + length - 1
            v = max(H[start - 1 : end])
            # Sanity checks
            assert 1 <= start <= end <= N
            assert 1 <= v <= A
            conditions.append((start, end, v))

        MOD = random.choice(self.mods)

        # Store state
        self.N = N
        self.A = A
        self.conditions = conditions
        self.mod = MOD

        # Compute the reference answer
        self.reference_answer = self._compute_reference(N, A, conditions, MOD)

        # Build problem prompt
        conditions_text = "\n".join(
            f"- max(H[{l} : {r} + 1]) = {v}" for (l, r, v) in conditions
        )
        problem_prompt = (
            f"Consider an array H[1], H[2], ..., H[{N}], where each H[i] is an integer in [1, {A}]. "
            f"We say max(H[l : r + 1]) denotes the maximum value in the subarray H[l], H[l+1], ..., H[r] "
            f"(1 ≤ l ≤ r ≤ {N}). How many arrays H satisfy all of the following conditions?\n"
            f"{conditions_text}\n\n"
            f"Output the number of valid arrays modulo {MOD}.\n"
            f"Remember to output your final result in \\boxed{{...}} format."
        )

        self.current_problem = problem_prompt
        obs = self._get_instructions() + problem_prompt
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the outcome."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check range relative to MOD
        info: Dict[str, Any] = {"user_answer": user_answer, "reference_answer": self.reference_answer}
        if self.mod is not None and not (0 <= user_answer < self.mod):
            info["error"] = "out_of_range"
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        info["correct"] = bool(is_correct)
        reward = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in boxed format."""
        if self.mod is None:
            rnd = 0
        else:
            rnd = random.randint(0, self.mod - 1)
        return f"\\boxed{{{rnd}}}"

    def _compute_reference(
        self,
        N: int,
        A: int,
        conditions: List[Tuple[int, int, int]],
        MOD: int,
    ) -> int:
        """Compute the reference answer using the original algorithm."""

        M = len(conditions)

        def calc(val: int, pts: List[int], eves: List[int], UNI: List[int], Q: List[Dict[str, int]]) -> int:
            # pts: list of segment indices i (1-based) where MX[i] == val
            # eves: list of event indices (1-based) where Q[id]['v'] == val
            if not pts:
                return 0
            L = len(pts)
            # 1-based for convenience; Aindex[0] = 0 as in the original logic
            Aindex = [0] + pts[:]

            # Precompute powers
            PPW = [1] * (L + 1)  # PPW[0] = 1
            for i in range(1, L + 1):
                seg_len = UNI[Aindex[i] + 1] - UNI[Aindex[i]]
                PPW[i] = pow(val - 1, seg_len, MOD)

            DP = [0] * (L + 1)
            DP[0] = 1

            for i in range(1, L + 1):
                seg_len = UNI[Aindex[i] + 1] - UNI[Aindex[i]]
                pw = (pow(val, seg_len, MOD) - pow(val - 1, seg_len, MOD) + MOD) % MOD
                mxL = 0
                for eid in eves:
                    if Q[eid]['r'] <= Aindex[i]:
                        if Q[eid]['l'] > mxL:
                            mxL = Q[eid]['l']
                j = i - 1
                while j >= 0 and Aindex[j] >= mxL:
                    DP[i] = (DP[i] + DP[j] * pw) % MOD
                    pw = (pw * PPW[j]) % MOD
                    j -= 1

            res = 0
            for i in range(0, L + 1):
                ok = True
                for eid in eves:
                    if Q[eid]['l'] > Aindex[i]:
                        ok = False
                        break
                if ok:
                    pw = 1
                    for j in range(i + 1, L + 1):
                        pw = (pw * PPW[j]) % MOD
                    res = (res + DP[i] * pw) % MOD
            return res

        def solve_one() -> int:
            # Prepare queries
            Q: List[Optional[Dict[str, int]]] = [None] * (M + 1)  # 1-based
            KEY: List[int] = []
            ST = set()
            for i, (l, r, v) in enumerate(conditions, start=1):
                r1 = r + 1
                Q[i] = {'l': l, 'r': r1, 'v': v}
                KEY.append(l)
                KEY.append(r1)
                ST.add(v)

            # Coordinate compression for boundaries
            KEY.sort()
            UNI: List[Optional[int]] = [None]  # 1-based: UNI[1..NUM] valid after compression
            prev: Optional[int] = None
            for x in KEY:
                if x != prev:
                    UNI.append(x)
                    prev = x
            NUM = len(UNI) - 1  # number of unique keys
            UNI.append(N + 1)   # UNI[NUM+1] = N+1; types below index 1.. used are ints

            # Map l, r to indices in UNI[1..NUM+1]
            for i in range(1, M + 1):
                lval = Q[i]['l']  # type: ignore[index]
                rval = Q[i]['r']  # type: ignore[index]
                li = bisect_left(UNI, lval, 1, NUM + 1)  # search within indices of ints
                ri = bisect_left(UNI, rval, 1, NUM + 1)
                Q[i]['l'] = li  # type: ignore[index]
                Q[i]['r'] = ri  # type: ignore[index]

            # Compute per-segment minimal maximum (INF if unconstrained)
            INF = A + 1
            MX = [INF] * (NUM + 2)  # 1-based up to NUM
            for i in range(1, M + 1):
                li = Q[i]['l']  # type: ignore[index]
                ri = Q[i]['r']  # type: ignore[index]
                v = Q[i]['v']   # type: ignore[index]
                for j in range(li, ri):
                    if v < MX[j]:
                        MX[j] = v

            # Sum of constrained lengths
            total_constrained = 0
            for i in range(1, NUM + 1):
                if MX[i] != INF:
                    total_constrained += (int(UNI[i + 1]) - int(UNI[i]))  # type: ignore[index]

            prd = pow(A, N - total_constrained, MOD)

            # Multiply contributions for each distinct maximum value
            for val in ST:
                pts = [i for i in range(1, NUM + 1) if MX[i] == val]
                eves = [i for i in range(1, M + 1) if Q[i]['v'] == val]  # type: ignore[index]
                # Cast UNI to List[int] for calc (ignore the leading None)
                UNI_int: List[int] = [0] + [int(x) for x in UNI[1:] if x is not None]  # 1-based alignment
                prd = (prd * calc(val, pts, eves, UNI_int, Q)) % MOD

            return prd

        return solve_one()