import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ChoHamstersEnv(Env):
    """ChoHamsters string construction environment - single-turn Q&A.

    Task:
      - Given N strings S[0..N-1] (none is a contiguous substring of another),
        construct a string T such that the sum over i of counting(T, S[i]) is at least M,
        where counting(T, s) is the number of (possibly overlapping) occurrences of s in T.
      - Minimize the length of T.
      - Output the minimum possible length as an integer in \\boxed{...} format.

    This environment generates a random instance with:
      - N strings of random lengths in [N * length_multiple_min, N * length_multiple_max],
      - Random Bernoulli letter generation per string with a random 'a' probability,
      - M uniformly at random from [1, MAX_M].

    The reference answer is computed using min-plus matrix exponentiation based on
    overlaps between strings (KMP-prefix for transitions).
    """

    def __init__(
        self,
        N: int = 3,
        MAX_M: int = 10,
        length_multiple_min: int = 2,
        length_multiple_max: int = 3,
        **kwargs: Any,
    ):
        super().__init__()
        # Parameters controlling instance generation
        self.N = N
        self.MAX_M = MAX_M
        self.length_multiple_min = length_multiple_min
        self.length_multiple_max = length_multiple_max

        # Validate parameters
        assert self.N >= 1, "N should be greater than or equal to 1"
        assert self.MAX_M >= 1, "MAX_M should be greater than or equal to 1"
        assert self.length_multiple_min >= 1, "length_multiple_min should be >= 1"
        assert self.length_multiple_max >= self.length_multiple_min, "length_multiple_max should be >= length_multiple_min"

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.S: Optional[List[str]] = None
        self.M: Optional[int] = None

        # Rewards
        self.REWARD_CORRECT: float = 1.0
        self.REWARD_WRONG: float = 0.0
        self.REWARD_FORMAT_ERROR: float = -0.1

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given N strings S[0..N-1], where for all i ≠ j, S[i] is NOT a contiguous substring of S[j].\n"
            "Your task is to construct a string T such that the sum over i of counting(T, S[i]) is at least M,\n"
            "where counting(T, s) is the number of (possibly overlapping) occurrences of s in T.\n"
            "You should minimize the length of such a string T and output the minimum possible length.\n"
            "Output Format: Provide a single integer as your final answer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate strings S with the property that none is a substring of another
        N = self.N
        while True:
            S: List[str] = []
            for _ in range(N):
                length = random.randint(N * self.length_multiple_min, N * self.length_multiple_max)
                a_probability = random.random()
                Si = "".join("a" if random.random() < a_probability else "b" for _ in range(length))
                S.append(Si)
            # Ensure none is a substring of another
            if all(Si not in Sj for i, Si in enumerate(S) for j, Sj in enumerate(S) if i != j):
                break

        # Sample M from [1, MAX_M]
        M = random.randint(1, self.MAX_M)

        # Compute reference answer using the same algorithm as original environment
        reference_answer = self._compute_reference_answer(S, M)

        # Build the problem prompt
        strings_listing = "\n".join(f"S[{i}]={Si}" for i, Si in enumerate(S))
        problem_text = (
            f"You are given {N} strings, listed below (it is guaranteed that for all i ≠ j, "
            f"the string S[i] is NOT a contiguous substring of S[j]):\n"
            f"{strings_listing}\n\n"
            f"Please construct a string T such that the sum (for all i) of counting(T, S[i]) "
            f"is at least {M}, where counting(T, s) is the number of (possibly overlapping) "
            f"occurrences of the string s in T.\n"
            f"Try your best to minimize the length of such a string T. "
            f"Output a single integer — the minimum possible length of T."
        )

        self.current_problem = problem_text
        self.reference_answer = reference_answer
        self.S = S
        self.M = M

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer and terminate."""
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, self.REWARD_FORMAT_ERROR, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, self.REWARD_WRONG, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not initialized. Call reset() before step()."
        is_correct = (user_answer == self.reference_answer)
        reward = self.REWARD_CORRECT if is_correct else self.REWARD_WRONG

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # Use a rough bound if S and M are available; otherwise default range.
        if self.S is not None and self.M is not None and len(self.S) > 0:
            max_len = max(len(s) for s in self.S)
            upper = max(1, self.M * max_len * 2)
        else:
            upper = 100
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, S: List[str], M: int) -> int:
        """Compute the minimal length of T satisfying the condition using min-plus matrix exponentiation."""
        N = len(S)
        assert N >= 1
        assert M >= 1

        # Compute prefix-function (KMP) for each string in S
        pi: List[List[int]] = []
        for s in S:
            L = len(s)
            p = [0] * L
            j = 0
            for i in range(1, L):
                while j > 0 and s[j] != s[i]:
                    j = p[j - 1]
                if s[j] == s[i]:
                    j += 1
                p[i] = j
            pi.append(p)

        # Determine an upper bound INF based on maximum possible cost
        max_len = max(len(s) for s in S)
        INF = M * max_len + 1

        # Build the transition matrix Tra of size (N+1) x (N+1)
        # Node 0 is the start; nodes 1..N correspond to S[0]..S[N-1]
        Tra = [[INF] * (N + 1) for _ in range(N + 1)]

        # From start (0) to each string x: cost = full length of string x
        for x in range(1, N + 1):
            Tra[0][x] = len(S[x - 1])

        # Precompute transition costs between strings
        # Tra[x][y] = extra letters needed to append S[y-1] after S[x-1]
        for x in range(1, N + 1):
            sx = S[x - 1]
            len_x = len(sx)
            for y in range(1, N + 1):
                sy = S[y - 1]
                len_y = len(sy)
                # Find overlap: longest suffix of sx matching prefix of sy
                j = 0
                for i in range(1, len_x):
                    while j > 0 and sy[j] != sx[i]:
                        j = pi[y - 1][j - 1]
                    if sy[j] == sx[i]:
                        j += 1
                # j is the overlap length
                Tra[x][y] = len_y - j

        def mat_mult(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
            """Matrix multiplication in min-plus (tropical) semiring."""
            C = [[INF] * (N + 1) for _ in range(N + 1)]
            for i in range(N + 1):
                row_i = C[i]
                for j in range(N + 1):
                    aij = A[i][j]
                    if aij == INF:
                        continue
                    bj = B[j]
                    for k in range(N + 1):
                        v = aij + bj[k]
                        if v < row_i[k]:
                            row_i[k] = v
            return C

        # Fast exponentiation: compute Ans = Tra^M (min-plus semiring)
        Ans = [row[:] for row in Tra]  # Tra^1
        exp = M - 1
        base = [row[:] for row in Tra]
        while exp > 0:
            if exp & 1:
                Ans = mat_mult(Ans, base)
            base = mat_mult(base, base)
            exp >>= 1

        # The answer is the minimum cost from start (0) to any string after M transitions
        result = min(Ans[0][1:])
        return result