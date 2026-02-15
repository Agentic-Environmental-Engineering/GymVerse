import math
import random
import re
from bisect import bisect_left, insort
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BoxSchedulingEnv(Env):
    """Box Scheduling problem environment - Single-turn Q&A."""

    def __init__(self, max_n: int = 100, **kwargs) -> None:
        """
        Initialize the BoxSchedulingEnv.

        Parameters:
        - max_n: The maximum value for N. Must be >= 3.

        Notes:
        - This environment generates a single-turn problem where the agent must
          output a sequence of positions satisfying certain constraints.
        """
        super().__init__()
        assert isinstance(max_n, int), "max_n must be an integer"
        assert max_n >= 3, "max_n should be greater than or equal to 3"
        self.max_n: int = max_n

        # Problem state
        self.N: Optional[int] = None
        self.C: Optional[List[int]] = None
        self.D: Optional[int] = None
        self.S: Optional[int] = None

        # Gold/reference answer
        self.gold_answer: Optional[List[int]] = None
        self.reference_answer: Optional[str] = None

        # Current problem description
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions in English."""
        return (
            "You are solving a box scheduling problem.\n"
            "Provide your answer in \\boxed{...} format, containing N-1 integers separated by spaces.\n"
            "For example: \\boxed{p1 p2 p3 ... p_{N-1}}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(3, self.max_n)
        C = [random.randint(0, N - 1) for _ in range(N - 1)]
        D = None
        for _ in range(int(N ** 0.5)):
            candidate = random.randint(1, N - 1)
            if math.gcd(candidate, N) > 1:
                D = candidate
                break
        if D is None:
            D = random.randint(1, N - 1)
        S = random.randint(0, N - 1)

        self.N = N
        self.C = C
        self.D = D
        self.S = S

        # Compute the gold answer using the original algorithm
        c = [0] + C

        # DSU for “next free” in a D-cycle
        parent = list(range(N))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Prepare the multiset st of (residue_mod_G, count)
        G = math.gcd(D, N)
        con = N // G
        tar = S % G

        # st will be a sorted list of (residue, remaining_slots)
        st: List[Tuple[int, int]] = []
        # positions array
        p = [0] * N

        # initialize
        for r in range(G):
            if r != tar:
                # all con slots available
                insort(st, (r, con))
            else:
                # reserve one for the empty slot at i=0
                p[0] = S
                # mark S as used by linking it to (S+D)%N
                parent[S] = find((S + D) % N)
                # if there are more in this class, keep (con-1)
                if con > 1:
                    insort(st, (r, con - 1))

        # assign positions for boxes 1..N-1
        for i in range(1, N):
            key = c[i] % G

            # find the first entry in st with residue >= key
            idx = bisect_left(st, (key, -1))
            if idx == len(st):
                # wrap around to the smallest residue
                idx = 0

            r, cnt = st.pop(idx)
            # if more remain in this residue-class, put it back
            if cnt > 1:
                insort(st, (r, cnt - 1))

            # compute the base position before DSU-skipping
            if r >= key:
                j = (c[i] + (r - key)) % N
            else:
                # jump up one multiple of G
                j = (((c[i] // G) + 1) * G + r) % N

            # find the actual next free slot in its D-cycle
            pj = find(j)
            p[i] = pj
            # mark pj used
            parent[pj] = find((pj + D) % N)

        self.gold_answer = p[1:]
        self.reference_answer = " ".join(map(str, self.gold_answer))

        # Build problem description
        problem_text = (
            f"You are given a sequence C: "
            + " ".join(f"C[{i + 1}]={Ci}" for i, Ci in enumerate(C))
            + "\n\n"
            f"Determine two non-negative integer sequences X[1], ..., X[{N - 1}] and Y[1], ..., Y[{N - 1}] such that:\n"
            f"- For 1 ≤ i ≤ {N - 1}, define: Pos[i] = (C[i] + {D} × X[i] + Y[i]) mod {N}\n"
            f"- The values Pos[1], ..., Pos[{N - 1}] must be all distinct.\n"
            f"- No Pos[i] can be equal to {S}.\n"
            f"Among all valid solutions:\n"
            f"- First, minimize the lexicographical order of sequence Y.\n"
            f"- If multiple solutions have the same Y, then choose the one with the smallest lexicographical order of X.\n\n"
            f"Output Format: A single line containing Pos[1], ..., Pos[{N - 1}], separated by spaces, "
            f"enclosed in \\boxed{{...}}.\n"
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": N,
            "D": D,
            "S": S,
            "C": C,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step to validate the boxed answer."""
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Expect a space-separated list of integers
        try:
            pos_list = list(map(int, boxed.strip().split()))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Basic validation checks
        assert self.N is not None and self.S is not None and self.gold_answer is not None
        if len(pos_list) != self.N - 1:
            info = {
                "correct": False,
                "reason": "invalid_length",
                "expected_length": self.N - 1,
                "user_length": len(pos_list),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        expected_set = set(range(self.N)) - {self.S}
        if set(pos_list) != expected_set:
            info = {
                "correct": False,
                "reason": "invalid_set",
                "expected_set": sorted(list(expected_set)),
                "user_set": sorted(list(set(pos_list))),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Exact match for the gold answer
        is_correct = (pos_list == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, pos_list)),
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action; by default, return the gold answer if available."""
        if self.reference_answer is not None:
            return f"\\boxed{{{self.reference_answer}}}"
        # Fallback: generate a random permutation satisfying constraints if possible
        if self.N is not None and self.S is not None:
            candidates = list(set(range(self.N)) - {self.S})
            random.shuffle(candidates)
            return f"\\boxed{{{' '.join(map(str, candidates))}}}"
        # Generic fallback
        return "\\boxed{}"