import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class StoneIntervalsGameEnv(Env):
    """Stone Intervals Game environment - single-turn Q&A.

    Players alternately collect stones from a pile whose adjacent pile has zero stones.
    Assuming optimal play, compute the number of stones Alice will collect.
    """

    def __init__(self, N: int = 3, **kwargs):
        """
        Initialize the StoneIntervalsGameEnv instance.

        Parameters:
            N (int): Number of piles. Must be >= 3.
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.A: List[int] = []

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving the Stone Intervals Game.\n"
            "Provide the number of stones Alice will collect assuming both players play optimally.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate piles
        N = self.N
        self.A = [random.randint(1, N * 2) for _ in range(N)]
        # Set a random number of piles to zero (at least 1, at most N-2)
        num_zeros = random.randint(1, N - 2)
        for zero_index in random.sample(range(N), num_zeros):
            self.A[zero_index] = 0

        # Compute reference answer using the core algorithm
        self.reference_answer = self._compute_reference_answer(self.A)

        # Build problem prompt
        piles_str = " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(self.A))
        self.current_problem = (
            f"There are {N} piles of stones. Initially, the i-th pile contains A[i] stones, given as: {piles_str}\n"
            "Alice and Bob play a game with the following rules:\n"
            "- Alice goes first. They alternate turns.\n"
            "- On each turn, a player selects a pile i such that at least one of its adjacent piles "
            "(i - 1 or i + 1, if within bounds) contains 0 stones (noting that the first/last pile has ONLY ONE adjacent pile). "
            "The player then collects all stones from pile i (pile i becomes 0).\n"
            "- The game ends when there are no piles with any stones remaining.\n\n"
            "Assuming both players play optimally to maximize their own total number of collected stones, "
            "output the number of stones Alice will collect.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {"N": self.N, "A": self.A}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem"}

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "A": self.A,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...}."""
        # A naive heuristic: Alice's total is between 0 and sum(A)
        upper = max(sum(self.A), 1)
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, A: List[int]) -> int:
        """Compute the reference answer using the original algorithm."""
        N = len(A)
        v = A.copy()
        SumVal = sum(v)

        # Mark which piles are non-zero
        tag = [x != 0 for x in v]

        # Doubly-linked list over indices 0..N-1
        prev_ = [i - 1 for i in range(N)]
        next_ = [i + 1 for i in range(N)]
        prev_[0] = None
        next_[N - 1] = None

        head = 0
        tail = N - 1

        # 1) Triple-compression: whenever three consecutive non-zero piles
        #    form a “peak” (middle >= both neighbors), merge them into the rightmost.
        i = head
        while i is not None:
            while (
                prev_[i] is not None
                and prev_[prev_[i]] is not None
                and tag[i]
                and tag[prev_[i]]
                and tag[prev_[prev_[i]]]
                and v[prev_[i]] >= v[prev_[prev_[i]]]
                and v[prev_[i]] >= v[i]
            ):
                p = prev_[i]
                pp = prev_[p]
                new_prev = prev_[pp]
                # merge: v[i] = v[pp] + v[i] − v[p]
                v[i] = v[pp] + v[i] - v[p]
                # remove pp and p by re-linking new_prev ↔ i
                prev_[i] = new_prev
                if new_prev is not None:
                    next_[new_prev] = i
                else:
                    head = i
            i = next_[i]

        # 2) Edge-peeling: greedily remove matching monotonic pairs at the ends,
        #    accumulating their difference into S
        L, R = head, tail
        S = 0
        # left side
        while True:
            nl = next_[L]
            if nl is None or not (tag[L] and tag[nl]) or v[L] < v[nl]:
                break
            S += v[nl] - v[L]
            L = next_[nl]
        # right side
        while True:
            pr = prev_[R]
            if pr is None or not (tag[R] and tag[pr]) or v[R] < v[pr]:
                break
            S += v[pr] - v[R]
            R = prev_[pr]

        # 3) Collect the remaining non-zero segments between L and R
        segments = []
        i = L
        while True:
            if tag[i]:
                segments.append(v[i])
            if i == R:
                break
            i = next_[i]

        # 4) Sort descending, append the peeled sum S, then do an alternating sum
        segments.sort(reverse=True)
        segments.append(S)
        score = 0
        for idx, val in enumerate(segments):
            score += val if idx % 2 == 0 else -val

        # 5) Recover Alice's total
        return (SumVal + score) // 2