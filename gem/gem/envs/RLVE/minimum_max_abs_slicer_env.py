from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from collections import defaultdict, deque
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Minimum_MaxAbsSlicerEnv(Env):
    """Environment for the Minimum Max Absolute Slicer problem - single-turn Q&A."""

    prompt_template = (
        "You are given two arrays A and B, each of length {N} (0-indexed). A is a permutation of [1, 2, ..., {N}], and each element of B is either +1 or -1. The values are as follows:\n"
        "{A_and_B}\n\n"
        "You must divide the indices [0, 1, ..., {N_minus_1}] into {M} consecutive batches. Let end[1], end[2], ..., end[{M}] (0 ≤ end[1] < end[2] < ... < end[{M}] = {N_minus_1}) represent the last index of each batch. This means:\n"
        "- Batch 1 contains indices from 0 to end[1]\n"
        "- Batch 2 contains indices from end[1] + 1 to end[2]\n"
        "- ...\n"
        "- Batch {M} contains indices from end[{M_minus_1}] + 1 to end[{M}] = {N_minus_1}\n\n"
        "For each batch i, let S[i] be the sum of B values in that batch. Your goal is to minimize the maximum absolute value among all batches, i.e., minimize max(|S[1]|, |S[2]|, ..., |S[{M}]|).\n"
        "Among all such optimal partitions, choose the one with the smallest lexicographical order of the sequence A[end[1]], A[end[2]], ..., A[end[{M}]].\n\n"
        "Output Format: Your final answer should be A[end[1]], A[end[2]], ..., A[end[{M}]], separated by spaces and wrapped in \\boxed{{...}}."
    )

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 50,
        M: Optional[int] = None,
        positive_probability: Optional[float] = None,
        wrong_format_reward: float = -0.1,
        invalid_answer_reward: float = 0.0,
        correct_reward: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, use this fixed N (must be >= 4). Otherwise sample uniformly from [min_N, max_N].
        - min_N: Minimum N when sampling randomly.
        - max_N: Maximum N when sampling randomly.
        - M: If provided, use this fixed M (must satisfy 3 <= M <= N-1). Otherwise sampled uniformly.
        - positive_probability: Probability that a B element is +1; if None, sampled uniformly in [0, 1].
        - wrong_format_reward: Reward when the answer format is incorrect (e.g., not in \\boxed{...}).
        - invalid_answer_reward: Reward when the parsed answer is invalid or incorrect.
        - correct_reward: Reward when the answer is fully correct (optimal and lexicographically minimal).
        """
        super().__init__()
        assert min_N >= 4, "min_N must be at least 4"
        assert max_N >= min_N, "max_N must be >= min_N"
        if N is not None:
            assert N >= 4, "N must be at least 4"
        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N

        self.fixed_M = M
        if positive_probability is not None:
            assert 0.0 <= positive_probability <= 1.0, "positive_probability must be in [0, 1]"
        self.positive_probability = positive_probability

        self.wrong_format_reward = wrong_format_reward
        self.invalid_answer_reward = invalid_answer_reward
        self.correct_reward = correct_reward

        # State variables
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.A: Optional[List[int]] = None  # 0-based permutation of [1..N]
        self.B: Optional[List[int]] = None  # 0-based list of +1/-1
        self.gold_answer_max_abs: Optional[int] = None
        self.gold_answer: Optional[List[int]] = None  # sequence of A[end[i]] values (length M)
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving the 'Minimum Max Absolute Slicer' problem.\n"
            "Please provide your final answer as a sequence of integers separated by single spaces, wrapped in \\boxed{...}.\n"
            "Example: \\boxed{3 7 10}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            self.N = self.fixed_N
        else:
            self.N = random.randint(self.min_N, self.max_N)
        assert self.N is not None and self.N >= 4, "N must be at least 4"

        # Determine M
        if self.fixed_M is not None:
            self.M = self.fixed_M
            assert 3 <= self.M <= self.N - 1, "M must satisfy 3 <= M <= N-1"
        else:
            self.M = random.randint(3, self.N - 1)

        # Generate A and B
        self.A = list(range(1, self.N + 1))
        random.shuffle(self.A)
        p = self.positive_probability if self.positive_probability is not None else random.random()
        self.B = [+1 if random.random() < p else -1 for _ in range(self.N)]

        # Build problem prompt
        A_and_B = "\n".join(f"A[{i}]={Ai} B[{i}]={Bi}" for i, (Ai, Bi) in enumerate(zip(self.A, self.B)))
        self.current_problem = self.prompt_template.format(
            N=self.N,
            N_minus_1=self.N - 1,
            M=self.M,
            M_minus_1=self.M - 1,
            A_and_B=A_and_B,
        )

        # Compute gold answer using the original algorithm
        gold_answer, gold_d = self._compute_gold_answer(self.A, self.B, self.N, self.M)
        self.gold_answer = gold_answer
        self.gold_answer_max_abs = gold_d
        self.reference_answer = " ".join(map(str, self.gold_answer))

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_gold_answer(self, A0: List[int], B0: List[int], N: int, M: int) -> Tuple[List[int], int]:
        """Compute the optimal partition's lexicographically smallest sequence of A[end[i]]."""
        # Convert to 1-based arrays for algorithm
        A = [0] * (N + 2)
        B = [0] * (N + 2)
        for i in range(1, N + 1):
            A[i] = A0[i - 1]
            B[i] = B0[i - 1]

        # Build suffix balance array
        SUF = [0] * (N + 3)  # SUF[i] = balance on [i … N]
        for i in range(N, 0, -1):
            SUF[i] = B[i] + SUF[i + 1]

        # Count perfectly balanced suffixes
        tot_zero = sum(1 for i in range(1, N + 1) if SUF[i] == 0)

        OFFSET = N  # shift to make indices non-negative

        # Minimal possible maximal monthly imbalance d
        if SUF[1] == 0:
            d = 1 if tot_zero < M else 0
        else:
            d = (abs(SUF[1]) - 1) // M + 1  # ceil(|SUF[1]| / M)

        # Monotone queues keyed by balance value
        queues: Dict[int, deque] = defaultdict(deque)  # balance → deque[(city, pos)]

        def push(pos: int) -> None:
            """Put position `pos` into queue of balance SUF[pos+1]."""
            key = SUF[pos + 1] + OFFSET
            dq = queues[key]
            rec = (A[pos], pos)  # ordered by city id
            while dq and rec[0] < dq[-1][0]:
                dq.pop()
            dq.append(rec)

        def best_from_queue(now_pos: int, key: int, cur_best: tuple) -> tuple:
            """Try improving cur_best using front of queue `key`."""
            dq = queues.get(key)
            if not dq:
                return cur_best
            while dq and dq[0][1] < now_pos:  # outdated endpoint
                dq.popleft()
            if dq and dq[0][0] < cur_best[0]:
                return dq[0]
            return cur_best

        # Construct the answer
        answer: List[int] = []

        if d == 0:
            # CASE 1: perfectly balanced plan possible
            C = [i for i in range(1, N + 1) if SUF[i + 1] == 0]  # candidate cuts
            tot_c = len(C)
            now = 1
            j = 0

            # decide the first M-1 months
            for month in range(1, M):
                # keep at least (M - month) candidates unpushed
                while tot_c - j > M - month:
                    push(C[j])
                    j += 1
                best = (N + 1, -1)  # (city id, pos)
                best = best_from_queue(now, OFFSET, best)
                answer.append(best[0])
                now = best[1] + 1  # next month starts here
        else:
            # CASE 2: need positive imbalance
            now = 1
            r = 1
            # preload all positions that may finish the first month
            while N - r >= M - 1:
                push(r)
                r += 1

            months_left = M
            while months_left > 1:
                best = (N + 1, -1)
                center = SUF[now] + OFFSET

                low = max(0, center - d)
                high = min(2 * N, center + d)

                for key in range(low, high + 1):
                    # |balance| must be small enough to finish the rest in (months_left-1) months
                    if abs(key - OFFSET) <= (months_left - 1) * d:
                        best = best_from_queue(now, key, best)

                answer.append(best[0])
                now = best[1] + 1
                months_left -= 1

                # make one more position available for the next round
                if r <= N:
                    push(r)
                r += 1

        # Last month ends at N
        answer.append(A[N])
        assert len(answer) == M, "The answer should have exactly M elements"

        return answer, d

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step by parsing and verifying the user's answer."""
        # Parse \\boxed{...}
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Validate environment state
        assert self.N is not None and self.M is not None and self.A is not None and self.B is not None
        assert self.gold_answer is not None and self.gold_answer_max_abs is not None

        # Convert boxed content to integer list
        try:
            user_list = list(map(int, boxed.split()))
        except Exception:
            return TERMINAL_STATE, self.invalid_answer_reward, True, False, {"error": "invalid_answer"}

        # Verify constraints
        N = self.N
        M = self.M
        A = self.A
        B = self.B

        if len(user_list) != M:
            info = {"error": "invalid_answer", "reason": "length_mismatch", "expected_length": M, "received_length": len(user_list)}
            return TERMINAL_STATE, self.invalid_answer_reward, True, False, info

        if not all(1 <= Ai <= N for Ai in user_list):
            info = {"error": "invalid_answer", "reason": "value_out_of_range", "N": N}
            return TERMINAL_STATE, self.invalid_answer_reward, True, False, info

        # Map city ids to indices (0-based)
        Ai2i = [None] * (N + 1)
        for idx, city in enumerate(A):
            Ai2i[city] = idx

        try:
            ends = [Ai2i[val] for val in user_list]
        except Exception:
            return TERMINAL_STATE, self.invalid_answer_reward, True, False, {"error": "invalid_answer", "reason": "mapping_failure"}

        if any(e is None for e in ends):
            return TERMINAL_STATE, self.invalid_answer_reward, True, False, {"error": "invalid_answer", "reason": "unknown_city"}

        # Check increasing and final endpoint equals N-1
        if any(not (0 <= ends[i] < N) for i in range(M)):
            return TERMINAL_STATE, self.invalid_answer_reward, True, False, {"error": "invalid_answer", "reason": "index_out_of_bounds"}
        for i in range(1, M):
            if not (ends[i - 1] < ends[i]):
                return TERMINAL_STATE, self.invalid_answer_reward, True, False, {"error": "invalid_answer", "reason": "non_increasing"}
        if ends[-1] != N - 1:
            return TERMINAL_STATE, self.invalid_answer_reward, True, False, {"error": "invalid_answer", "reason": "last_end_incorrect", "expected": N - 1, "received": ends[-1]}

        # Compute maximum absolute batch sum for user partition
        max_abs = abs(sum(B[index] for index in range(0, ends[0] + 1)))
        for i in range(1, M):
            s = sum(B[index] for index in range(ends[i - 1] + 1, ends[i] + 1))
            max_abs = max(max_abs, abs(s))

        gold_d = self.gold_answer_max_abs
        is_optimal_abs = (gold_d <= max_abs) and (gold_d == max_abs)
        is_lex_minimal = (user_list == self.gold_answer)

        is_correct = is_optimal_abs and is_lex_minimal
        reward = self.correct_reward if is_correct else self.invalid_answer_reward

        info = {
            "correct": is_correct,
            "user_answer": user_list,
            "gold_answer": self.gold_answer,
            "gold_answer_max_abs": self.gold_answer_max_abs,
            "N": self.N,
            "M": self.M,
            "A": self.A,
            "B": self.B,
            "reference_answer": self.reference_answer,
            "is_optimal_abs": is_optimal_abs,
            "is_lex_minimal": is_lex_minimal,
        }

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
        """Sample a random action (random sequence of M city ids)."""
        if self.N is None or self.M is None:
            # Fallback: random example
            seq = [random.randint(1, 10) for _ in range(3)]
        else:
            seq = sorted(random.sample(range(1, self.N + 1), self.M))
        return f"\\boxed{{{' '.join(map(str, seq))}}}"