from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxDifferentGroupPairDivisionEnv(Env):
    """Environment for the 'Max Different Group Pair Division' optimization problem (single-turn Q&A)."""

    prompt_template = (
        "You are given an array A of {N} integers: {A}\n\n"
        "Initially, the entire array is one single block. Let S = 0. You need to perform the following operation exactly {K} times:\n"
        "- Choose a position i such that A[i] and A[i + 1] are still in the same block.\n"
        "- Split the block into two parts: the first ends at A[i], the second starts at A[i + 1].\n"
        "- Let sum1 and sum2 be the sums of the two blocks. Then, update S += sum1 × sum2.\n\n"
        "After {K} operations, you will have {K} + 1 blocks. Try your best to maximize the final value of S.\n\n"
        "Output Format: Provide {K} integers — the positions i you chose in order, separated by spaces, in \\boxed{{...}} format.\n"
        "Example: \\boxed{{i1 i2 ... iK}}"
    )

    def __init__(self, N: int, **kwargs):
        """
        Initialize the environment.

        Parameters:
        - N: Length of the array A. Must be >= 4.
        """
        super().__init__()
        if N < 4:
            raise ValueError("N should be greater than or equal to 4")
        self.N: int = N

        # Problem instance state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_score: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.K: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You will be given an array and must choose K split positions to maximize a defined score S.\n"
            "Your final answer must be provided in \\boxed{...} format as K integers separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        K = random.randint(2, N - 2)
        A = [random.randint(0, N) for _ in range(N)]

        # Compute optimal score and one optimal set of splits via DP with convex-hull trick
        gold_score, reference_splits_zero_based = self._solve_max_score_and_splits(A, K)

        # Store state
        self.A = A
        self.K = K
        self.gold_score = gold_score
        self.reference_answer = " ".join(map(str, reference_splits_zero_based))

        # Build problem prompt
        array_str = " ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A))
        self.current_problem = self.prompt_template.format(N=N, A=array_str, K=K)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the submitted answer."""
        content = self._parse_answer(action)
        if content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse K integers from content
        if self.K is None or self.A is None or self.gold_score is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error"}

        try:
            parts = content.strip().split()
            user_splits = list(map(int, parts))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if len(user_splits) != self.K:
            info = {
                "error": "invalid_length",
                "expected_k": self.K,
                "received_k": len(user_splits),
                "reference_answer": self.reference_answer,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        valid, user_score = self._evaluate_splits(self.A, user_splits)
        if not valid:
            info = {
                "error": "invalid_solution",
                "reference_answer": self.reference_answer,
                "gold_score": self.gold_score,
                "user_score": user_score,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_score == self.gold_score)
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "gold_score": self.gold_score,
            "user_score": user_score,
            "N": self.N,
            "K": self.K,
            "A": self.A,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a naive action. Here we simply return the reference (optimal) answer when available."""
        if self.reference_answer is not None:
            return f"\\boxed{{{self.reference_answer}}}"
        # Fallback: random splits (may be invalid)
        if self.K is None or self.N is None:
            return "\\boxed{}"
        rand_splits = [str(random.randint(0, self.N - 2)) for _ in range(self.K)]
        return f"\\boxed{{{' '.join(rand_splits)}}}"

    def _solve_max_score_and_splits(self, A: List[int], K: int) -> Tuple[int, List[int]]:
        """
        Compute the maximum possible S and reconstruct one optimal sequence of split positions.
        Returns (gold_score, splits_zero_based).
        """
        N = len(A)
        B = K + 1  # number of blocks after K splits

        # Build prefix sums (1-based for convenience)
        prefix_sum = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]
        sum_N = prefix_sum[N]

        # pre[j][i] stores the split position for the j-th block ending at i (1-based indexing)
        pre = [[0] * (N + 1) for _ in range(B + 1)]

        prev_f = [0] * (N + 1)
        cur_f = [0] * (N + 1)

        # DP with convex-hull trick
        for j in range(1, B + 1):
            # Deque arrays for candidates
            qx = [0] * (N + 1)  # x = prefix_sum[p]
            qy = [0] * (N + 1)  # y = prev_f[p]
            qp = [0] * (N + 1)  # p = index (split position)

            head = tail = 0
            qx[0] = 0
            qy[0] = prev_f[0]
            qp[0] = 0

            for i in range(1, N + 1):
                psi = prefix_sum[i]
                s_rem = sum_N - psi

                # Pop from front while next candidate is better
                while head < tail and (qy[head + 1] - qy[head]) >= s_rem * (qx[head + 1] - qx[head]):
                    head += 1

                # Choose best candidate
                p = qp[head]
                pre[j][i] = p
                cur_f[i] = qy[head] + s_rem * (psi - qx[head])

                # Prepare new candidate from current i
                new_x = psi
                new_y = prev_f[i]

                # Pop from back while new candidate makes the last one obsolete
                while head < tail and (qy[tail] - qy[tail - 1]) * (new_x - qx[tail]) <= (new_y - qy[tail]) * (qx[tail] - qx[tail - 1]):
                    tail -= 1

                tail += 1
                qx[tail] = new_x
                qy[tail] = new_y
                qp[tail] = i

            prev_f, cur_f = cur_f, [0] * (N + 1)

        gold_score = prev_f[N]

        # Reconstruct split positions (1-based positions in path, convert to 0-based split indices)
        path = [0] * (B + 1)
        path[B] = N
        for j in range(B, 0, -1):
            path[j - 1] = pre[j][path[j]]
        splits_1_based_positions = path[1:B]  # positions in 1..N where blocks end
        # Convert to 0-based split indices: split at i means block ends at index i (0-based),
        # which corresponds to position (i + 1) in 1-based path. So i = position - 1.
        splits_zero_based = [pos - 1 for pos in splits_1_based_positions]

        return gold_score, splits_zero_based

    def _evaluate_splits(self, A: List[int], splits: List[int]) -> Tuple[bool, int]:
        """
        Evaluate the provided sequence of splits.
        Returns (valid, score). If invalid, score may be 0 or partial.
        """
        N = len(A)
        block_id = 0
        block_numbers = [0] * N
        score = 0

        for i in splits:
            # Basic index checks
            if not (0 <= i < N):
                return False, 0
            if not (0 <= (i + 1) < N):
                return False, 0
            # Must split within the same current block
            if block_numbers[i] != block_numbers[i + 1]:
                return False, 0

            # Sum for the left part (ending at i)
            sum1 = 0
            j = i
            while j >= 0:
                if block_numbers[j] != block_numbers[i]:
                    break
                sum1 += A[j]
                j -= 1

            # Create a new block id for the right part and sum it
            block_id += 1
            sum2 = 0
            j = i + 1
            while j < N:
                if block_numbers[j] != block_numbers[i]:
                    break
                sum2 += A[j]
                block_numbers[j] = block_id
                j += 1

            score += sum1 * sum2

        return True, score