import math
import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CleaningUpEnv(Env):
    """Environment for the 'Cleaning Up' partitioning problem - single-turn Q&A.

    You are given N numbers A[1..N]. You may partition them into consecutive non-empty batches.
    For each batch, define K as the number of distinct values in that batch. The total cost is
    the sum over all batches of K^2. The task is to find a partition that minimizes the total cost.
    The model should output the batch endpoints (last indices of each batch) as a space-separated
    list inside \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 100,
        **kwargs
    ):
        """
        Initialize the CleaningUpEnv instance.

        Parameters:
        - N: If provided, use this fixed N (must be >= 4). If None, N will be sampled in [min_N, max_N].
        - min_N: Minimum N to sample when N is None (must be >= 4).
        - max_N: Maximum N to sample when N is None (must be >= min_N).
        """
        super().__init__()
        if N is not None and N < 4:
            raise ValueError("N should be greater than or equal to 4")
        if min_N < 4:
            raise ValueError("min_N should be greater than or equal to 4")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N

        self.current_problem: Optional[str] = None
        self.reference_answer_cost: Optional[int] = None
        self.N: Optional[int] = None
        self.A: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task description and answer format instructions."""
        return (
            "You are given N numbers A[1..N]. You may partition them into consecutive non-empty batches.\n"
            "For each batch, define K as the number of distinct values in that batch. The total cost is "
            "the sum over all batches of K^2. Find a partition that minimizes the total cost.\n\n"
            "Output Format:\n"
            "Return the batch endpoints (last indices of each batch) as a space-separated list in \\boxed{...}.\n"
            "For example, \\boxed{3 7 N} means there are batches ending at indices 3, 7, and N.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 4:
            raise ValueError("N should be greater than or equal to 4")

        # Generate A and compute reference minimal cost
        A, gold_cost = self._generate_instance(N)

        # Build problem prompt
        problem_text = (
            f"You are given {N} numbers A[1], A[2], ..., A[{N}]. The values are: "
            + ", ".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A, start=1))
            + "\n"
            "You may divide these numbers (in order) into consecutive non-empty batches. "
            "Let the total number of batches be k, and let end[1], end[2], ..., end[k] "
            f"(1 ≤ end[1] < end[2] < ... < end[k] = {N}) denote the last index of each batch.\n"
            "- Batch 1 contains A[1] to A[end[1]]\n"
            "- Batch 2 contains A[end[1] + 1] to A[end[2]]\n"
            "- ...\n"
            "- Batch k contains A[end[k − 1] + 1] to A[end[k]] (with end[k] = {N})\n\n"
            "Define the cost of a division as follows:\n"
            "- For each batch i (1 <= i <= k), let K[i] be the number of distinct values in that batch.\n"
            "- The total cost is the sum of K[i]^2 over all batches.\n\n"
            "Can you find a division that minimizes the total cost?\n\n"
            "Output Format:\n"
            f"Output a single line: end[1] end[2] ... end[k] (space-separated, with end[k] = {N}). "
            "Your final answer must be provided in \\boxed{...} format."
        )

        self.current_problem = problem_text
        self.reference_answer_cost = gold_cost
        self.N = N
        self.A = A

        observation = self._get_instructions() + self.current_problem
        return observation, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and validate the answer, compute reward, and terminate."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error: cannot find boxed content
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse endpoints from boxed content
        ends: Optional[List[int]] = self._process_endpoints(boxed_content)
        if ends is None:
            # Invalid answer format (not a list of integers)
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate endpoints and compute cost
        if self.N is None or self.A is None or self.reference_answer_cost is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        valid, user_cost = self._validate_and_score(ends, self.A, self.N)
        if not valid:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        is_correct = (user_cost == self.reference_answer_cost)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_cost": self.reference_answer_cost,
            "user_cost": user_cost,
            "endpoints": ends,
            "N": self.N,
        }

        return TERMINAL_STATE, reward, True, False, info

    def sample_random_action(self) -> str:
        """Sample a random valid action (random partition) in boxed format."""
        if self.N is None:
            # Fallback: sample random N
            N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
        else:
            N = self.N

        # Create random endpoints: choose any number of cuts between 1 and N-1
        k = random.randint(1, max(1, N - 1))
        ends = sorted(random.sample(range(1, N), k=k)) + [N]
        return "\\boxed{" + " ".join(str(x) for x in ends) + "}"

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process_endpoints(self, content: str) -> Optional[List[int]]:
        """Process the boxed content into a list of integer endpoints."""
        try:
            tokens = content.strip().split()
            if not tokens:
                return None
            ends = [int(tok) for tok in tokens]
            return ends
        except ValueError:
            return None

    def _validate_and_score(self, ends: List[int], A: List[int], N: int) -> Tuple[bool, Optional[int]]:
        """Validate the endpoints and compute the partition cost if valid."""
        if not (1 <= len(ends) <= N):
            return False, None
        for i in range(len(ends)):
            if not (1 <= ends[i] <= N):
                return False, None
            if i and not (ends[i - 1] < ends[i]):
                return False, None
        if ends[-1] != N:
            return False, None

        # Compute the cost
        extended_A = [None] + A  # 1-indexed alignment
        total_cost = 0
        last = 0
        for end in ends:
            K = len(set(extended_A[last + 1 : end + 1]))
            total_cost += K ** 2
            last = end
        return True, total_cost

    def _generate_instance(self, N: int) -> Tuple[List[int], int]:
        """Generate a random instance A and compute the reference minimal cost."""
        # Loop until instance meets the filtering criteria
        while True:
            # Randomly choose endpoints and convert to segment lengths
            endpoints = random.sample(range(1, N), k=random.randint(1, N - 1))
            endpoints.sort()
            endpoints += [N]
            for i in range(len(endpoints) - 1, 0, -1):
                endpoints[i] -= endpoints[i - 1]

            # Build A: for each segment length x, choose up to sqrt(x) distinct numbers
            A: List[int] = []
            for x in endpoints:
                if x <= 0:
                    # Should not happen, but protect against invalid generation
                    continue
                number_range_size = 1
                while (number_range_size + 1) * (number_range_size + 1) <= x:
                    number_range_size += 1
                number_pool = random.sample(range(1, N + 1), k=number_range_size)
                A.extend([random.choice(number_pool) for _ in range(x)])
            assert len(A) == N

            # Compute reference minimal cost using the DP with move-to-front heuristic
            gold_cost = self._compute_gold_cost(A)

            # Filter to ensure a non-trivial instance
            if gold_cost > 0 and gold_cost < min(N, len(set(A)) ** 2):
                return A, gold_cost

    def _compute_gold_cost(self, A: List[int]) -> int:
        """Compute the minimal cost using the DP algorithm from the original environment."""
        N = len(A)

        # Build preferences P (1-indexed) with sentinel at position 0
        P = [0] * (N + 1)
        for i in range(1, N + 1):
            P[i] = A[i - 1]

        k = int(math.isqrt(N))  # sqrt(N)

        # Move-to-front list of last occurrences for up to k+1 distinct values
        last = [-1] * (k + 2)  # +2 to be safe
        last[0] = 0

        # DP: f[i] = minimal total cost for first i elements
        f: List[Optional[int]] = [None] * (N + 1)
        f[0] = 0

        for i in range(1, N + 1):
            x = P[i]

            # Find position j in move-to-front list for current type (or insertion point)
            j = 0
            while j <= k and last[j] != -1 and P[last[j]] != x:
                j += 1

            # Move-to-front: shift [0..j-1] right by one, put i at front
            while j > 0:
                last[j] = last[j - 1]
                j -= 1
            last[0] = i

            # Transition: consider segments ending at i with up to k distinct values
            best: Optional[int] = None
            j = 1
            while j <= k and last[j] != -1:
                prev = f[last[j]]
                cand = None if prev is None else prev + j * j
                if best is None or (cand is not None and cand < best):
                    best = cand
                j += 1

            f[i] = best

        assert f[N] is not None
        return int(f[N])