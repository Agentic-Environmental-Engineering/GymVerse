from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FaceRightWayEnv(Env):
    """Face Right Way environment (single-turn Q&A) converted to GEM format.

    Task:
    - Given an initial 0/1 array A of length N.
    - Choose a fixed positive integer K.
    - Perform M operations, each operation flips a contiguous subarray [l, l + K - 1].
    - After all operations, all elements of A must become 0.
    - Objectives:
        1) Minimize M.
        2) Among all strategies with minimal M, minimize K.
    - Output:
        - Output M lines, each line contains two integers "l r".
        - All intervals must have the same length K.
        - The answer must be wrapped in \\boxed{...}.
    """

    def __init__(self, N: int, **kwargs):
        """
        Initialize the environment.

        Args:
            N: Length of the array A. Must be >= 4.
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 4, "N should be greater than or equal to 4"
        self.N: int = N

        # State for the current episode
        self.initial_A: Optional[List[int]] = None
        self.reference_answer: Optional[str] = None  # Operations string for an optimal solution
        self.gold_K: Optional[int] = None
        self.gold_M: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a 0/1 array A and must turn all entries to 0 by flipping fixed-length subarrays.\n"
            "Rules:\n"
            "- First, pick a positive integer K; it must remain fixed for all operations.\n"
            "- Each operation chooses an index l (1 ≤ l ≤ N - K + 1) and flips all A[i] with l ≤ i ≤ l + K - 1.\n"
            "- After M operations, all elements of A must be 0.\n"
            "Objectives:\n"
            "1) Minimize M (the total number of operations).\n"
            "2) Among all strategies with minimal M, minimize K.\n\n"
            "Output Format:\n"
            "- Output M lines, each containing two integers l and r (separated by a space), representing the closed interval [l, r] with r = l + K - 1.\n"
            "- All intervals must have the same length K.\n"
            "- Wrap your entire multi-line answer inside \\boxed{...}.\n"
            "Example:\n"
            "\\boxed{\\n1 3\\n4 6\\n}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem instance, and return the observation."""
        super().reset(seed)

        # Generate initial array A by applying random K-length flips to an all-zero array
        N = self.N
        while True:
            A = [0] * N
            K_rand = random.randint(2, N)
            left_endpoints = list(range(0, N - K_rand + 1))
            # Choose a random non-empty subset of endpoints
            k_count = random.randint(1, len(left_endpoints))
            chosen = random.sample(left_endpoints, k=k_count)
            for l in chosen:
                for i in range(l, l + K_rand):
                    A[i] ^= 1
            if any(A):
                break

        self.initial_A = A

        # Compute optimal (M, K) and a reference optimal operations plan
        gold_K, gold_M, ref_ops = self._compute_optimal_plan(A)
        self.gold_K = gold_K
        self.gold_M = gold_M
        self.reference_answer = ref_ops

        # Build problem statement
        A_desc = "; ".join(f"A[{i}]={val}" for i, val in enumerate(A, start=1))
        problem = (
            f"There is a 0/1 array A of length {N}, and initially it is: {A_desc}\n\n"
            "Please do the following:\n"
            "- First, pick a positive integer K, which must remain fixed throughout the process.\n"
            "- Then, perform M operations. In each operation, you choose an index l (1 ≤ l ≤ N - K + 1) and flip all values A[i] with l ≤ i ≤ l + K - 1 (i.e., a contiguous subarray of length K).\n"
            "- Finally, all elements of A must become 0.\n\n"
            "Your goal is:\n"
            "1. Minimize M (the total number of operations).\n"
            "2. Among all strategies with minimal M, minimize K.\n\n"
            "Output Format: Output M lines, each containing two integers l and l + K - 1 (separated by a space), representing the closed interval [l, l + K - 1] flipped in that operation. All intervals must have the same length K.\n"
            "Wrap your multi-line answer in \\boxed{...}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted answer. Single-turn; always terminal."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse operations from boxed content
        try:
            operations = self._parse_operations(boxed_content)
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "parse_error"}

        # Validate and apply operations
        N = self.N
        A = list(self.initial_A) if self.initial_A is not None else []
        if not A or len(A) != N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error"}

        # Check indices and consistent K
        K: Optional[int] = None
        for (l, r) in operations:
            if not (1 <= l <= r <= N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_interval"}
            cur_len = r - l + 1
            if K is None:
                K = cur_len
            elif K != cur_len:
                return TERMINAL_STATE, 0.0, True, False, {"error": "inconsistent_K"}

        # Apply flips
        A_after = A[:]
        for (l, r) in operations:
            for i in range(l - 1, r):
                A_after[i] ^= 1

        success_zeroed = not any(A_after)
        user_M = len(operations)
        user_K = (operations[0][1] - operations[0][0] + 1) if operations else None

        # Correctness criterion: array becomes all zeros AND M == gold_M AND K == gold_K
        is_correct = (
            success_zeroed
            and (self.gold_M is not None and user_M == self.gold_M)
            and (self.gold_K is not None and user_K == self.gold_K)
        )
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "success_zeroed": success_zeroed,
            "user_M": user_M,
            "user_K": user_K,
            "gold_M": self.gold_M,
            "gold_K": self.gold_K,
            "reference_answer": self.reference_answer,
            "parsed_operations": operations,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content, supporting multi-line content."""
        import re
        # This pattern captures the last boxed content; it does not cross braces.
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_operations(self, text: str) -> List[Tuple[int, int]]:
        """Parse operations from the boxed content. Each non-empty line should contain 'l r'."""
        operations: List[Tuple[int, int]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError("Each line must contain exactly two integers")
            l, r = int(parts[0]), int(parts[1])
            operations.append((l, r))
        return operations

    def _compute_optimal_plan(self, A0: List[int]) -> Tuple[int, int, str]:
        """Compute the optimal (K, M) and a corresponding operations plan that zeroes the array.

        Returns:
            (ansK, ansM, operations_string)
        """
        N = len(A0)
        # Default solution with K=1: flip each 1 individually
        ansK = 1
        ansM = sum(A0)
        reference_ops = "\n".join(f"{i} {i}" for i, val in enumerate(A0, start=1) if val)

        # Convert A to 1-indexed for the greedy evaluation over K
        A = [None] + A0[:]  # 1-indexed

        for K in range(1, N + 1):
            flip = [0] * (N + 1)  # flip[i] == 1 if we start a K-flip at position i
            curr = 0  # parity of active flips affecting current position
            m = 0
            possible = True
            current_answer_lines: List[str] = []

            for i in range(1, N + 1):
                # Remove the effect of a flip that ends before i
                if i - K >= 1:
                    curr ^= flip[i - K]

                # If after applying current parity we still see a 1 at i, we need to flip starting at i
                need_flip = A[i] ^ (curr == 1)
                if need_flip:
                    # Cannot start a K-flip if it would exceed bounds
                    if i + K - 1 > N:
                        possible = False
                        break
                    current_answer_lines.append(f"{i} {i + K - 1}")
                    flip[i] = 1
                    curr ^= 1
                    m += 1

            if possible and m < ansM:
                ansM = m
                ansK = K
                reference_ops = "\n".join(current_answer_lines)

        return ansK, ansM, reference_ops

    def sample_random_action(self) -> str:
        """Sample a random action: either the reference optimal answer (if available) or a random guess."""
        if self.reference_answer:
            return f"\\boxed{{\n{self.reference_answer}\n}}"
        # Fallback: sample a random K and a random valid interval count (may be incorrect)
        N = self.N
        K = random.randint(1, N)
        max_l = max(1, N - K + 1)
        m = random.randint(0, max(0, max_l // 2))
        ops = []
        for _ in range(m):
            l = random.randint(1, max_l)
            r = l + K - 1
            ops.append(f"{l} {r}")
        content = "\n".join(ops)
        return f"\\boxed{{\n{content}\n}}"