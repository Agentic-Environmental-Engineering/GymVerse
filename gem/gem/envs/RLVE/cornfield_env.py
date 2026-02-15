import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CornfieldEnv(Env):
    """Cornfield environment: choose up to K range-increment operations to maximize the LNDS length."""

    prompt_template = (
        "You are given an array H of length {N}. The initial values of the array are: {H}\n"
        "You may perform at most {K} operations. In each operation, you choose an interval [L, R] "
        "(0 ≤ L ≤ R < {N}), and increment each element H[i] by 1 for all i in the range L ≤ i ≤ R. "
        "Try your best to maximize the length of the longest non-decreasing subsequence (not necessarily "
        "contiguous) in the final array after performing the operations.\n\n"
        "Output Format: Provide your chosen operations wrapped in a single \\boxed{{...}} block. "
        "Inside the box, write at most {K} lines. Each non-empty line should contain two integers L and R "
        "(0-indexed), separated by a space, indicating an interval you chose for an operation. "
        "If you choose to perform zero operations, submit \\boxed{{}}."
    )

    def __init__(self, N: int = 10):
        """
        Initialize the CornfieldEnv instance.

        Args:
            N: Length of the array H to generate. Must be >= 3.
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N

        # Per-episode state
        self.H: List[int] = []
        self.K: int = 0
        self.gold_answer: int = 0
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions for the agent."""
        return (
            "Task: Plan up to K range-increment operations to maximize the length of the longest "
            "non-decreasing subsequence (LNDS) of the final array.\n"
            "Answer format: Place your operations inside a single \\boxed{...} block. Write at most K lines, "
            "each line is two integers 'L R' (0-indexed). If you choose zero operations, submit \\boxed{}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Args:
            seed: Optional seed for randomness.

        Returns:
            observation: The instructions followed by the problem description.
            info: Auxiliary information dict (empty for this environment).
        """
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # Generate H with values in [1, 2N]
        self.H = [random.randint(1, 2 * self.N) for _ in range(self.N)]

        # Determine K per original logic:
        # K in [1, max(1, min(N, sum(max(H[i-1]-H[i], 0))))]
        descent_sum = sum(max(self.H[i - 1] - self.H[i], 0) for i in range(1, self.N))
        self.K = random.randint(1, max(1, min(self.N, descent_sum)))

        # Compute gold answer (maximum achievable LNDS) using 2-D BIT approach
        self.gold_answer = self._compute_gold(self.H, self.K)

        # Build problem statement
        H_str = " ".join(f"H[{i}]={val}" for i, val in enumerate(self.H))
        self.current_problem = self.prompt_template.format(N=self.N, K=self.K, H=H_str)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step by validating the user's proposed operations.

        Args:
            action: The textual response from the agent, which must contain a \\boxed{...} block.

        Returns:
            observation: TERMINAL_STATE (single-turn environment).
            reward: 1.0 if the plan reaches the gold LNDS; 0.0 otherwise; -0.1 for format errors.
            terminated: True (single-turn environment).
            truncated: False (no truncation in this environment).
            info: Additional information such as parsed operations and reference answer.
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse operations from boxed content
        parse_ok, operations_or_error = self._process_boxed_operations(boxed_content)
        if not parse_ok:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "details": operations_or_error}

        operations: List[Tuple[int, int]] = operations_or_error

        # Validate operation constraints
        if len(operations) > self.K:
            info = {
                "error": "invalid_solution",
                "reason": "too_many_operations",
                "num_operations": len(operations),
                "K": self.K,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        for (L, R) in operations:
            if not (0 <= L <= R < self.N):
                info = {
                    "error": "invalid_solution",
                    "reason": "out_of_bounds_interval",
                    "interval": (L, R),
                    "N": self.N,
                }
                return TERMINAL_STATE, 0.0, True, False, info

        # Apply operations via difference array
        delta = [0] * self.N
        for (L, R) in operations:
            delta[L] += 1
            if R + 1 < self.N:
                delta[R + 1] -= 1

        H_final = self.H.copy()
        for i in range(self.N):
            if i > 0:
                delta[i] += delta[i - 1]
            H_final[i] += delta[i]

        # Compute user's LNDS length (O(N^2) DP for non-decreasing)
        F = [0] * self.N
        for i in range(self.N):
            F[i] = 1
            for j in range(i):
                if H_final[j] <= H_final[i]:
                    F[i] = max(F[i], F[j] + 1)
        user_answer = max(F) if F else 0

        # Compare with gold
        is_correct = (user_answer == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.gold_answer,
            "user_answer": user_answer,
            "K": self.K,
            "num_operations": len(operations),
            "operations": operations,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract content inside the last \\boxed{...} block.

        Args:
            text: The raw agent response.

        Returns:
            The content inside the last \\boxed{...}, or None if not found.
        """
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process_boxed_operations(self, content: str) -> Tuple[bool, Any]:
        """
        Parse operations from boxed content.

        Args:
            content: The string inside \\boxed{...}.

        Returns:
            (True, operations) on success, where operations is a list of (L, R).
            (False, error_message) on parse failure.
        """
        content = content.strip()
        if content == "":
            return True, []

        operations: List[Tuple[int, int]] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                # Skip empty lines
                continue
            parts = line.split()
            if len(parts) != 2:
                return False, f"invalid_line_format: '{raw_line}'"
            try:
                L = int(parts[0])
                R = int(parts[1])
            except ValueError:
                return False, f"non_integer_values: '{raw_line}'"
            operations.append((L, R))
        return True, operations

    def _compute_gold(self, H: List[int], K: int) -> int:
        """
        Compute the maximum achievable LNDS length after at most K range-increment operations,
        using a 2-D Fenwick Tree (BIT) technique.

        Args:
            H: The initial heights array.
            K: Maximum number of operations.

        Returns:
            The gold (optimal) LNDS length.
        """
        def lowbit(x: int) -> int:
            return x & -x

        def add(bit: List[List[int]], X: int, Y: int, x: int, y: int, value: int) -> None:
            while x <= X:
                yy = y
                row = bit[x]
                while yy <= Y:
                    if value > row[yy]:
                        row[yy] = value
                    yy += lowbit(yy)
                x += lowbit(x)

        def query(bit: List[List[int]], x: int, y: int) -> int:
            res = 0
            while x:
                yy = y
                row = bit[x]
                while yy:
                    v = row[yy]
                    if v > res:
                        res = v
                    yy -= lowbit(yy)
                x -= lowbit(x)
            return res

        max_height = max(H) if H else 0
        X = K + 1
        Y = max_height + K

        # 2-D BIT initialized with 0 (1-based indexing)
        BIT = [[0] * (Y + 2) for _ in range(X + 2)]

        answer = 0
        for h in H:
            for j in range(K, -1, -1):
                cur_height = h + j
                best = query(BIT, j + 1, cur_height) + 1
                if best > answer:
                    answer = best
                add(BIT, X, Y, j + 1, cur_height, best)
        return answer

    def sample_random_action(self) -> str:
        """
        Sample a random valid action: pick a random number of operations and random intervals.

        Returns:
            A string containing the random action wrapped in \\boxed{...}.
        """
        if not self.H or self.K is None:
            # If called before reset, produce an empty plan.
            return "\\boxed{}"

        num_ops = random.randint(0, self.K)
        ops: List[Tuple[int, int]] = []
        for _ in range(num_ops):
            L = random.randint(0, self.N - 1)
            R = random.randint(L, self.N - 1)
            ops.append((L, R))
        inside = "\n".join(f"{L} {R}" for (L, R) in ops)
        return f"\\boxed{{{inside}}}"