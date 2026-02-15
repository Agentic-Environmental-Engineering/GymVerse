from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BitwiseOperationSequenceCountingEnv(Env):
    """Single-turn environment for counting operation sequences of AND/OR between binary strings."""

    def __init__(
        self,
        max_n_m: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n_m: Maximum value for both N (number of operations) and M (length of binary strings).
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        self.max_n_m = max_n_m

        # State variables
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.A: Optional[List[str]] = None
        self.R: Optional[str] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorics problem on bitwise operations over binary strings.\n"
            "Task: Insert an operation (AND or OR) between every pair of adjacent elements in A, evaluate left to right,\n"
            "and count the number of different operation sequences that result in the given target binary string R.\n"
            "Output Format: Your final answer must be a single integer enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameters
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)

        # Build A and generate a target R by simulating random AND/OR operations
        A = [None] * (N + 1)
        A[0] = "0" * M
        result = "0" * M
        AND_probability = random.random()

        for i in range(1, N + 1):
            one_probability = random.random()
            s = "".join(str(int(random.random() < one_probability)) for _ in range(M))
            A[i] = s
            operation = "AND" if random.random() < AND_probability else "OR"
            if operation == "AND":
                result = "".join(str(int(A[i][j]) & int(result[j])) for j in range(M))
            else:
                result = "".join(str(int(A[i][j]) | int(result[j])) for j in range(M))

        # Prepare matrix columns and stable partition order
        S = A[1:]
        rk = list(range(M))
        b = [[0] * N for _ in range(M)]

        for i in range(N):
            row_bits = [int(ch) for ch in S[i]]
            for j in range(M):
                b[j][i] = row_bits[j]

            # Stable partition: zeros first, then ones, preserving relative order
            new_rk: List[int] = []
            for k in rk:
                if row_bits[k] == 0:
                    new_rk.append(k)
            for k in rk:
                if row_bits[k] == 1:
                    new_rk.append(k)
            rk = new_rk

        # Compute column values as integers (most significant bit is the top row)
        Ans = [0] * M
        for j in range(M):
            val = 0
            for i in range(N - 1, -1, -1):
                val = val * 2 + b[j][i]
            Ans[j] = val

        # Compute the number of valid operation sequences
        def compute_count() -> int:
            s = result
            # First position in rk where s bit is '1'
            Rk_idx = M
            for idx in range(M):
                if s[rk[idx]] == '1':
                    Rk_idx = idx
                    break

            # Last position in rk where s bit is '0'
            Lk_idx = -1
            for idx in range(M - 1, -1, -1):
                if s[rk[idx]] == '0':
                    Lk_idx = idx
                    break

            if Rk_idx < Lk_idx:
                return 0
            x_val = 0 if Lk_idx == -1 else Ans[rk[Lk_idx]]
            y_val = (2 ** N) if Rk_idx == M else Ans[rk[Rk_idx]]
            return y_val - x_val

        ref_ans = compute_count()
        assert ref_ans > 0, "Generated instance must have at least one valid operation sequence."

        # Store state
        self.N = N
        self.M = M
        self.A = A
        self.R = result
        self.reference_answer = ref_ans

        # Build the problem statement
        problem = (
            f"You are given an array A of {N} + 1 binary strings, each of length {M}. The strings are:\n"
            + "\n".join(f"A[{i}]={Ai}" for i, Ai in enumerate(A))
            + "\n\n"
            f"You will insert an operation (`AND` or `OR`) between every pair of adjacent elements in A, resulting in {N} operations total, "
            f"to form an expression. You must evaluate the expression from left to right (without operator precedence).\n"
            f"Count the number of different ways to insert these operations such that the final result equals this binary string: {result}\n\n"
            f"Output Format: Provide your final integer answer in \\boxed{{...}}."
        )

        self.current_problem = problem
        obs = self._get_instructions() + problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_text)
            is_correct = (user_answer == self.reference_answer)
            reward = 1.0 if is_correct else 0.0
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "M": self.M,
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
        """Sample a random boxed integer answer."""
        # Use a simple range guess; not necessarily correct
        guess = random.randint(0, max(1, (2 ** (self.N or 2)) - 1))
        return f"\\boxed{{{guess}}}"