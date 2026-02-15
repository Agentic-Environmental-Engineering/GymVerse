import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RoundRobinEnv(Env):
    """Round-robin matrix construction environment - single turn Q&A.

    Task:
    Construct an N x N matrix with entries in {0, 1, 2} that satisfies:
    1) A[i][i] = 0 for all i.
    2) For all i != j, A[i][j] + A[j][i] = 2. Equivalently, one of:
       - (A[i][j], A[j][i]) = (0, 2)
       - (A[i][j], A[j][i]) = (2, 0)
       - (A[i][j], A[j][i]) = (1, 1)
    3) Define W[i] = 3 * (# of j with A[i][j] = 2) + 1 * (# of j with A[i][j] = 1).
       The final values of W must match the provided targets exactly.

    Answer format:
    Output exactly N lines, each containing N digits (0/1/2) with no separators.
    Wrap the entire matrix in \\boxed{ ... }.
    """

    prompt_template = (
        "Please construct an {N} × {N} matrix, where each element is either 0, 1, or 2. "
        "Denote the matrix as A (0-indexed), and it must satisfy the following conditions:\n"
        "1. A[i][i] = 0 for all i.\n"
        "2. For all i ≠ j (0 ≤ i, j < {N}), A[i][j] + A[j][i] = 2 "
        "(i.e., one of the following holds: A[i][j] = 0 and A[j][i] = 2; A[i][j] = 2 and A[j][i] = 0; or A[i][j] = A[j][i] = 1).\n"
        "3. Define W[i] = 3 × (number of positions j where A[i][j] = 2) + 1 × (number of positions j where A[i][j] = 1). "
        "The final values of W[0], ..., W[{N_minus_1}] must be exactly: {W}\n\n"
        "Output Format: Output {N} lines, each containing {N} digits (0, 1, or 2) with no separators. "
        "The i-th line should represent A[i][0], A[i][1], ..., A[i][{N_minus_1}].\n"
        "Answer Submission: Place the entire matrix inside \\boxed{{...}}."
    )

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 10,
        tie_probability: Optional[float] = None,
        **kwargs: Any
    ):
        """
        Initialize the RoundRobinEnv instance.

        Args:
            N: If provided, use this fixed matrix size (must be >= 3).
            min_N: Minimum N to sample when N is not provided (must be >= 3).
            max_N: Maximum N to sample when N is not provided (must be >= min_N).
            tie_probability: If provided, use this tie probability in generation; otherwise sample randomly in [0, 1).
            **kwargs: Ignored extra keyword arguments for compatibility.
        """
        super().__init__()
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if tie_probability is not None and not (0.0 <= tie_probability <= 1.0):
            raise ValueError("tie_probability must be in [0, 1]")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N
        self.tie_probability: Optional[float] = tie_probability

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.target_W: Optional[List[int]] = None
        self.current_N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a round-robin matrix construction problem.\n"
            "Please produce your final matrix in the following format:\n"
            "- Exactly N lines, each with N digits (0/1/2) and no separators.\n"
            "- Wrap the entire matrix inside \\boxed{ ... }.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        self.current_N = N

        # Generate matrix A and W
        tp = self.tie_probability if self.tie_probability is not None else random.random()
        A: List[List[int]] = [[0] * N for _ in range(N)]
        W: List[int] = [0] * N

        for i in range(N):
            for j in range(N):
                if i == j:
                    A[i][j] = 0
                elif i < j:
                    if random.random() < tp:
                        A[i][j] = 1
                    else:
                        A[i][j] = random.choice([0, 2])
                else:
                    # Enforce A[i][j] + A[j][i] = 2
                    A[i][j] = 2 - A[j][i]
                # Update W[i]
                W[i] += 3 * (A[i][j] == 2) + 1 * (A[i][j] == 1)

        # Prepare reference answer and target W
        self.reference_answer = "\n".join("".join(str(x) for x in row) for row in A)
        self.target_W = W[:]

        # Build problem statement
        W_str = " ".join(f"W[{i}]={Wi}" for i, Wi in enumerate(W))
        problem_text = self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            W=W_str,
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted matrix."""
        if self.current_N is None or self.target_W is None:
            # Environment was not properly reset
            info = {"error": "environment_not_initialized"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Process matrix lines
        lines = [line.strip() for line in boxed_content.splitlines() if line.strip()]
        N = self.current_N

        # Basic format checks
        if len(lines) != N:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "line_count_mismatch"}
        if any(len(row) != N for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "row_length_mismatch"}
        if any(any(c not in "012" for c in row) for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "invalid_characters"}

        # Validate constraints and compute W_user
        W_user = [0] * N
        try:
            for i in range(N):
                for j in range(N):
                    cij = int(lines[i][j])
                    if i == j:
                        if cij != 0:
                            info = {"correct": False, "reason": "diagonal_not_zero"}
                            return TERMINAL_STATE, 0.0, True, False, info
                    else:
                        cji = int(lines[j][i])
                        if cij + cji != 2:
                            info = {"correct": False, "reason": "pair_sum_constraint_violated", "i": i, "j": j}
                            return TERMINAL_STATE, 0.0, True, False, info
                    W_user[i] += 3 * (cij == 2) + 1 * (cij == 1)
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "parse_failure"}

        # Check W correctness
        is_correct = (W_user == self.target_W)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "target_W": self.target_W,
            "user_W": W_user,
            "reference_answer": self.reference_answer,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...}, supporting multiline content."""
        pattern = r'\\boxed\{([\s\S]+?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random (likely incorrect) action by generating a random matrix in the required boxed format."""
        if self.current_N is None:
            # If not reset yet, choose a random reasonable N
            N = self.fixed_N if self.fixed_N is not None else max(3, self.min_N)
        else:
            N = self.current_N

        matrix = []
        for _ in range(N):
            row = "".join(str(random.choice([0, 1, 2])) for _ in range(N))
            matrix.append(row)
        content = "\n".join(matrix)
        return f"\\boxed{{\n{content}\n}}"