from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class StringPartitionShuffleEnv(Env):
    """String Partition Shuffle environment - single-turn Q&A.

    Task:
    - Given a binary string S of length N and a target string T produced by shuffling
      a partition of S into K disjoint contiguous intervals, find K intervals [L[i], R[i])
      such that concatenating S[L[i]:R[i]] in order yields T. Intervals must be non-empty,
      disjoint, and cover the entire range [0, N).
    - The answer must be placed inside \\boxed{...} with exactly K lines, each containing two integers: L R.
    """

    prompt_template = (
        "You are given a string S of length {N} (0-indexed): {S}\n\n"
        "Please find {K} intervals [L[1], R[1]), ..., [L[{K}], R[{K}]) such that:\n"
        "- Each interval [L[i], R[i]) is non-empty and disjoint.\n"
        "- The intervals together cover the entire string S (each index appears in exactly one interval).\n"
        "- Concatenating all substrings S[L[i]: R[i]] (= S[L[i]] + S[L[i] + 1] + ... + S[R[i] - 1]) (in order) "
        "yields a new string T: {T}\n\n"
        "Output Format: Output {K} lines. The i-th line should contain two integers L[i] and R[i], "
        "separated by a space. Your entire answer must be inside \\boxed{{...}}."
    )

    def __init__(
        self,
        n: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 100,
        fixed_k: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the StringPartitionShuffleEnv.

        Parameters:
        - n: If provided, fixes the length of S to n (must be >= 3).
        - min_n: Minimum N when sampling randomly (must be >= 3).
        - max_n: Maximum N when sampling randomly (must be >= min_n).
        - fixed_k: If provided, fixes K (must satisfy 2 <= K <= N-1).
        """
        super().__init__()
        if min_n < 3:
            raise ValueError("min_n must be >= 3")
        if max_n < min_n:
            raise ValueError("max_n must be >= min_n")
        if n is not None and n < 3:
            raise ValueError("n must be >= 3 when provided")

        self.n = n
        self.min_n = min_n
        self.max_n = max_n
        self.fixed_k = fixed_k

        # Problem state
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.S: Optional[str] = None
        self.T: Optional[str] = None
        self.reference_intervals: Optional[List[Tuple[int, int]]] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the general task instructions."""
        return (
            "Task: Partition the string S into K disjoint, non-empty contiguous intervals that cover S exactly, "
            "so that when concatenated in the given order, the result is T.\n"
            "Answer format: Place exactly K lines inside \\boxed{...}. Each line contains two integers: L R.\n"
            "Indices are 0-based and intervals are half-open [L, R).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        N = self.n if self.n is not None else random.randint(self.min_n, self.max_n)
        if N < 3:
            raise ValueError("N must be >= 3")
        self.N = N

        # Determine K
        if self.fixed_k is not None:
            if not (2 <= self.fixed_k <= N - 1):
                raise ValueError("fixed_k must satisfy 2 <= fixed_k <= N - 1")
            K = self.fixed_k
        else:
            if N >= 4 and random.random() < 0.5:
                K = 3
            else:
                K = random.randint(2, N - 1)
        self.K = K

        # Generate S (binary string with random bias)
        one_probability = random.uniform(0.1, 0.9)
        S = "".join("1" if random.random() < one_probability else "0" for _ in range(N))
        self.S = S

        # Create a random partition of [0, N) into K intervals, then shuffle intervals
        endpoints = random.sample(range(1, N), K - 1)
        endpoints.sort()
        endpoints = [0] + endpoints + [N]
        intervals = [(endpoints[i], endpoints[i + 1]) for i in range(K)]
        random.shuffle(intervals)
        self.reference_intervals = intervals

        # Compute T by concatenating substrings in shuffled order
        T = "".join(S[L:R] for (L, R) in intervals)
        self.T = T

        # Build the problem statement
        self.current_problem = self.prompt_template.format(N=N, K=K, S=S, T=T)
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and return the result.

        Returns:
        - TERMINAL_STATE as observation (single-turn).
        - Reward: 1.0 if correct; 0.0 if wrong; -0.1 for format errors.
        - terminated: True (single-turn).
        - truncated: False.
        - info: details about correctness or error type.
        """
        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse intervals matrix from boxed content
        matrix = self._parse_intervals_matrix(boxed_content)
        if matrix is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate structure
        if self.K is None or self.N is None or self.S is None or self.T is None:
            # Environment was not properly reset
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_state_error"}

        if len(matrix) != self.K:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if not all(len(row) == 2 for row in matrix):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate intervals
        try:
            intervals: List[Tuple[int, int]] = [(int(L), int(R)) for L, R in matrix]
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if not all(0 <= L < R <= self.N for (L, R) in intervals):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        if sum(R - L for (L, R) in intervals) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        covered_indices = {i for (L, R) in intervals for i in range(L, R)}
        if covered_indices != set(range(self.N)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        # Construct T' based on user's intervals
        T_user = "".join(self.S[L:R] for (L, R) in intervals)
        is_correct = (T_user == self.T)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "user_intervals": intervals,
            "reference_intervals": self.reference_intervals,
            "S": self.S,
            "T": self.T,
            "N": self.N,
            "K": self.K,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...}. Supports multiline content."""
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_intervals_matrix(self, content: str) -> Optional[List[List[int]]]:
        """Parse a list of [L, R] from the boxed content.

        Returns None if the format is invalid (e.g., non-integer tokens or wrong structure).
        """
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return None
        matrix: List[List[int]] = []
        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                return None
            try:
                L = int(parts[0])
                R = int(parts[1])
            except ValueError:
                # Not integers -> treat as invalid answer rather than format error?
                # According to spec, use 0.0 for wrong answer, but we signal via step.
                return None
            matrix.append([L, R])
        return matrix

    def sample_random_action(self) -> str:
        """Return a random (often correct) action formatted inside \\boxed{...}.

        This uses the internal reference intervals to construct a valid correct answer
        if available; otherwise returns a random plausible partition.
        """
        if self.reference_intervals is not None and self.K is not None:
            answer_lines = [f"{L} {R}" for (L, R) in self.reference_intervals]
        else:
            # Fallback: construct a trivial partition [0, N) if K == 1 is not allowed,
            # so we produce a random valid partition into K parts if possible.
            if self.N is None:
                return r"\boxed{}"
            K = self.K if self.K is not None else min(3, max(2, self.N - 1))
            cut_points = sorted(random.sample(range(1, self.N), K - 1))
            endpoints = [0] + cut_points + [self.N]
            intervals = [(endpoints[i], endpoints[i + 1]) for i in range(K)]
            answer_lines = [f"{L} {R}" for (L, R) in intervals]
        return "\\boxed{" + "\n".join(answer_lines) + "}"