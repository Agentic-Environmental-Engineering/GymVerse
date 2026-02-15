import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from collections import defaultdict

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinSwapTwoPermutationsEnv(Env):
    """
    Environment for the "Minimum Swaps Between Two Arrays" problem.

    Task:
    - You are given two arrays A and B of the same length N.
    - You may choose a set of indices i1, i2, ..., ik and swap A[j] with B[j] for each chosen index j.
    - Your goal is to perform the minimum number of swaps such that both A and B contain no duplicate elements.
    - Provide the indices in 0-based indexing.

    Answer Format:
    - The answer must be provided in \\boxed{...} format.
    - Inside the box, provide the chosen indices separated by spaces or commas (e.g., \\boxed{0 2 5}).

    Rewards:
    - Correct answer (indices lead to no duplicates in both arrays and the count equals the minimal number): 1.0
    - Wrong answer: 0.0
    - Format error (cannot parse \\boxed{...} or invalid content): -0.1
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 20,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Optional fixed problem size. If None, N will be randomly chosen in [min_N, max_N].
        - min_N: Minimum allowed size for N (must be >= 3).
        - max_N: Maximum allowed size for N.
        - kwargs: Ignored, included for compatibility with original parameterization.
        """
        super().__init__()
        assert min_N >= 3, "min_N should be greater than or equal to 3"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"
        if N is not None:
            assert N >= 3, "N should be greater than or equal to 3"

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # State variables
        self.N: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.B: Optional[List[int]] = None
        self.current_problem: Optional[str] = None
        self.gold_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given two arrays A and B of equal length N.\n"
            "You may swap A[i] with B[i] for any chosen indices i.\n"
            "Your task is to choose the minimum number of indices so that after swapping at those indices, "
            "both A and B contain no duplicate elements.\n\n"
            "Answer Format:\n"
            "- Provide your chosen indices (0-based) inside \\boxed{...}.\n"
            "- Indices should be separated by spaces or commas, e.g., \\boxed{0 2 5}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Choose N
        if self.fixed_N is not None:
            self.N = self.fixed_N
        else:
            self.N = random.randint(self.min_N, self.max_N)
        assert self.N is not None and self.N >= 3

        # Generate arrays A and B with values 1..N
        A = list(range(1, self.N + 1))
        B = list(range(1, self.N + 1))

        # Shuffle and perform random index-wise swaps until at least one array has duplicates
        while True:
            random.shuffle(A)
            random.shuffle(B)
            num_swaps = random.randint(1, self.N - 1)
            swapped_indices = random.sample(range(self.N), num_swaps)
            A_copy = A[:]
            B_copy = B[:]
            for idx in swapped_indices:
                A_copy[idx], B_copy[idx] = B_copy[idx], A_copy[idx]
            # Exit when the configuration has at least one array with duplicates
            if not (len(set(A_copy)) == self.N and len(set(B_copy)) == self.N):
                A, B = A_copy, B_copy
                break

        self.A = A
        self.B = B

        # Compute minimal number of swaps needed using the original graph/parity algorithm
        self.gold_answer = self._compute_min_swaps(self.A, self.B)
        assert self.gold_answer is not None and 0 < self.gold_answer <= len(swapped_indices)

        # Build problem text
        A_repr = " ".join(f"A[{i}]={val}" for i, val in enumerate(self.A))
        B_repr = " ".join(f"B[{i}]={val}" for i, val in enumerate(self.B))
        self.current_problem = (
            f"You are given two arrays A and B of length {self.N}. Initially:\n"
            f"- A = {A_repr}\n"
            f"- B = {B_repr}\n\n"
            "Your task is to find the minimum number of indices i1, i2, ..., ik such that, after swapping "
            "A[i1] with B[i1], A[i2] with B[i2], ..., A[ik] with B[ik], both A and B contain no duplicate elements.\n"
            "Please output a single line containing the indices i1, ..., ik, separated by spaces, inside \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {
            "N": self.N,
            "A": self.A[:],
            "B": self.B[:],
            "gold_answer": self.gold_answer
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step: parse the answer, verify, and return the result."""
        # Parse indices from \\boxed{...}
        indices = self._parse_answer(action)

        if indices is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        assert self.A is not None and self.B is not None and self.N is not None and self.gold_answer is not None

        # Validate indices range
        for idx in indices:
            if not (0 <= idx < self.N):
                info = {
                    "error": "invalid_index",
                    "invalid_index": idx,
                    "N": self.N
                }
                return TERMINAL_STATE, 0.0, True, False, info

        # Apply swaps on copies
        A_copy = self.A[:]
        B_copy = self.B[:]
        for idx in indices:
            A_copy[idx], B_copy[idx] = B_copy[idx], A_copy[idx]

        # Check for no duplicates in both arrays
        success = (len(set(A_copy)) == self.N and len(set(B_copy)) == self.N)

        # Check minimality: length equals gold_answer
        is_minimal = (len(indices) == self.gold_answer)

        is_correct = success and is_minimal
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "success_no_duplicates": success,
            "is_minimal": is_minimal,
            "provided_indices": indices,
            "gold_answer": self.gold_answer,
            "N": self.N
        }

        return TERMINAL_STATE, reward, True, False, info

    def _compute_min_swaps(self, A: List[int], B: List[int]) -> int:
        """
        Compute the minimal number of swaps needed so that both arrays have no duplicates,
        using the original parity-based DFS on a graph constructed from mismatch positions.
        """
        N = len(A)
        # Map value to list of positions where A[i] != B[i] and the value appears in these mismatches
        p: Dict[int, List[int]] = defaultdict(list)
        for i in range(N):
            if A[i] != B[i]:
                p[A[i]].append(i)
                p[B[i]].append(i)

        # Build graph on positions 0..N-1 with edge weights 0 or 1
        graph: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
        for val, occ in p.items():
            if len(occ) == 2:
                u, v = occ
                w = 1 if (A[u] == A[v] or B[u] == B[v]) else 0
                graph[u].append((v, w))
                graph[v].append((u, w))

        visited = [False] * N
        ans = 0

        # Parity-based DFS on each connected component
        for i in range(N):
            if not visited[i]:
                stack: List[Tuple[int, int]] = [(i, 0)]
                cnt = [0, 0]
                while stack:
                    u, parity = stack.pop()
                    if visited[u]:
                        continue
                    visited[u] = True
                    cnt[parity] += 1
                    for v, w in graph[u]:
                        if not visited[v]:
                            stack.append((v, parity ^ w))
                ans += min(cnt)

        assert ans >= 0
        return ans

    def _parse_answer(self, text: str) -> Optional[List[int]]:
        """
        Extract indices from \\boxed{...} format.
        The content inside the box should be integers separated by spaces or commas.
        """
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if not matches:
            return None

        content = matches[-1].strip()
        if not content:
            # Empty content inside box
            return []

        # Split by spaces or commas
        tokens = [tok for tok in re.split(r'[,\s]+', content) if tok]
        indices: List[int] = []
        try:
            for tok in tokens:
                indices.append(int(tok))
        except ValueError:
            return None

        return indices

    def sample_random_action(self) -> str:
        """Sample a random action: a random set of indices inside \\boxed{...}."""
        if self.N is None:
            # Default sample if reset was not called
            return "\\boxed{}"

        k = random.randint(1, max(1, self.N // 2))
        indices = sorted(random.sample(range(self.N), k))
        content = " ".join(str(i) for i in indices)
        return f"\\boxed{{{content}}}"