import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ConcatenationPartitionCountingSumEnv(Env):
    """Single-turn environment for the Concatenation Partition Counting Sum problem."""

    def __init__(
        self,
        N: int = 10,
        M: int = 3,
        max_mod: int = 10000,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Length of the number string S (must be >= 2).
        - M: Transition dimension parameter (must be >= 1).
        - max_mod: Upper bound for the modulo (MOD) random selection, inclusive of upper bound via randint(2, max_mod).

        Raises:
        - AssertionError: If N < 2 or M < 1, or if max_mod < 2.
        """
        super().__init__()
        assert N >= 2, "N should be greater than or equal to 2"
        assert M >= 1, "M should be greater than or equal to 1"
        assert max_mod >= 2, "max_mod should be greater than or equal to 2"

        self.N = N
        self.M = M
        self.max_mod = max_mod

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_S: Optional[str] = None
        self.current_mod: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial dynamic programming problem on string partitions.\n"
            "Please provide your final answer in \\boxed{...} format. The answer should be a single integer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters
        N = self.N
        M = self.M
        MOD = random.randint(2, self.max_mod)
        S = "".join(random.choices("0123456789", k=N))

        # Store for later use
        self.current_S = S
        self.current_mod = MOD

        # Build problem prompt
        self.current_problem = (
            "Define F[n] as follows:\n"
            "- F[0] = 1\n"
            f"- For all n ≥ 1: F[n] = sum(F[n - m] for m in range(1, min(n, {M}) + 1))\n\n"
            f"You are given a number string S: {S}\n"
            "Consider all possible partitions of S into non-empty substrings s[1], s[2], ..., s[k] (for any k ≥ 1), "
            f"such that concatenating s[1] through s[k] gives exactly {S}. Leading zeros are allowed in any s[i].\n"
            "For each such partition, compute the value F[int(s[1]) + int(s[2]) + ... + int(s[k])]. "
            f"Please compute the total sum of this value over all such partitions, modulo {MOD}.\n\n"
            "Output Format: Provide your final answer as a single integer in \\boxed{...}."
        )

        # Compute reference answer via the original algorithm
        self.reference_answer = self._compute_reference_answer(S, M, MOD)

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": N,
            "M": M,
            "MOD": MOD,
            "S": S
        }
        return obs, info

    def _compute_reference_answer(self, S: str, M: int, MOD: int) -> int:
        """Compute the reference answer using the original matrix-based algorithm."""

        class Node:
            def __init__(self, init_zero: bool = True):
                # Initialize an MxM matrix of zeros
                self.a = [[0] * M for _ in range(M)] if init_zero else None

            def init(self) -> None:
                # Companion matrix for transitions: P[0]
                for i in range(M):
                    self.a[i][M - 1] = 1
                for i in range(1, M):
                    self.a[i][i - 1] = 1

            def init1(self) -> None:
                # Identity matrix
                for i in range(M):
                    self.a[i][i] = 1

            def __mul__(self, other: "Node") -> "Node":
                # Matrix multiplication modulo MOD
                z = Node()
                for i in range(M):
                    for k in range(M):
                        val = self.a[i][k]
                        if val == 0:
                            continue
                        row_z = z.a[i]
                        row_o = other.a[k]
                        for j in range(M):
                            row_z[j] = (row_z[j] + val * row_o[j]) % MOD
                return z

            def __add__(self, other: "Node") -> "Node":
                # Matrix addition modulo MOD
                z = Node()
                for i in range(M):
                    for j in range(M):
                        z.a[i][j] = (self.a[i][j] + other.a[i][j]) % MOD
                return z

        def ksm(mat: Node, exp: int) -> Node:
            # Fast exponentiation of matrix mat^exp
            res = Node()
            res.init1()
            base = mat
            e = exp
            while e > 0:
                if e & 1:
                    res = res * base
                base = base * base
                e >>= 1
            return res

        N = len(S)
        digits = [int(ch) for ch in S]

        # Precompute P[i] = P^(10^i)
        P: list[Optional[Node]] = [None] * N
        P[0] = Node()
        P[0].init()
        for i in range(1, N):
            # P[i] = (P[i-1])^10
            P[i] = ksm(P[i - 1], 10)

        # F[i][j]: transition matrix for substring S[i..j]
        F: list[list[Optional[Node]]] = [[None] * N for _ in range(N)]
        for j in range(N):
            for i in range(j, -1, -1):
                d = digits[i]
                if i == j:
                    F[i][j] = ksm(P[0], d)
                else:
                    # F[i][j] = F[i+1][j] * (P[j-i])^d
                    t = ksm(P[j - i], d)
                    F[i][j] = F[i + 1][j] * t

        # DP g: g[k] is matrix for prefix of length k
        g: list[Optional[Node]] = [None] * (N + 1)
        # g[0] = identity
        g[0] = Node()
        g[0].init1()
        for i in range(1, N + 1):
            cur = Node()
            # Sum over previous split points
            for j in range(i):
                cur = cur + (g[j] * F[j][i - 1])
            g[i] = cur

        # Answer: sum of the first row of g[N]
        return sum(g[N].a[0][i] for i in range(M)) % MOD

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step to verify the submitted answer."""
        # Parse the boxed answer
        boxed_answer = self._parse_answer(action)
        if boxed_answer is None:
            # Format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare answer
        try:
            user_answer = int(boxed_answer)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "S": self.current_S,
            "MOD": self.current_mod,
            "M": self.M,
            "N": self.N
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} format."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the expected \\boxed{...} format."""
        mod = self.current_mod if self.current_mod is not None else max(2, self.max_mod)
        random_answer = random.randint(0, mod - 1)
        return f"\\boxed{{{random_answer}}}"