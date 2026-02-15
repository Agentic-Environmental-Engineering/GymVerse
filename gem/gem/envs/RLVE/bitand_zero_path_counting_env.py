from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BitAndZero_PathCountingEnv(Env):
    """
    Environment for the path counting problem on an infinite directed graph defined by bitwise AND == 0 and s < t.
    Single-turn Q&A style, requiring answers in \\boxed{...} format.
    """

    def __init__(
        self,
        max_length: int = 20,
        modulo: int = 10000,
        ensure_nontrivial: bool = True,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_length: Maximum bit-length (number of bits) for the binary labels of S and T. Must be >= 1.
            modulo: Modulo for the answer.
            ensure_nontrivial: If True, regenerate problems until the answer is not 0 or 1.
        """
        super().__init__()
        assert max_length >= 1, "max_length should be greater than or equal to 1"
        self.max_length = max_length
        self.modulo = modulo
        self.ensure_nontrivial = ensure_nontrivial

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.S_str: Optional[str] = None
        self.T_str: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given a directed graph with an infinite number of vertices labeled by non-negative integers: 0, 1, 2, ...\n"
            "There is a directed edge from vertex s to vertex t if and only if s < t and (s & t) == 0 (bitwise AND).\n"
            "Your task is to compute the number of distinct paths from S to T, modulo a given number.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        while True:
            # Generate random binary strings for S and T (both start with '1')
            len_s = random.randint(1, self.max_length)
            len_t = random.randint(1, self.max_length)
            S = "1" + "".join(str(random.randint(0, 1)) for _ in range(len_s - 1))
            T = "1" + "".join(str(random.randint(0, 1)) for _ in range(len_t - 1))

            # Ensure S <= T under the numeric ordering by length then lexicographic
            if len(S) > len(T) or (len(S) == len(T) and S > T):
                S, T = T, S

            ans = self._compute_reference_answer(S, T, self.modulo)

            if not self.ensure_nontrivial or ans not in (0, 1):
                self.S_str = S
                self.T_str = T
                self.reference_answer = ans
                break

        # Build the problem statement
        self.current_problem = (
            "You are given a directed graph with an infinite number of vertices, where each vertex is labeled with a non-negative integer: 0, 1, 2, ...\n\n"
            "There is a directed edge from vertex s to vertex t if and only if:\n"
            "- s < t, and\n"
            "- s & t = 0 (where & denotes the bitwise AND operation).\n\n"
            f"Please compute the number of distinct paths from vertex {self.S_str} to vertex {self.T_str}. "
            f"Give the result modulo {self.modulo}. Note that the two vertex labels are provided in binary (base-2) representation.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result."""
        # Parse the answer from the \\boxed{...} format
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to integer and validate
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Check range: answer should be within [0, modulo)
        info: dict[str, Any] = {}
        if not (0 <= user_answer < self.modulo):
            info["error"] = "out_of_range"

        # Compare with reference answer
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info.update({
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "S": self.S_str,
            "T": self.T_str,
            "modulo": self.modulo
        })

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...} in the provided text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...}."""
        random_answer = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, S_str: str, T_str: str, mod: int) -> int:
        """
        Compute the number of distinct paths from S to T modulo mod,
        using the original dynamic programming approach from the RLVE environment.
        """
        def Add(a: int, b: int) -> int:
            s = a + b
            return s - mod if s >= mod else s

        def Mult(a: int, b: int) -> int:
            return (a * b) % mod

        S_bits: List[int] = list(map(int, S_str))
        T_bits: List[int] = list(map(int, T_str))

        N, M = len(S_bits), len(T_bits)
        if M > N:
            S_bits = [0] * (M - N) + S_bits
        else:
            # If M <= N, we assume the problem setup guarantees S <= T as strings,
            # and the DP expects both to have the same length.
            # When M < N, in principle we would pad T, but due to the generation logic, it should not happen.
            # If it does, we would pad T to N to match, but keeping consistent with the original code:
            assert M == N

        # Precompute G transitions
        G = [[[0, 0] for _ in range(M)] for __ in range(2)]
        for st in (0, 1):
            G[st][0][st] = 1
            for i in range(1, M):
                G[st][i][0] = Add(G[st][i - 1][0], G[st][i - 1][1])
                G[st][i][1] = G[st][i - 1][0]

        # Find the first position H where S_bits[H-1] == 1 (1-based index)
        H = 1
        while H <= M and S_bits[H - 1] == 0:
            H += 1

        # DP arrays
        F = [[0] * M for _ in range(M + 1)]
        F[1][0] = 1

        for i in range(2, M + 1):
            for x in range(0, i - 1):
                bit = T_bits[i - 1]
                if i <= H:
                    F[i][x + 1] = Add(F[i][x + 1], Mult(F[i - 1][x], G[1][x + 1][bit]))
                if i < H:
                    total = Add(G[0][x][bit], G[1][x][bit])
                    F[i][x] = Add(F[i][x], Mult(F[i - 1][x], total))
                if i > H:
                    F[i][x] = Add(F[i][x], Mult(F[i - 1][x], G[S_bits[i - 1]][x][bit]))

        ans = 0
        for x in range(0, M):
            ans = Add(ans, F[M][x])

        return ans