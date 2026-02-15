import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinSumPreXorEnv(Env):
    """
    Single-turn environment for the Min-Sum Prefix XOR array completion task.

    Task:
    - You are given an array P of length N with some entries equal to -1.
    - Replace each -1 with a non-negative integer (all other entries are fixed non-negative integers).
    - Your goal is to minimize the sum S = B[1] + B[2] + ... + B[N],
      where B[1] = P[1], and for i >= 2, B[i] = B[i - 1] XOR P[i].
    - Output the updated array P as N space-separated non-negative integers.

    Answer format:
    - Your answer must be provided in \\boxed{...} format, containing exactly N space-separated integers.
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 20,
        element_range: int = 2,
        **kwargs: Any,
    ):
        super().__init__()
        assert isinstance(min_n, int) and isinstance(max_n, int)
        assert min_n >= 3, "N should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be >= min_n"
        assert isinstance(element_range, int) and element_range >= 0

        self.min_n = min_n
        self.max_n = max_n
        self.element_range = element_range

        # Problem state
        self.N: Optional[int] = None
        self.initial_P: Optional[List[int]] = None
        self.current_problem: Optional[str] = None
        self.reference_sum: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions and answer format requirements."""
        return (
            "You are solving a Min-Sum Prefix XOR array completion problem.\n"
            "Replace each -1 in the given array P with a non-negative integer to minimize the sum of prefix XORs:\n"
            "  B[1] = P[1], and for i >= 2, B[i] = B[i-1] XOR P[i]. The objective is to minimize sum(B[1..N]).\n"
            "Output Format: Provide the completed array as N space-separated non-negative integers, wrapped in \\boxed{...}.\n"
            "Example: If N=3 and your array is [1, 2, 3], answer as \\boxed{1 2 3}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate N
        N = random.randint(self.min_n, self.max_n)
        self.N = N

        # Generate base array with values in [0, element_range * N]
        P = [random.randint(0, self.element_range * N) for _ in range(N)]

        # Randomly choose indices to set as -1 (at least 1 and at most N-1)
        k_remove = random.randint(1, N - 1)
        removed_indices = random.sample(range(N), k_remove)
        for idx in removed_indices:
            P[idx] = -1

        self.initial_P = P[:]

        # Build the problem prompt
        P_repr = " ".join(f"P[{i}]={val}" for i, val in enumerate(P, start=1))
        problem_text = (
            f"You are given an array P of length {N}: {P_repr}\n"
            f"Replace every entry P[i] that equals -1 (for 1 ≤ i ≤ {N}) with a non-negative integer "
            f"(all other entries are fixed non-negative integers), so as to minimize the sum: "
            f"B[1] + B[2] + ... + B[{N}], where B[1] = P[1] and for i ≥ 2, B[i] = B[i−1] XOR P[i] "
            f"(XOR is the bitwise exclusive OR).\n"
            f"Output the updated array P as {N} space-separated non-negative integers in one line.\n\n"
            f"Output Format: Put your final array inside \\boxed{{...}}, with exactly {N} space-separated integers."
        )
        self.current_problem = problem_text

        # Compute the reference minimal sum (as per the original algorithm)
        self.reference_sum = self._compute_reference_min_sum(P)

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": self.N,
            "initial_array": self.initial_P[:],
            "reference_sum": self.reference_sum,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Parse, validate, and score the submitted array."""
        # Parse answer from boxed format
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.N is None or self.initial_P is None or self.reference_sum is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        # Convert content to array of integers
        user_array = self._parse_array(boxed_content)
        if user_array is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer_format"}

        # Validate constraints
        if len(user_array) != self.N:
            info = {
                "error": "length_mismatch",
                "expected_length": self.N,
                "got_length": len(user_array),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        constraint_ok = all(
            ((orig >= 0 and val == orig) or (orig == -1 and val >= 0))
            for orig, val in zip(self.initial_P, user_array)
        )
        if not constraint_ok:
            info = {
                "error": "constraint_violation",
                "initial_array": self.initial_P[:],
                "user_array": user_array[:],
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute user's sum of prefix XORs
        user_sum = self._sum_prefix_xor(user_array)

        # Compare with reference minimal sum
        is_optimal = (user_sum == self.reference_sum)

        info = {
            "correct": is_optimal,
            "reference_sum": self.reference_sum,
            "user_sum": user_sum,
            "N": self.N,
            "initial_array": self.initial_P[:],
            "user_array": user_array[:],
        }
        reward: float = 1.0 if is_optimal else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_array(self, content: str) -> Optional[List[int]]:
        """Parse a space-separated array of integers from the boxed content."""
        # Allow commas or multiple spaces
        tokens = content.replace(",", " ").split()
        values: List[int] = []
        try:
            for t in tokens:
                values.append(int(t))
            return values
        except ValueError:
            return None

    def _sum_prefix_xor(self, arr: List[int]) -> int:
        """Compute the sum of prefix XORs of an array."""
        s = 0
        prefix = 0
        for v in arr:
            prefix ^= v
            s += prefix
        return s

    def _compute_reference_min_sum(self, P: List[int]) -> int:
        """
        Compute the minimal possible sum of prefix XORs according to the
        original RLVE environment logic. This preserves the algorithm as-is.
        """
        N = len(P)

        # Build A = list of (index, value) for known entries (1-indexed for indices)
        A: List[Tuple[int, int]] = []
        for i, ai in enumerate(P, start=1):
            if ai != -1:
                A.append((i, ai))
        A.sort()
        M = len(A)

        # Compute bit width from input instead of using a magic number.
        if M > 0:
            max_val = max(x for _, x in A)
            BIT = max(1, max_val.bit_length())
        else:
            BIT = 1

        F: List[List[int]] = []   # per-block counts of set bits for each bit position
        LEN: List[int] = []       # length of each block (number of known elements inside)
        tot = 0
        now = 0

        for idx in range(M):
            if idx == 0 or A[idx][0] != A[idx - 1][0] + 1:
                F.append([0] * BIT)
                LEN.append(0)
                tot += 1
                now = 0
            now ^= A[idx][1]
            for j in range(BIT):
                F[tot - 1][j] += (now >> j) & 1
            LEN[tot - 1] += 1

        ans = 0
        for i in range(tot):
            # Note: This logic directly mirrors the original environment code.
            if A[i][0] == 1:
                for j in range(BIT):
                    ans += (F[i][j] << j)
            else:
                for j in range(BIT):
                    ans += (min(F[i][j], LEN[i] - F[i][j] + 1) << j)

        return ans

    def sample_random_action(self) -> str:
        """Sample a random valid completion of the array, formatted in boxed form."""
        if self.N is None or self.initial_P is None:
            # If not initialized, sample a generic small answer
            return "\\boxed{0}"

        arr = []
        for orig in self.initial_P:
            if orig == -1:
                arr.append(random.randint(0, self.element_range * self.N))
            else:
                arr.append(orig)
        answer_str = " ".join(str(x) for x in arr)
        return f"\\boxed{{{answer_str}}}"