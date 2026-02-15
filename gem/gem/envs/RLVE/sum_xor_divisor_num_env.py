from typing import Any, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SumXorDivisorNumEnv(Env):
    """Environment for the sum of divisor counts with XOR over ranges problem (single-turn)."""

    def __init__(self, max_n_m: int = 1000, **kwargs) -> None:
        """
        Initialize the environment.

        Args:
            max_n_m: The maximum value for N and M (and X is chosen in [0, max_n_m]).
                     Must be greater than or equal to 3.
        """
        super().__init__()
        if max_n_m < 3:
            raise ValueError("max_n_m should be greater than or equal to 3")
        self.max_n_m: int = max_n_m
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.X: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a number theory problem involving divisor counts and bitwise XOR.\n"
            "Please provide your final answer as a single integer wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample parameters
        N = random.randint(3, self.max_n_m)
        M = random.randint(3, self.max_n_m)
        X = random.randint(0, self.max_n_m)

        # Store parameters
        self.N = N
        self.M = M
        self.X = X

        # Build the problem statement
        self.current_problem = (
            f"Let d(n) denote the number of positive divisors of n (with d(0) = 0). "
            f"What is the sum of d(i XOR j XOR {X}) (XOR means bitwise XOR) over all integer pairs (i, j) "
            f"such that 0 <= i <= {N} and 0 <= j <= {M}?\n\n"
            f"Output Format: Provide a single integer as your final answer in \\boxed{{...}}."
        )

        # Compute the reference answer
        self.reference_answer = self._compute_reference(N, M, X)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by parsing and checking the provided answer."""
        # Extract boxed answer
        extracted = self._parse_answer(action)
        if extracted is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(extracted)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Reference answer is not computed."

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the expected output format."""
        guess = 0 if self.reference_answer is None else random.randint(0, max(1, self.reference_answer))
        return f"\\boxed{{{guess}}}"

    def _compute_reference(self, N: int, M: int, X: int) -> int:
        """
        Compute the reference answer using the original algorithm.

        This computes:
        sum_{i=0..N} sum_{j=0..M} d(i XOR j XOR X),
        where d(0) = 0 and d(n) is the number of positive divisors of n for n >= 1.
        """
        A = N + 1
        B = M + 1

        # Build bit arrays (least significant bit first)
        a_bits = []
        while A:
            a_bits.append(A & 1)
            A >>= 1

        b_bits = []
        while B:
            b_bits.append(B & 1)
            B >>= 1

        x_bits = []
        tempX = X
        while tempX:
            x_bits.append(tempX & 1)
            tempX >>= 1

        # Pad to the same length
        L = max(len(a_bits), len(b_bits), len(x_bits))
        a_bits += [0] * (L - len(a_bits))
        b_bits += [0] * (L - len(b_bits))
        x_bits += [0] * (L - len(x_bits))

        # h[i] = integer value of bits (a xor b xor x) from position i..L-1
        h = [0] * (L + 1)
        for i in range(L - 1, -1, -1):
            h[i] = h[i + 1] + ((a_bits[i] ^ b_bits[i] ^ x_bits[i]) << i)

        # mi[k] = 2^k
        mi = [1] * L
        for i in range(1, L):
            mi[i] = mi[i - 1] * 2

        # Cache for the divisor-summatory function S(val) = sum_{k=1..val} d(k)
        sd: dict[int, int] = {}

        def D(val: int) -> int:
            """Return sum_{k=1..val} d(k), with d(0) = 0 and D(val<=0)=0."""
            if val <= 0:
                return 0
            if val in sd:
                return sd[val]
            res = 0
            l = 1
            # Use sqrt-decomposition to compute sum_{i=1..val} floor(val / i)
            while l <= val:
                t = val // l
                r = val // t
                cnt = r - l + 1
                res += cnt * t
                l = r + 1
            sd[val] = res
            return res

        # Main double loop over set bits in a_bits and b_bits
        ans = 0
        for i in range(L):
            if a_bits[i] == 0:
                continue
            for j in range(L):
                if b_bits[j] == 0:
                    continue
                s = max(i, j)
                t = min(i, j)

                # Flip the s-th bit in h[s], adjust if i == j
                H = h[s] ^ (1 << s)
                if i == j:
                    H ^= (1 << s)

                # Sum d(v) for v in [H, H + 2^s - 1]
                val = D(H + (1 << s) - 1) - D(H - 1)
                ans += val * mi[t]

        assert ans > 0, "The computed answer should be greater than 0"
        return ans