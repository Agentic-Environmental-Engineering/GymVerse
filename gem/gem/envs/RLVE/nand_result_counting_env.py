import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, Dict, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NANDResultCountingEnv(Env):
    """NAND Result Counting Environment - Single-turn Q&A

    Task:
      - Given K-bit numbers and the ability to combine them using the NAND operation
        any number of times in any order (with all results masked to K bits),
        compute how many distinct K-bit results within the interval [L, R] (inclusive)
        can be formed.

    Answer format:
      - The final answer must be provided in \\boxed{...} format containing a single integer.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        K: Optional[int] = None,
        N_min: int = 2,
        N_max: int = 10,
        K_min: int = 1,
        K_max: int = 20,
        **kwargs
    ):
        """Initialize the NANDResultCountingEnv.

        Parameters:
            N: If provided, use this fixed number of input numbers (N >= 2).
            K: If provided, use this fixed bit-width (K >= 1).
            N_min: Minimum N when sampling randomly (inclusive).
            N_max: Maximum N when sampling randomly (inclusive).
            K_min: Minimum K when sampling randomly (inclusive).
            K_max: Maximum K when sampling randomly (inclusive).
        """
        super().__init__()
        # Validate ranges
        assert N_min >= 2, "N_min should be greater than or equal to 2"
        assert K_min >= 1, "K_min should be greater than or equal to 1"
        assert N_max >= N_min, "N_max should be greater than or equal to N_min"
        assert K_max >= K_min, "K_max should be greater than or equal to K_min"

        if N is not None:
            assert N >= 2, "N should be greater than or equal to 2"
        if K is not None:
            assert K >= 1, "K should be greater than or equal to 1"

        self.fixed_N = N
        self.fixed_K = K
        self.N_min = N_min
        self.N_max = N_max
        self.K_min = K_min
        self.K_max = K_max

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.L: Optional[int] = None
        self.R: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a NAND result counting problem.\n"
            "All intermediate and final values are treated as K-bit numbers (masking to K bits).\n"
            "Your final answer must be a single integer placed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N and K
        self.N = self.fixed_N if self.fixed_N is not None else random.randint(self.N_min, self.N_max)
        self.K = self.fixed_K if self.fixed_K is not None else random.randint(self.K_min, self.K_max)

        # Generate components over bit positions
        K = self.K
        N = self.N
        assert N is not None and K is not None
        component_num = random.randint(1, K)
        endpoints = random.sample(range(1, K), component_num - 1) if component_num > 1 else []
        endpoints.sort()
        endpoints = [0] + endpoints + [K]
        allbits = list(range(K))
        random.shuffle(allbits)
        assert len(endpoints) == component_num + 1, "Endpoints should be of length component_num + 1"
        assert all(0 <= endpoints[i] < endpoints[i + 1] <= K for i in range(component_num)), (
            "Endpoints should be in the range [0, K] and strictly increasing"
        )
        components = [allbits[endpoints[i]: endpoints[i + 1]] for i in range(component_num)]

        def generate_number() -> int:
            number = 0
            existence_probability = random.random()
            for component in components:
                if random.random() < existence_probability:
                    number |= sum(1 << bit for bit in component)
            return number

        self.A = [generate_number() for _ in range(N)]

        # Generate query interval [L, R]
        full = (1 << K) - 1
        L = random.randint(0, full)
        R = random.randint(0, full)
        if L > R:
            L, R = R, L
        self.L, self.R = L, R

        # Compute reference answer using the original algorithm
        self.reference_answer = self._compute_reference_answer(self.A, K, L, R)

        # Build problem prompt
        numbers_str = " ".join(map(str, self.A))
        self.current_problem = (
            f"From now on, all numbers are treated as {K}-bit binary strings "
            f"(i.e., only the lowest {K} bits are considered, and leading zeros may be added to fill up to {K} bits).\n\n"
            "The NAND operation is defined as:\n"
            "- 0 NAND 0 = 1\n"
            "- 0 NAND 1 = 1 NAND 0 = 1\n"
            "- 1 NAND 1 = 0\n\n"
            f"You are given the following {N} numbers: {numbers_str}\n"
            "You may combine them arbitrarily using the NAND operation and brackets (i.e., in any order, any number of times).\n\n"
            f"How many distinct numbers in the range [{L}, {R}] (inclusive) can be obtained by such combinations? "
            f"Note: all intermediate and final results are considered as {K}-bit binary strings, so only numbers within the {K}-bit range are valid.\n\n"
            "Output Format: Provide a single integer as your final answer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, A: List[int], K: int, L: int, R: int) -> int:
        """Compute the number of reachable K-bit values in [L, R] using the original algorithm."""
        full = (1 << K) - 1
        lk = [0] * K
        num = [0] * K
        have = 0

        # Build the 'basis' masks
        for i in range(K - 1, -1, -1):
            if ((have >> i) & 1) == 0:
                now_mask = full
                for a in A:
                    if (a >> i) & 1:
                        now_mask &= a
                    else:
                        now_mask &= (~a) & full
                lk[i] = now_mask
                num[i] = 1
                have |= now_mask

        # Prefix-sum the counts
        for i in range(1, K):
            num[i] += num[i - 1]

        def count_upto(x: int) -> int:
            # How many reachable values ≤ x
            if x < 0:
                return 0
            if x >= full:
                return 1 << num[K - 1]
            ans = 0
            for i in range(K - 1, -1, -1):
                if x < 0:
                    break
                if (x >> i) & 1:
                    if lk[i] != 0:
                        ans += 1 << (num[i] - 1)
                        x -= lk[i]
                    else:
                        ans += 1 << num[i]
                        break
            if x == 0:
                ans += 1
            return ans

        return count_upto(R) - count_upto(L - 1)

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step: parse the boxed answer, verify, and return reward."""
        answer_str = self._parse_answer(action)
        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_str.strip())
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
            "A": self.A,
            "L": self.L,
            "R": self.R,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...}.

        Returns:
            The content inside the last \\boxed{...} if present, otherwise None.
        """
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted in \\boxed{...}.

        For this problem, the answer is a count of distinct reachable values in [L, R].
        We sample a uniform random integer in [0, max_possible], where max_possible is (R - L + 1).
        """
        if self.L is None or self.R is None:
            # Fallback if not initialized
            random_answer = random.randint(0, 1)
        else:
            max_possible = max(0, self.R - self.L + 1)
            random_answer = random.randint(0, max_possible)
        return f"\\boxed{{{random_answer}}}"