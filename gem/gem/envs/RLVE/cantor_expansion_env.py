from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CantorExpansionEnv(Env):
    """Environment for counting lexicographically smaller distinct permutations modulo MOD."""

    def __init__(
        self,
        N: int = 10,
        max_MOD: int = 100000,
        **kwargs
    ):
        """
        Initialize the CantorExpansionEnv.

        Parameters:
            N (int): Length of the sequence to generate. Must be >= 3.
            max_MOD (int): Maximum modulus value used when generating MOD (uniformly chosen from [2, max_MOD]).
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N
        self.max_MOD = max_MOD

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.sequence: Optional[List[int]] = None
        self.modulo: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Given a sequence of integers, count the number of distinct permutations of the sequence "
            "that are lexicographically smaller than the original sequence. Return the count modulo MOD.\n"
            "Important notes:\n"
            "- Permutations that only differ by positions of equal elements are considered the same.\n"
            "- Your answer must be a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate problem parameters
        N = self.N
        M = random.randint(2, N)
        A = [random.randint(1, M) for _ in range(N)]
        MOD = random.randint(2, self.max_MOD)

        # Store state
        self.sequence = A
        self.modulo = MOD

        # Build problem prompt
        seq_str = ", ".join(map(str, A))
        self.current_problem = (
            f"Given a sequence of integers: {seq_str}\n\n"
            f"Please count the number of distinct permutations of this sequence that are lexicographically "
            f"smaller than the original sequence. Output a single integer — the number of such permutations "
            f"modulo {MOD}.\n"
            f"Note: Permutations that only differ by the positions of equal elements are considered the same.\n\n"
            f"Output Format: Provide your final answer in \\boxed{{...}}."
        )

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(A, MOD)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, A: List[int], MOD: int) -> int:
        """Compute the number of lexicographically smaller distinct permutations modulo MOD."""
        N = len(A)
        M = max(A)

        # 1. Factor MOD and compute Euler's totient phi(MOD)
        ph = MOD
        nt = MOD
        p_list: List[int] = []
        i = 2
        while i * i <= nt:
            if nt % i == 0:
                p_list.append(i)
                ph = ph // i * (i - 1)
                while nt % i == 0:
                    nt //= i
            i += 1
        if nt > 1:
            p_list.append(nt)
            ph = ph // nt * (nt - 1)
        pc = len(p_list)

        # 2. Fenwick tree (BIT)
        T = [0] * (M + 1)

        def bit_add(x: int) -> None:
            while x <= M:
                T[x] += 1
                x += x & -x

        def bit_sum(x: int) -> int:
            s = 0
            while x > 0:
                s += T[x]
                x -= x & -x
            return s

        # 3. State for multiplicative tracking
        iv = [0] * (N + 2)  # modular inverses for co-prime parts, lazily filled
        iv[1] = 1

        def inv_mod(x: int) -> int:
            # Lazily compute inverse for co-prime part using Euler's theorem
            if iv[x] == 0:
                iv[x] = pow(x, ph - 1, MOD)
            return iv[x]

        tp = [0] * pc  # prime exponents for each prime in p_list
        tc = 1  # product of co-prime parts modulo MOD
        cnt = [0] * (M + 1)

        ans = 0

        # Initialize with the last element
        bit_add(A[N - 1])
        cnt[A[N - 1]] += 1

        for idx in range(N - 2, -1, -1):
            # Count how many suffix elements are strictly smaller than A[idx]
            w = bit_sum(A[idx] - 1)

            # 1) Multiply in the next factorial factor: (suffix length)!
            k = (N - 1) - idx
            tmp = k
            for j, pj in enumerate(p_list):
                while tmp % pj == 0:
                    tmp //= pj
                    tp[j] += 1
            tc = (tc * tmp) % MOD

            # 2) Add this element into the BIT and update its count
            bit_add(A[idx])
            iv[k + 1] = pow(k + 1, ph - 1, MOD)  # precompute inverse for potential use
            cnt[A[idx]] += 1

            # 3) Divide out the new multiplicity factorial factor
            tmp = cnt[A[idx]]
            for j, pj in enumerate(p_list):
                while tmp % pj == 0:
                    tmp //= pj
                    tp[j] -= 1
            tc = (tc * inv_mod(tmp)) % MOD

            # 4) If there are smaller choices w, add w * (remaining permutations) to the rank
            if w > 0:
                # Multiply by w (co-prime part) and increase prime exponents
                tmp = w
                for j, pj in enumerate(p_list):
                    while tmp % pj == 0:
                        tmp //= pj
                        tp[j] += 1
                tc = (tc * tmp) % MOD

                # Build current contribution: co-prime part times prime part
                cur = tc
                for j, pj in enumerate(p_list):
                    if tp[j]:
                        cur = (cur * pow(pj, tp[j], MOD)) % MOD
                ans = (ans + cur) % MOD

                # Divide back by w to restore state
                tmp = w
                for j, pj in enumerate(p_list):
                    while tmp % pj == 0:
                        tmp //= pj
                        tp[j] -= 1
                tc = (tc * inv_mod(tmp)) % MOD

        return ans % MOD

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step to verify the answer."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check [0, MOD)
        if self.modulo is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_ready"}

        if not (0 <= user_answer < self.modulo):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "modulo": self.modulo,
                "error": "out_of_range"
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "modulo": self.modulo
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
        """Sample a random action in the required \\boxed{...} format."""
        if self.modulo is None:
            # Fallback before reset: use a generic small modulo
            guess = random.randint(0, 999)
        else:
            guess = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{guess}}}"