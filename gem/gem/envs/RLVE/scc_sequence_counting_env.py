import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SCC_Sequence_CountingEnv(Env):
    """Environment for counting distinct SCC sequences in a growing directed graph."""

    def __init__(
        self,
        max_MOD: int = 1000000,
        fixed_N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 50,
        **kwargs
    ):
        """
        Initialize the SCC_Sequence_CountingEnv instance.

        Parameters:
        - max_MOD: The maximum modulo value. MOD will be randomly selected in [2, max_MOD].
        - fixed_N: If provided, the number of vertices N will be fixed to this value (must be >= 3).
        - min_N: Minimum value for N if fixed_N is not provided.
        - max_N: Maximum value for N if fixed_N is not provided.
        """
        super().__init__()
        self.max_MOD = max_MOD
        self.fixed_N = fixed_N
        self.min_N = min_N
        self.max_N = max_N

        self.current_problem: Optional[str] = None
        self.reference_answer_list: Optional[List[int]] = None
        self.reference_answer_string: Optional[str] = None
        self.N: Optional[int] = None
        self.MOD: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a directed graph SCC sequence counting problem.\n"
            "Please output your result as a single line of integers inside \\boxed{...}.\n"
            "The integers must be separated by single spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            if self.fixed_N < 3:
                raise ValueError("fixed_N should be greater than or equal to 3")
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
            if N < 3:
                N = 3

        MOD = random.randint(2, self.max_MOD)

        self.N = N
        self.MOD = MOD

        # Build problem prompt
        max_edges = N * (N - 1)
        self.current_problem = (
            f"Consider a directed graph with {N} vertices, initially with no edges. "
            f"You may choose an arbitrary list of E directed edges to add to the graph, under the following constraints:\n"
            "- Each edge connects two distinct vertices (i.e., no self-loops).\n"
            "- No two edges in the list are the same.\n"
            "- The edges are added one by one in the given order of the list.\n\n"
            "After adding each edge, compute the number of strongly connected components (SCCs) in the current graph (with the edges added so far) and record it; "
            "this produces a sequence of E integers — we call this an SCC sequence. "
            "Your task is to compute, for each possible value of E from 1 to {N} × ({N} - 1), how many distinct SCC sequences can be produced.\n\n"
            f"Output {max_edges} integers in one line, separated by spaces. "
            f"The i-th number (1 ≤ i ≤ {max_edges}) is the number of distinct SCC sequences that can be obtained when E = i, modulo {MOD}.\n\n"
            "Output Format: Your final answer should be the entire sequence in \\boxed{a1 a2 ... a_{N*(N-1)}}."
        )

        # Compute reference answer using the provided algorithm
        reference_list = self._compute_reference_answer(N, MOD)
        self.reference_answer_list = reference_list
        self.reference_answer_string = " ".join(map(str, reference_list))

        obs = self._get_instructions() + self.current_problem
        return obs, {"N": N, "MOD": MOD, "max_edges": max_edges}

    def _compute_reference_answer(self, N: int, MOD: int) -> List[int]:
        """Compute the reference answer (list of counts mod MOD) for given N and MOD."""
        # Precompute the p_limit array
        p_limit = [0] * (N + 1)
        for i in range(1, N + 1):
            p_limit[i] = (N - i + 1) * (N - 1) + (i - 1) * (i - 2) // 2

        # f and sf are 2×(N+2)×(N+2)
        f = [[[0] * (N + 2) for _ in range(N + 2)] for _ in range(2)]
        sf = [[[0] * (N + 2) for _ in range(N + 2)] for _ in range(2)]

        # g and sg are 2×(N+2)
        g = [[0] * (N + 2) for _ in range(2)]
        sg = [[0] * (N + 2) for _ in range(2)]

        # ans[E] will hold the answer for sequence-length E
        ans = [0] * (N * (N - 1) + 2)

        # --- initialize for E = 1 ---
        f[1][N][1] = 1
        ans[1] = 1
        for i in range(1, N + 1):
            sf[1][i][1] = 1

        # --- first phase: E = 2 … min(N*(N-1), 2*N) ---
        maxE = min(N * (N - 1), N << 1)
        for E in range(2, maxE + 1):
            op = E & 1
            prev = op ^ 1

            # zero out f[op]
            for j in range(1, N + 1):
                for k in range(1, N + 1):
                    f[op][j][k] = 0

            # DP recurrence
            for j in range(1, N + 1):
                if E <= p_limit[j]:
                    for k in range(1, N + 1):
                        if E + j >= N + k - 1:
                            f[op][j][k] = (f[prev][j][k] + sf[prev][j + 1][k - 1]) % MOD

            # build sf[op] and accumulate ans[E]
            total = 0
            for j in range(N, 0, -1):
                for k in range(1, N + 1):
                    sf[op][j][k] = (sf[op][j + 1][k] + f[op][j][k]) % MOD
                    total = (total + f[op][j][k]) % MOD
            ans[E] = total

        # --- prepare g[0] and sg[0] from f[0] ---
        for j in range(1, N + 1):
            s = 0
            for k in range(1, N + 1):
                s = (s + f[0][j][k]) % MOD
            g[0][j] = s

        for j in range(N, 0, -1):
            sg[0][j] = (sg[0][j + 1] + g[0][j]) % MOD

        # --- second phase: E = 2*N+1 … N*(N-1) ---
        for E in range((N << 1) + 1, N * (N - 1) + 1):
            op = E & 1
            prev = op ^ 1

            # zero out g[op]
            for j in range(1, N + 1):
                g[op][j] = 0

            # recurrence for g
            for j in range(1, N + 1):
                if E <= p_limit[j]:
                    g[op][j] = sg[prev][j]

            # build sg[op] and accumulate ans[E]
            total = 0
            for j in range(N, 0, -1):
                sg[op][j] = (sg[op][j + 1] + g[op][j]) % MOD
                total = (total + g[op][j]) % MOD
            ans[E] = total

        # return ans[1..N*(N-1)]
        return ans[1: N * (N - 1) + 1]

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step: parse and verify the user's answer."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate internal state
        if self.reference_answer_list is None or self.N is None or self.MOD is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error"}

        # Parse answer list
        try:
            user_list = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        expected_len = self.N * (self.N - 1)
        if len(user_list) != expected_len:
            return TERMINAL_STATE, 0.0, True, False, {
                "error": "length_mismatch",
                "expected_length": expected_len,
                "received_length": len(user_list),
            }

        is_correct = (user_list == self.reference_answer_list)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_string,
            "user_answer": " ".join(map(str, user_list)),
            "N": self.N,
            "MOD": self.MOD,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random sequence of integers inside \\boxed{...})."""
        if self.N is None or self.MOD is None:
            # Fallback: return a simple random integer
            random_answer = random.randint(0, 9)
            return f"\\boxed{{{random_answer}}}"
        count = self.N * (self.N - 1)
        random_sequence = [str(random.randint(0, self.MOD - 1)) for _ in range(count)]
        return f"\\boxed{{{' '.join(random_sequence)}}}"