import random
from functools import cmp_to_key
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ProtectingFlowersEnv(Env):
    """Protecting Flowers scheduling environment - single-turn Q&A.

    Task:
      Given arrays T and D of length N, output a permutation p[1..N].
      Define S[i] as the sum of T[p[j]] for all 1 ≤ j < i (so S[1] = 0).
      The objective is to minimize the total sum of S[i] * D[p[i]] over i = 1..N.

    Answer format:
      The answer must be provided as N integers (the permutation 1..N) inside \\boxed{...},
      separated by spaces. Example: \\boxed{3 1 2}
    """

    def __init__(
        self,
        n: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 50,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
          n: If provided, use this fixed N for all problems. Must be >= 3.
          min_n: Minimum N to sample when n is None (default 3).
          max_n: Maximum N to sample when n is None (default 50).
        """
        super().__init__()
        if n is not None and n < 3:
            raise ValueError("n should be greater than or equal to 3")
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")

        self.n_fixed: Optional[int] = n
        self.min_n: int = min_n
        self.max_n: int = max_n

        self.N: Optional[int] = None
        self.T: Optional[List[int]] = None
        self.D: Optional[List[int]] = None
        self.reference_sum: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a scheduling optimization problem.\n"
            "Given arrays T and D, output a permutation of 1..N to minimize the sum of S[i] * D[p[i]],\n"
            "where S[i] is the sum of T[p[j]] for all 1 ≤ j < i.\n"
            "Output Format: Provide N integers (a permutation of 1..N) inside \\boxed{...}, separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.n_fixed is not None:
            N = self.n_fixed
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"

        # Generate T and D with values in [1, N]
        T = [random.randint(1, N) for _ in range(N)]
        D = [random.randint(1, N) for _ in range(N)]

        # Prepare pairs and sort by t/d ascending using cross-multiplication
        A = [(t, d) for t, d in zip(T, D)]

        def cmp(x, y):
            left = x[0] * y[1]
            right = x[1] * y[0]
            if left < right:
                return -1
            elif left > right:
                return 1
            else:
                return 0

        A.sort(key=cmp_to_key(cmp))

        # Compute prefix sums of D in sorted order
        prefix = [0] * (N + 1)
        for i in range(N):
            prefix[i + 1] = prefix[i] + A[i][1]

        # Compute minimal total sum (reference answer)
        total_d = prefix[N]
        ans = 0
        for i in range(N):
            t_i, d_i = A[i]
            # All cows after position i keep eating while fetching i
            ans += t_i * (total_d - prefix[i + 1])

        assert ans > 0, "The reference answer should be positive"

        # Store state
        self.N = N
        self.T = T
        self.D = D
        self.reference_sum = ans

        # Build problem description
        t_and_d_lines = "\n".join(
            f"T[{i}]={Ti} D[{i}]={Di}" for i, (Ti, Di) in enumerate(zip(T, D), start=1)
        )
        self.current_problem = (
            f"You are given two arrays T and D, each containing {N} integers:\n"
            f"{t_and_d_lines}\n\n"
            f"Please output a permutation of 1 to {N}, denoted as p[1], p[2], ..., p[{N}].\n"
            f"- Define S[i] as the sum of T[p[j]] for all 1 ≤ j < i (so S[1] = 0).\n"
            f"- The objective is to minimize the total sum of S[i] * D[p[i]] for i from 1 to {N}.\n\n"
            f"Output Format: Provide your permutation as N space-separated integers inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the submitted permutation."""
        # Extract boxed content
        content = self._parse_answer(action)
        if content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers inside the box
        try:
            tokens = content.replace(",", " ").split()
            p_list = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate permutation
        N = self.N if self.N is not None else 0
        if len(p_list) != N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_length", "expected_length": N}

        if any(pi < 1 or pi > N for pi in p_list):
            return TERMINAL_STATE, 0.0, True, False, {"error": "out_of_range"}

        if set(p_list) != set(range(1, N + 1)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_a_permutation"}

        # Compute user's objective value
        assert self.T is not None and self.D is not None and self.reference_sum is not None
        T, D = self.T, self.D
        s_prev = 0
        user_sum = 0
        for pi in p_list:
            user_sum += s_prev * D[pi - 1]
            s_prev += T[pi - 1]

        # Check correctness
        is_correct = (user_sum == self.reference_sum)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_sum,
            "user_answer_sum": user_sum,
            "N": N,
            "T": T,
            "D": D,
            "permutation": p_list,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random permutation action in boxed format."""
        if self.N is None:
            # If reset has not been called, sample a default small N
            N = self.n_fixed if self.n_fixed is not None else max(self.min_n, 3)
        else:
            N = self.N
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        content = " ".join(map(str, perm))
        return f"\\boxed{{{content}}}"