from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LandformGenerationCountingEnv(Env):
    """Landform Generation Counting environment - single-turn Q&A.

    Task:
      Given arrays H and C of length N, consider permutations p of indices 0..N-1.
      A permutation p is valid iff for every i (0 <= i < N), the number of j (j < i)
      with H[p[j]] > H[p[i]] is strictly less than C[p[i]].
      Count the number of distinct sequences H[p[0]], ..., H[p[N-1]] obtainable by valid permutations.
      Two permutations that produce the same H-sequence are counted once.
      Output the result modulo MOD.

    Answer format:
      The answer must be provided in \\boxed{...}.
    """

    def __init__(
        self,
        max_MOD: int = 1_000_000_000,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 20,
        **kwargs: Any,
    ):
        super().__init__()
        # Parameters
        self.max_MOD: int = max_MOD
        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Validate parameters
        if self.fixed_N is not None:
            if self.fixed_N < 3:
                raise ValueError("N should be greater than or equal to 3")
        if self.min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if self.max_N < self.min_N:
            raise ValueError("max_N should be greater than or equal to min_N")
        if self.max_MOD < 2:
            raise ValueError("max_MOD should be at least 2")

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.MOD: Optional[int] = None
        self.A_shuffled: Optional[List[Tuple[int, int]]] = None  # List of (H[i], C[i])

    def _get_instructions(self) -> str:
        """Return task description and answer format instructions."""
        return (
            "You are solving a counting problem over permutations with constraints.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)

        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        self.N = N

        # Generate H and C (embedded in tuples A)
        example_H = [random.randint(1, N) for _ in range(N)]
        A: List[Tuple[int, int]] = [None] * N  # type: ignore

        # Build (H[i], C[i]) using the same logic as the original environment
        for i, Hi in enumerate(example_H):
            greater_before = sum(1 for Hj in example_H[:i] if Hj > Hi)
            greater_total = sum(1 for Hj in example_H if Hj > Hi)
            Ci = random.randint(greater_before + 1, greater_total + 1)
            A[i] = (Hi, Ci)

        # Keep a shuffled copy for presenting to the user
        random.shuffle(A)
        self.A_shuffled = A.copy()

        # Choose MOD
        MOD = random.randint(2, self.max_MOD)
        self.MOD = MOD

        # Compute reference answer following the original algorithm
        # Work on a sorted copy: sort by height desc, key asc
        A_sorted = sorted(self.A_shuffled, key=lambda x: (-x[0], x[1]))

        ans_heights = 1
        start = 0
        while start < N:
            end = start
            h_cur = A_sorted[start][0]
            while end + 1 < N and A_sorted[end + 1][0] == h_cur:
                end += 1

            processed = start + 1  # 1-based count as in the original code
            dp = [0] * (processed + 2)  # dp[0..processed]

            first_key = A_sorted[start][1]
            for j in range(1, min(processed, first_key) + 1):
                dp[j] = 1

            for i in range(start + 1, end + 1):
                key = A_sorted[i][1]
                limit = min(processed, key)
                for j in range(1, limit + 1):
                    dp[j] = (dp[j] + dp[j - 1]) % MOD

            last_key = A_sorted[end][1]
            res = sum(dp[1 : min(processed, last_key) + 1]) % MOD
            ans_heights = (ans_heights * res) % MOD

            start = end + 1

        self.reference_answer = ans_heights

        # Build problem prompt
        H_str = " ".join(f"H[{i}]={Ai[0]}" for i, Ai in enumerate(self.A_shuffled))
        C_str = " ".join(f"C[{i}]={Ai[1]}" for i, Ai in enumerate(self.A_shuffled))
        N_minus_1 = N - 1

        problem_text = (
            f"You are given two arrays H and C, each of length {N}:\n"
            f"H: {H_str}\n"
            f"C: {C_str}\n\n"
            f"A permutation p of the indices 0 to {N_minus_1} (i.e., p[0], p[1], ..., p[{N_minus_1}]) "
            f"is considered valid if and only if the following condition holds for every index i from 0 to {N_minus_1}: "
            f"there are fewer than C[p[i]] indices j (j < i) such that H[p[j]] > H[p[i]].\n"
            f"Please count the number of distinct sequences H[p[0]], H[p[1]], ..., H[p[{N_minus_1}]] that can be obtained "
            f"by a valid permutation p. (Two permutations producing the same H-sequence count as one.) "
            f"Output the result modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and return the result. Single-turn episode."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check according to original environment
        mod_value = self.MOD if self.MOD is not None else 1
        if not (0 <= user_answer < mod_value):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "error": "out_of_range",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (self.reference_answer is not None) and (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action."""
        mod_value = self.MOD if self.MOD is not None else 1000000007
        random_answer = random.randint(0, max(0, mod_value - 1))
        return f"\\boxed{{{random_answer}}}"