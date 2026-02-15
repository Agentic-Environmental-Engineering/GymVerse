from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SalesmanFatigueEnv(Env):
    """
    Salesman Fatigue problem environment (single-turn Q&A).

    Task:
    - Given N pairs (S[i], A[i]) with S sorted non-decreasing, select k distinct pairs
      to maximize: max(S[i_1], ..., S[i_k]) * 2 + A[i_1] + ... + A[i_k], for each k = 1..N.
    - Output a single line of N integers (answers for k = 1..N) separated by spaces.
    - The answer must be enclosed in \\boxed{...}.

    Reward:
    - Correct: 1.0
    - Wrong: 0.0
    - Format error: -0.1

    Parameters:
    - N: If provided, the environment uses this fixed N (must be >= 3).
         If None, N will be randomly sampled in [min_N, max_N] at reset.
    - min_N, max_N: Bounds for random N when N is None.
    """

    prompt_template = (
        "You are given {N} pairs of integers (S[i], A[i]) for 0 <= i < {N}, provided as:\n"
        "{S_and_A}\n\n"
        "Note: The array S is sorted in non-decreasing order: S[0] <= S[1] <= ... <= S[{N_minus_1}].\n\n"
        "Please select k distinct pairs i_1, i_2, ..., i_k and maximize the following expression:\n"
        "max(S[i_1], S[i_2], ..., S[i_k]) * 2 + A[i_1] + A[i_2] + ... + A[i_k].\n"
        "Please compute the maximum value of this expression for each k = 1 to {N}.\n\n"
        "Output Format: Your final answer should be a single line containing {N} integers —\n"
        "the maximum value for each k = 1 to {N} in order, separated by spaces, enclosed in \\boxed{{...}}."
    )

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 50,
        **kwargs
    ):
        super().__init__()
        self.N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        self.current_problem: Optional[str] = None
        self.reference_answer_list: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.current_N: Optional[int] = None
        self.S: Optional[List[int]] = None
        self.A: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a combinatorial optimization problem involving pairs (S[i], A[i]).\n"
            "Please provide your final answer as N integers separated by spaces, enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N is not None:
            if self.N < 3:
                raise ValueError("N should be greater than or equal to 3")
            N = self.N
        else:
            N = random.randint(self.min_N, self.max_N)
            if N < 3:
                N = 3

        self.current_N = N

        # Generate S and A
        S = [random.randint(1, max(1, N * N // 2)) for _ in range(N)]
        S.sort()
        A = [random.randint(1, N) for _ in range(N)]

        self.S = S
        self.A = A

        # Compute reference answers
        answers = self._compute_answers(S, A)
        self.reference_answer_list = answers
        self.reference_answer_str = " ".join(map(str, answers))

        # Build problem prompt
        pairs_str = "\n".join(f"S[{i}]={S[i]} A[{i}]={A[i]}" for i in range(N))
        self.current_problem = self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            S_and_A=pairs_str
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the submitted answer and return the result."""
        # Parse the boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from boxed content
        try:
            parts = boxed_content.strip().split()
            user_answer_list = [int(x) for x in parts]
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length
        if self.current_N is None or self.reference_answer_list is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem"}
        if len(user_answer_list) != self.current_N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer_length", "expected_length": self.current_N}

        # Compare with reference
        is_correct = user_answer_list == self.reference_answer_list
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_list,
            "user_answer": user_answer_list
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def _compute_answers(S: List[int], A: List[int]) -> List[int]:
        """Compute the maximum values for each k = 1..N based on the provided algorithm."""
        N = len(S)
        v = list(zip(S, A))
        v.sort(key=lambda x: -x[1])

        P = [0] * (N + 1)
        for i in range(N):
            P[i + 1] = P[i] + v[i][1]

        q = [0] * N
        max_q = 0
        for i in range(N):
            max_q = max(max_q, 2 * v[i][0])
            q[i] = max_q

        h = [0] * N
        max_h = 0
        for i in range(N - 1, -1, -1):
            max_h = max(max_h, 2 * v[i][0] + v[i][1])
            h[i] = max_h

        answers: List[int] = []
        for X in range(1, N + 1):
            idx = X - 1
            option1 = P[X] + q[idx]
            option2 = P[X - 1] + h[idx]
            answers.append(max(option1, option2))

        return answers

    def sample_random_action(self) -> str:
        """Sample a random action by generating N random integers as a guess."""
        if self.current_N is None or self.S is None or self.A is None:
            # Fallback: random short guess
            return "\\boxed{0}"

        N = self.current_N
        max_val = 2 * (max(self.S) if self.S else 1) + (sum(self.A) if self.A else 0)
        if max_val < 1:
            max_val = 1
        guess = [str(random.randint(0, max_val)) for _ in range(N)]
        return f"\\boxed{{{' '.join(guess)}}}"