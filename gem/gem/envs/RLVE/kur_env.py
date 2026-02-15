import math
import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class KUREnv(Env):
    """KUR environment converted to GEM format - single turn Q&A.

    The task: Given parameters N, A, B, P (with gcd(N, A) = 1), construct a binary string C of length N:
    - C[i] = 0 iff (A * i + B) mod N < P
    - otherwise C[i] = 1
    Then, count how many times a given binary string T appears as a contiguous substring in C.
    The answer must be provided in \\boxed{...} format.
    """

    prompt_template = (
        "You are given a binary string C of length {N}, defined as C[0], C[1], ..., C[{N_minus_1}].\n"
        "For each index i (0 ≤ i < {N}):\n"
        "- C[i] = 0 if and only if ({A} × i + {B}) mod {N} < {P}. It is guaranteed that {A} and {N} are coprime.\n"
        "- Otherwise, C[i] = 1.\n\n"
        "Please output how many times the following binary string appears (as a contiguous substring) in the string C: {T}\n\n"
        "Output Format: Your final answer should be a single integer in \\boxed{{...}}."
    )

    def __init__(
        self,
        max_n: int = 1000,
        max_m: int = 50,
        **kwargs
    ):
        """Initialize KUREnv with difficulty parameters.

        Args:
            max_n: Maximum value for N (must be >= 8).
            max_m: Maximum length for T (must be >= 2).
        """
        super().__init__()
        assert max_n >= 8, "max_n should be greater than or equal to 8"
        assert max_m >= 2, "max_m should be greater than or equal to 2"
        self.max_n = max_n
        self.max_m = max_m

        # Problem attributes
        self.N: Optional[int] = None
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.P: Optional[int] = None
        self.T: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a binary string counting problem defined by modular arithmetic.\n"
            "Provide your final answer in \\boxed{...} format. Do not include any extra text.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: The problem description string.
            info: Additional info dict (empty for this environment).
        """
        super().reset(seed)

        # Generate parameters with required constraints
        N = random.randint(8, self.max_n)
        while True:
            A = random.randint(2, N - 1)
            B = random.randint(0, N - 1)
            P = random.randint(1, N - 1)
            if math.gcd(N, A) == 1:
                break

        # Function to compute the number of occurrences of T in C
        def compute_answer(T: str) -> int:
            M = len(T)
            intervals = []
            for x, ch in enumerate(T):
                ax = (A * x) % N
                if ch == '0':
                    l = (P - ax - B) % N
                    r = (N - ax - B) % N
                else:
                    l = (-ax - B) % N
                    r = (P - ax - B) % N
                # l, r are in [0, N-1]
                if l <= r:
                    intervals.append((l, r - 1))
                else:
                    intervals.append((0, r - 1))
                    intervals.append((l, N - 1))

            # account for the tail positions
            for i in range(N - M + 1, N):
                ai = (A * i) % N
                intervals.append((ai, ai))

            intervals.sort()
            ans = N
            mx = -1

            for l, r in intervals:
                if l <= mx:
                    removed = max(0, r - mx)
                    ans -= removed
                    mx = max(mx, r)
                else:
                    ans -= (r - l + 1)
                    mx = r

            return ans

        # Construct T by taking a substring from C starting at a random index
        start_i = random.randint(0, N - 2)
        T = ""
        answer_to_ts: Dict[int, list[str]] = {}
        for i in range(start_i, min(N, start_i + self.max_m)):
            T += "0" if (A * i + B) % N < P else "1"
            answer = compute_answer(T)
            assert answer >= 1, "Answer should be at least 1"
            if answer not in answer_to_ts:
                answer_to_ts[answer] = []
            answer_to_ts[answer].append(T)

        reference_answer = random.choice(list(answer_to_ts.keys()))
        T = random.choice(answer_to_ts[reference_answer])

        # Save parameters
        self.N = N
        self.A = A
        self.B = B
        self.P = P
        self.T = T
        self.reference_answer = reference_answer

        # Build problem text
        self.current_problem = self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            A=A,
            B=B,
            P=P,
            T=T,
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Validate the submitted answer.

        Args:
            action: The agent's response text containing \\boxed{...}.

        Returns:
            observation: TERMINAL_STATE (since single-turn).
            reward: 1.0 if correct, 0.0 if wrong, -0.1 if format error.
            terminated: True.
            truncated: False.
            info: Dict with correctness and parameters.
        """
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None) and (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "A": self.A,
            "B": self.B,
            "P": self.P,
            "T": self.T,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        # If parameters are not set yet, default to 0
        if self.N is None or self.reference_answer is None:
            return "\\boxed{0}"
        random_answer = random.randint(0, max(self.N, 1))
        return f"\\boxed{{{random_answer}}}"