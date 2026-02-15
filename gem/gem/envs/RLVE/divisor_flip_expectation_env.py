from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DivisorFlipExpectationEnv(Env):
    """Environment for the Divisor Flip Expectation problem - single turn Q&A.

    Task:
    - Given N lights with initial on/off states and switches where pressing switch i toggles all lights whose indices divide i.
    - Repeatedly press a random switch until all lights are off, but if it becomes possible to finish in at most K presses, switch to the optimal (shortest) strategy.
    - Compute the expected total number of presses E, and output E × N! modulo MOD.

    Answer format:
    - The answer must be provided as a single integer inside \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 1,
        max_n: int = 100,
        modulo: int = 10**9 + 7,
        **kwargs
    ):
        super().__init__()
        self.fixed_N = N
        self.min_n = min_n
        self.max_n = max_n
        self.MOD = modulo

        # Runtime variables
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.state: Optional[List[Optional[int]]] = None  # 1-indexed, with state[0] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a combinatorial expected value problem with modular arithmetic.\n"
            "You must provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            if self.fixed_N < 1:
                raise ValueError("N should be greater than or equal to 1")
            N = self.fixed_N
        else:
            N = random.randint(max(1, self.min_n), max(self.min_n, self.max_n))
        if N < 1:
            raise ValueError("N should be greater than or equal to 1")

        # Determine K uniformly from [0, N]
        K = random.randint(0, N)

        # Generate random initial state with a random probability of 1's
        one_probability = random.random()
        B = [None] + [1 if random.random() < one_probability else 0 for _ in range(N)]

        # Store current parameters
        self.N = N
        self.K = K
        self.state = B.copy()

        # Compute reference answer using the original algorithm
        reference_answer = self._compute_reference_answer(N, K, B.copy(), self.MOD)
        self.reference_answer = reference_answer

        # Build problem prompt
        state_str = "\n".join(f"Light {i}: {self.state[i]}" for i in range(1, N + 1))
        self.current_problem = (
            f"You are given {N} lights labeled from 1 to {N}, each in an initial state: 1 (on) or 0 (off). "
            f"The initial state is:\n{state_str}\n\n"
            f"Each light can be toggled by pressing switches. There are {N} switches, and pressing switch i will "
            f"toggle the state of all lights whose indices divide i (including 1 and i itself). Toggling means "
            f"changing from 0 to 1 or from 1 to 0.\n\n"
            f"You play the following game:\n"
            f"- Repeatedly select a switch uniformly at random and press it, until the state of all lights is 0.\n"
            f"- However, if at any point it becomes possible to turn off all lights using at most {K} switch presses, "
            f"you stop random pressing and directly use an optimal (shortest-length) sequence of switches (≤ {K} presses) "
            f"to turn off all lights.\n\n"
            f"Let E be the expected number of total switch presses under this strategy. "
            f"Compute the integer value of E × {N}! modulo {self.MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted answer and terminate."""
        # Extract the boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate integer and range
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if not (0 <= user_answer < self.MOD):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "error": "out_of_range"
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compare to reference answer
        is_correct = (user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "K": self.K,
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
        """Sample a random action formatted as \\boxed{...}."""
        random_answer = random.randint(0, self.MOD - 1)
        return f"\\boxed{{{random_answer}}}"

    def _compute_reference_answer(self, N: int, K: int, B: List[Optional[int]], MOD: int) -> int:
        """Compute the reference answer E × N! mod MOD using the original algorithm."""
        # Precompute modular inverses for 1..N under prime MOD
        inv = [0] * (N + 1)
        inv[1] = 1
        for i in range(2, N + 1):
            inv[i] = (MOD - MOD // i) * inv[MOD % i] % MOD

        # Build divisor lists: g[j] = list of divisors of j
        g: List[List[int]] = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                g[j].append(i)

        # Greedy elimination from N down to 1 to count minimal needed toggles
        tp = 0
        for i in range(N, 0, -1):
            if B[i] == 1:
                for d in g[i]:
                    B[d] ^= 1
                tp += 1

        # Compute expected presses based on whether optimal finish threshold is reached
        if tp <= K:
            ans = tp % MOD
        else:
            f = [0] * (N + 1)
            f[N] = 1
            for i in range(N - 1, 0, -1):
                ans_term = (f[i + 1] + 1) % MOD
                f[i] = (1 + (N - i) * ans_term * inv[i]) % MOD

            ans = 0
            for i in range(tp, K, -1):
                ans = (ans + f[i]) % MOD
            ans = (ans + K) % MOD

        # Multiply by N! modulo MOD
        fact = 1
        for i in range(1, N + 1):
            fact = (fact * i) % MOD
        ans = (ans * fact) % MOD

        return ans