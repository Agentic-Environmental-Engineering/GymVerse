from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PrefixProductMODDistinctPermutationEnv(Env):
    """Environment for finding a permutation with distinct prefix products modulo N (single-turn)."""

    def __init__(
        self,
        max_n: int = 100,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        wrong_format_reward: float = -0.1,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            max_n: Maximum value of N to sample from (must be >= 3).
            correct_reward: Reward for a correct answer.
            wrong_reward: Reward for an incorrect but well-formatted answer.
            wrong_format_reward: Reward for a format error (no valid \\boxed{...} content).
        """
        super().__init__()
        if max_n < 3:
            raise ValueError("max_n should be greater than or equal to 3")
        self.max_n = max_n
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.wrong_format_reward = wrong_format_reward

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Find a permutation of numbers 1..N such that all N prefix products "
            "(the product of the first i numbers for i = 1..N) are distinct modulo N.\n"
            "Output Format: Provide the permutation as N integers separated by spaces inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)
        # Generate N such that N is prime or N == 4, as per original construction
        while True:
            N = random.randint(3, self.max_n)

            def is_composite(x: int) -> bool:
                """Return True if x is composite (has a non-trivial divisor), False otherwise."""
                for i in range(2, int(x**0.5) + 1):
                    if x % i == 0:
                        return True
                return False

            if N == 4:
                reference_perm = "1 3 2 4"
                self.N = N
                self.reference_answer = reference_perm
                break
            elif is_composite(N):
                # Skip composite N (except 4) to ensure existence of a simple construction
                continue
            else:
                # N is prime, construct a valid permutation
                inv = [0] * (N + 1)
                inv[0] = inv[1] = 1
                for i in range(2, N + 1):
                    inv[i] = ((N - N // i) * inv[N % i]) % N
                perm: List[int] = [1]
                for i in range(1, N - 1):
                    perm.append(((i + 1) * inv[i]) % N)
                perm.append(N)
                reference_perm = " ".join(map(str, perm))
                self.N = N
                self.reference_answer = reference_perm
                break

        problem = (
            f"Please find a permutation of the numbers from 1 to {self.N} such that all {self.N} "
            f"prefix products (i.e., the product of the first i numbers for all i from 1 to {self.N}) "
            f"are distinct modulo {self.N}.\n\n"
            f"Output Format: Provide the permutation as {self.N} integers in one line, separated by spaces, "
            f"wrapped in \\boxed{{...}}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted permutation."""
        if self.N is None:
            # Environment not properly reset
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Parse the permutation
        try:
            tokens = boxed_content.replace(",", " ").split()
            user_perm = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, self.wrong_reward, True, False, {"error": "invalid_answer_format"}

        # Basic validation
        if len(user_perm) != self.N:
            return TERMINAL_STATE, self.wrong_reward, True, False, {"error": "wrong_length", "expected_length": self.N}

        if set(user_perm) != set(range(1, self.N + 1)):
            return TERMINAL_STATE, self.wrong_reward, True, False, {"error": "not_a_permutation"}

        # Check distinct prefix products modulo N
        existing = [False] * self.N
        prefix_product = 1
        for x in user_perm:
            prefix_product = (prefix_product * x) % self.N
            if not (0 <= prefix_product < self.N):
                return TERMINAL_STATE, self.wrong_reward, True, False, {"error": "modulo_range_error"}
            existing[prefix_product] = True

        satisfied = sum(existing)
        is_correct = (satisfied == self.N)
        reward = self.correct_reward if is_correct else self.wrong_reward

        info = {
            "correct": is_correct,
            "satisfied_count": satisfied,
            "N": self.N,
            "reference_answer": self.reference_answer,
            "user_answer": user_perm,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} block."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random permutation wrapped in \\boxed{...}."""
        N = self.N if self.N is not None else 3
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        content = " ".join(map(str, perm))
        return f"\\boxed{{{content}}}"