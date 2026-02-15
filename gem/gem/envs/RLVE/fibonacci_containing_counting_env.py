from typing import Any, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class FibonacciContainingCountingEnv(Env):
    """
    Environment for counting the number of positive integer pairs (a, b) such that
    for the sequence f defined by f(0)=a, f(1)=b, and f(n)=f(n−1)+f(n−2) for n≥2,
    there exists an n≥2 with f(n)=K. Single-turn question-answer environment.
    """

    prompt_template = (
        "How many pairs of positive integers (a, b) are there such that, defining f by "
        "f(0)=a, f(1)=b, and f(n)=f(n−1)+f(n−2) for n≥2, there exists an n≥2 with f(n)={K}?"
    )

    def __init__(
        self,
        max_k: int = 1000000,
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(min/max)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 10.0,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
            max_k: Maximum value for K (K will be sampled uniformly from [2, max_k]).
                   Must be >= 2.
            wrong_format: Preserved parameter from original environment (not used in GEM scoring).
            rewarding_strategy: Preserved parameter from original environment (not used in GEM scoring).
            rewarding_weight: Preserved parameter from original environment (not used in GEM scoring).
            rewarding_beta: Preserved parameter from original environment (not used in GEM scoring).
        """
        super().__init__()
        assert max_k >= 2, "max_k should be greater than or equal to 2"
        self.max_k = max_k

        # Preserved parameters (not used in GEM scoring, but kept to maintain compatibility)
        self.wrong_format = wrong_format
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_k: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a counting problem related to Fibonacci-like sequences.\n"
            "Please provide your final answer as a single integer wrapped in \\boxed{...}.\n"
            "Example: \\boxed{42}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: A string containing instructions and the problem statement.
            info: An empty dict for compatibility.
        """
        super().reset(seed)

        # Sample problem parameter K
        K = random.randint(2, self.max_k)
        self.current_k = K

        # Build problem statement
        problem_statement = self.prompt_template.format(K=K)
        problem_statement += "\n\nOutput Format: Your final answer should be a single integer in \\boxed{...}."

        # Compute reference answer
        self.reference_answer = self._compute_reference_answer(K)

        # Construct full observation
        self.current_problem = problem_statement
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step: parse and check the submitted answer.

        Parameters:
            action: The agent's text output containing \\boxed{...} with an integer.

        Returns:
            observation: TERMINAL_STATE for single-turn environments.
            reward: 1.0 if correct; 0.0 if incorrect; -0.1 if format error.
            terminated: True (single-turn).
            truncated: False.
            info: Dict containing result details.
        """
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "K": self.current_k,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside the last \\boxed{...} occurrence from the text.

        Parameters:
            text: The agent's response.

        Returns:
            The extracted string inside \\boxed{...}, or None if not found.
        """
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _compute_reference_answer(self, K: int) -> int:
        """
        Compute the number of pairs (a, b) of positive integers such that the sequence
        defined by f(0)=a, f(1)=b, f(n)=f(n−1)+f(n−2) for n≥2 has some term f(n)=K for n≥2.

        This replicates the original algorithm from the RLVE environment.
        """
        def gcd(a: int, b: int) -> int:
            return gcd(b, a % b) if b else a

        def lcm(a: int, b: int) -> int:
            return a // gcd(a, b) * b

        fib = [1, 1]
        e = 1
        while fib[e] + fib[e - 1] <= K:
            fib.append(fib[e] + fib[e - 1])
            e += 1

        ans = 0
        for i in range(1, e):
            a = fib[i - 1]
            b = fib[i]
            x = 1
            while (K - b * x) % a != 0 and K > b * x:
                x += 1
            if K <= b * x:
                continue
            ans += (K - b * x - 1) // lcm(a, b) + 1

        assert ans > 0, "The answer should be positive."
        return ans

    def sample_random_action(self) -> str:
        """
        Sample a random action by producing a random integer wrapped in \\boxed{...}.
        This does not attempt to be correct; it is for testing purposes.
        """
        # Choose a random integer within a reasonable range
        upper = max(10, (self.reference_answer or 10))
        random_answer = random.randint(0, upper)
        return f"\\boxed{{{random_answer}}}"