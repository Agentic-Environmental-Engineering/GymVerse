from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SquSquarksEnv(Env):
    """Environment for the SquSquarks problem - single-turn question answering.

    Task:
    Given a list of pairwise sums from N distinct positive integers, find the original N integers.
    The answer must be provided as N space-separated positive integers inside \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 10,
        number_multiple: int = 2,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(intersection/union)^beta",
        rewarding_beta: float = 5.0,
        rewarding_weight: float = +1.0,
        **kwargs
    ):
        super().__init__()
        # Problem size configuration
        self.N_fixed = N
        self.min_N = min_N
        self.max_N = max_N

        # Generation control
        self.number_multiple = number_multiple

        # Original reward configuration preserved as attributes (not used for final reward)
        self.rewards_config = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_beta": rewarding_beta,
            "rewarding_weight": rewarding_weight,
        }

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_numbers: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.sums: Optional[List[int]] = None
        self.N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are solving a pairwise-sum reconstruction problem.\n"
            "Given N distinct positive integers, consider all N*(N-1)/2 distinct pair sums.\n"
            "You are given these sums in arbitrary order. Your task is to recover the original N integers.\n"
            "Output Format: Provide the N integers, separated by single spaces, inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            assert isinstance(self.N_fixed, int), "N must be an integer if provided"
            assert self.N_fixed >= 3, "N should be greater than or equal to 3"
            self.N = self.N_fixed
        else:
            assert self.min_N >= 3, "min_N should be greater than or equal to 3"
            assert self.max_N >= self.min_N, "max_N should be at least min_N"
            self.N = random.randint(self.min_N, self.max_N)

        # Generate reference numbers
        range_max = self.N * self.number_multiple
        numbers = random.sample(range(1, range_max + 1), self.N)
        self.reference_numbers = numbers
        self.reference_answer_str = " ".join(map(str, numbers))

        # Compute all pairwise sums and shuffle
        sums: List[int] = []
        for i, xi in enumerate(numbers):
            for xj in numbers[i + 1:]:
                sums.append(xi + xj)
        assert len(sums) == self.N * (self.N - 1) // 2, "sums should have exactly N * (N - 1) / 2 elements"
        random.shuffle(sums)
        self.sums = sums

        # Build problem description
        problem_text = (
            f"Please find {self.N} distinct positive integers such that the sums of all "
            f"{self.N} * ({self.N} - 1) / 2 distinct pairs among them (in any order) are exactly: "
            f"{', '.join(map(str, self.sums))}\n\n"
            f"Output these {self.N} integers, separated by spaces, in \\boxed{{...}} format."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert boxed content to list of integers
        try:
            answer_array = list(map(int, boxed_content.strip().split()))
        except ValueError:
            # Not all items are integers
            info = {"error": "invalid_answer"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate structure: length, distinctness, positivity
        if self.N is None or self.sums is None:
            # Environment was not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}
        if len(answer_array) != self.N:
            info = {
                "error": "wrong_length",
                "expected_length": self.N,
                "received_length": len(answer_array),
            }
            return TERMINAL_STATE, 0.0, True, False, info
        if len(set(answer_array)) != self.N:
            info = {"error": "not_distinct"}
            return TERMINAL_STATE, 0.0, True, False, info
        if not all(x >= 1 for x in answer_array):
            info = {"error": "non_positive"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute intersection/union following original logic
        intersection, union = 0, 0
        gold_basket: Dict[int, int] = {}
        for s in self.sums:
            gold_basket[s] = gold_basket.get(s, 0) + 1
            union += 1
        for i, xi in enumerate(answer_array):
            for xj in answer_array[i + 1:]:
                s = xi + xj
                if gold_basket.get(s, 0) > 0:
                    gold_basket[s] -= 1
                    intersection += 1
                else:
                    union += 1
        assert intersection <= union, "intersection should not exceed union"

        # Correctness: exact multiset match of sums
        is_correct = (intersection == union)

        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_numbers,
            "reference_answer_str": self.reference_answer_str,
            "user_answer": answer_array,
            "sums": self.sums,
            "intersection": intersection,
            "union": union,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid-form answer."""
        if self.N is None:
            # Default to a small N if not yet reset
            N = 3
        else:
            N = self.N
        range_max = N * self.number_multiple
        sample = random.sample(range(1, range_max + 1), N)
        return f"\\boxed{{{' '.join(map(str, sample))}}}"