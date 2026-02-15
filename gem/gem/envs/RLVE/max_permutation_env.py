from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from functools import cmp_to_key
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
import re


class MaxPermutationEnv(Env):
    """Environment for arranging array elements to form the largest concatenated integer - single-turn Q&A"""

    def __init__(
        self,
        N: int = 5,
        MAX_DIGIT_NUM: int = 3,
        proportion_being_prefix: float = 0.6,
        **kwargs
    ):
        """
        Initialize the MaxPermutationEnv instance.

        Args:
            N (int): Number of elements in the array (must be >= 2).
            MAX_DIGIT_NUM (int): Maximum number of digits for base elements (must be >= 1).
            proportion_being_prefix (float): Proportion of numbers that are prefixes of other numbers (in [0.0, 1.0)).
        """
        super().__init__()
        assert isinstance(N, int) and N >= 2, "N should be greater than or equal to 2"
        assert isinstance(MAX_DIGIT_NUM, int) and MAX_DIGIT_NUM >= 1, "MAX_DIGIT_NUM should be greater than or equal to 1"
        assert 0.0 <= proportion_being_prefix < 1.0, "proportion_being_prefix should be in [0.0, 1.0)"

        self.N = N
        self.max_digit_num = MAX_DIGIT_NUM
        self.proportion_being_prefix = proportion_being_prefix

        self.current_problem: Optional[str] = None
        self.array: Optional[List[str]] = None
        self.reference_indices: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.gold_value: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a maximum concatenation ordering problem.\n"
            "Arrange all indices of the given array to form the largest possible integer when concatenated.\n"
            "Treat the numbers as strings during concatenation (not as digits or arithmetic values).\n"
            "Output Format: Provide your indices separated by spaces inside \\boxed{...}.\n"
            "Example: \\boxed{0 2 1} means use A[0], A[2], A[1] in that order.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem."""
        super().reset(seed)

        # Generate array with controlled prefixes
        N = self.N
        MAX_DIGIT_NUM = self.max_digit_num
        M = N - int(N * self.proportion_being_prefix)
        assert M >= 1, "M should be greater than or equal to 1"

        # Base elements: strings of length MAX_DIGIT_NUM using digits '1' or '2'
        array = ["".join(str(random.randint(1, 2)) for _ in range(MAX_DIGIT_NUM)) for _ in range(M)]
        # Add prefix elements
        for _ in range(N - M):
            prefix = random.choice(array[:M])
            assert len(prefix) == MAX_DIGIT_NUM, "prefix should have the same length as MAX_DIGIT_NUM"
            array.append(prefix[: random.randint(1, MAX_DIGIT_NUM)])
        random.shuffle(array)
        self.array = array

        # Compute reference optimal order using comparator: a+b vs b+a
        def cmp(a: dict, b: dict) -> int:
            av, bv = a["value"], b["value"]
            if av + bv > bv + av:
                return -1
            elif av + bv < bv + av:
                return 1
            else:
                return 0

        items = [dict(index=i, value=a) for i, a in enumerate(array)]
        items.sort(key=cmp_to_key(cmp))
        self.reference_indices = [item["index"] for item in items]
        self.reference_answer_str = " ".join(str(idx) for idx in self.reference_indices)
        self.gold_value = int("".join(item["value"] for item in items))

        # Build problem prompt
        all_indices_example = " ".join(str(i) for i in range(N - 1, -1, -1))
        all_items_example = ", ".join(f"A[{i}]" for i in range(N - 1, -1, -1))
        self.current_problem = (
            f"You are given an array A of {N} positive integers:\n" +
            "\n".join(f"A[{i}]={a}" for i, a in enumerate(array)) + "\n\n" +
            "Your task is to rearrange all the elements of the array (each number must be used exactly once) "
            "to form the largest possible integer when the numbers are concatenated in order. "
            "Treat the numbers as strings during concatenation (not as digits or arithmetic values).\n\n"
            "Output Format:\n"
            "Your final answer should be indices separated by spaces inside \\boxed{...}.\n"
            f"Example: \\boxed{{{all_indices_example}}} means the numbers are used in the order: {all_items_example}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse, validate, and score the submitted answer."""
        answer_content = self._parse_answer(action)
        if answer_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices separated by spaces
        try:
            tokens = answer_content.strip().split()
            user_indices = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate structure
        if self.array is None or self.reference_indices is None or self.gold_value is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        N = self.N
        if len(user_indices) != N:
            info = {
                "error": "invalid_solution",
                "user_indices": " ".join(map(str, user_indices)),
                "reference_indices": self.reference_answer_str,
            }
            return TERMINAL_STATE, 0.0, True, False, info
        if len(set(user_indices)) != N:
            info = {
                "error": "invalid_solution",
                "user_indices": " ".join(map(str, user_indices)),
                "reference_indices": self.reference_answer_str,
            }
            return TERMINAL_STATE, 0.0, True, False, info
        if not all(0 <= i < N for i in user_indices):
            info = {
                "error": "invalid_solution",
                "user_indices": " ".join(map(str, user_indices)),
                "reference_indices": self.reference_answer_str,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute user's concatenated integer
        try:
            user_value = int("".join(self.array[i] for i in user_indices))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        is_correct = (user_value == self.gold_value)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_indices": self.reference_answer_str,
            "user_indices": " ".join(map(str, user_indices)),
            "gold_value": self.gold_value,
            "user_value": user_value,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random permutation of indices wrapped in \\boxed{...}."""
        indices = list(range(self.N))
        random.shuffle(indices)
        return f"\\boxed{{{' '.join(map(str, indices))}}}"