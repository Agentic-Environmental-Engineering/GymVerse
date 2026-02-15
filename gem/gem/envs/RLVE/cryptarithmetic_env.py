import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


def Add(a_digits: List[int], b_digits: List[int], base: int) -> List[int]:
    """Add two numbers represented by digit arrays in least-significant-first order under a given base."""
    c_digits = []

    carry = 0
    for i in range(max(len(a_digits), len(b_digits))):
        a = a_digits[i] if i < len(a_digits) else 0
        b = b_digits[i] if i < len(b_digits) else 0

        c = a + b + carry
        carry = c // base
        c_digits.append(c % base)
    if carry > 0:
        c_digits.append(carry)

    return c_digits


class CryptarithmeticEnv(Env):
    """Cryptarithmetic environment in GEM format (single-turn Q&A).

    In a base-N number system, digits are represented by symbols d[0], d[1], ..., d[N-1].
    Each d[i] corresponds to a unique integer in [0, N-1], but their assignments are unknown.
    Given an addition equation written using these symbols, the task is to output a valid assignment
    of values (in decimal) for d[0], d[1], ..., d[N-1] that makes the equation correct.

    The answer must be provided in \\boxed{...} format, containing N integers separated by spaces,
    which represent the values of d[0], d[1], ..., d[N-1] respectively.
    """

    def __init__(
        self,
        N: int = 10,
        addend_length: int = 3,
        wrong_format: float = -1.0,
        not_permutation: float = -0.5,
        rewarding_strategy: str = "mean([gold=answer])^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 3.0,
        **kwargs
    ):
        super().__init__()
        # Parameter validation (preserved from original environment logic)
        assert N >= 2, "N should be greater than or equal to 2"
        assert addend_length >= 1, "addend_length should be greater than or equal to 1"

        # Core parameters
        self.N: int = N
        self.addend_length: int = addend_length

        # Reward-related parameters (stored for completeness; fixed reward scheme is used in step)
        self.wrong_format: float = wrong_format
        self.not_permutation: float = not_permutation
        self.rewarding_strategy: str = rewarding_strategy
        self.rewarding_weight: float = rewarding_weight
        self.rewarding_beta: float = rewarding_beta

        # State variables for current instance
        self.digits: List[int] = []  # assignment for d[0..N-1]: index -> actual digit
        self.addend_1: List[int] = []
        self.addend_2: List[int] = []
        self.sum_result: List[int] = []  # gold result in actual digits
        self.reference_answer: str = ""
        self.current_problem: str = ""

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a cryptarithmetic addition problem in a base-N number system.\n"
            "The system uses symbols d[0], d[1], ..., d[N-1], each corresponding to a unique integer in [0, N-1].\n"
            "A number written as d[i0]d[i1]...d[ik] represents the value d[i0] * N^k + d[i1] * N^(k-1) + ... + d[ik] * N^0.\n"
            "Your task is to find one possible assignment of values (in decimal) for d[0], d[1], ..., d[N-1] such that the given equation holds.\n\n"
            "Output Format: Provide your final answer as N integers in \\boxed{...}, separated by spaces.\n"
            "Example: \\boxed{0 1 2 3 4 5 6 7 8 9}\n"
            "This means d[0] = 0, d[1] = 1, ..., d[N-1] = N-1.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new cryptarithmetic problem."""
        super().reset(seed)

        N = self.N

        # Generate a random assignment of actual digits for d[0..N-1]
        digits = list(range(N))
        random.shuffle(digits)
        self.digits = digits[:]  # index -> actual digit
        self.reference_answer = " ".join(str(digits[i]) for i in range(N))

        # Generate two addends (least-significant-first), ensuring the most significant digit is non-zero
        addend_length = self.addend_length
        self.addend_1 = [random.randint(0 if idx < addend_length - 1 else 1, N - 1) for idx in range(addend_length)]
        self.addend_2 = [random.randint(0 if idx < addend_length - 1 else 1, N - 1) for idx in range(addend_length)]

        # Compute the sum in actual digits (least-significant-first)
        self.sum_result = Add(self.addend_1, self.addend_2, N)

        # Build the human-readable equation using d[index] symbols
        gold_digit2i = {digit: i for i, digit in enumerate(self.digits)}

        def print_dis(ds: List[int]) -> str:
            # Convert least-significant-first actual digits into a string of d[index] from most to least significant
            return "".join(f"d[{gold_digit2i[ds[i]]}]" for i in range(len(ds) - 1, -1, -1))

        addend_1_str = print_dis(self.addend_1)
        addend_2_str = print_dis(self.addend_2)
        sum_result_str = print_dis(self.sum_result)

        # Construct problem prompt
        problem = (
            f"Now consider a number system with base {N}, which uses digits d[0], d[1], ..., d[{N - 1}].\n"
            f"Each d[i] is a unique integer in the range [0, {N - 1}], but their actual values are unknown.\n\n"
            "We define the number `d[i0]d[i1]...d[ik]` to represent the value "
            f"`d[i0] * {N}^k + d[i1] * {N}^(k-1) + ... + d[ik] * {N}^0`,\n"
            "where `d[i]` is the actual digit assigned to index `i`, and the number is visually written using the digits `d[i0]`, `d[i1]`, ..., `d[ik]`.\n\n"
            "You are given the following equation in this unknown base digit system:\n"
            f"{addend_1_str}\n"
            "+\n"
            f"{addend_2_str}\n"
            "=\n"
            f"{sum_result_str}\n\n"
            "Your task is to find one possible assignment of values (in decimal) for d[0], d[1], ..., d[{N - 1}] such that the equation holds true.\n\n"
            "Output Format:\n"
            "Your final answer should be N integers for d[0], d[1], ..., d[N-1], in order, separated by spaces, "
            "and written in \\boxed{...} format.\n"
            f"Example: \\boxed{{{' '.join(str(i) for i in range(N))}}}\n"
        )

        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the submitted assignment and return the result."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to list of integers
        try:
            answer_array = list(map(int, answer_text.split()))
        except ValueError:
            # Content inside boxed is not purely integers
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        N = self.N

        # Validate permutation format
        if len(answer_array) != N:
            info = {"error": "not_permutation", "detail": "length_mismatch", "expected_length": N, "received_length": len(answer_array)}
            return TERMINAL_STATE, 0.0, True, False, info
        if len(set(answer_array)) != N:
            info = {"error": "not_permutation", "detail": "duplicates_present"}
            return TERMINAL_STATE, 0.0, True, False, info
        if any((x < 0 or x >= N) for x in answer_array):
            info = {"error": "not_permutation", "detail": "out_of_range_values"}
            return TERMINAL_STATE, 0.0, True, False, info

        # Check correctness using core verification logic
        digits = answer_array[:]  # index -> actual digit (candidate assignment)

        gold_digit2i = {digit: i for i, digit in enumerate(self.digits)}
        # Map addends from gold actual digits to user's actual digits via indices
        addend_1_user = [digits[gold_digit2i[digit]] for digit in self.addend_1]
        addend_2_user = [digits[gold_digit2i[digit]] for digit in self.addend_2]
        user_sum_result = Add(addend_1_user, addend_2_user, N)
        gold_sum_result = self.sum_result.copy()

        # Align lengths if needed (handle carry differences)
        if len(user_sum_result) < len(gold_sum_result):
            user_sum_result.append(0)
        elif len(user_sum_result) > len(gold_sum_result):
            gold_sum_result.append(0)

        # Convert actual digits back to index representation for comparison
        digit2i = {digit: i for i, digit in enumerate(digits)}
        user_sum_indices = [digit2i[d] for d in user_sum_result]
        gold_sum_indices = [gold_digit2i[d] for d in gold_sum_result]

        is_correct = all(a == b for a, b in zip(user_sum_indices, gold_sum_indices))
        reward = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_assignment": self.reference_answer,
            "user_assignment": answer_array,
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
        """Sample a random valid action (random permutation of digits in boxed format)."""
        arr = list(range(self.N))
        random.shuffle(arr)
        return f"\\boxed{{{' '.join(map(str, arr))}}}"