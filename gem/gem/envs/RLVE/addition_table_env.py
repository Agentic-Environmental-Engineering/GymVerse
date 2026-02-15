from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class AdditionTableEnv(Env):
    """Unknown base-N addition table environment - Single turn Q&A.

    The environment generates an unknown base-N number system (N in [3, 26]),
    shuffles N distinct digit symbols (letters a..), and provides the complete
    addition table in that system using the shuffled digits. The agent must
    recover N (in decimal) and the mapping from letters a.. to decimal digits
    0..N-1, outputting the answer in \\boxed{...} format.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 26,
        **kwargs: Any,
    ):
        super().__init__()
        # Parameter validation
        assert isinstance(min_N, int) and isinstance(max_N, int), "min_N and max_N must be integers"
        assert 3 <= min_N <= max_N <= 26, "N should be in the range [3, 26]"
        if N is not None:
            assert isinstance(N, int), "N must be an integer when provided"
            assert min_N <= N <= max_N, "N should be in the range [min_N, max_N]"

        self.min_N = min_N
        self.max_N = max_N
        self.fixed_N = N

        # Runtime state
        self.current_N: Optional[int] = None
        self.digit2letter: Optional[List[str]] = None
        self.letter2digit: Optional[Dict[str, int]] = None
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a puzzle about an unknown base-N positional number system.\n"
            "Please provide your final answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample or set N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        self.current_N = N

        # Create digit symbols (letters a..)
        letters = [chr(i) for i in range(97, 97 + N)]  # ['a', 'b', ..., ]
        digit2letter = letters[:]
        random.shuffle(digit2letter)
        self.digit2letter = digit2letter
        self.letter2digit = {letter: digit for digit, letter in enumerate(digit2letter)}

        # Build equations for the addition table
        equations: List[str] = []
        for a_ascii in range(97, 97 + N):
            for b_ascii in range(a_ascii, 97 + N):
                a = chr(a_ascii)
                b = chr(b_ascii)
                sum_value = self.letter2digit[a] + self.letter2digit[b]
                equations.append(f"{a} + {b} = {self._convert_to_expression(sum_value)}")
        equations_text = "\n".join(equations)

        # Build reference answer: "N d_a d_b ... d_<last_letter>"
        gold_digits = [self.letter2digit[chr(i)] for i in range(97, 97 + N)]
        self.reference_answer = f"{N} " + " ".join(str(x) for x in gold_digits)

        # Construct the problem text
        all_letters_text = ", ".join(letters)
        example_digits = " ".join(str(i) for i in range(N))
        example_mapping = ", ".join([f"{chr(i)}={i-97}" for i in range(97, 97 + N)])

        problem_text = (
            f"You are given an unknown base-N number system (N is an integer ≥ 3), and {N} distinct digits "
            f"{all_letters_text} in that system. The digits satisfy the following equations in base-N:\n\n"
            f"{equations_text}\n\n"
            "Note:\n"
            f"- {all_letters_text} are distinct digits in the range [0, N−1].\n"
            "- Expressions like ba represent base-N numbers formed by concatenation. For example, if a=1 and b=2, then ba = \"21\" in base-N.\n\n"
            "Your task is to find the correct base N (in decimal), and the values of these digits (also in decimal) that satisfy all the equations.\n\n"
            "Output Format:\n"
            "Your final answer should be a single line containing N followed by the values of the digits a.. in order, separated by spaces, all wrapped in \\boxed{...}.\n"
            "For example: \\boxed{N d_a d_b d_c ...}, meaning N is the base and d_a is the value of 'a', d_b is the value of 'b', etc.\n"
            f"Illustration example: \\boxed{{{N} {example_digits}}} means N={N}, and {example_mapping}.\n"
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Process the submitted answer and return the terminal result."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure current problem exists
        if self.current_N is None or self.letter2digit is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Parse integers
        try:
            parts = boxed_content.strip().split()
            numbers = list(map(int, parts))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length
        expected_len = self.current_N + 1
        if len(numbers) != expected_len:
            info = {
                "error": "invalid_answer_length",
                "expected_length": expected_len,
                "received_length": len(numbers),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        user_N = numbers[0]
        user_digits = numbers[1:]
        gold_digits = [self.letter2digit[chr(i)] for i in range(97, 97 + self.current_N)]
        gold_N = self.current_N

        is_correct = (user_N == gold_N) and all(a == b for a, b in zip(gold_digits, user_digits))
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": boxed_content.strip(),
            "user_N": user_N,
            "user_digits": user_digits,
            "gold_N": gold_N,
            "gold_digits": gold_digits,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} in the text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _convert_to_expression(self, n: int) -> str:
        """Convert a non-negative integer to its representation in the unknown base using current digit2letter."""
        assert self.current_N is not None and self.digit2letter is not None, "Problem not initialized"
        N = self.current_N
        if n == 0:
            return self.digit2letter[0]
        expression = ""
        while n > 0:
            digit = n % N
            expression = self.digit2letter[digit] + expression
            n //= N
        return expression

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        if self.current_N is None:
            # Fallback if called before reset
            return "\\boxed{5 0 1 2 3 4}"
        N = self.current_N
        # Randomly choose whether to use the correct N or a random N in range
        use_correct_N = random.random() < 0.5
        chosen_N = N if use_correct_N else random.randint(self.min_N, self.max_N)
        # Random digits (may not be a valid permutation; it's just a random guess)
        digits = [random.randint(0, max(1, N - 1)) for _ in range(N)]
        return "\\boxed{" + " ".join([str(chosen_N)] + [str(x) for x in digits]) + "}"