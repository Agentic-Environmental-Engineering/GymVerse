import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaxMultSplitEnv(Env):
    """Maximize the product by splitting a digit string into exactly K+1 parts - single-turn environment.

    The agent is given a digit string S (digits 1-9, no zeros) of length N.
    The task is to split S into exactly K+1 non-empty contiguous parts (from left to right, preserving order)
    such that the product of the integer values of these parts is maximized.

    Answer format: The agent must output the K+1 parts separated by spaces, inside \\boxed{...}.
    Example: \\boxed{31 2} means the string "312" is split into two parts "31" and "2".
    """

    def __init__(
        self,
        N: Optional[int] = None,
        K: Optional[int] = None,
        min_length: int = 1,
        max_length: int = 20,
        digit_low: int = 1,
        digit_high: int = 9,
        # Preserved reward-related parameters from the original RLVE environment (not used in GEM scoring)
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 3.0,
        **kwargs
    ):
        """Initialize the MaxMultSplitEnv.

        Parameters:
        - N: If provided, the length of the digit string.
        - K: If provided, the number of splits; total parts = K + 1.
        - min_length, max_length: Range for random N generation when N is not provided.
        - digit_low, digit_high: Range for digits used to generate the string (inclusive).
        - wrong_format, invalid_solution, rewarding_strategy, rewarding_weight, rewarding_beta:
          Preserved parameters from the original environment for compatibility; they are not used in GEM scoring.

        Note:
        - If both N and K are provided, the environment will validate that N >= 1, K >= 1, and K + 1 <= N.
        - If N or K is not provided, they will be randomly generated with the constraint K + 1 <= N.
        """
        super().__init__()
        self.fixed_N = N
        self.fixed_K = K
        self.min_length = min_length
        self.max_length = max_length
        self.digit_low = digit_low
        self.digit_high = digit_high

        # Preserve original reward parameters (not used in GEM scoring)
        self.rewards_preserved = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # State variables for the current problem instance
        self.current_problem: Optional[str] = None
        self.string: Optional[str] = None
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.gold_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are given a string of digits S.\n"
            "Your task is to split S into exactly K+1 non-empty contiguous parts (from left to right), "
            "such that the product of the resulting integer values is maximized.\n"
            "Answer format: Provide the K+1 parts separated by spaces inside \\boxed{...}.\n"
            "Example: \\boxed{31 2}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate N and K with validation
        if self.fixed_N is not None and self.fixed_K is not None:
            # Validation for fixed parameters
            assert self.fixed_N >= 1, "N should be greater than or equal to 1"
            assert self.fixed_K >= 1, "K should be greater than or equal to 1"
            assert self.fixed_K + 1 <= self.fixed_N, "K + 1 should be less than or equal to N"
            self.N = self.fixed_N
            self.K = self.fixed_K
        else:
            # Randomly generate N and K with constraints
            N = random.randint(self.min_length, self.max_length)
            # Ensure K >= 1 and K + 1 <= N, i.e., K in [1, N - 1] if N >= 2
            if N == 1:
                # The original logic requires K >= 1 and K + 1 <= N, which is impossible for N == 1.
                # To preserve logic, ensure N >= 2.
                N = max(2, self.min_length + 1)
            K = random.randint(1, N - 1)
            self.N = N
            self.K = K

        # Generate the digit string S (digits in [digit_low, digit_high], non-zero by default)
        self.string = "".join([str(random.randint(self.digit_low, self.digit_high)) for _ in range(self.N)])

        # Compute the maximum product using dynamic programming
        # dpF[k][i] = max product for prefix string[:i+1] with exactly k splits,
        # where the last part is string[j:i+1] and previous product is dpF[k-1][j-1] for j in [1, i]
        dpF: List[List[int]] = [[0] * self.N for _ in range(self.K + 1)]
        for k in range(0, self.K + 1):
            for i in range(self.N):
                if k == 0:
                    dpF[0][i] = int(self.string[: i + 1])
                else:
                    best = 0
                    for j in range(1, i + 1):
                        part_val = int(self.string[j: i + 1])
                        candidate = part_val * dpF[k - 1][j - 1]
                        if candidate > best:
                            best = candidate
                    dpF[k][i] = best

        self.gold_answer = dpF[self.K][self.N - 1]

        # Build problem text
        self.current_problem = (
            f"You are given a string of digits S of length {self.N}:\n"
            f"{self.string}\n\n"
            f"Your task is to divide this string into exactly {self.K + 1} non-empty, non-overlapping parts "
            f"(from left to right, maintaining original order), such that the product of the resulting integer values "
            f"is maximized.\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single line containing the {self.K + 1} parts, separated by spaces, "
            f"wrapped in \\boxed{{...}}.\n"
            f"Example: \\boxed{{31 2}} (do NOT include the quotes); this means the string \"312\" is split "
            f"into two parts: \"31\" and \"2\"."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": self.N,
            "K": self.K,
            "string": self.string,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step: parse and verify the answer."""
        # Extract the boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the split parts from the boxed content
        parts_text = boxed_content.strip()
        if not parts_text:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        tokens = parts_text.split()
        # Validate number of parts
        if self.K is None or self.string is None or self.gold_answer is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        if len(tokens) != self.K + 1:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "wrong_number_of_parts"}

        # Convert tokens to integers
        try:
            parts = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate concatenation equals original string
        if "".join(str(p) for p in parts) != self.string:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "wrong_concatenation"}

        # Compute product and compare to gold
        product_val = 1
        for val in parts:
            product_val *= val

        is_correct = (product_val == self.gold_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.gold_answer,
            "user_answer": product_val,
            "string": self.string,
            "N": self.N,
            "K": self.K,
            "parts": parts,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid split action in \\boxed{...} format."""
        if self.string is None or self.N is None or self.K is None:
            # If environment is not reset, produce a random numeric attempt
            return "\\boxed{1}"

        # Choose K split positions among N-1 possible positions
        split_positions = sorted(random.sample(range(1, self.N), self.K))
        parts: List[str] = []
        prev = 0
        for pos in split_positions + [self.N]:
            parts.append(self.string[prev:pos])
            prev = pos
        return f"\\boxed{{{' '.join(parts)}}}"