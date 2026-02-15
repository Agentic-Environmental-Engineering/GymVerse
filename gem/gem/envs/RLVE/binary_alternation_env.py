from typing import Any, List, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BinaryAlternationEnv(Env):
    """Binary Alternation problem environment - single-turn Q&A.

    Task:
    - You are given a binary string consisting of '0' and '1', indexed from 0.
    - In one operation, you may swap the characters at indices i and j.
    - Transform the string into an alternating binary string (no two adjacent characters are the same)
      using the minimum number of operations.

    Answer format:
    - Write each operation on its own line in the form: i j
    - Wrap the entire list of operations inside \\boxed{...}
    - Do not include quotes or backticks.
    - If no operation is needed, submit \\boxed{} (an empty box).
    """

    def __init__(
        self,
        zero_count: int = 2,
        **kwargs
    ):
        super().__init__()
        # Validate parameters
        assert zero_count >= 2, "zero_count should be greater than or equal to 2"
        self.zero_count: int = zero_count

        # State for current episode
        self.initial_string: Optional[str] = None
        self.reference_operations: Optional[List[str]] = None
        self.gold_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task description and output requirements."""
        return (
            "You are solving a binary alternation transformation problem.\n"
            "In one operation, you may swap characters at indices i and j (0-indexed).\n"
            "Transform the given string into an alternating binary string using the minimum number of operations.\n"
            "Output Format:\n"
            "- Write each operation on a single line as: i j\n"
            "- Wrap the entire list of operations inside \\boxed{...}\n"
            "- Do not include quotes or backticks.\n"
            "- If no operation is needed, submit \\boxed{} (an empty box).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        zero_count = self.zero_count
        one_count = random.randint(zero_count - 1, zero_count + 1)

        # Construct and shuffle string
        string_list = ["0"] * zero_count + ["1"] * one_count
        random.shuffle(string_list)
        string = "".join(string_list)

        # Compute reference minimal solution
        reference_lines, gold = self._compute_minimal_solution(string)

        # Store episode state
        self.initial_string = string
        self.reference_operations = reference_lines
        self.gold_answer = gold

        # Build problem prompt
        N = len(string)
        problem = (
            f"You are given a binary string of length {N}, consisting of '0's and '1's.\n"
            f"It is 0-indexed: {string}\n\n"
            f"In one operation, you may swap the characters at indices i and j (0 ≤ i, j < {N}).\n"
            f"Please transform the string into an alternating binary string (no two adjacent characters are the same)\n"
            f"using the minimum number of operations.\n\n"
            f"Output Format: Write each operation on a single line as: i j\n"
            f"Wrap the entire list of operations inside \\boxed{{...}}. Do NOT include backticks or quotes.\n"
            f"If no operation is needed, submit \\boxed{{}}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted solution."""
        if self.initial_string is None or self.gold_answer is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: no valid \boxed{...} found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        parse_result = self._parse_operations(boxed_content)
        if parse_result is None:
            # Format error inside the boxed content
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        actions = parse_result  # List of (i, j)
        # Apply actions to a copy of the string
        s_list = list(self.initial_string)
        n = len(s_list)

        # Validate indices during simulation
        for i, j in actions:
            if not (0 <= i < n and 0 <= j < n):
                # Invalid indices -> wrong answer
                info = self._build_info(False, actions, "".join(s_list))
                return TERMINAL_STATE, 0.0, True, False, info
            s_list[i], s_list[j] = s_list[j], s_list[i]

        final_string = "".join(s_list)
        is_alternating = self._is_alternating(final_string)
        is_minimal = (len(actions) == self.gold_answer)

        is_correct = (is_alternating and is_minimal)
        reward = 1.0 if is_correct else 0.0

        info = self._build_info(is_correct, actions, final_string)
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_operations(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """Parse operations from the boxed content.

        Each non-empty line must contain exactly two integers: i and j.
        Returns a list of (i, j) tuples, or None if the format is invalid.
        """
        lines = text.splitlines()
        actions: List[Tuple[int, int]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                return None
            try:
                i = int(parts[0])
                j = int(parts[1])
            except ValueError:
                return None
            actions.append((i, j))
        return actions

    def _is_alternating(self, s: str) -> bool:
        """Check if no two adjacent characters in s are the same."""
        return all(s[i] != s[i + 1] for i in range(len(s) - 1))

    def _compute_minimal_solution(self, s: str) -> Tuple[List[str], int]:
        """Compute a minimal set of swaps to make s alternating, returning (operations, count)."""

        def compute_with_start(start_char: str) -> List[str]:
            zero_to_one: List[int] = []
            one_to_zero: List[int] = []
            expected = start_char
            for idx, ch in enumerate(s):
                if ch != expected:
                    if ch == '0':
                        zero_to_one.append(idx)
                    else:
                        one_to_zero.append(idx)
                expected = '1' if expected == '0' else '0'
            assert len(zero_to_one) == len(one_to_zero), "Mismatch lists must have the same length"
            ops: List[str] = []
            for i, j in zip(zero_to_one, one_to_zero):
                ops.append(f"{i} {j}")
            return ops

        zero_cnt = s.count('0')
        one_cnt = len(s) - zero_cnt

        best_ops: Optional[List[str]] = None

        if zero_cnt >= one_cnt:
            ops0 = compute_with_start('0')
            best_ops = ops0

        if one_cnt >= zero_cnt:
            ops1 = compute_with_start('1')
            if best_ops is None or len(ops1) < len(best_ops):
                best_ops = ops1

        assert best_ops is not None, "Reference solution could not be computed"
        return best_ops, len(best_ops)

    def _build_info(self, correct: bool, actions: List[Tuple[int, int]], final_string: str) -> dict[str, Any]:
        """Build the info dictionary for the step result."""
        return {
            "correct": correct,
            "initial_string": self.initial_string,
            "final_string": final_string,
            "user_operations_count": len(actions),
            "minimal_operations": self.gold_answer,
            "reference_solution": None if self.reference_operations is None else "\n".join(self.reference_operations),
        }

    def sample_random_action(self) -> str:
        """Sample a random action: either empty (no operations) or a single random swap."""
        if self.initial_string is None:
            return r"\boxed{}"
        n = len(self.initial_string)
        if n < 2:
            return r"\boxed{}"
        # Randomly choose to return empty or a single random swap
        if random.random() < 0.5:
            return r"\boxed{}"
        i, j = random.randrange(n), random.randrange(n)
        return f"\\boxed{{{i} {j}}}"