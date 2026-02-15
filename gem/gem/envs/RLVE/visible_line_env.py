import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List, Set
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class VisibleLineEnv(Env):
    """Environment for the 'Visible Lines from y = +∞' problem - single-turn Q&A."""

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 20,
        **kwargs,
    ):
        """
        Initialize the environment.

        Parameters:
        - min_n: Minimum number of lines (must be >= 3).
        - max_n: Maximum number of lines (must be >= min_n).

        Note:
        - Rewards are fixed: correct = 1.0, wrong = 0.0, format error = -0.1.
        """
        super().__init__()
        if min_n < 3:
            raise ValueError("min_n should be greater than or equal to 3")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")

        self.min_n = min_n
        self.max_n = max_n

        # Problem state
        self.N: int = 0
        self.lines: List[Tuple[int, int]] = []
        self.gold_indices_set: Set[int] = set()
        self.reference_answer_list: List[int] = []
        self.reference_answer_str: str = ""
        self.current_problem: str = ""

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are given N lines on the 2D plane.\n"
            "A line is visible if any portion of it can be seen when viewed from y = +∞ "
            "(looking vertically downward). That is, there exists at least one x-coordinate "
            "such that this line lies on top (has the maximum y-value) at that x among all lines.\n\n"
            "Output Format: Provide the indices (0-based) of all visible lines, in any order, "
            "separated by spaces, and put them in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # Generate number of lines
        self.N = random.randint(self.min_n, self.max_n)

        # Generate unique lines: y = A x + B with A, B in [-N, N]
        unique_lines = set()
        while len(unique_lines) < self.N:
            A = random.randint(-self.N, self.N)
            B = random.randint(-self.N, self.N)
            unique_lines.add((A, B))
        self.lines = list(unique_lines)
        random.shuffle(self.lines)

        # Compute visible lines using the upper hull method
        # Prepare list with indices
        P: List[Tuple[int, int, int]] = [(A, B, i) for i, (A, B) in enumerate(self.lines)]

        # Sort by slope ascending, and for ties by intercept descending
        P.sort(key=lambda x: (x[0], -x[1]))

        # Build the "upper hull" of visible lines
        BIN: List[Tuple[int, int, int]] = []
        prevA: Optional[int] = None
        for A, B, idx in P:
            # Skip duplicate slopes (only keep the one with highest intercept)
            if A == prevA:
                continue
            prevA = A

            # While the last segment and the new point make a non-left turn,
            # pop the last line (it's covered)
            while len(BIN) >= 2:
                A1, B1, _ = BIN[-2]
                A2, B2, _ = BIN[-1]
                # Cross product of vectors (A2-A1, B2-B1) and (A-A2, B-B2)
                if (A2 - A1) * (B - B2) - (B2 - B1) * (A - A2) >= 0:
                    BIN.pop()
                else:
                    break

            BIN.append((A, B, idx))

        # Sort visible lines by original input order (their indices)
        BIN.sort(key=lambda x: x[2])

        # Prepare gold/reference answer
        self.reference_answer_list = [idx for _, _, idx in BIN]
        self.gold_indices_set = set(self.reference_answer_list)
        self.reference_answer_str = " ".join(map(str, self.reference_answer_list))

        # Build the problem prompt
        lines_desc = "\n".join(
            f"Line {i}: y = {A}x + {B}"
            for i, (A, B) in enumerate(self.lines)
        )
        self.current_problem = (
            f"You are given {self.N} lines on the 2D plane:\n"
            f"{lines_desc}\n\n"
            "We say a line is visible if any portion of it can be seen when viewed from y = +∞ (i.e., looking vertically downward). "
            "That is, a line is visible if there exists at least one x-coordinate such that this line lies on top (i.e., has the maximum y-value) "
            "at that x among all lines.\n\n"
            "Output Format: A single line containing the indices (0-based) of all visible lines, in any order, separated by spaces, enclosed in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the user's answer."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Process into a set of integers
        try:
            tokens = boxed_content.strip().split()
            if len(tokens) == 0:
                user_set: Set[int] = set()
            else:
                user_set = set(map(int, tokens))
        except ValueError:
            # Non-integer tokens present
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate index range
        if not all(0 <= x < self.N for x in user_set):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Check correctness (set equality; order does not matter)
        is_correct = (user_set == self.gold_indices_set)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "gold_indices": self.reference_answer_list,
            "user_answer": sorted(user_set),
            "N": self.N,
            "lines": self.lines,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random boxed action (random subset of indices)."""
        if self.N <= 0:
            # In case called before reset, default to empty
            return r"\boxed{}"
        k = random.randint(0, self.N)
        indices = sorted(random.sample(range(self.N), k))
        content = " ".join(map(str, indices))
        return f"\\boxed{{{content}}}"