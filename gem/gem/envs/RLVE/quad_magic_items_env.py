from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class QuadMagicItemsEnv(Env):
    """Quad Magic Items environment - single-turn Q&A.

    Given N items with positive integer values X[1..N], count for each item how many magic formations
    it participates in as types A, B, C, and D, where a magic formation consists of indices a < b < c < d
    such that:
      - X[a] < X[b] < X[c] < X[d]
      - X[b] - X[a] = 2 × (X[d] - X[c])
      - X[b] - X[a] < (X[c] - X[b]) / 3

    The output should be N lines, where the i-th line contains four integers separated by spaces:
    the counts of item i being used as A, B, C, and D respectively.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 5,
        max_n: int = 100,
        weight_range_multiple: int = 1,
        # Legacy reward configuration parameters preserved (not used in GEM step, kept for compatibility)
        wrong_format: float = -1.0,
        rewarding_strategy: str = "mean([gold=answer])^beta",
        rewarding_beta: float = 10.0,
        rewarding_weight: float = +1.0,
        **kwargs,
    ):
        """Initialize the QuadMagicItemsEnv.

        Parameters:
            N: If provided, fixes the number of items; must be >= 5.
            min_n: Minimum N when sampling randomly; must be >= 5.
            max_n: Maximum N when sampling randomly; must be >= min_n.
            weight_range_multiple: Multiplier controlling the range of item values (values in [1, N * weight_range_multiple]).
            wrong_format, rewarding_strategy, rewarding_beta, rewarding_weight: Preserved legacy parameters (unused in GEM reward).
        """
        super().__init__()
        if N is not None and N < 5:
            raise ValueError("N should be greater than or equal to 5")
        if min_n < 5:
            raise ValueError("min_n should be greater than or equal to 5")
        if max_n < min_n:
            raise ValueError("max_n should be greater than or equal to min_n")
        if weight_range_multiple < 1:
            raise ValueError("weight_range_multiple should be a positive integer")

        self.N_fixed: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n
        self.weight_range_multiple: int = weight_range_multiple

        # Legacy reward config (unused but preserved)
        self.rewards_legacy = {
            "wrong_format": wrong_format,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_beta": rewarding_beta,
            "rewarding_weight": rewarding_weight,
        }

        # State variables
        self.current_problem: Optional[str] = None
        self.reference_answer_str: Optional[str] = None
        self.gold_answer: Optional[List[Tuple[int, int, int, int]]] = None
        self.X: Optional[List[int]] = None
        self.current_N: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions for the environment."""
        return (
            "You are given a set of items with positive values.\n"
            "Your task is to count, for each item, how many magic formations it participates in as A, B, C, and D.\n"
            "A magic formation consists of indices a < b < c < d such that:\n"
            "- X[a] < X[b] < X[c] < X[d]\n"
            "- X[b] - X[a] = 2 × (X[d] - X[c])\n"
            "- X[b] - X[a] < (X[c] - X[b]) / 3\n\n"
            "Output Format:\n"
            "- Your answer must contain N lines.\n"
            "- The i-th line should be four integers: counts of item i being A, B, C, and D.\n"
            "- Wrap your entire output inside \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance.

        Returns:
            A tuple containing the observation string and an info dictionary.
        """
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            N = self.N_fixed
        else:
            N = random.randint(self.min_n, self.max_n)
        self.current_N = N

        # Generate item values X
        X = [random.randint(1, N * self.weight_range_multiple) for _ in range(N)]
        self.X = X

        # Compute the gold answer using the original algorithm
        gold_list, reference_str = self._compute_magic_counts(X)
        self.gold_answer = gold_list
        self.reference_answer_str = reference_str

        # Build the problem prompt
        prompt = (
            f"You are given {N} items, each with a positive value. The values of the items are:\n"
            + " ".join(f"X[{i + 1}]={x}" for i, x in enumerate(X))
            + "\n\n"
            "We say that four items with indices a, b, c, d form a magic formation if their values satisfy:\n"
            "- X[a] < X[b] < X[c] < X[d]\n"
            "- X[b] - X[a] = 2 × (X[d] - X[c])\n"
            "- X[b] - X[a] < (X[c] - X[b]) / 3\n\n"
            "In such a formation, items a, b, c, and d are called type A, B, C, and D respectively.\n\n"
            "Output Format: Output N lines. The i-th line should contain four integers, representing the number of times "
            "the i-th item is used as an A, B, C, and D item in any valid magic formation. The four values should be "
            "separated by spaces.\n\n"
            "Please provide your entire output wrapped in \\boxed{...}.\n"
        )

        self.current_problem = prompt

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step to verify the user's answer.

        Returns:
            TERMINAL_STATE as observation, reward, terminated=True, truncated=False, and info dict.
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        parsed_matrix = self._process_boxed_content(boxed_content)
        if parsed_matrix is None or self.gold_answer is None or self.current_N is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if len(parsed_matrix) != self.current_N:
            return TERMINAL_STATE, -0.1, True, False, {"error": "length_mismatch"}

        is_correct = parsed_matrix == self.gold_answer
        reward = 1.0 if is_correct else 0.0

        info: dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "user_answer": parsed_matrix,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns the last boxed content if multiple are found."""
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process_boxed_content(self, content: str) -> Optional[List[Tuple[int, int, int, int]]]:
        """Parse the boxed content into a list of 4-tuples per line."""
        try:
            matrix: List[Tuple[int, int, int, int]] = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) != 4:
                        return None
                    a, b, c, d = map(int, parts)
                    matrix.append((a, b, c, d))
            return matrix
        except Exception:
            return None

    def _compute_magic_counts(
        self, X: List[int]
    ) -> Tuple[List[Tuple[int, int, int, int]], str]:
        """Compute the counts for each item used as A, B, C, and D according to the magic formation rules.

        Returns:
            A tuple containing:
              - gold_list: a list of 4-tuples for each item in input order
              - reference_str: a newline-separated string representation of the counts
        """
        MAX = max(X)
        cnt = [0] * (MAX + 1)
        for xi in X:
            cnt[xi] += 1

        # ans_val[v][0] = times value v is used as A
        # ans_val[v][1] = times value v is used as B
        # ans_val[v][2] = times value v is used as C
        # ans_val[v][3] = times value v is used as D
        ans_val: List[List[int]] = [[0, 0, 0, 0] for _ in range(MAX + 1)]

        # Enumerate t ensuring indices stay valid according to the original algorithm
        for t in range(1, (MAX - 2) // 9 + 1):
            # Forward pass: accumulate over d increasing
            s = 0
            for d in range(9 * t + 2, MAX + 1):
                a = d - 9 * t - 1
                b = a + 2 * t
                c = d - t
                s += cnt[a] * cnt[b]
                # add all new magic arrays ending at (c, d)
                ans_val[c][2] += s * cnt[d]   # as C
                ans_val[d][3] += s * cnt[c]   # as D

            # Backward pass: accumulate over a decreasing
            s = 0
            for a in range(MAX - 9 * t - 1, 0, -1):
                b = a + 2 * t
                c = b + 6 * t + 1
                d = c + t
                s += cnt[c] * cnt[d]
                # add all new magic arrays starting at (a, b)
                ans_val[a][0] += s * cnt[b]   # as A
                ans_val[b][1] += s * cnt[a]   # as B

        # Build gold answer list and reference string in input order
        gold_list: List[Tuple[int, int, int, int]] = []
        reference_str_lines: List[str] = []
        for xi in X:
            A_cnt, B_cnt, C_cnt, D_cnt = ans_val[xi]
            gold_list.append((A_cnt, B_cnt, C_cnt, D_cnt))
            reference_str_lines.append(f"{A_cnt} {B_cnt} {C_cnt} {D_cnt}")
        reference_str = "\n".join(reference_str_lines)

        return gold_list, reference_str

    def sample_random_action(self) -> str:
        """Sample a random action: produce N lines of zeros wrapped in \\boxed{...}."""
        n = self.current_N if self.current_N is not None else self.min_n
        lines = ["0 0 0 0" for _ in range(n)]
        return f"\\boxed{{\n" + "\n".join(lines) + "\n}}"