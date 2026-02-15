from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LIZ_LollipopEnv(Env):
    """
    LIZ Lollipop problem environment (single-turn Q&A).
    Given an array A of length N with elements in {1, 2} and total sum S,
    for each i in [1..S], the agent must output a pair (l, r) (0-indexed, inclusive)
    such that sum(A[l..r]) = i if such a subarray exists, otherwise output -1 -1.
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 50,
        fixed_n: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            min_n: Minimum N to sample when fixed_n is None. Must be >= 3.
            max_n: Maximum N to sample when fixed_n is None. Must be >= min_n.
            fixed_n: If provided, use this fixed N for all episodes. Must be >= 3.
        """
        super().__init__()
        assert min_n >= 3, "min_n should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"
        if fixed_n is not None:
            assert fixed_n >= 3, "fixed_n should be greater than or equal to 3"

        self.min_n = min_n
        self.max_n = max_n
        self.fixed_n = fixed_n

        # Problem state
        self.N: int = 0
        self.A: List[int] = []
        self.S: int = 0
        self.existence: List[bool] = []
        self.reference_answer_lines: List[str] = []
        self.reference_answer_text: str = ""
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Given an array A of 1s and 2s, you must determine, for each i from 1 to S, "
            "whether there exists a contiguous subarray whose sum equals i. If it exists, output "
            "the 0-indexed inclusive bounds l and r; otherwise output -1 -1.\n"
            "Important details:\n"
            "- A[l:r+1] denotes a Python-style slice including A[l] and A[r].\n"
            "- Indices l and r are 0-indexed and inclusive.\n"
            "- You must produce exactly S lines, where S = sum(A).\n"
            "- The i-th line (1-based) corresponds to sum i.\n"
            "- If no such subarray exists for i, output -1 -1 on that line.\n"
            "Output Format:\n"
            "- Put all S lines inside a single \\boxed{...} block.\n"
            "- Inside the box, each line should contain two integers separated by a space.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_n is not None:
            self.N = self.fixed_n
        else:
            self.N = random.randint(self.min_n, self.max_n)

        # Generate A with a single shared probability for value 2
        two_probability = random.random()
        self.A = [2 if random.random() < two_probability else 1 for _ in range(self.N)]
        self.S = sum(self.A)

        # Build reference answers and existence using the original algorithm
        self._build_reference()

        # Build problem statement
        self.current_problem = self._build_problem_text()

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the submitted solution."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the boxed content into list of (l, r)
        parsed_pairs = self._process_answer_text(boxed_content)
        if parsed_pairs is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate length: must be exactly S lines
        if len(parsed_pairs) != self.S:
            info = {
                "error": "wrong_length",
                "expected_lines": self.S,
                "received_lines": len(parsed_pairs),
                "reference_answer": self.reference_answer_text,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate content correctness
        is_fully_correct, num_correct = self._score_answer(parsed_pairs)

        reward = 1.0 if is_fully_correct else 0.0
        info = {
            "correct": is_fully_correct,
            "num_correct": num_correct,
            "total": self.S,
            "reference_answer": self.reference_answer_text,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the last \\boxed{...} content from the text.
        Supports multi-line content inside the box.
        """
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _process_answer_text(self, boxed_text: str) -> Optional[List[Tuple[int, int]]]:
        """
        Parse the multi-line boxed content into a list of (l, r) pairs.
        Returns None on format error.
        """
        pairs: List[Tuple[int, int]] = []
        lines = boxed_text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                return None
            try:
                l = int(parts[0])
                r = int(parts[1])
            except Exception:
                return None
            pairs.append((l, r))
        return pairs

    def _build_problem_text(self) -> str:
        """Construct the problem prompt string."""
        A_str = " ".join(f"A[{i}]={v}" for i, v in enumerate(self.A))
        return (
            f"You are given an array A of length {self.N}: {A_str}\n"
            f"Each element in A is either 1 or 2, and the total sum of the array is {self.S}.\n\n"
            f"You need to output {self.S} lines. For the i-th line (1 ≤ i ≤ {self.S}), output two integers l and r "
            f"(0-indexed, inclusive), separated by a space:\n"
            f"- If there exists a contiguous subarray A[l : r + 1] such that the sum of its elements is exactly i, output l and r.\n"
            f"- If no such subarray exists, output -1 -1.\n\n"
            f"Output Format: Put all {self.S} lines inside a single \\boxed{{...}} block."
        )

    def _build_reference(self) -> None:
        """
        Reproduce the original algorithm to compute one valid interval for reachable sums,
        and record existence and a reference answer.
        """
        N = self.N
        A0 = self.A[:]  # 0-indexed copy for reference and verification

        # Convert to 1-indexed for algorithm
        A = [0] + A0.copy()

        # Prefix sums (1-indexed)
        pref = [0] * (N + 1)
        for i in range(1, N + 1):
            pref[i] = pref[i - 1] + A[i]
        S = pref[N]
        self.S = S

        # l[k], r[k] store 1-indexed inclusive interval for sum k if known
        l = [0] * (S + 3)
        r = [0] * (S + 3)
        # Max[0] = max even sum seen, Max[1] = max odd sum seen
        Max = [-1, -1]

        def up(val: int, ll: int, rr: int) -> None:
            p = val & 1
            if val > Max[p]:
                Max[p] = val
                l[val] = ll
                r[val] = rr

        # Record all prefixes and suffixes
        for i in range(1, N):
            up(S - pref[i], i + 1, N)  # suffix sum
            up(pref[i], 1, i)          # prefix sum
        # Whole array
        up(S, 1, N)

        # Propagate downward from S to 1 by removing 1 or 2 from ends from known k+2
        for k in range(S, 0, -1):
            if l[k] == 0 and r[k] == 0:
                pl, pr = l[k + 2], r[k + 2]
                if pl and pr:
                    ll, rr = pl, pr
                    if A[pl] == 2:
                        ll += 1
                    elif A[pr] == 2:
                        rr -= 1
                    else:
                        ll += 1
                        rr -= 1
                    l[k], r[k] = ll, rr

        # Build existence and reference answer (converted to 0-indexed)
        existence: List[bool] = []
        ref_lines: List[str] = []
        for x in range(1, S + 1):
            if x > S or x > Max[x & 1]:
                ref_lines.append("-1 -1")
                existence.append(False)
            else:
                ref_lines.append(f"{l[x] - 1} {r[x] - 1}")
                existence.append(True)

        self.existence = existence
        self.reference_answer_lines = ref_lines
        self.reference_answer_text = "\n".join(ref_lines)

    def _score_answer(self, pairs: List[Tuple[int, int]]) -> Tuple[bool, int]:
        """
        Check the provided pairs against the problem instance.
        Returns (is_fully_correct, number_of_correct_lines).
        """
        N = self.N
        A = self.A

        # Prefix sums for 0-indexed checking: prefix[i] = sum of A[:i]
        prefix = [0] * (N + 1)
        for i in range(N):
            prefix[i + 1] = prefix[i] + A[i]

        total = len(pairs)
        correct = 0

        for idx in range(total):
            x = idx + 1  # target sum
            l, r = pairs[idx]
            exists = self.existence[idx]

            # Validate pair format: either (-1, -1) or valid index range
            if (l, r) != (-1, -1):
                if not (0 <= l <= r < N):
                    # Invalid indices -> wrong
                    return False, correct

            if exists:
                # Must provide a valid interval with sum exactly x
                if (l, r) == (-1, -1):
                    # Missed existing sum
                    continue
                seg_sum = prefix[r + 1] - prefix[l]
                if seg_sum == x:
                    correct += 1
                else:
                    # Wrong segment sum
                    continue
            else:
                # Must provide -1 -1 to be correct
                if (l, r) == (-1, -1):
                    correct += 1
                else:
                    # Provided an interval when none should exist for x
                    continue

        return (correct == total), correct

    def sample_random_action(self) -> str:
        """
        Sample a naive random action: outputs all -1 -1 lines inside a single box.
        This is unlikely to be correct but demonstrates the expected format.
        """
        lines = ["-1 -1"] * self.S
        return "\\boxed{" + "\n".join(lines) + "}"