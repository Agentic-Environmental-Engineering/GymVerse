from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RoundTableAssignmentEnv(Env):
    """Round Table Assignment environment - single-turn Q&A.

    Task:
    - There are M groups and N tables.
    - The i-th group consists of R[i] people.
    - The j-th table can seat up to C[j] people.
    - Assign each person to a table such that:
        * No table contains more than one person from the same group.
        * No table exceeds its total capacity.

    Output:
    - Provide M lines, where the i-th line contains R[i] integers (separated by spaces),
      representing the table indices assigned to each person in the i-th group.
    - The entire output must be wrapped in \\boxed{...}.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
        - max_n_m: Upper bound for the number of groups and base tables (must be >= 3).
        """
        super().__init__()
        if max_n_m < 3:
            raise ValueError("max_n_m should be greater than or equal to 3")
        self.max_n_m = max_n_m

        # Problem state
        self.M: Optional[int] = None
        self.N: Optional[int] = None
        self.R: Optional[List[int]] = None
        self.C: Optional[List[int]] = None
        self.reference_answer_matrix: Optional[List[List[int]]] = None
        self.reference_answer_str: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a round table assignment problem.\n"
            "Rules:\n"
            "- No table contains more than one person from the same group.\n"
            "- No table exceeds its total capacity.\n\n"
            "Answer Format:\n"
            "- Output M lines. The i-th line (0-indexed) should contain R[i] integers separated by spaces,\n"
            "  representing the table indices assigned to each person in the i-th group.\n"
            "- Wrap the entire output inside \\boxed{...}. Newlines inside the box are allowed.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate problem parameters
        M = random.randint(2, self.max_n_m)
        R: List[int] = []
        # Create base tables indexed by range(max_n_m)
        tables: List[List[int]] = [[] for _ in range(self.max_n_m)]

        for group_index in range(M):
            # Each group size between 2 and max_n_m (inclusive)
            group_size = random.randint(2, self.max_n_m)
            R.append(group_size)
            # Assign this group's members to distinct tables (sample without replacement)
            table_indices = random.sample(range(self.max_n_m), group_size)
            for table_index in table_indices:
                tables[table_index].append(group_index)

        # Filter out empty tables; remaining tables get reindexed from 0 to N-1
        tables = [table for table in tables if len(table) > 0]
        N = len(tables)
        C = [len(table) for table in tables]

        # Construct a valid reference assignment: each group member goes to a selected table
        reference_answer: List[List[int]] = [[] for _ in range(M)]
        for table_index, table in enumerate(tables):
            for group_index in table:
                reference_answer[group_index].append(table_index)

        # Sanity checks
        assert len(R) == M, "R should have length M"
        assert len(C) == N, "C should have length N"
        assert all(len(answer) == R[group_index] for group_index, answer in enumerate(reference_answer)), \
            "Reference answer does not match the group sizes"

        # Save state
        self.M = M
        self.N = N
        self.R = R
        self.C = C
        self.reference_answer_matrix = reference_answer
        self.reference_answer_str = "\n".join(" ".join(map(str, ans)) for ans in reference_answer)

        # Build problem prompt
        R_str = " ".join(f"R[{i}]={ri}" for i, ri in enumerate(R))
        C_str = " ".join(f"C[{j}]={cj}" for j, cj in enumerate(C))
        self.current_problem = (
            f"There are {M} groups of people and {N} tables.\n"
            f"- The i-th group consists of R[i] people. Array R: {R_str}\n"
            f"- The j-th table can seat up to C[j] people. Array C: {C_str}\n\n"
            "You need to assign each person to a table such that:\n"
            "- No table contains more than one person from the same group.\n"
            "- No table exceeds its total capacity.\n\n"
            "Output Format:\n"
            f"Output {M} lines. The i-th line (0-indexed) should contain R[i] integers (separated by spaces), "
            "representing the table indices assigned to each person in the i-th group.\n"
            "Please wrap your entire output inside \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "M": M,
            "N": N,
            "R": R,
            "C": C,
            "format": "boxed_lines"
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: verify the user's assignment."""
        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse matrix from boxed content
        processed_result = self._process_matrix(boxed_content)
        if processed_result is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        assert isinstance(processed_result, list), "processed_result should be a list"

        # Validate dimensions
        if self.M is None or self.N is None or self.R is None or self.C is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_state_error"}

        if len(processed_result) != self.M:
            info = {"error": "invalid_solution", "reason": "group_count_mismatch", "expected_M": self.M}
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate group-wise assignments and collect table usage
        countings = [0] * self.N
        for group_idx, (answer_line, Ri) in enumerate(zip(processed_result, self.R)):
            if len(answer_line) != Ri:
                info = {
                    "error": "invalid_solution",
                    "reason": "group_line_length_mismatch",
                    "group_index": group_idx,
                    "expected_len": Ri,
                    "actual_len": len(answer_line)
                }
                return TERMINAL_STATE, 0.0, True, False, info

            # Check table index range
            if not all(isinstance(i, int) and 0 <= i < self.N for i in answer_line):
                info = {
                    "error": "invalid_solution",
                    "reason": "table_index_out_of_range",
                    "group_index": group_idx
                }
                return TERMINAL_STATE, 0.0, True, False, info

            # No duplicate tables for a group
            if len(set(answer_line)) != Ri:
                info = {
                    "error": "invalid_solution",
                    "reason": "duplicate_tables_in_group",
                    "group_index": group_idx
                }
                return TERMINAL_STATE, 0.0, True, False, info

            # Count table assignments
            for table_index in answer_line:
                countings[table_index] += 1

        # Capacity check
        satisfied = sum(int(counting <= Ci) for counting, Ci in zip(countings, self.C))
        is_correct = (satisfied == self.N)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied_tables": satisfied,
            "total_tables": self.N,
            "reference_answer": self.reference_answer_str,
            "user_answer": "\n".join(" ".join(map(str, line)) for line in processed_result)
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Supports multi-line content."""
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process_matrix(self, answer: Optional[str]) -> Optional[List[List[int]]]:
        """Parse the boxed content into a matrix of integers."""
        if answer is None:
            return None
        try:
            matrix: List[List[int]] = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    matrix.append(list(map(int, line.split())))
            return matrix
        except ValueError:
            return None

    def sample_random_action(self) -> str:
        """Sample a random action. Returns the reference solution to ensure correctness."""
        if self.reference_answer_str is not None:
            return f"\\boxed{{\n{self.reference_answer_str}\n}}"
        # Fallback: produce a random boxed number (not a valid solution), if state is not ready
        random_answer = random.randint(0, 100)
        return f"\\boxed{{{random_answer}}}"