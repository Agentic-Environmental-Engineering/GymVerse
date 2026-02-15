from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
import re


class KingSortingEnv(Env):
    """King Sorting problem environment - single-turn Q&A.

    The task:
    You are given N + 1 pairs of integers: (A[0], B[0]), (a[1], b[1]), ..., (a[N], b[N]).
    You need to rearrange the N pairs (a[i], b[i]) for 1 ≤ i ≤ N in some order.
    After rearrangement, the new sequence becomes: (A[0], B[0]), (A[1], B[1]), ..., (A[N], B[N]),
    where (A[i], B[i]) for i ≥ 1 comes from the chosen permutation.

    Your goal is to minimize:
    max( A[0] * A[1] * ... * A[i - 1] // B[i] | 1 ≤ i ≤ N )
    where // denotes integer division (floor division in Python).

    Output format:
    Provide the permutation as space-separated integers from 1 to N inside \\boxed{...}.
    """

    def __init__(
        self,
        N: int = 5,
        max_a_b: int = 10,
        **kwargs: Any
    ):
        super().__init__()
        # Parameter validation
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")
        if max_a_b < 1:
            raise ValueError("max_a_b should be greater than or equal to 1")

        self.N: int = N
        self.max_a_b: int = max_a_b

        # Problem state
        self.array: List[Dict[str, int]] = []
        self.gold_answer: Optional[int] = None
        self.reference_permutation: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving the King Sorting problem.\n"
            "Please provide your answer as a single permutation of integers from 1 to N, "
            "space-separated, in \\boxed{...} format.\n"
            "Example: If N = 5, a valid submission is \\boxed{5 4 3 2 1}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate the array of pairs
        self.array = [{"index": index, "A": random.randint(1, self.max_a_b), "B": random.randint(1, self.max_a_b)}
                      for index in range(0, self.N + 1)]

        # Compute the optimal (gold) answer and a reference permutation by sorting by A*B (for indices 1..N)
        array_copy = self.array.copy()
        array_copy[1:] = sorted(array_copy[1:], key=lambda x: x["A"] * x["B"])

        ans = 0
        mult = array_copy[0]["A"]
        for i in range(1, self.N + 1):
            ans = max(ans, mult // array_copy[i]["B"])
            mult *= array_copy[i]["A"]

        self.gold_answer = ans
        self.reference_permutation = [item["index"] for item in array_copy[1:]]
        self.reference_answer_str = " ".join(map(str, self.reference_permutation))

        # Build the problem prompt
        values_lines = ["(A[0], B[0]) = ({}, {})".format(self.array[0]["A"], self.array[0]["B"])]
        values_lines += ["(a[{}], b[{}]) = ({}, {})".format(i, i, self.array[i]["A"], self.array[i]["B"])
                         for i in range(1, self.N + 1)]
        reverse_indices_example = " ".join(str(i) for i in range(self.N, 0, -1))

        self.current_problem = (
            f"You are given `{self.N} + 1 = {self.N + 1}` pairs of integers: "
            f"(A[0], B[0]), (a[1], b[1]), (a[2], b[2]), ..., (a[{self.N}], b[{self.N}])\n"
            f"{chr(10).join(values_lines)}\n\n"
            f"Your task is to rearrange the {self.N} pairs (a[i], b[i]) for 1 ≤ i ≤ {self.N} in some order "
            f"(there are {self.N}! possible permutations). After rearrangement, define the new sequence of "
            f"{self.N + 1} pairs as: (A[0], B[0]), (A[1], B[1]), ..., (A[{self.N}], B[{self.N}]), where (A[i], B[i]) "
            f"comes from the chosen permutation for i ≥ 1.\n\n"
            f"Your goal is to minimize the following value:\n"
            f"max( A[0] * A[1] * ... * A[i - 1] // B[i] | 1 ≤ i ≤ {self.N} )\n"
            f"Note: // means integer division (floor division in Python).\n\n"
            f"For each i from 1 to {self.N}, compute the product of all previous A values (A[0] to A[i - 1]) divided by B[i], "
            f"take the maximum of these, and find a permutation that minimizes this maximum.\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single line containing a permutation of integers from 1 to {self.N} "
            f"(space-separated) in \\boxed{{...}} format.\n"
            f"Example: \\boxed{{{reverse_indices_example}}} (do NOT include the backticks or quotes); this means: "
            f"(A[1], B[1]) = (a[{self.N}], b[{self.N}]), (A[2], B[2]) = (a[{self.N - 1}], b[{self.N - 1}]), ..., "
            f"(A[{self.N}], B[{self.N}]) = (a[1], b[1])\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted permutation and compute reward."""
        # Parse boxed answer content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert boxed content to a list of integers
        try:
            permutation = list(map(int, boxed_content.split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate permutation
        if len(permutation) != self.N or len(set(permutation)) != self.N or any(not (1 <= i <= self.N) for i in permutation):
            info = {
                "error": "invalid_solution",
                "user_permutation": permutation,
                "expected_length": self.N
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute the objective value for the submitted permutation
        array_ordered = [self.array[0]] + [self.array[i] for i in permutation]
        answer_value = 0
        mult = array_ordered[0]["A"]
        for i in range(1, self.N + 1):
            answer_value = max(answer_value, mult // array_ordered[i]["B"])
            mult *= array_ordered[i]["A"]

        is_correct = (answer_value == self.gold_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "gold_answer": self.gold_answer,
            "computed_answer": answer_value,
            "reference_permutation": self.reference_permutation,
            "reference_answer_str": self.reference_answer_str,
            "user_permutation": permutation
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid permutation action."""
        perm = list(range(1, self.N + 1))
        random.shuffle(perm)
        return f"\\boxed{{{' '.join(map(str, perm))}}}"