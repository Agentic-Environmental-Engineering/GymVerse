from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SetSplittingEnv(Env):
    """Set Splitting environment - single-turn Q&A.

    The task is to split the full set S = {0, 1, ..., N-1} into two disjoint subsets S1 and S2,
    such that each provided subset intersects both S1 and S2 (i.e., not fully contained in either).
    The agent must output S1 as space-separated integers inside \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        M: Optional[int] = None,
        max_N: int = 20,
        max_M: int = 12,
        **kwargs,
    ):
        """
        Initialize the SetSplittingEnv instance.

        Parameters:
            N: Fixed size of the full set S. If None, N will be sampled at reset.
            M: Fixed number of subsets. If None, M will be sampled at reset.
            max_N: Upper bound for sampling N when N is None. Must be >= 3.
            max_M: Upper bound for sampling M when M is None. Must be >= 2.
        """
        super().__init__()
        self.fixed_N = N
        self.fixed_M = M
        self.max_N = max_N
        self.max_M = max_M

        # Internal state for the current problem
        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.Sets: List[List[int]] = []
        self.reference_partition_S1: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Set Splitting problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Validate sampling bounds
        assert self.max_N >= 3, "max_N must be >= 3"
        assert self.max_M >= 2, "max_M must be >= 2"

        # Determine N and M
        self.N = self.fixed_N if self.fixed_N is not None else random.randint(3, self.max_N)
        self.M = self.fixed_M if self.fixed_M is not None else random.randint(2, self.max_M)

        # Validation logic from the original environment
        assert self.N >= 3, "N should be greater than or equal to 3"
        assert self.M >= 2, "M should be greater than or equal to 2"

        # Generate a valid partition S1, S2
        k = random.randint(1, self.N - 1)
        S1 = random.sample(range(self.N), k=k)
        S2 = list(set(range(self.N)) - set(S1))
        assert S1 and S2, "S1 and S2 must be non-empty"
        self.reference_partition_S1 = S1[:]  # keep a reference solution (not used for scoring)

        # Generate subsets that must be split across S1 and S2
        self.Sets = []
        for _ in range(self.M):
            # Must pick at least one from S1 and one from S2
            part1 = random.sample(S1, k=random.randint(1, len(S1)))
            part2 = random.sample(S2, k=random.randint(1, len(S2)))
            subset = part1 + part2
            random.shuffle(subset)
            self.Sets.append(subset)

        # Build problem prompt
        sets_str = "\n".join("{ " + ", ".join(map(str, subset)) + " }" for subset in self.Sets)
        self.current_problem = (
            f"Define the full set S as all {self.N} integers from 0 to {self.N - 1}.\n\n"
            "Your task is to partition S into two disjoint subsets S1 and S2 such that:\n"
            "- S1 ∪ S2 = S and S1 ∩ S2 = ∅\n"
            "- For each of the following subsets (each a subset of S), the subset is not fully contained in either S1 or S2. "
            "That is, each subset must contain at least one element from S1 and at least one element from S2.\n\n"
            f"The list of {self.M} subsets is as follows:\n{sets_str}\n\n"
            "Output Format: Your final answer should be a single line containing the elements of S1, separated by spaces, "
            "enclosed in a single \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided partition."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure problem exists
        if self.N is None or self.M is None or not self.Sets:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Process the boxed content into a list of integers
        try:
            if boxed_content.strip() == "":
                # Empty S1 is invalid
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_partition", "reason": "empty_S1"}
            answer_array = list(map(int, boxed_content.split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate integers are within range and unique
        if not all(0 <= x < self.N for x in answer_array):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_partition", "reason": "out_of_range"}
        if len(set(answer_array)) != len(answer_array):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_partition", "reason": "duplicates"}

        S1 = set(answer_array)
        S2 = set(range(self.N)) - S1

        # S1 and S2 must be non-empty
        if len(S1) == 0 or len(S2) == 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_partition", "reason": "empty_S1_or_S2"}

        # Check the constraints for each subset
        satisfied_count = 0
        for subset in self.Sets:
            subset_set = set(subset)
            # The subset must not be fully contained in S1 or S2,
            # equivalently it must intersect both S1 and S2.
            if bool(subset_set & S1) and bool(subset_set & S2):
                satisfied_count += 1

        all_satisfied = (satisfied_count == self.M)
        reward = 1.0 if all_satisfied else 0.0

        info = {
            "correct": all_satisfied,
            "satisfied_count": satisfied_count,
            "total_subsets": self.M,
            "N": self.N,
            "M": self.M,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by proposing a random S1 inside \\boxed{...}."""
        if self.N is None:
            # No active problem, propose an empty box to trigger format error or no problem error
            return "\\boxed{}"

        k = random.randint(1, self.N - 1)
        proposal = random.sample(range(self.N), k=k)
        proposal.sort()
        return "\\boxed{" + " ".join(map(str, proposal)) + "}"