from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NewNimGameEnv(Env):
    """Nim-like game environment - single-turn Q&A.

    The player is given N heaps with sizes A[1..N]. The first round has two phases:
    1) Player may remove any number of entire heaps (possibly zero), but not all heaps.
    2) Opponent may then remove any number of entire heaps (possibly zero), but likewise cannot remove all remaining heaps.
    From the second round onward, standard Nim rules apply on the remaining heaps.

    The task is to choose which heaps to remove in the first move to guarantee a win.
    If multiple winning choices exist, choose one that minimizes the total number of matches removed.
    The answer must be provided as 1-based indices of removed heaps, separated by spaces, in \\boxed{...}.
    If you remove no heap, use \\boxed{} (empty content).
    """

    def __init__(
        self,
        N: int,
        match_number_range_coefficient: int = 2,
        **kwargs
    ):
        """Initialize the environment with parameters.

        Args:
            N: Number of heaps (must be at least 3).
            match_number_range_coefficient: Upper bound scaling for heap sizes. Each heap size is sampled uniformly from [1, N * match_number_range_coefficient].
            **kwargs: Extra arguments (unused).
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N must be at least 3"
        assert isinstance(match_number_range_coefficient, int) and match_number_range_coefficient >= 1, "match_number_range_coefficient must be a positive integer"

        self.N: int = N
        self.match_number_range_coefficient: int = match_number_range_coefficient

        self.A: List[int] = []
        self.current_problem: Optional[str] = None
        self.reference_min_removed_sum: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Nim-like game with a special first round.\n"
            "Provide your chosen heap indices to remove in the first move inside \\boxed{...}.\n"
            "Use 1-based indices, separated by spaces. If removing none, use \\boxed{}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: A string containing instructions and the problem statement.
            info: An empty dict.
        """
        super().reset(seed)

        # Generate heap sizes
        self.A = [random.randint(1, self.N * self.match_number_range_coefficient) for _ in range(self.N)]

        # Compute reference minimal sum of removed matches ensuring remaining heaps form an xor-basis (linear independence under xor)
        self.reference_min_removed_sum = self._compute_min_removed_sum(self.A)

        # Build problem statement
        heaps_description = ", ".join(f"the size of heap {i} is {Ai}" for i, Ai in enumerate(self.A, start=1))
        self.current_problem = (
            f"You are given a Nim-like game with heaps of matches. There are {self.N} heaps with the following sizes (1-indexed): {heaps_description}.\n"
            "Game rules:\n"
            "- First round has two phases:\n"
            "  1) Your move (first player): You may remove any number of entire heaps (possibly zero), but you are not allowed to remove all heaps.\n"
            "  2) Opponent's move (second player): Then the opponent may remove any number of entire heaps (possibly zero), but likewise cannot remove all remaining heaps.\n"
            "- From the second round onward: Standard Nim rules apply on the remaining heaps: players alternate; a move removes any positive number of matches from exactly one heap; the player who takes the last match wins.\n"
            "- Both players play optimally.\n\n"
            "Your task: Choose which heaps to remove in your first move so that you guarantee a win; if multiple winning choices exist, choose one that minimizes the total number of matches you remove (i.e., the sum of sizes of the heaps you remove).\n"
            "Output Format: Put the distinct indices (1-based) of the heaps you remove in \\boxed{...}, separated by spaces. If you remove none, use \\boxed{}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step to verify the user's answer.

        Args:
            action: The model's output containing \\boxed{...} with chosen indices.

        Returns:
            observation: TERMINAL_STATE.
            reward: 1.0 if correct, 0.0 if wrong, -0.1 if format error.
            terminated: True (single-turn environment).
            truncated: False.
            info: Additional information about correctness and errors if any.
        """
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices
        indices: List[int] = []
        content = boxed_content.strip()
        if content == "":
            indices = []
        else:
            tokens = content.split()
            try:
                indices = list(map(int, tokens))
            except ValueError:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate indices
        if len(indices) != len(set(indices)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "duplicate_indices"}
        if not all(1 <= idx <= self.N for idx in indices):
            return TERMINAL_STATE, 0.0, True, False, {"error": "index_out_of_range"}
        if len(indices) == self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "cannot_remove_all"}

        # Check if the remaining heaps form an xor-basis (i.e., guarantee victory)
        if not self._remaining_is_basis(self.A, indices):
            info = {
                "correct": False,
                "expected_min_removed_sum": self.reference_min_removed_sum,
                "user_removed_sum": sum(self.A[i - 1] for i in indices),
                "indices": indices,
                "heaps": self.A,
                "error": "unsuccessful_solution"
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check minimal sum condition
        user_removed_sum = sum(self.A[i - 1] for i in indices)
        is_minimal = (self.reference_min_removed_sum == user_removed_sum)
        reward = 1.0 if is_minimal else 0.0

        info = {
            "correct": is_minimal,
            "expected_min_removed_sum": self.reference_min_removed_sum,
            "user_removed_sum": user_removed_sum,
            "indices": indices,
            "heaps": self.A
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random subset of heap indices inside \\boxed{...}."""
        k = random.randint(0, max(0, self.N - 1))  # cannot remove all heaps
        subset = sorted(random.sample(range(1, self.N + 1), k))
        content = " ".join(map(str, subset))
        return f"\\boxed{{{content}}}"

    @staticmethod
    def _compute_min_removed_sum(A: List[int]) -> int:
        """Compute the minimal total matches to remove so that remaining heaps form an xor-basis.

        This follows the original algorithm:
        - Sort A in descending order.
        - Build a xor linear basis; if an element cannot be added, it contributes to the minimal removal sum.
        """
        if not A:
            return 0
        A_sorted = sorted(A, reverse=True)
        max_bit = max(A_sorted).bit_length()
        D = [0] * max_bit

        def add(x: int) -> bool:
            for i in range(max_bit - 1, -1, -1):
                if (x >> i) & 1:
                    if D[i]:
                        x ^= D[i]
                    else:
                        D[i] = x
                        return True
            return False

        ans = 0
        for x in A_sorted:
            if not add(x):
                ans += x
        return ans

    @staticmethod
    def _remaining_is_basis(A: List[int], removed_indices_1based: List[int]) -> bool:
        """Check if all remaining heaps can be inserted into a xor-basis (i.e., no dependence)."""
        if not A:
            return True
        max_bit = max(A).bit_length()
        D = [0] * max_bit

        removed_flags = [False] * len(A)
        for idx in removed_indices_1based:
            removed_flags[idx - 1] = True

        def add(x: int) -> bool:
            for i in range(max_bit - 1, -1, -1):
                if (x >> i) & 1:
                    if D[i]:
                        x ^= D[i]
                    else:
                        D[i] = x
                        return True
            return False

        for i, Ai in enumerate(A):
            if not removed_flags[i]:
                if not add(Ai):
                    return False
        return True