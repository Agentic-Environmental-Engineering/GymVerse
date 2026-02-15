from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ValueDiminishingSelectionEnv(Env):
    """Value Diminishing Selection environment - single-turn Q&A.
    
    You are given N items. Each item i has a base value W[i] and a diminishing factor R[i].
    You must select a sequence of distinct items (order matters) to maximize total gain.
    
    Answer format: indices in order, separated by spaces, wrapped in \\boxed{...}.
    For selecting no items, use an empty box: \\boxed{}.
    """

    def __init__(self, N: int, **kwargs) -> None:
        """Initialize the environment with problem size N.
        
        Args:
            N: Number of items (must be >= 2).
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 2, "N should be greater than or equal to 2"
        self.N: int = N

        # Problem state
        self.W: List[int] = []
        self.R: List[int] = []
        self.current_problem: Optional[str] = None
        self.reference_gain: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Select a sequence of distinct items to maximize total gain.\n"
            "- You are given N items labeled from 0 to N-1, each with base value W[i] and diminishing factor R[i].\n"
            "- When selecting the k-th item in your sequence (0-indexed order within your selection), its effective gain is:\n"
            "  W[item] - (sum of R[j] over all previously selected items j).\n"
            "- Equivalently, each item i reduces the gain of every item chosen after it by R[i].\n"
            "- Your goal is to choose the order and subset to maximize total gain.\n\n"
            "Output Format:\n"
            "- Provide the indices of the selected items in order, separated by spaces, in \\boxed{...}.\n"
            "- For selecting no items, submit an empty box: \\boxed{} or \\boxed{ }.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        # Generate W and R following the original logic
        self.W = [random.randint(0, N * N // 2) for _ in range(N)]
        self.R = [random.randint(1, N) for _ in range(N)]

        # Compute the optimal total gain (reference) using the original DP approach
        P = [(Wi, Ri) for Wi, Ri in zip(self.W, self.R)]
        # Sort by R descending
        P.sort(key=lambda x: x[1], reverse=True)

        # dp[j] = best gain picking j items from considered prefix
        dp: List[Optional[int]] = [None] * (N + 1)
        dp[0] = 0
        best = 0  # at least 0 by taking nothing

        for i in range(N):
            Wi, Ri = P[i]
            new_dp = dp.copy()
            # up to i+1 items can be chosen now
            for j in range(1, i + 2):
                prev = dp[j - 1]
                if prev is None:
                    continue
                cand = prev + Wi - Ri * (j - 1)
                if new_dp[j] is None or cand > new_dp[j]:
                    new_dp[j] = cand
                    if cand > best:
                        best = cand
            dp = new_dp

        self.reference_gain = best

        # Build the problem statement
        W_and_R_lines = "\n".join(
            f"W[{i}]={self.W[i]} R[{i}]={self.R[i]}" for i in range(N)
        )
        self.current_problem = (
            f"You are given {N} items labeled from 0 to {N - 1}. "
            f"Each item has a base value W[i] and a diminishing factor R[i]. "
            f"The list of values and diminishing factors is given as:\n{W_and_R_lines}\n\n"
            "You must select a sequence of distinct items (the order matters). When selecting the i-th item:\n"
            "- Its effective value is W[item] minus the total of R[j] for all previously selected items j.\n"
            "- In other words, each item selected after it will lose R[item] from their gain due to the diminishing effect.\n\n"
            "Your goal is to select a sequence of items to maximize the total gain.\n\n"
            "Output Format: Output the indices of the selected items in order, separated by spaces, "
            "wrapped in \\boxed{...}. For selecting no items, use an empty box: \\boxed{}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer and return the result."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error: no valid \\boxed{...} found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices from boxed content
        try:
            content = boxed_content.strip()
            if content == "":
                indices: List[int] = []
            else:
                indices = list(map(int, content.split()))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate indices: distinct and within range
        if len(indices) != len(set(indices)):
            info = {
                "error": "invalid_solution_duplicate_indices",
                "indices": indices,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if not all(0 <= i < self.N for i in indices):
            info = {
                "error": "invalid_solution_index_out_of_range",
                "indices": indices,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Compute user gain following the original scoring logic
        user_gain = 0
        sum_R = 0
        for i in indices:
            Wi, Ri = self.W[i], self.R[i]
            user_gain += Wi - sum_R
            sum_R += Ri
        user_gain = max(0, user_gain)

        assert self.reference_gain is not None, "Reference gain must be computed in reset()"
        correct = (user_gain == self.reference_gain)
        reward: float = 1.0 if correct else 0.0

        info = {
            "correct": correct,
            "reference_gain": self.reference_gain,
            "user_gain": user_gain,
            "indices": indices,
            "N": self.N,
            "W": self.W,
            "R": self.R,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random subset in a random order."""
        idxs = list(range(self.N))
        random.shuffle(idxs)
        k = random.randint(0, self.N)
        chosen = idxs[:k]
        content = " ".join(map(str, chosen))
        return f"\\boxed{{{content}}}"