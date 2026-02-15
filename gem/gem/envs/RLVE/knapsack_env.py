import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class KnapsackEnv(Env):
    """0/1 Knapsack environment - single-turn Q&A.

    The agent is given N items with weights W[i] and values V[i], and a maximum weight capacity W_max.
    The task is to select a subset of distinct item indices such that:
      - The total weight does not exceed W_max.
      - The total value is maximized.

    The answer must be provided in \\boxed{...} format, where the content is a single line of
    space-separated item indices, e.g., \\boxed{0 3 5}.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        n_min: int = 2,
        n_max: int = 10,
        value_range_multiple: int = 2,
        **kwargs: Any,
    ):
        """
        Initialize the Knapsack environment.

        Args:
            n: If provided, use this fixed number of items. Must be >= 2.
            n_min: Minimum number of items when sampling N. Must be >= 2.
            n_max: Maximum number of items when sampling N. Must be >= n_min.
            value_range_multiple: Controls the range of values relative to weights. Must be >= 1.
        """
        super().__init__()
        # Validate parameters
        if n is not None:
            assert isinstance(n, int) and n >= 2, "n should be an integer and >= 2"
        else:
            assert isinstance(n_min, int) and n_min >= 2, "n_min should be an integer and >= 2"
            assert isinstance(n_max, int) and n_max >= n_min, "n_max should be an integer and >= n_min"
        assert isinstance(value_range_multiple, int) and value_range_multiple >= 1, \
            "value_range_multiple should be an integer and >= 1"

        self.n_fixed: Optional[int] = n
        self.n_min: int = n_min
        self.n_max: int = n_max
        self.value_range_multiple: int = value_range_multiple

        # Current problem state
        self.N: Optional[int] = None
        self.W: Optional[List[int]] = None
        self.V: Optional[List[int]] = None
        self.W_max: Optional[int] = None
        self.gold_value: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return generic task instructions."""
        return (
            "You are solving a 0/1 knapsack problem.\n"
            "Select a subset of distinct item indices such that the total weight does not exceed the capacity, "
            "and the total value is maximized.\n"
            "Please provide your final selection in \\boxed{...} format. The content should be a single line of "
            "space-separated item indices. Example: \\boxed{0 3}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new knapsack problem.

        Returns:
            observation: The problem description string (instructions + instance).
            info: An optional information dictionary (empty for this environment).
        """
        super().reset(seed)

        # Determine number of items
        if self.n_fixed is not None:
            N = self.n_fixed
        else:
            N = random.randint(self.n_min, self.n_max)
        assert N >= 2, "N should be greater than or equal to 2"

        # Generate weights and values
        W = [random.randint(1, N) for _ in range(N)]
        V = [random.randint(1, Wi * self.value_range_multiple) for Wi in W]
        W_max = random.randint(min(W), sum(W))

        # Compute optimal total value via 0/1 knapsack DP
        F = [0] * (W_max + 1)
        for Wi, Vi in zip(W, V):
            for w in range(W_max, Wi - 1, -1):
                F[w] = max(F[w], F[w - Wi] + Vi)
        gold_value = F[W_max]
        assert gold_value > 0, "The optimal value should be positive."

        # Save state
        self.N = N
        self.W = W
        self.V = V
        self.W_max = W_max
        self.gold_value = gold_value

        # Build problem string
        items_desc = "\n".join(f"W[{i}]={W[i]} V[{i}]={V[i]}" for i in range(N))
        problem = (
            f"You are given {N} items labeled from 0 to {N - 1}. Each item has a weight W[i] and a value V[i]:\n"
            f"{items_desc}\n\n"
            f"Please select a subset of distinct items i_1, i_2, ..., i_k such that:\n"
            f"- The total weight W[i_1] + W[i_2] + ... + W[i_k] is less than or equal to {W_max}, and\n"
            f"- Try your best to maximize the total value V[i_1] + V[i_2] + ... + V[i_k].\n\n"
            f"Output Format: Your final answer should be a single line containing the indices of the selected items, "
            f"separated by spaces, wrapped in \\boxed{{...}}.\n"
            f"Example: \\boxed{{0 {N - 1}}} (do not include quotes); this means you selected items 0 and {N - 1}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the submitted answer.

        Args:
            action: The agent's output text containing \\boxed{...} with indices.

        Returns:
            observation: TERMINAL_STATE for single-turn environment.
            reward: 1.0 if optimal valid solution; 0.0 if invalid or suboptimal; -0.1 for format error.
            terminated: Always True (single-turn).
            truncated: Always False.
            info: Additional diagnostic information.
        """
        # Ensure a problem has been generated
        assert self.N is not None and self.W is not None and self.V is not None and self.W_max is not None and self.gold_value is not None, \
            "Environment not initialized. Call reset() before step()."

        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices list from boxed content
        content = boxed_content.strip()
        indices: List[int] = []
        if content != "":
            tokens = content.split()
            try:
                indices = [int(tok) for tok in tokens]
            except ValueError:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate solution constraints
        N = self.N
        W = self.W
        V = self.V
        W_max = self.W_max
        gold = self.gold_value

        # Distinctness check
        if len(indices) != len(set(indices)):
            info = self._build_info(False, indices, W, V, W_max, gold)
            return TERMINAL_STATE, 0.0, True, False, {**info, "error": "duplicate_indices"}

        # Range check
        if not all(isinstance(i, int) and 0 <= i < N for i in indices):
            info = self._build_info(False, indices, W, V, W_max, gold)
            return TERMINAL_STATE, 0.0, True, False, {**info, "error": "index_out_of_range"}

        total_weight = sum(W[i] for i in indices)
        total_value = sum(V[i] for i in indices)

        # Capacity check
        if total_weight > W_max:
            info = self._build_info(False, indices, W, V, W_max, gold, total_weight, total_value)
            return TERMINAL_STATE, 0.0, True, False, {**info, "error": "over_capacity"}

        # Correctness: must achieve optimal total value
        is_optimal = (total_value == gold)
        reward: float = 1.0 if is_optimal else 0.0

        info = self._build_info(is_optimal, indices, W, V, W_max, gold, total_weight, total_value)
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _build_info(
        self,
        correct: bool,
        indices: List[int],
        W: List[int],
        V: List[int],
        W_max: int,
        gold: int,
        total_weight: Optional[int] = None,
        total_value: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build an information dictionary for diagnostics."""
        if total_weight is None:
            total_weight = sum(W[i] for i in indices if 0 <= i < len(W))
        if total_value is None:
            total_value = sum(V[i] for i in indices if 0 <= i < len(V))
        return {
            "correct": correct,
            "selection": indices,
            "user_total_weight": total_weight,
            "user_total_value": total_value,
            "reference_optimal_value": gold,
            "capacity": W_max,
            "weights": W,
            "values": V,
        }

    def sample_random_action(self) -> str:
        """Sample a random feasible subset and return it in \\boxed{...} format."""
        if self.N is None or self.W is None or self.W_max is None:
            # If not initialized, return empty selection
            return "\\boxed{}"

        indices = list(range(self.N))
        random.shuffle(indices)
        sel: List[int] = []
        current_w = 0
        for i in indices:
            if current_w + self.W[i] <= self.W_max and random.random() < 0.5:
                sel.append(i)
                current_w += self.W[i]
        sel.sort()
        content = " ".join(map(str, sel))
        return f"\\boxed{{{content}}}"