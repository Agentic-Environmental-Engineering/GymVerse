from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LIS_LDS_ConcatenationEnv(Env):
    """
    Bitonic subsequence environment (strictly increasing then strictly decreasing, indices strictly increasing).
    Single-turn QA: the agent must output a sequence of indices that forms a valid bitonic subsequence
    and achieves the maximum possible length.
    """

    def __init__(self, N: int = 10, MAX: int = 100, **kwargs):
        """
        Initialize the environment.

        Args:
            N: Length of the array (must be >= 1).
            MAX: Maximum value for elements in the array (inclusive, must be >= 1).
        """
        super().__init__()
        assert isinstance(N, int) and N >= 1, "N should be greater than or equal to 1"
        assert isinstance(MAX, int) and MAX >= 1, "MAX should be greater than or equal to 1"
        self.N: int = N
        self.MAX: int = MAX

        self.array: Optional[List[int]] = None
        self.gold_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """
        Return general task instructions.
        """
        return (
            "You are solving a sequence selection problem.\n"
            "You must select a strictly increasing sequence of indices (0-based) such that their values first strictly increase and then strictly decrease (a bitonic sequence).\n"
            "It is also allowed for the sequence to be entirely strictly increasing or entirely strictly decreasing.\n"
            "Your goal is to maximize the length of the selected sequence.\n"
            "Output Format: Put your selected indices, separated by spaces, inside \\boxed{...}.\n"
            "Example: \\boxed{0 2 3}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment and generate a new problem instance.

        Returns:
            observation: The problem statement as a string.
            info: Additional info dict (empty for this environment).
        """
        super().reset(seed)

        # Generate the random array
        self.array = [random.randint(0, self.MAX) for _ in range(self.N)]

        # Compute the gold answer (maximum length of a bitonic subsequence using indices in increasing order)
        F = [0] * self.N  # LIS ending at i
        G = [0] * self.N  # LDS starting at i
        for i in range(self.N):
            F[i] = 1
            for j in range(i):
                if self.array[j] < self.array[i]:
                    F[i] = max(F[i], F[j] + 1)
        for i in range(self.N - 1, -1, -1):
            G[i] = 1
            for j in range(i + 1, self.N):
                if self.array[i] > self.array[j]:
                    G[i] = max(G[i], G[j] + 1)

        self.gold_answer = 0
        for i in range(self.N):
            self.gold_answer = max(self.gold_answer, F[i] + G[i] - 1)

        # Build the problem statement
        array_str = " ".join(map(str, self.array))
        self.current_problem = (
            f"You are given an array A of length {self.N}. The values are as follows (indexing starts at 0):\n"
            f"{array_str}\n\n"
            "Your task is to select a strictly increasing sequence of indices i1, i2, ..., ik such that:\n"
            f"- 0 ≤ i1 < i2 < ... < ik < {self.N}\n"
            "- Let a[1], a[2], ..., a[k] be the values of A at the selected indices (i.e., a[1] = A[i1], a[2] = A[i2], ..., a[k] = A[ik]).\n"
            "  We want the sequence a[1] < a[2] < ... < a[m] > a[m + 1] > ... > a[k] for some m that satisfies 1 <= m <= k. "
            "In other words, it is allowed for the sequence to first be strictly increasing, then strictly decreasing. "
            "It is also allowed for the sequence to be entirely strictly increasing or entirely strictly decreasing.\n"
            "- Your goal is to maximize the length of the selected sequence k.\n\n"
            "Output Format: Your final answer should be the selected indices separated by spaces, wrapped in \\boxed{...}.\n"
            "Example: \\boxed{0 2 3}\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Validate the submitted answer.

        Args:
            action: The agent's textual output.

        Returns:
            observation: TERMINAL_STATE.
            reward: 1.0 if a valid sequence with maximum possible length is provided; 0.0 otherwise;
                    -0.1 for format errors.
            terminated: True (single-turn).
            truncated: False.
            info: Additional info including correctness and references.
        """
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse indices from boxed content
        indices: List[int] = []
        content = boxed_content.strip()
        if content:
            try:
                indices = list(map(int, content.split()))
            except ValueError:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate indices and sequence
        info: Dict[str, Any] = {}
        if self.array is None or self.gold_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        # Empty sequence is invalid for this task
        if len(indices) == 0:
            info.update({
                "correct": False,
                "reason": "empty_sequence",
                "gold_answer": self.gold_answer,
                "user_length": 0
            })
            return TERMINAL_STATE, 0.0, True, False, info

        # Check indices are within range and strictly increasing
        for i, idx in enumerate(indices):
            if not (0 <= idx < self.N):
                info.update({
                    "correct": False,
                    "reason": "index_out_of_range",
                    "bad_index": idx,
                    "gold_answer": self.gold_answer,
                    "user_length": len(indices)
                })
                return TERMINAL_STATE, 0.0, True, False, info
            if i > 0 and not (indices[i - 1] < idx):
                info.update({
                    "correct": False,
                    "reason": "indices_not_strictly_increasing",
                    "gold_answer": self.gold_answer,
                    "user_length": len(indices)
                })
                return TERMINAL_STATE, 0.0, True, False, info

        values = [self.array[idx] for idx in indices]

        # Check bitonic validity (prefix strictly increasing and suffix strictly decreasing meet at some position)
        k = len(values)
        increasing = [False] * k
        decreasing = [False] * k

        for i in range(k):
            if i == 0:
                increasing[i] = True
            else:
                increasing[i] = increasing[i - 1] and (values[i - 1] < values[i])

        found = False
        for i in range(k - 1, -1, -1):
            if i == k - 1:
                decreasing[i] = True
            else:
                decreasing[i] = decreasing[i + 1] and (values[i] > values[i + 1])
            if increasing[i] and decreasing[i]:
                found = True

        if not found:
            info.update({
                "correct": False,
                "reason": "not_bitonic",
                "gold_answer": self.gold_answer,
                "user_length": k,
                "selected_indices": indices,
                "selected_values": values
            })
            return TERMINAL_STATE, 0.0, True, False, info

        # Determine correctness: must achieve maximal length
        is_correct = (k == self.gold_answer)
        reward: SupportsFloat = 1.0 if is_correct else 0.0

        info.update({
            "correct": is_correct,
            "gold_answer": self.gold_answer,
            "user_length": k,
            "selected_indices": indices,
            "selected_values": values
        })

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside the last \\boxed{...}.
        """
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random action: picks a random non-empty subsequence of indices and returns them boxed.
        This does not guarantee validity or optimality; it is for random exploration only.
        """
        if self.N <= 0:
            return r"\boxed{}"
        # Random length between 1 and N, then pick sorted unique indices
        k = random.randint(1, self.N)
        idxs = sorted(random.sample(range(self.N), k))
        content = " ".join(map(str, idxs))
        return f"\\boxed{{{content}}}"