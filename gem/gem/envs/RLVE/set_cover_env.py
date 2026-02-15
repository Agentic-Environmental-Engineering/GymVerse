from typing import Any, Optional, SupportsFloat, Tuple, List, Set
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SetCoverEnv(Env):
    """
    Set Cover environment (exact cover variant) - single-turn Q&A.

    The agent must select a collection of sets such that each item from 0 to N-1
    is covered by exactly one of the selected sets. The answer must be submitted
    in \\boxed{...} format containing space-separated set indices.

    Reward:
    - Correct (exact cover): 1.0
    - Wrong (valid format but incorrect or invalid solution): 0.0
    - Format error (no or invalid \\boxed{...}): -0.1
    """

    def __init__(
        self,
        N: Optional[int] = None,
        max_N: int = 20,
        MAX_M_multiple: int = 2,
        **kwargs,
    ):
        super().__init__()
        # Parameters
        self.fixed_N: Optional[int] = N
        self.max_N: int = max_N
        self.MAX_M_multiple: int = MAX_M_multiple

        # Validate parameters
        if self.fixed_N is not None:
            assert isinstance(self.fixed_N, int), "N must be an integer if provided"
            assert self.fixed_N >= 3, "N should be greater than or equal to 3"
        assert isinstance(self.max_N, int) and self.max_N >= 3, "max_N should be an integer >= 3"
        assert isinstance(self.MAX_M_multiple, int) and self.MAX_M_multiple >= 1, "MAX_M_multiple should be an integer >= 1"

        # State
        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.sets: List[List[int]] = []
        self.reference_indices: List[int] = []

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Set Cover problem with exact coverage.\n"
            "You will be given N items labeled from 0 to N-1 and M sets.\n"
            "Your task is to select a collection of sets such that every item is covered by exactly one of the selected sets.\n"
            "Output Format: Your final answer should be space-separated indices inside \\boxed{...}, for example: \\boxed{0 3 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(3, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N

        # Construct M and sets following the original generation logic
        M = random.randint(3, N * self.MAX_M_multiple)
        constructed_M = random.randint(2, M - 1)

        # First create 'constructed_M' sets that partition all items (exact cover exists)
        Sets: List[List[int]] = [[] for _ in range(constructed_M)]
        for item in range(N):
            Sets[random.randint(0, constructed_M - 1)].append(item)

        # Add remaining random sets
        for _ in range(M - constructed_M):
            existence_probability = random.random()
            Sets.append([item for item in range(N) if random.random() < existence_probability])

        # Pair each set with a flag indicating whether it belongs to the exact partition
        sets_with_flag = [(s, True) for s in Sets[:constructed_M]] + [(s, False) for s in Sets[constructed_M:]]
        # Filter out empty sets
        sets_with_flag = [(s, f) for (s, f) in sets_with_flag if len(s) > 0]
        random.shuffle(sets_with_flag)

        # Store sets and reference (indices of sets that form the exact cover)
        self.sets = [s for (s, _) in sets_with_flag]
        self.reference_indices = [idx for idx, (_, f) in enumerate(sets_with_flag) if f]

        # Build problem prompt
        lines = []
        lines.append(f"You are given {N} items labeled from 0 to {N - 1}, and {len(self.sets)} sets labeled from 0 to {len(self.sets) - 1}. Each set is a subset of the items:")
        for index, s in enumerate(self.sets):
            content = ", ".join(map(str, s))
            lines.append(f"Set {index}: {{ {content} }}")
        lines.append(
            "Your task is to select a collection of sets such that every item is covered by exactly one of the selected sets."
        )
        lines.append(
            "Output Format: Your final answer should be a single line containing the indices of the selected sets, separated by spaces, in \\boxed{...}. Example: \\boxed{0 3}"
        )
        self.current_problem = "\n".join(lines)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided answer and return the result."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse indices from boxed content
        try:
            indices: List[int] = []
            tokens = boxed_content.strip().split()
            if len(tokens) > 0:
                indices = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate and score according to original logic (without partial credit)
        info: dict[str, Any] = {}
        is_correct = False

        if self.N is None or not self.sets:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem"}

        M = len(self.sets)
        chosen: Set[int] = set(indices)

        # Validate indices
        for idx in chosen:
            if not (0 <= idx < M):
                info["error"] = "index_out_of_range"
                return TERMINAL_STATE, 0.0, True, False, info

        # Check for overlaps and build union
        union: Set[int] = set()
        for idx in chosen:
            current = set(self.sets[idx])
            if union & current:
                info["error"] = "overlap_violation"
                return TERMINAL_STATE, 0.0, True, False, info
            union |= current

        # Check exact coverage
        if len(union) == self.N:
            is_correct = True
            reward = 1.0
        else:
            reward = 0.0

        info.update(
            {
                "correct": is_correct,
                "reference_answer": " ".join(map(str, self.reference_indices)),
                "user_answer": " ".join(map(str, indices)),
                "N": self.N,
                "M": M,
            }
        )
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        import re

        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in \\boxed{...} format."""
        if not self.sets:
            # Fallback random action
            return "\\boxed{}"
        M = len(self.sets)
        k = random.randint(0, M)  # number of indices to include
        indices = random.sample(range(M), k) if k > 0 else []
        return f"\\boxed{{{' '.join(map(str, sorted(indices)))}}}"