from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TopologicalSortEnv(Env):
    """Topological ordering environment - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 10,
        **kwargs
    ):
        """
        Initialize the TopologicalSortEnv.

        Args:
            N: If provided, use this fixed problem size. Otherwise sample from [min_N, max_N].
            min_N: Minimum N when sampling problems (inclusive).
            max_N: Maximum N when sampling problems (inclusive).
        """
        super().__init__()
        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Runtime state
        self.current_N: Optional[int] = None
        self.before_conditions: List[tuple[int, int]] = []
        self.reference_permutation: List[int] = []
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

        # Validate initialization arguments
        if self.fixed_N is not None:
            assert self.fixed_N >= 3, "N should be greater than or equal to 3"
        assert self.min_N >= 3, "min_N should be greater than or equal to 3"
        assert self.min_N <= self.max_N, "min_N should be less than or equal to max_N"

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a topological ordering problem.\n"
            "Given a set of precedence constraints between integers 0..N-1, "
            "your goal is to output a permutation that satisfies all constraints.\n"
            "Please provide your final permutation inside \\boxed{...}.\n"
            "The permutation must be a single line of N integers separated by spaces.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Choose N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"
        self.current_N = N

        # Generate a hidden reference permutation that will satisfy the constraints
        permutation = list(range(N))
        random.shuffle(permutation)
        self.reference_permutation = permutation[:]
        self.reference_answer = " ".join(map(str, permutation))

        # Generate precedence constraints based on the reference permutation
        before_conditions: List[tuple[int, int]] = []
        for i in range(N):
            if i == 0:
                continue
            # For each position i, randomly choose a non-empty subset of previous indices as predecessors
            predecessors = random.sample(range(i), random.randint(1, i))
            for j in predecessors:
                before_conditions.append((permutation[j], permutation[i]))
        random.shuffle(before_conditions)
        self.before_conditions = before_conditions

        # Build problem statement
        conditions_text = "\n".join(
            f"{j} must be before {i}" for j, i in self.before_conditions
        )
        problem_text = (
            f"Please find a permutation of 0 to {N-1} ({N} integers in total) such that the following conditions are satisfied:\n"
            f"{conditions_text}\n\n"
            "Output Format: Your final answer should be a single line containing the permutation "
            f"p(0), p(1), ..., p({N-1}), separated by spaces, and wrapped inside \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step by validating the provided permutation."""
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure environment has a current problem
        if self.current_N is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Try to parse the permutation from the boxed content
        try:
            tokens = boxed.strip().split()
            submitted_perm = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.current_N
        info: dict[str, Any] = {}

        # Validate permutation structure
        is_length_ok = (len(submitted_perm) == N)
        is_range_ok = all(0 <= x < N for x in submitted_perm)
        is_unique_ok = (len(set(submitted_perm)) == N)

        info["is_length_ok"] = is_length_ok
        info["is_range_ok"] = is_range_ok
        info["is_unique_ok"] = is_unique_ok

        if not (is_length_ok and is_range_ok and is_unique_ok):
            info.update({
                "correct": False,
                "satisfied_count": 0,
                "total_conditions": len(self.before_conditions),
                "reference_answer": self.reference_answer,
                "user_answer": " ".join(map(str, submitted_perm)),
            })
            return TERMINAL_STATE, 0.0, True, False, info

        # Build positions map
        positions = [0] * N
        for idx, p in enumerate(submitted_perm):
            positions[p] = idx

        # Count satisfied constraints
        satisfied = sum(1 for j, i in self.before_conditions if positions[j] < positions[i])
        total = len(self.before_conditions)
        satisfies_all = (satisfied == total)

        reward: float = 1.0 if satisfies_all else 0.0

        info.update({
            "correct": satisfies_all,
            "satisfied_count": satisfied,
            "total_conditions": total,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, submitted_perm)),
        })

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action by generating a random permutation."""
        N = self.current_N if self.current_N is not None else max(self.min_N, 3)
        perm = list(range(N))
        random.shuffle(perm)
        return f"\\boxed{{{' '.join(map(str, perm))}}}"