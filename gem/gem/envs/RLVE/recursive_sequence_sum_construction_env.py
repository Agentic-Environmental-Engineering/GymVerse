import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class RecursiveSequenceSumConstructionEnv(Env):
    """Environment for a recursive sequence sum construction problem (single-turn Q&A).

    Task:
    Define a sequence F by:
    - F(0) = F0
    - For every integer n >= 1, F(n) = A * F(n - 1) + B

    Output any number of distinct positive (F(0) cannot be included) indices n1, n2, ..., nk (k >= 1),
    separated by spaces, such that: F(n1) + F(n2) + ... + F(nk) = S.

    The answer must be provided in \\boxed{...} format, e.g., \\boxed{1 3 5}.
    """

    def __init__(
        self,
        max_f0: int = 10,
        max_a: int = 10,
        max_b: int = 10,
        n: int = 20,
        a_is_1_probability: float = 0.3,
        **kwargs: Any,
    ):
        super().__init__()
        # Parameter validation (preserves the original constraints)
        if max_f0 < 1:
            raise ValueError("max_f0 should be greater than or equal to 1")
        if max_a < 2:
            raise ValueError("max_a should be greater than or equal to 2")
        if max_b < 1:
            raise ValueError("max_b should be greater than or equal to 1")
        if n < 1:
            raise ValueError("n should be greater than or equal to 1")
        if not (0.0 <= a_is_1_probability <= 1.0):
            raise ValueError("a_is_1_probability should be in [0.0, 1.0]")

        self.max_f0 = max_f0
        self.max_a = max_a
        self.max_b = max_b
        self.n = n
        self.a_is_1_probability = a_is_1_probability

        # Problem state
        self.current_problem: Optional[str] = None
        self.F0: Optional[int] = None
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.S: Optional[int] = None
        self.reference_indices: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return general instructions for the task."""
        return (
            "You are solving a recursive sequence sum construction problem.\n"
            "Provide any set of distinct positive indices n1, n2, ..., nk (k >= 1) such that "
            "their corresponding F(n) values sum to the given target S.\n"
            "Answer format: put the space-separated indices inside \\boxed{...}, e.g., \\boxed{1 3 5}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters for the sequence
        self.F0 = random.randint(0, self.max_f0)
        self.A = 1 if random.random() < self.a_is_1_probability else random.randint(2, self.max_a)
        self.B = random.randint(0, self.max_b)
        N = self.n

        # Build the sequence values up to N to compute S for a random subset of indices
        F = [self.F0]
        for idx in range(1, N + 1):
            F.append(self.A * F[idx - 1] + self.B)

        # Choose a random valid subset of indices and compute the target sum S
        self.reference_indices = random.sample(range(1, N + 1), k=random.randint(1, N))
        self.S = sum(F[n] for n in self.reference_indices)

        # Build the problem description
        self.current_problem = (
            f"Define a sequence F by:\n"
            f"- F(0) = {self.F0}\n"
            f"- For every integer n >= 1, F(n) = {self.A} * F(n - 1) + {self.B}\n\n"
            f"Output any number of distinct positive (F(0) cannot be included) indices n1, n2, ..., nk (k >= 1), "
            f"in one line separated by spaces, such that: F(n1) + F(n2) + ... + F(nk) = {self.S}.\n\n"
            f"Output Format: Put your indices inside \\boxed{{...}}, separated by single spaces."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted indices and provide a reward."""
        # Parse the boxed answer
        inside = self._parse_answer(action)
        if inside is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        inside = inside.strip()
        if not inside:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse integers separated by whitespace
        tokens = inside.split()
        try:
            indices = list(map(int, tokens))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if len(indices) == 0:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validation checks (duplicates and positivity)
        if len(indices) != len(set(indices)) or not all(n >= 1 for n in indices):
            info = {
                "correct": False,
                "reason": "invalid_indices",
                "user_indices": indices,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Core verification logic (preserving the original environment's scorer behavior)
        assert self.F0 is not None and self.A is not None and self.B is not None and self.S is not None
        assert self.reference_indices is not None

        target_S = self.S
        computed_S = 0
        max_user_n = max(indices)

        # Anti-abuse bound from the original scorer
        max_ref = max(self.reference_indices)
        if max_user_n > max_ref * 10:
            info = {
                "correct": False,
                "reason": "index_out_of_bound",
                "user_indices": indices,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        index_set = set(indices)
        Fn_minus_1 = self.F0
        for n in range(1, max_user_n + 1):
            Fn = self.A * Fn_minus_1 + self.B
            if computed_S + Fn > target_S:
                info = {
                    "correct": False,
                    "reason": "sum_exceeds_target",
                    "user_indices": indices,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            if n in index_set:
                computed_S += Fn
            Fn_minus_1 = Fn

        is_correct = (computed_S == target_S)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "user_indices": indices,
            "target_S": target_S,
            "F0": self.F0,
            "A": self.A,
            "B": self.B,
            "N": self.n,
            "reference_indices": self.reference_indices,
            "reference_S": target_S,
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
        """Sample a random action: a random non-empty subset of indices inside \\boxed{...}."""
        k = random.randint(1, self.n)
        indices = sorted(random.sample(range(1, self.n + 1), k=k))
        return f"\\boxed{{{' '.join(map(str, indices))}}}"