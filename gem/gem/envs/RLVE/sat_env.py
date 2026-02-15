import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SATEnv(Env):
    """Single-turn SAT problem environment in GEM format."""

    prompt_template = (
        "There are {N} boolean (0/1) values x[0], x[1], ..., x[{N_minus_1}]. "
        "Each of the following {M} expressions (`|` means OR, `!` means NOT) must equal 1:\n"
        "{expressions}\n\n"
        "Please find any solution x[0], x[1], ..., x[{N_minus_1}] that satisfies the conditions above.\n\n"
        "Output Format: Your final answer should be a single line containing x[0], x[1], ..., x[{N_minus_1}], "
        "separated by spaces, wrapped in \\boxed{{...}}.\n"
        "Example: \\boxed{{{N_boolean}}}"
    )

    def __init__(
        self,
        N: int = 5,
        M: int = 5,
        density: float = 0.5,
        # The following parameters are preserved for compatibility with the original environment,
        # but are not used to compute rewards in GEM (see requirements).
        wrong_format: float = -1.0,
        rewarding_strategy: str = "(satisfied/all)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 10.0,
        **kwargs: Any
    ):
        """
        Initialize the SATEnv instance.

        Parameters:
        - N: number of boolean variables (N >= 2)
        - M: number of clauses/expressions (M >= 1)
        - density: probability that a variable appears in a clause (0 < density <= 1)

        Notes:
        - Rewards in GEM are:
            correct answer: 1.0
            wrong answer: 0.0
            format error: -0.1
        - The compatibility arguments (wrong_format, rewarding_strategy, rewarding_weight, rewarding_beta)
          are preserved as attributes but not used in reward calculation.
        """
        super().__init__()
        self.N = N
        self.M = M
        self.density = density

        # Compatibility fields (not used for reward calculation in GEM step)
        self.wrong_format = wrong_format
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_weight = rewarding_weight
        self.rewarding_beta = rewarding_beta

        # Runtime fields
        self.current_problem: Optional[str] = None
        self.reference_answer_list: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None
        self.clauses: Optional[List[List[Tuple[int, bool]]]] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are solving a Boolean satisfiability (SAT) problem.\n"
            "Find any assignment to x[0..N-1] such that all given OR-clauses evaluate to 1.\n"
            "Please provide your final answer in \\boxed{...} format, containing N space-separated 0/1 values.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new SAT instance."""
        super().reset(seed)

        # Parameter validation (preserved from original logic)
        assert isinstance(self.N, int) and self.N >= 2, "N should be an integer >= 2"
        assert isinstance(self.M, int) and self.M >= 1, "M should be an integer >= 1"
        assert isinstance(self.density, float) or isinstance(self.density, int), "density must be a number"
        assert 0 < float(self.density) <= 1, "density should be in (0, 1]"

        N = self.N
        M = self.M
        density = float(self.density)

        # Generate a satisfying assignment x
        x = [random.randint(0, 1) for _ in range(N)]
        self.reference_answer_list = x
        self.reference_answer_str = " ".join(map(str, x))

        # Generate clauses such that the reference assignment satisfies all of them
        clauses: List[List[Tuple[int, bool]]] = []
        for _ in range(M):
            while True:
                clause: List[Tuple[int, bool]] = []
                any_true = False
                for index in range(N):
                    if random.random() < density:
                        is_positive = (random.random() < 0.5)
                        clause.append((index, is_positive))
                        xi_true = (x[index] == 1)
                        any_true |= (xi_true if is_positive else (not xi_true))
                if len(clause) >= 2 and any_true:
                    break
            clauses.append(clause)
        self.clauses = clauses

        # Build expressions string for prompt
        expressions = "\n".join(
            " | ".join(
                ("(x[{}])".format(index) if is_positive else "(!x[{}])".format(index))
                for index, is_positive in clause
            )
            for clause in clauses
        )

        # Build problem description
        problem = self.prompt_template.format(
            N=N,
            N_minus_1=N - 1,
            M=M,
            expressions=expressions,
            N_boolean=" ".join(str(i % 2) for i in range(N)),
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the submitted assignment in a single step."""
        # Parse the answer from \boxed{...}
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate that the boxed content is a sequence of integers
        try:
            tokens = boxed.strip().split()
            user_answer_list = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate length and binary constraint
        if self.reference_answer_list is None or self.clauses is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}
        if len(user_answer_list) != self.N:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
        if not all(xi in (0, 1) for xi in user_answer_list):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Compute the number of satisfied clauses
        satisfied = 0
        for clause in self.clauses:
            clause_value = any(
                (user_answer_list[index] == 1) if is_positive else (user_answer_list[index] == 0)
                for index, is_positive in clause
            )
            satisfied += int(clause_value)

        total_clauses = len(self.clauses)
        is_correct = (satisfied == total_clauses)

        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied": satisfied,
            "total_clauses": total_clauses,
            "reference_answer": self.reference_answer_str,
            "user_answer": " ".join(map(str, user_answer_list)),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns None if not found."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random binary assignment wrapped in \\boxed{...}."""
        n = self.N if isinstance(self.N, int) and self.N >= 2 else 2
        rand_bits = [str(random.randint(0, 1)) for _ in range(n)]
        return f"\\boxed{{{' '.join(rand_bits)}}}"