from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TwoSATEnv(Env):
    """Two-SAT environment - single-turn Q&A.

    Task:
      - There are N boolean (0/1) variables x[0], x[1], ..., x[N-1].
      - We generate M disjunctive clauses with two literals each.
      - Each clause is of the form (lit_a | lit_b), where lit is either x[i] or !x[i].
      - The instance is guaranteed to be satisfiable by a hidden assignment generated internally.
      - The agent must provide any assignment that satisfies all clauses.

    Answer format:
      - The final answer must be provided as a space-separated list of N integers (0 or 1),
        enclosed in \\boxed{ ... }.
      - Example: \\boxed{0 1 1 0}
    """

    def __init__(
        self,
        N: Optional[int] = None,
        M: Optional[int] = None,
        min_N: int = 2,
        max_N: int = 20,
        min_M: int = 1,
        max_M: int = 64,
        **kwargs
    ):
        super().__init__()
        # Parameter bounds
        assert isinstance(min_N, int) and isinstance(max_N, int) and min_N >= 2 and max_N >= min_N
        assert isinstance(min_M, int) and isinstance(max_M, int) and min_M >= 1 and max_M >= min_M
        self.min_N = min_N
        self.max_N = max_N
        self.min_M = min_M
        self.max_M = max_M

        # Fixed or randomizable parameters
        if N is not None:
            assert isinstance(N, int) and N >= 2, "N should be an integer >= 2"
        if M is not None:
            assert isinstance(M, int) and M >= 1, "M should be an integer >= 1"
        self.fixed_N = N
        self.fixed_M = M

        # Internal state for current episode
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.current_N: Optional[int] = None
        self.current_M: Optional[int] = None
        self.current_clauses: Optional[List[List[Tuple[int, bool]]]] = None
        self.hidden_solution: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a boolean satisfiability (2-SAT) problem.\n"
            "- There are N boolean (0/1) variables x[0], x[1], ..., x[N-1].\n"
            "- Each clause is a disjunction of two literals. The symbol '|' means OR and '!' means NOT.\n"
            "- Your goal is to provide any assignment that satisfies all clauses.\n"
            "Output Format: Provide a single line with N integers (0 or 1) separated by spaces, enclosed in \\boxed{...}.\n"
            "Example: \\boxed{0 1 0 1}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new 2-SAT instance."""
        super().reset(seed)

        # Determine N and M
        N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
        M = self.fixed_M if self.fixed_M is not None else random.randint(self.min_M, self.max_M)

        assert N >= 2, "N should be greater than or equal to 2"
        assert M >= 1, "M should be greater than or equal to 1"

        # Generate a hidden satisfying assignment
        x = [random.randint(0, 1) for _ in range(N)]
        reference_answer = " ".join(map(str, x))

        # Generate clauses that are satisfied by x
        clauses: List[List[Tuple[int, bool]]] = []
        for _ in range(M):
            while True:
                clause: List[Tuple[int, bool]] = []
                indices = random.sample(range(N), 2)
                all_or = False
                for index in indices:
                    is_positive = (random.random() < 0.5)
                    clause.append((index, is_positive))
                    # Evaluate literal under x
                    lit_val = bool(x[index]) if is_positive else (not bool(x[index]))
                    all_or |= lit_val
                if len(clause) == 2 and all_or:
                    break
            clauses.append(clause)

        # Store internal state
        self.current_N = N
        self.current_M = M
        self.current_clauses = clauses
        self.hidden_solution = x
        self.reference_answer = reference_answer

        # Build problem description
        expressions_str = "\n".join(
            " | ".join(("({}x[{}])".format("" if is_pos else "!", idx)) for (idx, is_pos) in clause)
            for clause in clauses
        )
        example_assignment = " ".join(str(i % 2) for i in range(N))

        self.current_problem = (
            f"There are {N} boolean (0/1) values x[0], x[1], ..., x[{N-1}]. "
            f"Each of the following {M} expressions (`|` means OR, `!` means NOT) must equal 1:\n"
            f"{expressions_str}\n\n"
            f"Please find any solution x[0], x[1], ..., x[{N-1}] that satisfies the conditions above.\n\n"
            f"Output Format: Your final answer should be a single line containing x[0], x[1], ..., x[{N-1}], "
            f"separated by spaces, enclosed in \\boxed{{...}}.\n"
            f"Example: \\boxed{{{example_assignment}}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate the provided assignment."""
        if self.current_N is None or self.current_clauses is None:
            # Environment not properly reset
            return TERMINAL_STATE, -0.1, True, False, {"error": "environment_not_initialized"}

        # Extract content from \boxed{...}
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the assignment from the boxed content
        try:
            tokens = boxed_content.strip().split()
            assignment = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate format: length and values must be 0/1
        if len(assignment) != self.current_N:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
        if not all(v in (0, 1) for v in assignment):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Verify all clauses are satisfied
        def literal_value(idx: int, is_positive: bool) -> bool:
            val = bool(assignment[idx])
            return val if is_positive else (not val)

        all_satisfied = all(
            any(literal_value(idx, is_pos) for (idx, is_pos) in clause)
            for clause in self.current_clauses
        )

        reward: float = 1.0 if all_satisfied else 0.0

        info = {
            "correct": all_satisfied,
            "reference_answer": self.reference_answer,  # One possible satisfying assignment
            "user_answer": assignment,
            "N": self.current_N,
            "M": self.current_M,
            "clauses": self.current_clauses,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within \\boxed{...} from the text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random assignment enclosed in \\boxed{...}."""
        N = self.current_N if self.current_N is not None else self.max_N
        assignment = [str(random.randint(0, 1)) for _ in range(N)]
        return f"\\boxed{{{' '.join(assignment)}}}"