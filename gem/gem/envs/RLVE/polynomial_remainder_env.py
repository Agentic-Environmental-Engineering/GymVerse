import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PolynomialRemainderEnv(Env):
    """Polynomial remainder problem environment - single-turn Q&A.

    Given polynomials P(x) and Q(x) with integer coefficients, there exists a unique
    polynomial R(x) and a remainder polynomial S(x) such that:
      P(x) = Q(x) * R(x) + S(x),
    where deg(S) < deg(Q) = M. The task is to output the integer coefficients of S(x),
    padded to length M if needed (though in generation we always produce length M).
    """

    def __init__(
        self,
        N: Optional[int] = None,
        M: Optional[int] = None,
        max_weight: int = 5,
        N_min: int = 2,
        N_max: int = 8,
        **kwargs
    ):
        """
        Initialize the PolynomialRemainderEnv instance.

        Parameters:
        - N: Optional degree of P(x), must satisfy N >= 2 if provided.
        - M: Optional degree of Q(x), must satisfy N >= M >= 2 if provided.
        - max_weight: Maximum absolute value of coefficients (except leading term positivity).
        - N_min: Minimum degree for random N generation (inclusive), must be >= 2.
        - N_max: Maximum degree for random N generation (inclusive), must be >= N_min.
        """
        super().__init__()
        # Core parameterization
        if N_min < 2:
            raise ValueError("N_min should be greater than or equal to 2")
        if N_max < N_min:
            raise ValueError("N_max should be greater than or equal to N_min")

        self.N = N
        self.M = M
        self.max_weight = max_weight
        self.N_min = N_min
        self.N_max = N_max

        # State holders for a generated problem
        self.current_problem: Optional[str] = None
        self.reference_answer_list: Optional[List[int]] = None  # List of s_i
        self.reference_answer_str: Optional[str] = None  # "s0 s1 ... s_{M-1}"

        # Coefficients of polynomials for info/debugging
        self.P_coeffs: Optional[List[int]] = None
        self.Q_coeffs: Optional[List[int]] = None
        self.R_coeffs: Optional[List[int]] = None
        self.S_coeffs: Optional[List[int]] = None

        # Degrees for the current problem
        self.current_N: Optional[int] = None
        self.current_M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a polynomial remainder problem.\n"
            "Please provide your answer as the space-separated coefficients of S(x) in \\boxed{...} format.\n"
            "For example: \\boxed{1 -2 3}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N and M
        N = self.N if self.N is not None else random.randint(self.N_min, self.N_max)
        if N < 2:
            raise ValueError("N should be greater than or equal to 2")

        M = self.M if self.M is not None else random.randint(2, N)
        if not (N >= M >= 2):
            raise ValueError("M should be less than or equal to N and greater than or equal to 2")

        self.current_N = N
        self.current_M = M

        # Generate Q(x) of degree M: coefficients of size M + 1
        # Leading coefficient (degree M) must be positive
        Q_coeffs = [random.randint(-self.max_weight, self.max_weight) for _ in range(M)] + [random.randint(1, self.max_weight)]

        # Generate R(x) of degree N - M: coefficients of size (N - M) + 1
        # Leading coefficient must be positive
        R_coeffs = [random.randint(-self.max_weight, self.max_weight) for _ in range(N - M)] + [random.randint(1, self.max_weight)]

        # Generate S(x) with degree < M: coefficients of size M
        S_coeffs = [random.randint(-self.max_weight, self.max_weight) for _ in range(M)]

        # Compute P(x) = Q(x) * R(x) + S(x)
        P_coeffs = [0] * (N + 1)
        for Qi in range(M + 1):
            for Ri in range(N - M + 1):
                P_coeffs[Qi + Ri] += Q_coeffs[Qi] * R_coeffs[Ri]
        for Si in range(M):
            P_coeffs[Si] += S_coeffs[Si]

        self.Q_coeffs = Q_coeffs
        self.R_coeffs = R_coeffs
        self.S_coeffs = S_coeffs
        self.P_coeffs = P_coeffs

        # Reference answer preparations
        reference_answer_list = S_coeffs[:]  # Already of length M
        reference_answer_str = " ".join(map(str, reference_answer_list))
        self.reference_answer_list = reference_answer_list
        self.reference_answer_str = reference_answer_str

        # Build problem statement
        P_repr = " + ".join("({}) * x^{}".format(coefficient, i) for i, coefficient in enumerate(P_coeffs) if coefficient != 0)
        Q_repr = " + ".join("({}) * x^{}".format(coefficient, i) for i, coefficient in enumerate(Q_coeffs) if coefficient != 0)

        problem_text = (
            f"You are given two polynomials:\n"
            f"- P(x) of degree {N}: P(x) = {P_repr}\n"
            f"- Q(x) of degree {M}: Q(x) = {Q_repr}\n\n"
            f"There exists a unique polynomial R(x) such that: P(x) = Q(x) * R(x) + S(x), "
            f"where S(x) is the remainder polynomial and its degree is less than {M}. "
            f"Let the coefficients of S(x) be s_0, ..., s_{M - 1} "
            f"(if the degree of S(x) is less than {M - 1}, pad the remaining coefficients with zeros); "
            f"the coefficients of S(x) are all integers.\n\n"
            f"Output Format: Your final answer should be a single line containing s_0 ... s_{M - 1}, "
            f"separated by spaces, inside \\boxed{{...}} (do NOT include additional quotes)."
        )

        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the provided answer."""
        # Parse boxed answer
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert boxed content to a list of integers
        boxed = boxed.strip()
        if boxed == "":
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        try:
            user_answer_list = list(map(int, boxed.split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate length
        assert self.current_M is not None, "current_M should be set after reset"
        if len(user_answer_list) != self.current_M:
            info = {
                "error": "invalid_length",
                "expected_length": self.current_M,
                "received_length": len(user_answer_list),
                "reference_answer": self.reference_answer_str,
                "user_answer": user_answer_list,
                "correct": False
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Check correctness
        assert self.reference_answer_list is not None, "reference_answer_list should be set after reset"
        is_correct = (user_answer_list == self.reference_answer_list)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "user_answer": user_answer_list,
            "N": self.current_N,
            "M": self.current_M,
            "P_coeffs": self.P_coeffs,
            "Q_coeffs": self.Q_coeffs,
            "R_coeffs": self.R_coeffs,
            "S_coeffs": self.S_coeffs
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random list of M integers within range, boxed."""
        if self.current_M is None:
            # Default to 3 coefficients if reset was not called yet
            m = 3
        else:
            m = self.current_M
        random_coeffs = [random.randint(-self.max_weight, self.max_weight) for _ in range(m)]
        return f"\\boxed{{{' '.join(map(str, random_coeffs))}}}"