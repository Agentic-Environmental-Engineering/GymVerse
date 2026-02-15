from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinXorPairEnv(Env):
    """Environment for finding a pair (i, j) that minimizes (A[i] AND A[j]) XOR (A[i] OR A[j]).
    
    Single-turn Q&A environment. The agent must output the indices as two integers inside \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 50,
        max_bit_length: int = 10,
        **kwargs
    ):
        """Initialize the MinXorPairEnv.
        
        Parameters:
        - N: If provided, the array length to use; otherwise sampled uniformly from [min_n, max_n].
        - min_n: Minimum N when sampling (must be >= 3).
        - max_n: Maximum N when sampling (must be >= min_n).
        - max_bit_length: Maximum bit length for elements of A (must be >= 1). Elements are sampled from [0, 2^max_bit_length).
        
        Notes:
        - This environment uses fixed reward settings: correct = 1.0, wrong = 0.0, format error = -0.1.
        """
        super().__init__()
        self.fixed_N = N
        self.min_n = min_n
        self.max_n = max_n
        self.max_bit_length = max_bit_length

        # State variables for current instance
        self.current_problem: Optional[str] = None
        self.A: Optional[List[int]] = None
        self.N: Optional[int] = None
        self.reference_pair: Optional[Tuple[int, int]] = None
        self.reference_value: Optional[int] = None

        # Validate init parameters
        if self.min_n < 3:
            raise ValueError("min_n must be greater than or equal to 3")
        if self.max_n < self.min_n:
            raise ValueError("max_n must be greater than or equal to min_n")
        if self.max_bit_length < 1:
            raise ValueError("max_bit_length must be greater than or equal to 1")
        if self.fixed_N is not None and self.fixed_N < 3:
            raise ValueError("N must be greater than or equal to 3 when provided")

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a minimization problem on an integer array.\n"
            "Task: Given an array A of length N (indexed from 0), find a pair of indices (i, j) such that 0 <= i < j < N,\n"
            "that minimizes (A[i] AND A[j]) XOR (A[i] OR A[j]). AND, OR, and XOR denote bitwise operations.\n\n"
            "Output Format: Provide your final answer as two integers i and j separated by a space inside \\boxed{...}.\n"
            "Example: \\boxed{0 2}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_n, self.max_n)
        if N < 3:
            raise ValueError("N must be greater than or equal to 3")

        # Validate bit length
        if self.max_bit_length < 1:
            raise ValueError("max_bit_length must be greater than or equal to 1")

        max_value = 1 << self.max_bit_length
        if N > max_value:
            raise ValueError("N must be less than or equal to 2^max_bit_length to allow sampling unique elements")

        # Generate array A with unique elements
        A = random.sample(range(max_value), N)
        random.shuffle(A)

        # Compute the minimal value and a reference pair using the sorted-adjacent property
        indices = list(range(N))
        indices.sort(key=lambda x: A[x])

        i_ref, j_ref = indices[0], indices[1]
        res_ref = self.compute(A, i_ref, j_ref)
        for _i, _j in zip(indices, indices[1:]):
            _res = self.compute(A, _i, _j)
            if _res < res_ref:
                i_ref, j_ref, res_ref = _i, _j, _res

        # Save state
        self.N = N
        self.A = A
        self.reference_pair = (min(i_ref, j_ref), max(i_ref, j_ref))
        self.reference_value = res_ref

        # Build problem text
        array_lines = "\n".join(f"A[{index}]={a}" for index, a in enumerate(A))
        self.current_problem = (
            f"Given an array of length {N} (index starting from 0):\n"
            f"{array_lines}\n\n"
            "Please find a pair of (i, j) such that 0 <= i < j < {N}, and minimize the value of "
            "(A[i] AND A[j]) XOR (A[i] OR A[j]).\n"
            "Your final answer should be two integers i and j separated by a space in \\boxed{{...}}.\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": N,
            "max_bit_length": self.max_bit_length,
        }

    def compute(self, A: List[int], i: int, j: int) -> int:
        """Compute (A[i] AND A[j]) XOR (A[i] OR A[j]) for given indices."""
        return (A[i] & A[j]) ^ (A[i] | A[j])

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and return the terminal state."""
        if self.A is None or self.N is None or self.reference_value is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Parse answer from \\boxed{...}
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Extract two integers i and j
        try:
            parts = boxed_content.strip().split()
            if len(parts) != 2:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            i = int(parts[0])
            j = int(parts[1])
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate indices
        if not (0 <= i < j < self.N):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_indices"}

        # Compute user's value and compare with reference minimal value
        user_value = self.compute(self.A, i, j)
        is_correct = (user_value == self.reference_value)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_pair": self.reference_pair,
            "reference_value": self.reference_value,
            "user_pair": (i, j),
            "user_value": user_value,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the provided text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action in \\boxed{...} format."""
        if self.N is None:
            # Fallback random small pair if called before reset
            i, j = 0, 1
        else:
            i = random.randint(0, self.N - 2)
            j = random.randint(i + 1, self.N - 1)
        return f"\\boxed{{{i} {j}}}"