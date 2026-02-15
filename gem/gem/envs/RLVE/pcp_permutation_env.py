from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PCPPermutationEnv(Env):
    """
    PCP Permutation environment (single-turn Q&A).

    The task:
    Given two arrays of strings A and B (each of length N), find a permutation p_0, ..., p_{N-1}
    of indices 0..N-1 such that concatenating A in that order equals concatenating B in that order.

    Answer format:
    The final answer must be provided inside \\boxed{...} where the content is the permutation
    as N integers separated by spaces, e.g., \\boxed{0 2 1}.
    """

    def __init__(self, N: int = 5, average_length: float = 2.0, **kwargs):
        super().__init__()
        self.N: int = N
        self.average_length: float = average_length

        # Episode state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.A: Optional[List[str]] = None
        self.B: Optional[List[str]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given two arrays of strings A and B, each containing N strings.\n"
            "Your goal is to find a permutation p_0, ..., p_{N-1} of indices 0..N-1 such that:\n"
            "A[p_0] + ... + A[p_{N-1}] equals B[p_0] + ... + B[p_{N-1}] (where + denotes string concatenation).\n\n"
            "Output Format: Your final answer must be the permutation as N integers separated by spaces, "
            "enclosed in \\boxed{...}. For example: \\boxed{0 2 1}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Validate parameters
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 2, "N should be greater than or equal to 2"
        assert isinstance(self.average_length, float) or isinstance(self.average_length, int), "average_length must be a number"
        assert self.average_length >= 1.0, "average_length should be greater than or equal to 1.0"

        N = self.N

        # Generate base string S and split points
        sum_length = max(N + 1, random.randint(N, int(N * self.average_length)))
        probability = random.random()
        S = "".join("ab"[random.random() < probability] for _ in range(sum_length))

        arrays = {}
        for array_name in ("A", "B"):
            endpoints = random.sample(range(1, sum_length), N - 1)
            endpoints.sort()
            endpoints = [0] + endpoints + [sum_length]
            assert len(endpoints) == N + 1, "endpoints should have length N + 1"
            arrays[array_name] = [S[endpoints[i]: endpoints[i + 1]] for i in range(N)]

        # Apply a random permutation to both A and B
        permutation = list(range(N))
        random.shuffle(permutation)
        for array_name in ("A", "B"):
            arrays[array_name] = [arrays[array_name][i] for i in permutation]

        # Compute inverse permutation as the reference answer
        inv_permutation = [None] * N
        for i, p in enumerate(permutation):
            inv_permutation[p] = i
        reference_answer = " ".join(map(str, inv_permutation))

        # Save episode state
        self.A = arrays["A"]
        self.B = arrays["B"]
        self.reference_answer = reference_answer

        # Build problem prompt
        A_and_B_lines = "\n".join(f"A[{i}]={self.A[i]} B[{i}]={self.B[i]}" for i in range(N))
        problem_text = (
            f"You are given two arrays of strings, A and B, each containing {N} strings:\n"
            f"{A_and_B_lines}\n\n"
            f"Find a permutation p_0, ..., p_{N-1} of the indices 0 to {N-1} such that: "
            f"A[p_0] + ... + A[p_{N-1}] is equal to B[p_0] + ... + B[p_{N-1}] "
            f"(here, + denotes string concatenation).\n\n"
            f"Output Format: Your final answer should be the permutation 'p_0 ... p_{N-1}', separated by spaces, "
            f"inside \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Check the submitted answer and return the result."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: no \\boxed{...} found
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse permutation from boxed content
        try:
            permutation = list(map(int, boxed_content.strip().split()))
        except ValueError:
            # Content inside box is not a valid sequence of integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate permutation
        N = self.N
        if len(permutation) != N or len(set(permutation)) != N or not all(0 <= i < N for i in permutation):
            return TERMINAL_STATE, 0.0, True, False, {
                "correct": False,
                "error": "invalid_permutation",
                "user_answer": boxed_content,
                "reference_answer": self.reference_answer
            }

        assert self.A is not None and self.B is not None, "Environment not properly initialized. Call reset() first."

        concatenated_A = "".join(self.A[i] for i in permutation)
        concatenated_B = "".join(self.B[i] for i in permutation)
        is_correct = (concatenated_A == concatenated_B)

        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": boxed_content,
            "permutation": permutation
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
        """Sample a random permutation action in boxed format."""
        perm = list(range(self.N))
        random.shuffle(perm)
        content = " ".join(map(str, perm))
        return f"\\boxed{{{content}}}"