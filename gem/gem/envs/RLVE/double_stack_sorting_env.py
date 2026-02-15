import random
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DoubleStackSortingEnv(Env):
    """Double Stack Sorting environment - single-turn Q&A."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 4,
        max_N: int = 100,
        **kwargs
    ):
        """
        Initialize the DoubleStackSortingEnv.

        Parameters:
        - N: If provided, use this fixed problem size (must be >= 4).
        - min_N: Minimum value for N when sampling (must be >= 4).
        - max_N: Maximum value for N when sampling (must be >= min_N).
        """
        super().__init__()
        if N is not None:
            assert N >= 4, "N should be greater than or equal to 4"
        assert min_N >= 4, "min_N should be greater than or equal to 4"
        assert max_N >= min_N, "max_N should be greater than or equal to min_N"

        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N

        self.current_problem: Optional[str] = None
        self.N: Optional[int] = None
        self.reference_operations: Optional[str] = None
        self.reference_output_sequence: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task description and output format."""
        return (
            "You are given a queue of integers containing N elements: 0 at the front and N-1 at the back. "
            "You also have two empty stacks, S1 and S2, and an initially empty output sequence. You may perform the following operations:\n"
            "- a: Pop the front of the queue and push it onto S1.\n"
            "- b: Pop the top of S1 and append it to the output sequence.\n"
            "- c: Pop the front of the queue and push it onto S2.\n"
            "- d: Pop the top of S2 and append it to the output sequence.\n\n"
            "Find a sequence of operations that transforms the initial queue into the provided output sequence.\n\n"
            "Output Format: Provide the sequence of operations (a, b, c, d) as a single string without spaces, enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 4, "N should be greater than or equal to 4"
        self.N = N

        # Generate operation distribution and reference solution by simulation
        operation_distribution = [random.randint(1, N) for _ in range(4)]
        total = sum(operation_distribution)
        operation_weights = [w / total for w in operation_distribution]

        self.reference_operations = ""
        S1: List[int] = []
        S2: List[int] = []
        output_sequence: List[int] = []
        queue_front = 0

        while len(output_sequence) < N:
            op = random.choices(["a", "b", "c", "d"], weights=operation_weights, k=1)[0]
            if op == "a" and queue_front < N:
                self.reference_operations += "a"
                S1.append(queue_front)
                queue_front += 1
            elif op == "b" and S1:
                self.reference_operations += "b"
                output_sequence.append(S1.pop())
            elif op == "c" and queue_front < N:
                self.reference_operations += "c"
                S2.append(queue_front)
                queue_front += 1
            elif op == "d" and S2:
                self.reference_operations += "d"
                output_sequence.append(S2.pop())

        assert len(self.reference_operations) == N * 2, "reference_operations should have length 2 * N"
        self.reference_output_sequence = output_sequence

        # Build problem statement
        problem_text = (
            f"You are given a queue of integers containing {N} elements: 0 at the front and {N - 1} at the back.\n"
            f"Please find a sequence of operations that transforms the initial queue into the output sequence:\n"
            f"{' '.join(map(str, self.reference_output_sequence))}\n\n"
            f"Output Format: A single string containing only the letters a, b, c, d, enclosed in \\boxed{{...}}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the answer."""
        # Parse boxed answer
        answer = self._parse_answer(action)
        if answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.N is None or self.reference_output_sequence is None or self.reference_operations is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_problem"}

        user_ops = answer.strip()

        # Simulate the user's operations
        S1: List[int] = []
        S2: List[int] = []
        output_sequence: List[int] = []
        queue_front = 0
        invalid = False

        for op in user_ops:
            if op == "a":
                if queue_front >= self.N:
                    invalid = True
                    break
                S1.append(queue_front)
                queue_front += 1
            elif op == "b":
                if not S1:
                    invalid = True
                    break
                output_sequence.append(S1.pop())
            elif op == "c":
                if queue_front >= self.N:
                    invalid = True
                    break
                S2.append(queue_front)
                queue_front += 1
            elif op == "d":
                if not S2:
                    invalid = True
                    break
                output_sequence.append(S2.pop())
            else:
                invalid = True
                break

        if invalid or len(output_sequence) != self.N:
            info = {
                "error": "invalid_solution",
                "user_operations": user_ops,
                "user_output_sequence": output_sequence,
                "reference_operations": self.reference_operations,
                "reference_output_sequence": self.reference_output_sequence,
                "N": self.N,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (output_sequence == self.reference_output_sequence)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "user_operations": user_ops,
            "user_output_sequence": output_sequence,
            "reference_operations": self.reference_operations,
            "reference_output_sequence": self.reference_output_sequence,
            "N": self.N,
        }

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
        """Sample a random valid action by generating a valid operations sequence."""
        if self.N is None:
            # Fallback: generate a short random operations string
            length = 8
            ops = "".join(random.choice("abcd") for _ in range(length))
            return f"\\boxed{{{ops}}}"

        # Generate a valid operations sequence using a random policy
        operation_distribution = [random.randint(1, self.N) for _ in range(4)]
        total = sum(operation_distribution)
        operation_weights = [w / total for w in operation_distribution]

        S1: List[int] = []
        S2: List[int] = []
        output_sequence: List[int] = []
        queue_front = 0
        ops = ""

        while len(output_sequence) < self.N:
            op = random.choices(["a", "b", "c", "d"], weights=operation_weights, k=1)[0]
            if op == "a" and queue_front < self.N:
                ops += "a"
                S1.append(queue_front)
                queue_front += 1
            elif op == "b" and S1:
                ops += "b"
                output_sequence.append(S1.pop())
            elif op == "c" and queue_front < self.N:
                ops += "c"
                S2.append(queue_front)
                queue_front += 1
            elif op == "d" and S2:
                ops += "d"
                output_sequence.append(S2.pop())

        return f"\\boxed{{{ops}}}"