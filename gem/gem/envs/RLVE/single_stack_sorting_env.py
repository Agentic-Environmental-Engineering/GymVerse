import random
from typing import Any, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class SingleStackSortingEnv(Env):
    """Single Stack Sorting environment - single-turn Q&A.

    The agent must produce a sequence of operations (consisting of 'a' and 'b')
    that transforms an initial queue of integers 0..N-1 into a given target output
    sequence using a single stack. The answer must be submitted in \\boxed{...} format.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 50,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "mean([gold=answer])^beta",
        rewarding_beta: float = 10.0,
        rewarding_weight: float = +1.0,
        **kwargs,
    ):
        super().__init__()
        # Problem size configuration
        self.N_fixed = N
        self.min_N = min_N
        self.max_N = max_N

        # Keep original reward-related parameters for compatibility (not used in GEM scoring)
        self.wrong_format = wrong_format
        self.invalid_solution = invalid_solution
        self.rewarding_strategy = rewarding_strategy
        self.rewarding_beta = rewarding_beta
        self.rewarding_weight = rewarding_weight

        # Runtime state
        self.N: Optional[int] = None
        self.target_output: Optional[list[int]] = None
        self.reference_operations: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task description."""
        return (
            "You are solving Single Stack Sorting problems.\n"
            "Your task is to provide a valid sequence of operations that transforms "
            "the initial queue into the given target output sequence using a single stack.\n"
            "Please provide your answer inside \\boxed{...} and ensure the content consists "
            "only of the characters 'a' and 'b' with no spaces or extra characters.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.N_fixed is not None:
            assert self.N_fixed >= 3, "N should be greater than or equal to 3"
            self.N = self.N_fixed
        else:
            assert self.min_N >= 3, "min_N should be greater than or equal to 3"
            assert self.max_N >= self.min_N, "max_N should be greater than or equal to min_N"
            self.N = random.randint(self.min_N, self.max_N)

        # Generate target output sequence by simulating random operations
        operation_distribution = [random.randint(1, self.N) for _ in range(2)]
        total_weight = sum(operation_distribution)
        operation_distribution = [w / total_weight for w in operation_distribution]

        self.reference_operations = ""
        S: list[int] = []
        output_sequence: list[int] = []
        queue_front = 0

        while len(output_sequence) < self.N:
            operation = random.choices(["a", "b"], weights=operation_distribution, k=1)[0]
            if operation == "a" and queue_front < self.N:
                self.reference_operations += "a"
                S.append(queue_front)
                queue_front += 1
            elif operation == "b" and S:
                self.reference_operations += "b"
                output_sequence.append(S.pop())

        assert len(self.reference_operations) == self.N * 2, "reference_operations should have length 2 * N"
        self.target_output = output_sequence

        # Build problem prompt
        sequence_str = " ".join(map(str, self.target_output))
        self.current_problem = (
            f"You are given a queue of integers containing {self.N} elements in increasing order "
            f"from 0 (at the front) to {self.N - 1} (at the back). You also have an empty stack S "
            f"and an initially empty output sequence. You may perform the following operations:\n"
            f"- a: Pop the front element of the queue and push it onto the stack S.\n"
            f"- b: Pop the top element from the stack S and append it to the output sequence.\n\n"
            f"Please produce the following target output sequence:\n{sequence_str}\n\n"
            f"Please output a valid sequence of operations (a string consisting of the characters 'a' and 'b' only) "
            f"that transforms the initial queue into the given output sequence using the rules above.\n\n"
            f"Output Format: A single line containing the sequence of operations inside \\boxed{{...}}, "
            f"with no spaces or extra characters."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the answer."""
        # Parse boxed answer
        operations = self._parse_answer(action)
        if operations is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and simulate operations
        assert self.N is not None and self.target_output is not None

        S: list[int] = []
        produced_output: list[int] = []
        queue_front = 0

        for op in operations:
            if op == "a":
                if queue_front >= self.N:
                    info = {
                        "error": "push_overflow",
                        "user_operations": operations,
                        "produced_output": produced_output,
                        "target_output": self.target_output,
                        "reference_operations": self.reference_operations,
                    }
                    return TERMINAL_STATE, 0.0, True, False, info
                S.append(queue_front)
                queue_front += 1
            elif op == "b":
                if not S:
                    info = {
                        "error": "pop_from_empty_stack",
                        "user_operations": operations,
                        "produced_output": produced_output,
                        "target_output": self.target_output,
                        "reference_operations": self.reference_operations,
                    }
                    return TERMINAL_STATE, 0.0, True, False, info
                produced_output.append(S.pop())
            else:
                info = {
                    "error": "invalid_character",
                    "invalid_char": op,
                    "user_operations": operations,
                    "produced_output": produced_output,
                    "target_output": self.target_output,
                    "reference_operations": self.reference_operations,
                }
                return TERMINAL_STATE, 0.0, True, False, info

        # Must produce exactly N outputs
        if len(produced_output) != self.N:
            info = {
                "error": "incomplete_output",
                "user_operations": operations,
                "produced_output": produced_output,
                "target_output": self.target_output,
                "reference_operations": self.reference_operations,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = produced_output == self.target_output
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "user_operations": operations,
            "produced_output": produced_output,
            "target_output": self.target_output,
            "reference_operations": self.reference_operations,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re

        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if not matches:
            return None
        content = matches[-1].strip()
        return content

    def sample_random_action(self) -> str:
        """Sample a random valid action (uses the reference operations)."""
        if self.reference_operations is None:
            # Fallback: produce an empty boxed answer if not initialized
            return "\\boxed{}"
        return f"\\boxed{{{self.reference_operations}}}"