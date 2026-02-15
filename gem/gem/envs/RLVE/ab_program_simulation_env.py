from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class ABProgramSimulationEnv(Env):
    """A::B token rewriting environment - single-turn Q&A.

    This environment presents a token rewriting system with four tokens: 'A#', '#A', 'B#', '#B'.
    A program is a sequence of tokens. Neighbor tokens whose '#' face each other must be
    rewritten according to the rules until no more rewrites are possible.

    Rewriting rules (for adjacent pairs):
    - A# #A -> (delete both)
    - A# #B -> #B A#
    - B# #A -> #A B#
    - B# #B -> (delete both)

    The task is to compute the final state (normal form) of a randomly generated program.
    The answer must be provided in \\boxed{...} format, where the content is the final sequence
    of tokens separated by single spaces. For an empty final sequence, answer \\boxed{}.
    """

    def __init__(
        self,
        N: int = 3,
        max_steps: int = 100,
        **kwargs    
    ):
        super().__init__()
        assert N >= 1, "N should be greater than or equal to 1"
        assert max_steps >= 1, "max_steps should be greater than or equal to 1"
        self.N: int = N
        self.max_steps: int = max_steps

        self.allowed_tokens: List[str] = ["A#", "#A", "B#", "#B"]

        self.current_problem: Optional[str] = None
        self.program: Optional[List[str]] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[List[str]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "A::B is a system with 4 tokens: 'A#', '#A', 'B#' and '#B'.\n"
            "An A::B program is a sequence of tokens, e.g., 'B# A# #B #A B#'.\n"
            "To compute a program, rewrite neighbor tokens whose '#' face each other using the rules:\n"
            "- A# #A becomes (nothing)\n"
            "- A# #B becomes #B A#\n"
            "- B# #A becomes #A B#\n"
            "- B# #B becomes (nothing)\n"
            "Repeat until no more rules apply.\n\n"
            "Output Format: Put the final sequence of tokens inside \\boxed{...}, tokens separated by single spaces.\n"
            "For example: \\boxed{B# A# A#}. If the final sequence is empty, answer \\boxed{}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Generate a program and its final normal form within max_steps
        while True:
            # Random distribution for token selection
            weights = [random.randint(1, self.N) for _ in range(4)]
            total = sum(weights)
            probs = [w / total for w in weights]

            # Sample the initial program of length N
            token_indices = random.choices(range(4), weights=probs, k=self.N)
            program = [self.allowed_tokens[i] for i in token_indices]

            # Simulate rewriting up to max_steps
            current = program.copy()
            final: Optional[List[str]] = None

            for _step in range(self.max_steps):
                new_program: Optional[List[str]] = None
                # Scan left-to-right and apply the first applicable rewrite
                for i in range(len(current) - 1):
                    a, b = current[i], current[i + 1]
                    if a == "A#" and b == "#A":
                        new_program = current[:i] + current[i + 2:]
                    elif a == "A#" and b == "#B":
                        new_program = current[:i] + ["#B", "A#"] + current[i + 2:]
                    elif a == "B#" and b == "#A":
                        new_program = current[:i] + ["#A", "B#"] + current[i + 2:]
                    elif a == "B#" and b == "#B":
                        new_program = current[:i] + current[i + 2:]

                    if new_program is not None:
                        break

                if new_program is None:
                    final = current
                    break
                else:
                    current = new_program

            # Accept this instance only if a final state was reached within max_steps
            if final is not None:
                self.program = program
                self.gold_answer = final
                self.reference_answer = " ".join(final)
                break

        # Build the problem statement
        program_str = " ".join(self.program) if self.program is not None else ""
        self.current_problem = (
            f"Please give the final state of the following program:\n{program_str}\n\n"
            "Remember to present your final sequence inside \\boxed{...} with tokens separated by spaces.\n"
            "If the final sequence is empty, answer \\boxed{}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: missing or malformed boxed answer
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        content = boxed_content.strip()
        tokens: List[str] = [] if content == "" else content.split()

        # Validate tokens
        if not all(t in self.allowed_tokens for t in tokens):
            # Format error: invalid tokens
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Compare with gold answer
        is_correct = (tokens == (self.gold_answer or []))
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "reference_answer_tokens": self.gold_answer,
            "user_answer_tokens": tokens
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...}. Allows empty content."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required \\boxed{...} format."""
        length = random.randint(0, self.N)
        if length == 0:
            return "\\boxed{}"
        tokens = [random.choice(self.allowed_tokens) for _ in range(length)]
        return f"\\boxed{{{' '.join(tokens)}}}"