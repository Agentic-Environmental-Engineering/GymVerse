import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LASEnv(Env):
    """
    Local Assignment Stability (LAS) environment.

    Problem:
    - There are N people labeled from 1 to N, and N foods also labeled from 1 to N.
    - The i-th food has C[i] calories.
    - Person i (1 ≤ i < N) can choose either food i or food i+1.
    - Person N can choose either food N or food 1.
    - If a food is chosen by only one person, that person receives all of its calories.
      If a food is chosen by two people, they share the calories of that food equally.

    Requirement:
    - Find a valid assignment (one food per person from their two choices), such that for every person,
      if this person switches to their other option (while all others keep their choices unchanged),
      that person does NOT receive more calories than they currently do.

    Answer format:
    - Provide N integers (space-separated), the chosen food for each person 1..N, inside \\boxed{...}.
      Example: \\boxed{1 2 3 4}
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 50,
        **kwargs,
    ):
        """
        Initialize the LAS environment.

        Args:
            N: If provided, use this exact N for every reset. Must be >= 3.
            min_n: Minimum N when sampling N randomly (inclusive). Must be >= 3.
            max_n: Maximum N when sampling N randomly (inclusive). Must be >= min_n.
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N must be at least 3 when provided.")
        if min_n < 3:
            raise ValueError("min_n must be at least 3.")
        if max_n < min_n:
            raise ValueError("max_n must be >= min_n.")

        self.fixed_N: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.N: Optional[int] = None
        self.A: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a local assignment stability problem on a cycle.\n"
            "Rules:\n"
            "- There are N people and N foods, both labeled 1..N.\n"
            "- Person i (1 ≤ i < N) can choose food i or i+1; person N can choose food N or 1.\n"
            "- If a food is chosen by exactly one person, that person receives all of its calories.\n"
            "- If a food is chosen by two people, they share its calories equally.\n"
            "- A valid assignment must be stable: for every person, switching to their other option (while all others keep their choices) does NOT increase their calories.\n"
            "Output Format: Provide N integers (space-separated) inside \\boxed{...}, representing the chosen food for persons 1..N.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            self.N = self.fixed_N
        else:
            self.N = random.randint(self.min_n, self.max_n)

        assert self.N is not None and self.N >= 3

        # Generate calories array A (1..2N inclusive)
        self.A = [random.randint(1, 2 * self.N) for _ in range(self.N)]

        # Build problem prompt
        c_list_str = ", ".join(f"C[{i + 1}]={Ci}" for i, Ci in enumerate(self.A))
        self.current_problem = (
            f"There are {self.N} people labeled from 1 to {self.N}, and {self.N} foods also labeled from 1 to {self.N}. "
            f"The i-th food has C[i] calories, and the array C is: {c_list_str}\n\n"
            f"Each person chooses one food as follows:\n"
            f"- Person i (1 ≤ i < {self.N}) can choose either food i or food i+1.\n"
            f"- Person {self.N} can choose either food {self.N} or food 1.\n"
            f"- If a food is chosen by only one person, that person receives all of its calories. "
            f"If a food is chosen by two people, they share the calories of that food equally.\n\n"
            f"You are to find a valid food assignment (i.e., choose one food between the two choices for each person), such that for every person, "
            f"if this person switches to the other food choice (while all other people keep their choices unchanged), this person does NOT receive more calories than this person currently does.\n"
            f"Output Format: Output a single line with {self.N} integers — the food chosen by person 1, 2, ..., {self.N}, separated by spaces, "
            f"and put the entire line inside \\boxed{{...}}."
        )

        # Compute reference answer using the original algorithm
        self.reference_answer = self._compute_reference_answer(self.A)

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _compute_reference_answer(self, A: List[int]) -> str:
        """
        Compute a stable assignment using the original dynamic programming algorithm.
        Returns a space-separated string of choices for persons 1..N (1-indexed).
        """
        N = len(A)

        # B is 1-indexed, with B[N+1] = B[1]
        B = [0] * (N + 2)
        for i in range(1, N + 1):
            B[i] = A[i - 1]
        B[N + 1] = B[1]

        # C is a DP table: (N+2) x 5, initialized to 0
        C = [[0] * 5 for _ in range(N + 2)]

        def dynamic_programming(s: int) -> bool:
            # Reset DP table
            for i in range(N + 2):
                for j in range(5):
                    C[i][j] = 0

            # Base case
            C[1][s] = 1

            # Transition
            for i in range(2, N + 2):
                if C[i - 1][1] and B[i - 1] <= B[i] * 2:
                    C[i][1] = 1
                if C[i - 1][1] and B[i - 1] <= B[i]:
                    C[i][3] = 1
                if C[i - 1][2] and B[i] <= B[i - 1] * 2:
                    C[i][2] = 2
                if C[i - 1][2] and B[i] <= B[i - 1]:
                    C[i][4] = 2
                if C[i - 1][3] and B[i] <= B[i - 1]:
                    C[i][2] = 3
                if C[i - 1][3] and B[i] * 2 <= B[i - 1]:
                    C[i][4] = 3
                if C[i - 1][4] and B[i - 1] <= B[i]:
                    C[i][1] = 4
                if C[i - 1][4] and B[i - 1] * 2 <= B[i]:
                    C[i][3] = 4

            return C[N + 1][s] != 0

        # D stores the final choices (1-indexed)
        D = [0] * (N + 2)

        # Try all 4 possible states
        for s in range(1, 5):
            if dynamic_programming(s):
                x = s
                for j in range(N + 1, 0, -1):
                    if x == 1:
                        D[j - 1] = ((j - 1) % N) + 1
                    if x == 2:
                        D[j] = ((j - 1) % N) + 1
                    if x == 3:
                        D[j - 1] = ((j - 1) % N) + 1
                        D[j] = ((j - 1) % N) + 1
                    # Note: original code does not have explicit case for x == 4
                    x = C[j][x]

                return " ".join(str(D[i]) for i in range(1, N + 1))

        # If no state works (should not happen for valid instances), raise an error
        raise RuntimeError("Failed to compute a valid reference assignment.")

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Evaluate the submitted assignment.

        Rewards:
        - 1.0 if the assignment is valid and stable (all persons satisfied).
        - 0.0 if the assignment is invalid or not stable.
        - -0.1 if the answer format is incorrect.
        """
        if self.N is None or self.A is None or self.reference_answer is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse integers from boxed content
        try:
            user_choices_1based = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Length check
        if len(user_choices_1based) != self.N:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Convert to 0-based indices
        choices = [c - 1 for c in user_choices_1based]

        # Validate choices are within allowed options
        if not all(choice in (person, (person + 1) % self.N) for person, choice in enumerate(choices)):
            info = {
                "correct": False,
                "reason": "invalid_choice",
                "reference_answer": self.reference_answer,
                "user_answer": " ".join(map(str, user_choices_1based)),
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Count how many people chose each food
        counting = [0] * self.N
        for choice in choices:
            if not (0 <= choice < self.N):
                info = {
                    "correct": False,
                    "reason": "choice_out_of_range",
                    "reference_answer": self.reference_answer,
                    "user_answer": " ".join(map(str, user_choices_1based)),
                }
                return TERMINAL_STATE, 0.0, True, False, info
            counting[choice] += 1

        # Helper to compute calories (scaled by 2 to avoid fractions)
        def get_calories(choice_index: int) -> int:
            if counting[choice_index] == 1:
                return self.A[choice_index] * 2
            elif counting[choice_index] == 2:
                return self.A[choice_index] * 1
            else:
                # Invalid state (a food chosen by 0 or >2 people)
                raise ValueError(f"Invalid counting for food {choice_index}: {counting[choice_index]}")

        # Compute satisfied count using the original logic
        satisfied = 0
        for person, choice in enumerate(choices):
            current = get_calories(choice)

            # The other choice (original logic preserved)
            other_choice = ((person + (person + 1)) - choice) % self.N

            # Simulate switching (original code increments only the other choice)
            counting[other_choice] += 1
            changed = get_calories(other_choice)
            counting[other_choice] -= 1

            satisfied += int(current >= changed)

        is_correct = (satisfied == self.N)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied": satisfied,
            "N": self.N,
            "calories": self.A,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, user_choices_1based)),
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
        """Sample a random valid action: each person picks randomly from their two options."""
        if self.N is None:
            # If called before reset, assume a small default N
            n = self.fixed_N if self.fixed_N is not None else max(self.min_n, 3)
        else:
            n = self.N
        choices = []
        for i in range(n):
            option = random.choice([i, (i + 1) % n])
            choices.append(option + 1)  # convert to 1-based
        return f"\\boxed{{{' '.join(map(str, choices))}}}"