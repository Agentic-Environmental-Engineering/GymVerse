from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class EightDigitPuzzleEnv(Env):
    """Eight-Digit (Sliding) Puzzle environment - single turn Q&A using boxed format."""

    action2delta = {
        "L": (0, -1),
        "R": (0, +1),
        "U": (-1, 0),
        "D": (+1, 0),
    }

    def __init__(
        self,
        max_n_m: int = 4,
        steps: int = 10,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - max_n_m: maximum value for both N and M (N and M are sampled from [2, max_n_m])
        - steps: number of random moves applied to generate the destination grid
        """
        super().__init__()
        self.max_n_m = max_n_m
        self.steps = steps

        # Internal state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.start_grid: Optional[List[List[int]]] = None
        self.destination_grid: Optional[List[List[int]]] = None
        self.zero_i: Optional[int] = None
        self.zero_j: Optional[int] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an N × M grid containing digits 0 to N×M−1. "
            "You can swap the 0 with one of its four neighbors: U (up), D (down), L (left), R (right).\n"
            "Your task is to provide the sequence of moves that transforms the start grid into the destination grid.\n"
            "Output Format: Provide the sequence in \\boxed{...} as a string of characters using only U, D, L, R.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Parameter validation
        assert self.max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        assert self.steps >= 1, "steps should be greater than or equal to 1"

        # Sample grid dimensions
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Generate a random start grid as a permutation of 0..N*M-1
        start_permutation = list(range(self.N * self.M))
        random.shuffle(start_permutation)
        self.start_grid = [
            [start_permutation[i * self.M + j] for j in range(self.M)]
            for i in range(self.N)
        ]

        # Locate the position of zero
        zero_positions = [(i, j) for i in range(self.N) for j in range(self.M) if self.start_grid[i][j] == 0]
        self.zero_i, self.zero_j = zero_positions[0]

        # Prepare destination grid by applying random valid moves to the start grid
        self.destination_grid = [row.copy() for row in self.start_grid]
        zero_i, zero_j = self.zero_i, self.zero_j

        # Random action distribution for move generation
        action_distribution = [random.randint(1, self.N * self.M) for _ in range(4)]
        total = sum(action_distribution)
        action_distribution = [weight / total for weight in action_distribution]
        actions = ["U", "D", "L", "R"]

        # Build the reference answer while generating the destination grid
        self.reference_answer = ""
        for _ in range(self.steps):
            while True:
                action = random.choices(actions, weights=action_distribution, k=1)[0]
                di, dj = self.action2delta[action]
                new_zero_i, new_zero_j = zero_i + di, zero_j + dj
                if 0 <= new_zero_i < self.N and 0 <= new_zero_j < self.M:
                    self.reference_answer += action
                    self.destination_grid[zero_i][zero_j], self.destination_grid[new_zero_i][new_zero_j] = (
                        self.destination_grid[new_zero_i][new_zero_j],
                        self.destination_grid[zero_i][zero_j],
                    )
                    zero_i, zero_j = new_zero_i, new_zero_j
                    break

        # Build the problem prompt
        start_grid_str = "\n".join(" ".join(map(str, row)) for row in self.start_grid)
        destination_grid_str = "\n".join(" ".join(map(str, row)) for row in self.destination_grid)
        self.current_problem = (
            f"You are given a {self.N} × {self.M} grid, where each cell contains a digit from 0 to {self.N * self.M - 1}.\n"
            "At any time, you can swap the 0 with one of its four (existing) neighbors:\n"
            "- U = up\n"
            "- D = down\n"
            "- L = left\n"
            "- R = right\n\n"
            f"You start with the following grid:\n{start_grid_str}\n\n"
            f"Your goal is to reach the following grid:\n{destination_grid_str}\n\n"
            "Output Format: Output a single line containing the sequence of moves made by the 0, "
            "represented by a string of characters (U, D, L, R) inside \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided sequence of moves."""
        # Parse the boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: missing or invalid boxed content
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and simulate the moves
        assert self.start_grid is not None and self.destination_grid is not None
        assert self.N is not None and self.M is not None
        assert self.zero_i is not None and self.zero_j is not None

        # Start from the initial grid and zero position
        simulated_grid = [row.copy() for row in self.start_grid]
        zero_i, zero_j = self.zero_i, self.zero_j

        # Process each move
        for ch in boxed_content.strip():
            if ch not in self.action2delta:
                # Invalid character in moves; considered wrong answer
                info = {
                    "correct": False,
                    "reference_answer": self.reference_answer,
                    "user_answer": boxed_content,
                    "reason": "invalid_move_character",
                }
                return TERMINAL_STATE, 0.0, True, False, info
            di, dj = self.action2delta[ch]
            new_zero_i, new_zero_j = zero_i + di, zero_j + dj
            if 0 <= new_zero_i < self.N and 0 <= new_zero_j < self.M:
                simulated_grid[zero_i][zero_j], simulated_grid[new_zero_i][new_zero_j] = (
                    simulated_grid[new_zero_i][new_zero_j],
                    simulated_grid[zero_i][zero_j],
                )
                zero_i, zero_j = new_zero_i, new_zero_j
            else:
                # Move goes out of bounds; considered wrong answer
                info = {
                    "correct": False,
                    "reference_answer": self.reference_answer,
                    "user_answer": boxed_content,
                    "reason": "out_of_bounds_move",
                }
                return TERMINAL_STATE, 0.0, True, False, info

        # Check if simulated grid matches the destination grid
        is_correct = self._grids_equal(simulated_grid, self.destination_grid)
        reward = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": boxed_content,
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

    @staticmethod
    def _grids_equal(a: List[List[int]], b: List[List[int]]) -> bool:
        """Check if two grids are equal."""
        if len(a) != len(b):
            return False
        for row_a, row_b in zip(a, b):
            if len(row_a) != len(row_b):
                return False
            for x, y in zip(row_a, row_b):
                if x != y:
                    return False
        return True

    def sample_random_action(self) -> str:
        """Sample a random action in boxed format."""
        # If reference answer is available, return it as a valid sample action
        if self.reference_answer is not None:
            return f"\\boxed{{{self.reference_answer}}}"
        # Otherwise, generate a random move sequence of length 'steps'
        moves = ''.join(random.choice(list(self.action2delta.keys())) for _ in range(self.steps))
        return f"\\boxed{{{moves}}}"