from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NinePuzzleEnv(Env):
    """Nine-Puzzle environment: shift rows/columns of a grid to reach a target configuration.
    
    Single-turn QA format. The agent must output a sequence of actions inside \\boxed{...},
    one action per line, where each action follows:
      [row_or_column] [index] [shifts]
    """

    def __init__(
        self,
        max_n_m: int = 5,
        steps: int = 3,
        **kwargs
    ):
        """
        Initialize the NinePuzzleEnv instance.

        Args:
            max_n_m: Upper bound for both N and M (grid dimensions). Must be >= 2.
            steps: Number of random actions used to generate the target configuration. Must be >= 1.
            **kwargs: Reserved for compatibility.
        """
        super().__init__()
        self.max_n_m: int = max_n_m
        self.steps: int = steps

        # Validations mimicking original logic
        if self.max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")
        if self.steps < 1:
            raise ValueError("steps should be greater than or equal to 1")

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.row_K: Optional[int] = None
        self.col_K: Optional[int] = None
        self.start_grid: Optional[List[List[int]]] = None
        self.destination_grid: Optional[List[List[int]]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a grid-based puzzle. You may shift entire rows or columns by a limited number of cells.\n"
            "Provide your sequence of actions inside \\boxed{...}. Each action must be on its own line.\n"
            "Do NOT include backticks or quotes. Example action lines: 'row 0 2' or 'column 1 -3'.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new puzzle instance."""
        super().reset(seed)

        # Sample grid dimensions
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Sample allowed shift ranges
        self.row_K = random.randint(1, self.M - 1)
        self.col_K = random.randint(1, self.N - 1)

        # Generate start grid as a random permutation of 0..N*M-1
        start_perm = list(range(self.N * self.M))
        random.shuffle(start_perm)
        self.start_grid = [
            [start_perm[i * self.M + j] for j in range(self.M)]
            for i in range(self.N)
        ]

        # Apply random steps to produce destination grid and record reference actions
        destination_grid = [row.copy() for row in self.start_grid]
        actions_lines: List[str] = []

        for _ in range(self.steps):
            row_or_column = random.choice(["row", "column"])
            if row_or_column == "row":
                index = random.randint(0, self.N - 1)
                # Non-zero shift within [-row_K, row_K]
                while True:
                    shifts = random.randint(-self.row_K, self.row_K)
                    if shifts != 0:
                        break
                # Apply row shift
                new_grid = [row.copy() for row in destination_grid]
                for j in range(self.M):
                    new_grid[index][j] = destination_grid[index][((j - shifts) % self.M + self.M) % self.M]
                destination_grid = new_grid
            else:
                # column
                index = random.randint(0, self.M - 1)
                # Non-zero shift within [-col_K, col_K]
                while True:
                    shifts = random.randint(-self.col_K, self.col_K)
                    if shifts != 0:
                        break
                # Apply column shift
                new_grid = [row.copy() for row in destination_grid]
                for i in range(self.N):
                    new_grid[i][index] = destination_grid[((i - shifts) % self.N + self.N) % self.N][index]
                destination_grid = new_grid

            actions_lines.append(f"{row_or_column} {index} {shifts}")

        self.destination_grid = destination_grid
        self.reference_answer = "\n".join(actions_lines) + ("\n" if actions_lines else "")

        # Build prompt
        self.current_problem = self._build_problem_prompt()

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted sequence of actions and compute reward."""
        # Extract boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse actions from boxed content
        parse_ok, actions_or_error = self._parse_actions(boxed_content)
        if not parse_ok:
            # Parsing error -> format error
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": actions_or_error}

        actions: List[List[Any]] = actions_or_error  # [ [str, int, int], ... ]

        # Apply actions to the start grid
        assert self.start_grid is not None and self.destination_grid is not None
        assert self.N is not None and self.M is not None
        assert self.row_K is not None and self.col_K is not None

        candidate_grid = [row.copy() for row in self.start_grid]

        for act in actions:
            kind, index, shifts = act[0], act[1], act[2]
            new_grid = [row.copy() for row in candidate_grid]

            if kind == "row":
                if not (0 <= index < self.N):
                    info = {"error": "invalid_action", "reason": "row_index_out_of_range"}
                    return TERMINAL_STATE, 0.0, True, False, info
                if not (-self.row_K <= shifts <= self.row_K):
                    info = {"error": "invalid_action", "reason": "row_shift_out_of_range"}
                    return TERMINAL_STATE, 0.0, True, False, info
                for j in range(self.M):
                    new_grid[index][j] = candidate_grid[index][((j - shifts) % self.M + self.M) % self.M]
            else:
                # column
                if not (0 <= index < self.M):
                    info = {"error": "invalid_action", "reason": "column_index_out_of_range"}
                    return TERMINAL_STATE, 0.0, True, False, info
                if not (-self.col_K <= shifts <= self.col_K):
                    info = {"error": "invalid_action", "reason": "column_shift_out_of_range"}
                    return TERMINAL_STATE, 0.0, True, False, info
                for i in range(self.N):
                    new_grid[i][index] = candidate_grid[((i - shifts) % self.N + self.N) % self.N][index]

            candidate_grid = new_grid

        # Compare with target
        is_correct = self._grids_equal(candidate_grid, self.destination_grid)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_actions": self.reference_answer,
            "user_actions": ["{} {} {}".format(a[0], a[1], a[2]) for a in actions],
            "N": self.N,
            "M": self.M,
            "row_K": self.row_K,
            "col_K": self.col_K,
            "start_grid": self.start_grid,
            "target_grid": self.destination_grid,
            "final_grid": candidate_grid,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract content inside \\boxed{...}. Returns the last occurrence if multiple exist."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_actions(self, content: str) -> Tuple[bool, Any]:
        """Parse action lines from boxed content.
        
        Returns:
            (True, actions) on success, where actions is a list of [kind, index, shifts].
            (False, error_message) on failure.
        """
        actions: List[List[Any]] = []
        lines = content.splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                return False, "each action must contain exactly 3 tokens"
            kind = parts[0]
            if kind not in ("row", "column"):
                return False, "first token must be 'row' or 'column'"
            try:
                idx = int(parts[1])
                sh = int(parts[2])
            except ValueError:
                return False, "index and shift must be integers"
            actions.append([kind, idx, sh])
        return True, actions

    def _build_problem_prompt(self) -> str:
        """Build the problem description based on current parameters and grids."""
        assert self.N is not None and self.M is not None
        assert self.row_K is not None and self.col_K is not None
        assert self.start_grid is not None and self.destination_grid is not None

        start_grid_str = "\n".join(" ".join(map(str, row)) for row in self.start_grid)
        destination_grid_str = "\n".join(" ".join(map(str, row)) for row in self.destination_grid)

        prompt = (
            f"You are given a {self.N} × {self.M} grid, where each cell contains a digit from 0 to {self.N * self.M - 1}.\n\n"
            f"At any time, you may perform one of the following actions:\n"
            f"- Pick a row i (0 ≤ i < {self.N}) and shift it left or right by at most {self.row_K} cells.\n"
            f"- Pick a column j (0 ≤ j < {self.M}) and shift it up or down by at most {self.col_K} cells.\n\n"
            f"You start with the following grid:\n{start_grid_str}\n\n"
            f"Your goal is to transform it into the following grid:\n{destination_grid_str}\n\n"
            f"Output Format: Write each action on its own line as: [row_or_column] [index] [shifts]\n"
            f"- 'row_or_column' is either 'row' or 'column'\n"
            f"- 'index' is the 0-based index of the row or column\n"
            f"- 'shifts' is a signed integer: positive for right/down, negative for left/up\n"
            f"- Example: row 0 2 or column 1 -3\n\n"
            f"Submit your entire sequence wrapped in \\boxed{{...}}. Do NOT include backticks or quotes.\n"
        )
        return prompt

    def _grids_equal(self, A: List[List[int]], B: List[List[int]]) -> bool:
        """Check if two grids are equal."""
        if len(A) != len(B):
            return False
        for ra, rb in zip(A, B):
            if len(ra) != len(rb):
                return False
            for a, b in zip(ra, rb):
                if a != b:
                    return False
        return True

    def sample_random_action(self) -> str:
        """Sample a random (possibly invalid) action sequence wrapped in \\boxed{...}."""
        # If parameters are not initialized yet, provide a minimal example action.
        if self.N is None or self.M is None or self.row_K is None or self.col_K is None:
            return "\\boxed{row 0 1}"

        num_actions = random.randint(0, max(1, self.steps))
        lines: List[str] = []
        for _ in range(num_actions):
            kind = random.choice(["row", "column"])
            if kind == "row":
                idx = random.randint(0, self.N - 1)
                sh = random.randint(-self.row_K, self.row_K)
                if sh == 0:
                    sh = 1 if self.row_K >= 1 else 0
            else:
                idx = random.randint(0, self.M - 1)
                sh = random.randint(-self.col_K, self.col_K)
                if sh == 0:
                    sh = 1 if self.col_K >= 1 else 0
            lines.append(f"{kind} {idx} {sh}")
        content = "\n".join(lines)
        return f"\\boxed{{{content}}}"