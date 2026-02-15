import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class HitoriPuzzleEnv(Env):
    """Hitori puzzle environment - single-turn Q&A using GEM interface."""

    prompt_template = (
        "You are given a {N} × {M} matrix. Each cell contains an integer. Please \"black out\" some cells such that:\n"
        "1. In each row and each column, no number appears more than once among the remaining (non-blacked-out) cells.\n"
        "2. No two blacked-out cells are adjacent (horizontally or vertically).\n"
        "3. All remaining cells must form a single connected region — you must be able to reach any remaining cell from any other by moving up, down, left, or right.\n\n"
        "The matrix is given in row-major order, with each row represented as a list of integers separated by spaces:\n"
        "{matrix}\n\n"
        "Output Format: Output {N} lines, each containing {M} characters with no separators (also in row-major order). "
        "Use '.' for a remaining cell and '*' for a blacked-out cell.\n\n"
        "Important: Submit your final answer wrapped in \\boxed{{...}}. Inside the box you should provide exactly {N} lines, "
        "each with exactly {M} characters using only '.' and '*'."
    )

    def __init__(
        self,
        max_n_m: int = 5,
        correct_reward: float = 1.0,
        wrong_answer_reward: float = 0.0,
        format_error_reward: float = -0.1,
        **kwargs,
    ):
        """
        Initialize the HitoriPuzzleEnv instance.

        Args:
            max_n_m: Maximum dimension for N and M (both sampled in [2, max_n_m]).
            correct_reward: Reward for a fully correct solution.
            wrong_answer_reward: Reward for an incorrect but well-formatted solution.
            format_error_reward: Reward for a format error (e.g., missing or malformed \\boxed{...}).
        """
        super().__init__()
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        self.max_n_m = max_n_m

        # Rewards according to the conversion requirements
        self.correct_reward = correct_reward
        self.wrong_answer_reward = wrong_answer_reward
        self.format_error_reward = format_error_reward

        # State variables
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.matrix: Optional[List[List[int]]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a Hitori puzzle.\n"
            "Provide your final grid using '.' for white cells and '*' for blacked-out cells.\n"
            "Your submission must be wrapped in \\boxed{...} with exactly N lines, each of length M.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Sample dimensions
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        self.N, self.M = N, M

        # Generate matrix and a valid reference answer
        matrix, reference_answer = self._generate_puzzle(N, M)
        self.matrix = matrix
        # Store reference answer as a single string with newlines
        self.reference_answer = "\n".join("".join(row) for row in reference_answer)

        # Build the problem prompt
        matrix_str = "\n".join(" ".join(map(str, row)) for row in matrix)
        self.current_problem = self.prompt_template.format(N=N, M=M, matrix=matrix_str)

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def _generate_puzzle(self, N: int, M: int) -> Tuple[List[List[int]], List[List[str]]]:
        """Generate a Hitori puzzle matrix with at least one valid solution (reference answer)."""

        def check_connected(grid: List[List[str]]) -> bool:
            return self.check_connected(grid, N, M)

        matrix: List[List[Optional[int]]] = [[None] * M for _ in range(N)]
        reference_answer: List[List[str]] = [["."] * M for _ in range(N)]

        all_cells = [(i, j) for i in range(N) for j in range(M)]
        random.shuffle(all_cells)

        def backtrack(idx: int) -> bool:
            if idx == len(all_cells):
                return True
            i, j = all_cells[idx]

            remaining_numbers = set(
                matrix[i][_j] for _j in range(M) if reference_answer[i][_j] == "." and matrix[i][_j] is not None
            ) | set(
                matrix[_i][j] for _i in range(N) if reference_answer[_i][j] == "." and matrix[_i][j] is not None
            )

            for color in random.sample([".", "*"], 2):
                if color == ".":
                    num = 0
                    while num in remaining_numbers:
                        num += 1
                    matrix[i][j] = num
                else:
                    if not remaining_numbers:
                        continue
                    ok = True
                    for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M and reference_answer[ni][nj] == "*":
                            ok = False
                            break
                    if not ok:
                        continue
                    reference_answer[i][j] = "*"
                    # Ensure connectivity of remaining '.' cells
                    if not check_connected(reference_answer):
                        reference_answer[i][j] = "."
                        continue
                    matrix[i][j] = random.choice(list(remaining_numbers))
                assert backtrack(idx + 1)
                return True

            return False

        assert backtrack(0), "Failed to generate a valid matrix"

        # Convert matrix from Optional[int] to int
        final_matrix: List[List[int]] = [[int(matrix[i][j]) for j in range(M)] for i in range(N)]
        return final_matrix, reference_answer

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}, supporting multi-line content."""
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process_answer_block(self, boxed_text: str) -> Optional[List[str]]:
        """
        Process the boxed content into a list of non-empty lines.
        Each line should be a string of '.' and '*'.
        """
        try:
            lines: List[str] = []
            for line in boxed_text.splitlines():
                s = line.strip()
                if s:
                    lines.append(s)
            return lines
        except Exception:
            return None

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        assert self.N is not None and self.M is not None and self.matrix is not None

        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        processed = self._process_answer_block(boxed)
        if processed is None:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        N, M = self.N, self.M
        solution = processed

        # Format checks
        if len(solution) != N or any(len(row) != M for row in solution):
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error", "detail": "dimension_mismatch"}
        if not all(c in ".*" for row in solution for c in row):
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error", "detail": "invalid_characters"}

        # Constraint checks
        # 1) No two '*' adjacent (4-neighborhood)
        for i in range(N):
            for j in range(M):
                if solution[i][j] == "*":
                    for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M and solution[ni][nj] == "*":
                            info = {"error": "invalid_solution", "violated": "adjacent_black_cells"}
                            return TERMINAL_STATE, self.wrong_answer_reward, True, False, info

        # 2) Connectivity of '.' cells
        try:
            if not self.check_connected(solution, N, M):
                info = {"error": "invalid_solution", "violated": "connectivity"}
                return TERMINAL_STATE, self.wrong_answer_reward, True, False, info
        except AssertionError:
            # Triggered when there is no '.' cell at all in the solution
            info = {"error": "invalid_solution", "violated": "no_white_cells"}
            return TERMINAL_STATE, self.wrong_answer_reward, True, False, info

        # 3) Row and column uniqueness among '.' cells
        satisfied = 0
        for i in range(N):
            row_numbers = [self.matrix[i][j] for j in range(M) if solution[i][j] == "."]
            if len(row_numbers) == len(set(row_numbers)):
                satisfied += 1
        for j in range(M):
            col_numbers = [self.matrix[i][j] for i in range(N) if solution[i][j] == "."]
            if len(col_numbers) == len(set(col_numbers)):
                satisfied += 1

        total_constraints = N + M
        is_correct = (satisfied == total_constraints)
        reward: float = self.correct_reward if is_correct else self.wrong_answer_reward

        info = {
            "correct": is_correct,
            "satisfied": satisfied,
            "total_constraints": total_constraints,
            "completion_ratio": satisfied / total_constraints if total_constraints > 0 else 0.0,
            "reference_answer": self.reference_answer,
            "N": N,
            "M": M,
            "matrix": self.matrix,
        }

        return TERMINAL_STATE, reward, True, False, info

    def check_connected(self, grid: List[List[str]], N: int, M: int) -> bool:
        """
        Check connectivity of '.' cells using DFS. Returns True if all '.' are connected.
        Note: If there is no '.' in the grid, this function asserts False (preserved from original logic).
        """
        visited = [[False] * M for _ in range(N)]

        def dfs(x: int, y: int) -> None:
            visited[x][y] = True
            for dx, dy in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < M and not visited[nx][ny] and grid[nx][ny] == ".":
                    dfs(nx, ny)

        for i in range(N):
            for j in range(M):
                if grid[i][j] == ".":
                    dfs(i, j)
                    return all(visited[_i][_j] for _i in range(N) for _j in range(M) if grid[_i][_j] == ".")
        assert False

    def sample_random_action(self) -> str:
        """Sample a random action: a random grid of '.' and '*' with proper dimensions, wrapped in \\boxed{...}."""
        if self.N is None or self.M is None:
            # Default small grid if called before reset
            N, M = 2, 2
        else:
            N, M = self.N, self.M
        grid = []
        for _ in range(N):
            row = "".join(random.choice(".*") for _ in range(M))
            grid.append(row)
        content = "\n".join(grid)
        return f"\\boxed{{\n{content}\n}}"