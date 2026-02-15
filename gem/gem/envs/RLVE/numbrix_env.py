import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class NumbrixEnv(Env):
    """Numbrix puzzle environment - Single-turn Q&A.

    The task is to fill an N × M grid with numbers from 0 to N*M-1 such that:
    1. Each number appears exactly once.
    2. Consecutive numbers are orthogonally adjacent (Manhattan distance equals 1).
    Some cells are pre-filled, and empty cells are represented by -1.
    """

    def __init__(
        self,
        max_n_m: int = 10,
        sparsity: float = 0.5,
        **kwargs
    ):
        """
        Initialize the Numbrix environment.

        Parameters:
        - max_n_m: Maximum dimension for N and M (grid size will be between 2 and max_n_m).
        - sparsity: Fraction of cells to hide (-1). Must satisfy 0 < sparsity < 1.
        """
        super().__init__()
        assert isinstance(max_n_m, int), "max_n_m must be an integer"
        assert max_n_m >= 2, "max_n_m should be greater than or equal to 2"
        assert 0.0 < sparsity < 1.0, "sparsity should be between 0 and 1"

        self.max_n_m = max_n_m
        self.sparsity = sparsity

        self.current_problem: Optional[str] = None
        self.reference_solution: Optional[List[List[int]]] = None
        self.reference_answer: Optional[str] = None
        self.puzzle_matrix: Optional[List[List[int]]] = None
        self.N: Optional[int] = None
        self.M: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return the task description and output format instructions."""
        return (
            "You are given an N × M matrix with some cells filled with numbers from 0 to N×M-1, and some cells empty (represented by -1).\n"
            "Please fill the empty cells with numbers from 0 to N×M-1 such that:\n"
            "1. Each number from 0 to N×M-1 appears exactly once in the matrix.\n"
            "2. Each number is horizontally or vertically adjacent to the next number (every number x is adjacent to x + 1).\n\n"
            "Output Format: Your final answer should contain N lines, each with M numbers separated by spaces, representing the completed matrix in row-major order.\n"
            "Wrap your entire matrix in \\boxed{...}. For example:\n"
            "\\boxed{\\n0 1\\n2 3\\n}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new Numbrix puzzle."""
        super().reset(seed)

        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        self.N, self.M = N, M

        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def is_inside(x: int, y: int) -> bool:
            return 0 <= x < N and 0 <= y < M

        def count_unvisited_degree(x: int, y: int, visited: List[List[bool]]) -> int:
            cnt = 0
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if is_inside(nx, ny) and not visited[nx][ny]:
                    cnt += 1
            return cnt

        def check_connectivity(remain: int, visited: List[List[bool]]) -> bool:
            start = None
            for i in range(N):
                for j in range(M):
                    if not visited[i][j]:
                        start = (i, j)
                        break
                if start:
                    break
            if not start:
                return True
            stack = [start]
            seen = {start}
            count = 1
            while stack:
                x, y = stack.pop()
                for dx, dy in dirs:
                    xx, yy = x + dx, y + dy
                    if is_inside(xx, yy) and not visited[xx][yy] and (xx, yy) not in seen:
                        seen.add((xx, yy))
                        stack.append((xx, yy))
                        count += 1
            return count == remain

        def generate_random_hamiltonian_path(n: int, m: int) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
            while True:
                sx = random.randint(0, n - 1)
                sy = random.randint(0, m - 1)
                visited = [[False] * m for _ in range(n)]
                order = [[-1] * m for _ in range(n)]
                path: List[Tuple[int, int]] = []
                visited[sx][sy] = True
                order[sx][sy] = 0
                path = [(sx, sy)]

                def DFS(step: int, x: int, y: int) -> bool:
                    if step == n * m:
                        return True
                    cand: List[Tuple[int, int]] = []
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if is_inside(nx, ny) and not visited[nx][ny]:
                            cand.append((nx, ny))
                    if not cand:
                        return False
                    random.shuffle(cand)
                    cand_scores: List[Tuple[int, int, int]] = []
                    for nx, ny in cand:
                        deg = count_unvisited_degree(nx, ny, visited)
                        cand_scores.append((deg, nx, ny))
                    cand_scores.sort(key=lambda t: t[0])
                    for _, nx, ny in cand_scores:
                        visited[nx][ny] = True
                        order[nx][ny] = step
                        path.append((nx, ny))
                        remain = n * m - (step + 1)
                        if check_connectivity(remain, visited):
                            if DFS(step + 1, nx, ny):
                                return True
                        visited[nx][ny] = False
                        order[nx][ny] = -1
                        path.pop()
                    return False

                if DFS(1, sx, sy):
                    return path, order

        # Generate full solution order matrix via a Hamiltonian path
        _, order = generate_random_hamiltonian_path(N, M)
        self.reference_solution = order
        self.reference_answer = "\n".join(" ".join(map(str, row)) for row in order)

        # Create puzzle matrix by masking some cells to -1
        puzzle = [row[:] for row in order]
        empty_count = max(1, int(N * M * self.sparsity))
        empty_cells = random.sample(range(N * M), empty_count)
        for cell in empty_cells:
            row, column = divmod(cell, M)
            puzzle[row][column] = -1
        self.puzzle_matrix = puzzle

        NM_minus_1 = N * M - 1
        self.current_problem = (
            f"You are given a {N} × {M} matrix with some cells filled with numbers from 0 to {NM_minus_1}, "
            f"and some cells empty (represented by -1). Please fill the empty cells with numbers from 0 to {NM_minus_1} such that:\n"
            f"1. Each number from 0 to {NM_minus_1} appears exactly once in the matrix.\n"
            f"2. Each number is horizontally or vertically adjacent to the next number (i.e., every number x is adjacent to x + 1).\n\n"
            f"The matrix is given as follows:\n"
            f"{self._matrix_to_string(self.puzzle_matrix)}\n\n"
            f"Output Format: Your final answer should contain {N} lines, each with {M} numbers, separated by spaces. "
            f"The numbers should represent the completed matrix in row-major order, matching the format of the given input.\n"
            f"Wrap your entire matrix within \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step by verifying the provided solution."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the matrix from the boxed content
        user_matrix = self._parse_matrix(boxed_content)
        if user_matrix is None or self.N is None or self.M is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate dimensions
        if len(user_matrix) != self.N or any(len(row) != self.M for row in user_matrix):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate against pre-filled cells and constraints
        is_valid = self._validate_solution(user_matrix)

        reward = 1.0 if is_valid else 0.0
        info = {
            "correct": is_valid,
            "reference_answer": self.reference_answer,
            "user_answer": "\n".join(" ".join(map(str, row)) for row in user_matrix),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _parse_matrix(self, text: str) -> Optional[List[List[int]]]:
        """Parse a matrix from text: lines of space-separated integers."""
        try:
            matrix: List[List[int]] = []
            lines = text.splitlines()
            for line in lines:
                line = line.strip()
                if line:
                    row = list(map(int, line.split()))
                    matrix.append(row)
            if not matrix:
                return None
            return matrix
        except ValueError:
            return None

    def _validate_solution(self, solution: List[List[int]]) -> bool:
        """Check if the provided solution satisfies all constraints."""
        assert self.N is not None and self.M is not None, "Environment not initialized"
        assert self.puzzle_matrix is not None, "Puzzle not initialized"

        N, M = self.N, self.M

        # Check consistency with pre-filled cells and valid range
        location: List[Optional[Tuple[int, int]]] = [None] * (N * M)
        for i in range(N):
            for j in range(M):
                original_value = self.puzzle_matrix[i][j]
                solution_value = solution[i][j]
                if original_value != -1 and original_value != solution_value:
                    return False
                if not (0 <= solution_value < N * M):
                    return False
                if location[solution_value] is not None:
                    return False  # duplicate
                location[solution_value] = (i, j)

        # Ensure all values are present
        if any(loc is None for loc in location):
            return False

        # Check adjacency for consecutive numbers
        for value in range(N * M - 1):
            x1, y1 = location[value]
            x2, y2 = location[value + 1]
            if abs(x1 - x2) + abs(y1 - y2) != 1:
                return False

        return True

    def _matrix_to_string(self, matrix: List[List[int]]) -> str:
        """Convert a matrix to a printable string with rows on separate lines."""
        return "\n".join(" ".join(map(str, row)) for row in matrix)

    def sample_random_action(self) -> str:
        """Sample a random action. Here, we provide the reference solution for convenience."""
        if self.reference_answer is None:
            # Fallback: output a random matrix of the correct shape
            if self.N is None or self.M is None:
                return "\\boxed{}"
            rand_matrix = []
            nums = list(range(self.N * self.M))
            random.shuffle(nums)
            for i in range(self.N):
                row = nums[i * self.M:(i + 1) * self.M]
                rand_matrix.append(" ".join(map(str, row)))
            return "\\boxed{\n" + "\n".join(rand_matrix) + "\n}"
        return "\\boxed{\n" + self.reference_answer + "\n}"