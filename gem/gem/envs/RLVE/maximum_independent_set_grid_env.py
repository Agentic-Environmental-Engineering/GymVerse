import random
from typing import Any, Optional, SupportsFloat, Tuple, List
import networkx as nx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MaximumIndependentSetGridEnv(Env):
    """Maximum Independent Set on a Grid with weights - single-turn environment.

    The task is to select cells in an N x M grid with integer weights such that:
    - No two selected cells share a horizontal or vertical edge (no adjacency).
    - The sum of selected cell weights is maximized.

    The agent must output an N-line selection matrix (0/1 digits, no separators),
    placed inside a single \\boxed{...} block.

    Reward scheme:
    - Correct optimal solution: 1.0
    - Wrong solution (feasible but not optimal) or invalid solution: 0.0
    - Format error: -0.1
    """

    def __init__(
        self,
        max_n_m: int = 10,
        **kwargs
    ):
        """Initialize the environment.

        Args:
            max_n_m: Maximum size for N and M (both sampled in [2, max_n_m]).
                     Must be >= 2.
        """
        super().__init__()
        if max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")

        self.max_n_m: int = max_n_m

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.matrix: Optional[List[List[int]]] = None
        self.gold_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a matrix of size N × M with positive integer weights.\n"
            "Select some cells such that no two selected cells are adjacent "
            "(i.e., no two selected cells share a horizontal or vertical edge). "
            "Your goal is to maximize the sum of the values in the selected cells.\n\n"
            "Answer Format: Provide N lines, each with M digits (0 or 1) and no separators, "
            "inside a single \\boxed{...}. A '1' means the cell is selected; a '0' means it is not.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem.

        Returns:
            observation: The instructions plus the specific problem description.
            info: Additional info dict (empty for this environment).
        """
        super().reset(seed)

        # Sample dimensions
        self.N = random.randint(2, self.max_n_m)
        self.M = random.randint(2, self.max_n_m)

        # Generate matrix weights in [1, max(N, M)]
        upper = max(self.N, self.M)
        self.matrix = [[random.randint(1, upper) for _ in range(self.M)] for _ in range(self.N)]

        # Compute optimal value via min-cut on bipartite graph formulation
        self.gold_answer = self._compute_optimal_sum(self.N, self.M, self.matrix)

        # Build problem description
        matrix_str = "\n".join(" ".join(str(x) for x in row) for row in self.matrix)
        self.current_problem = (
            f"You are given a matrix of size {self.N} × {self.M}. "
            "Select some cells such that no two selected cells are adjacent "
            "(i.e., no two selected cells share a horizontal or vertical edge). "
            "Try your best to maximize the sum of the values in the selected cells. "
            "The matrix is given below (in row-major order):\n"
            f"{matrix_str}\n\n"
            f"Output Format: Output {self.N} lines, each with {self.M} digits (0 or 1) and no separators. "
            "A '1' means the corresponding cell is selected; a '0' means it is not. "
            "Place your answer inside a single \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided solution.

        Args:
            action: The agent's answer text containing a single \\boxed{...} block
                    with N lines of M digits (0/1), no separators.

        Returns:
            observation: TERMINAL_STATE for single-turn environments.
            reward: 1.0 if optimal and valid, 0.0 if wrong or invalid, -0.1 if format error.
            terminated: True (single-turn).
            truncated: False.
            info: Dictionary containing details about correctness and values.
        """
        # Ensure a problem has been generated
        if self.N is None or self.M is None or self.matrix is None or self.gold_answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "no_problem_generated"}

        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Process boxed content into a list of lines (solution matrix)
        solution_lines = self._process_solution_text(boxed_content)
        if solution_lines is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate dimensions and characters
        if len(solution_lines) != self.N or any(len(row) != self.M for row in solution_lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
        if any(c not in '01' for row in solution_lines for c in row):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Check feasibility (no adjacent selected cells), compute sum
        user_sum = 0
        for i in range(self.N):
            for j in range(self.M):
                if solution_lines[i][j] == '1':
                    user_sum += self.matrix[i][j]
                    for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.N and 0 <= nj < self.M and solution_lines[ni][nj] == '1':
                            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "violation": "adjacency"}

        # If somehow exceeds known optimum, mark invalid
        if user_sum > self.gold_answer:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "violation": "exceeds_optimum"}

        is_optimal = (user_sum == self.gold_answer)
        reward: float = 1.0 if is_optimal else 0.0

        info = {
            "correct": is_optimal,
            "reference_answer": self.gold_answer,
            "user_answer": user_sum,
            "N": self.N,
            "M": self.M,
            "matrix": self.matrix,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} block.

        Supports multi-line content inside the box.
        """
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process_solution_text(self, text: str) -> Optional[List[str]]:
        """Convert the boxed content into a list of non-empty, trimmed lines."""
        try:
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
            return lines
        except Exception:
            return None

    def _compute_optimal_sum(self, N: int, M: int, NUM: List[List[int]]) -> int:
        """Compute the maximum weight independent set value via min-cut on a bipartite graph."""
        total = sum(sum(row) for row in NUM)
        inf = total

        G = nx.DiGraph()
        source, sink = 's', 't'

        for i in range(N):
            for j in range(M):
                u = (i, j)
                weight = NUM[i][j]

                if (i + j) % 2 == 1:
                    # Odd parity: source -> u
                    G.add_edge(source, u, capacity=weight)
                    # Connect odd-parity cell to neighbors with infinite capacity
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M:
                            v = (ni, nj)
                            G.add_edge(u, v, capacity=inf)
                else:
                    # Even parity: u -> sink
                    G.add_edge(u, sink, capacity=weight)

        flow_value, _ = nx.maximum_flow(G, source, sink)
        max_weight_independent_set = total - flow_value
        if max_weight_independent_set <= 0:
            # In extremely degenerate cases (should not happen with positive weights),
            # ensure non-negative value.
            max_weight_independent_set = max(0, max_weight_independent_set)
        return max_weight_independent_set

    def sample_random_action(self) -> str:
        """Sample a random (not necessarily valid or optimal) action in \\boxed{...} format."""
        if self.N is None or self.M is None:
            # Default to a trivial random small matrix
            n, m = 2, 2
        else:
            n, m = self.N, self.M

        # Create a random 0/1 matrix (independent of adjacency feasibility)
        lines = []
        for _ in range(n):
            row = ''.join(str(random.randint(0, 1)) for _ in range(m))
            lines.append(row)
        content = "\n".join(lines)
        return f"\\boxed{{\n{content}\n}}"