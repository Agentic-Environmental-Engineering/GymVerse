import random
import re
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Minimum_DominatingSet_GridEnv(Env):
    """
    Minimum Dominating Set on a Grid with Costs - Single-turn QA Environment.

    Task:
    - You are given an N x M grid with positive integer costs F[i][j].
    - Select a set of distinct cells S (1-based indexing), such that every cell is either in S
      or has at least one orthogonally adjacent selected neighbor (up, down, left, or right).
    - Minimize the total cost of selected cells (sum of F[i][j] for all (i, j) in S).
    - Output K lines inside \\boxed{...}, each line containing two integers "i j" (1-based),
      representing the selected cells (in any order).

    Reward:
    - Correct (optimal) solution: 1.0
    - Wrong solution: 0.0
    - Format error (cannot extract \\boxed{...}): -0.1
    """

    def __init__(self, max_n_m: int = 8, **kwargs) -> None:
        """
        Initialize the environment.

        Args:
            max_n_m: Maximum size for both N and M (N, M are sampled uniformly from [2, max_n_m]).
                     Must be >= 2.
        """
        super().__init__()
        if max_n_m < 2:
            raise ValueError("max_n_m should be greater than or equal to 2")
        self.max_n_m: int = max_n_m

        # Problem state
        self.N: Optional[int] = None
        self.M: Optional[int] = None
        self.F: Optional[List[List[int]]] = None
        self.gold_answer: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a minimum dominating set problem on a grid with costs.\n"
            "Select a set of distinct cells S such that every cell is either in S or has at least one orthogonally adjacent selected neighbor (up, down, left, or right).\n"
            "Your goal is to minimize the total cost of selected cells.\n"
            "Output Format: Put your answer inside a single \\boxed{...} block. Inside the box, write K lines, each line with two integers 'i j' (1-based indices), representing the selected cells.\n"
            "Example:\n"
            "\\boxed{ \n"
            "1 2\n"
            "3 4\n"
            "}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Sample grid size
        N = random.randint(2, self.max_n_m)
        M = random.randint(2, self.max_n_m)
        self.N = N
        self.M = M

        # Generate cost matrix F with values in [1, N*M]
        F = [[random.randint(1, N * M) for _ in range(M)] for _ in range(N)]
        self.F = F

        # Compute the optimal (minimum) total cost using DP over row bitmasks
        gold = self._compute_optimal_cost(N, M, F)
        if not (gold is not None and gold > 0):
            raise RuntimeError("Failed to compute a valid optimal cost")
        self.gold_answer = gold

        # Build the problem prompt
        F_str = "\n".join(
            " ".join(f"F[{i}][{j}]={F[i-1][j-1]}" for j in range(1, M + 1))
            for i in range(1, N + 1)
        )
        self.current_problem = (
            f"We have a grid with {N} rows and {M} columns (1-based indices). The cost of cell (i, j) is F[i][j]:\n"
            f"{F_str}\n\n"
            "Select a set of distinct cells S such that every cell is either in S or has at least one orthogonally adjacent selected neighbor (up, down, left, or right). "
            "Minimize the total cost of selected cells (i.e., the sum of F[i][j] for all (i,j) ∈ S). "
            "Output K (the number of selected cells) lines: each line contains two integers 'i j' (1-based), the row and column of a selected cell (in any order)."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided solution and return the reward."""
        # Extract boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.N is None or self.M is None or self.F is None or self.gold_answer is None:
            # Should never happen after a proper reset
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_initialized"}

        # Parse lines "i j" from the boxed content
        try:
            parsed_cells = self._parse_cells(boxed)
        except ValueError:
            # Parsing failed (e.g., malformed lines)
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate indices and uniqueness
        selected = [[False] * self.M for _ in range(self.N)]
        for (i, j) in parsed_cells:
            if not (1 <= i <= self.N and 1 <= j <= self.M):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "index_out_of_bounds"}
            if selected[i - 1][j - 1]:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "duplicate_cell"}
            selected[i - 1][j - 1] = True

        # Check domination property (orthogonal adjacency + itself)
        dxs = [0, 0, 0, -1, +1]
        dys = [0, -1, +1, 0, 0]
        for i in range(self.N):
            for j in range(self.M):
                covered = False
                for dx, dy in zip(dxs, dys):
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < self.N and 0 <= nj < self.M and selected[ni][nj]:
                        covered = True
                        break
                if not covered:
                    return TERMINAL_STATE, 0.0, True, False, {"error": "unsuccessful_solution"}

        # Compute total cost of selected cells
        user_cost = sum(self.F[i - 1][j - 1] for (i, j) in parsed_cells)
        gold_cost = self.gold_answer

        # Reward: 1.0 if optimal, else 0.0
        is_optimal = (user_cost == gold_cost)
        info = {
            "correct": is_optimal,
            "reference_answer": gold_cost,
            "user_answer": user_cost,
            "selected_count": len(parsed_cells),
        }
        reward: float = 1.0 if is_optimal else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """
        Extract the content inside the last \\boxed{...} occurrence.
        Supports multi-line content inside the box.
        """
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if not matches:
            return None
        return matches[-1].strip()

    def _parse_cells(self, boxed_content: str) -> List[Tuple[int, int]]:
        """
        Parse a list of cell coordinates from the boxed content.
        Each non-empty line should contain two integers: i j.
        """
        cells: List[Tuple[int, int]] = []
        lines = boxed_content.splitlines()
        for line in lines:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 2:
                raise ValueError("Each line must contain exactly two integers 'i j'")
            i, j = int(parts[0]), int(parts[1])
            cells.append((i, j))
        return cells

    def _compute_optimal_cost(self, N: int, M: int, F: List[List[int]]) -> int:
        """
        Compute the minimal total cost for a dominating set on an N x M grid with costs F,
        using dynamic programming over bitmasks per row.

        This reproduces the algorithmic core of the original environment.
        """
        S = 1 << M
        ALL = S - 1

        # Precompute popcount for each mask
        ones = [0] * S
        for m in range(S):
            # Using int.bit_count() for efficiency
            ones[m] = m.bit_count()

        # Coverage within a row (left/right neighbors + itself)
        shift_cov = [0] * S
        for m in range(S):
            shift_cov[m] = (m | ((m << 1) & ALL) | (m >> 1)) & ALL

        # Map single-bit to column index
        bit_to_idx = {1 << c: c for c in range(M)}

        # Row sums: row_sums[i][mask] = cost of choosing 'mask' on row i
        # We add a dummy row N+1 with zero costs
        row_sums = [[0] * S for _ in range(N + 2)]
        for i in range(1, N + 1):
            costs = F[i - 1]
            rs = row_sums[i]
            for mask in range(S):
                total = 0
                x = mask
                while x:
                    t = x & -x
                    total += costs[bit_to_idx[t]]
                    x -= t
                rs[mask] = total
        # row_sums[N+1] remains zeros

        # Precompute supersets: for each 'need' mask, list all supersets p
        supersets: List[List[int]] = [[] for _ in range(S)]
        for need in range(S):
            rem = ALL ^ need
            x = rem
            while True:
                supersets[need].append(need | x)
                if x == 0:
                    break
                x = (x - 1) & rem

        INF = float('inf')

        # DP arrays: f[p][j] minimal cost, g[p][j] number of selected cells (tie-breaker)
        f = [[INF] * S for _ in range(S)]
        g = [[INF] * S for _ in range(S)]

        # Initialize first row (previous row k is 0)
        rs1 = row_sums[1]
        for j in range(S):
            f[j][0] = rs1[j]
            g[j][0] = ones[j]

        # Transition for rows 2..N+1 (N+1 is dummy zero-cost row to flush coverage)
        for i in range(2, N + 2):
            nf = [[INF] * S for _ in range(S)]
            ng = [[INF] * S for _ in range(S)]
            rsi = row_sums[i]
            for j in range(S):  # mask for row i-1
                sj = shift_cov[j]
                fj = f[j]
                gj = g[j]
                for k in range(S):  # mask for row i-2
                    base_cost = fj[k]
                    if base_cost == INF:
                        continue
                    base_cnt = gj[k]
                    need = ALL ^ (sj | k)  # columns that still need coverage on row i-1
                    for p in supersets[need]:  # mask for row i
                        v = base_cost + rsi[p]
                        c = base_cnt + ones[p]
                        if v < nf[p][j]:
                            nf[p][j] = v
                            ng[p][j] = c
                        elif v == nf[p][j] and c < ng[p][j]:
                            ng[p][j] = c
            f, g = nf, ng

        # Finalize at the dummy last row: p = 0; scan any j
        best_cost = INF
        best_cnt = INF
        for j in range(S):
            v = f[0][j]
            if v < best_cost:
                best_cost = v
                best_cnt = g[0][j]
            elif v == best_cost and g[0][j] < best_cnt:
                best_cnt = g[0][j]

        if best_cost == INF:
            raise RuntimeError("DP failed to compute a valid solution")
        if best_cost <= 0:
            raise RuntimeError("Optimal cost should be positive")
        return int(best_cost)

    def sample_random_action(self) -> str:
        """
        Sample a simple (not necessarily optimal) valid action.
        This returns selecting all cells, which is always a dominating set but not optimal.
        """
        if self.N is None or self.M is None:
            # If no problem has been generated yet, return an empty selection
            return "\\boxed{}"
        lines = []
        for i in range(1, self.N + 1):
            for j in range(1, self.M + 1):
                lines.append(f"{i} {j}")
        content = "\n".join(lines)
        return f"\\boxed{{\n{content}\n}}"