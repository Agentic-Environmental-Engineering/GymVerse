from typing import Any, List, Optional, SupportsFloat, Tuple
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class DisinfectionEnv(Env):
    """3D Disinfection problem environment - single-turn Q&A.

    The agent is given a 3D 0-1 cube and must submit a set of sub-cube zeroing operations
    that set all 1-cells to 0 with minimum total cost. Each operation zeroes all cells in
    a half-open sub-cube [x1, x2) × [y1, y2) × [z1, z2) and costs min(x2-x1, y2-y1, z2-z1).
    The final answer must be placed inside \\boxed{...}, consisting of multiple lines,
    each line containing six integers: x1 x2 y1 y2 z1 z2.
    """

    def __init__(
        self,
        max_a_b_c: int = 10,
        # Rewards follow GEM specification for single-turn tasks
        reward_correct: float = 1.0,
        reward_wrong: float = 0.0,
        reward_format_error: float = -0.1,
        **kwargs: Any,
    ):
        super().__init__()
        self.max_a_b_c: int = max_a_b_c
        self.reward_correct: float = reward_correct
        self.reward_wrong: float = reward_wrong
        self.reward_format_error: float = reward_format_error

        # Problem data
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.C: Optional[int] = None
        self.one_coordinates: Optional[List[Tuple[int, int, int]]] = None
        self.gold_cost: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given a 3D cube with 0-1 values and a list of coordinates of cells equal to 1.\n"
            "In one operation, you may choose a half-open sub-cube [x1, x2) × [y1, y2) × [z1, z2) and set all its cells to 0.\n"
            "The cost of each operation is min(x2 - x1, y2 - y1, z2 - z1).\n"
            "Goal: set all 1-cells to 0 with the minimum total cost.\n"
            "Output Format: Place your entire answer inside \\boxed{...}. Inside the box, output multiple lines.\n"
            "Each line contains six integers: x1 x2 y1 y2 z1 z2, separated by spaces, representing one operation.\n"
            "Example:\n"
            "\\boxed{0 1 0 2 0 3\\n1 2 0 2 0 3}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        assert self.max_a_b_c >= 2, "max_a_b_c should be greater than or equal to 2"

        # Generate dimensions A, B, C (avoid trivial 1×1×1)
        while True:
            A = random.randint(1, self.max_a_b_c)
            B = random.randint(1, self.max_a_b_c)
            C = random.randint(1, self.max_a_b_c)
            if not (A == 1 and B == 1 and C == 1):
                break

        # Choose subsets of indices for each axis and then pick a random non-empty subset of their Cartesian product
        subA = random.sample(range(A), random.randint(1, A))
        subB = random.sample(range(B), random.randint(1, B))
        subC = random.sample(range(C), random.randint(1, C))

        cells = [(x, y, z) for x in subA for y in subB for z in subC]
        one_coordinates = random.sample(cells, random.randint(1, len(cells)))
        random.shuffle(one_coordinates)

        self.A, self.B, self.C = A, B, C
        self.one_coordinates = one_coordinates

        # Compute optimal (gold) cost using the original algorithm
        self.gold_cost = self._compute_optimal_cost(A, B, C, one_coordinates)
        assert self.gold_cost is not None and self.gold_cost > 0, "Gold answer should be greater than 0"

        # Build problem statement
        coords_str = "\n".join(f"({x},{y},{z})" for x, y, z in one_coordinates)
        problem_template = (
            f"You are given a 3D cube of dimensions {A} × {B} × {C} (0-indexed). "
            f"Some cells in the cube contain the value 1, and the rest are 0. "
            f"The coordinates of the cells with value 1 are:\n"
            f"{coords_str}\n\n"
            "In one operation, you may select a contiguous sub-cube defined by ranges: "
            f"x ∈ [x1, x2) y ∈ [y1, y2) z ∈ [z1, z2), where 0 ≤ x1 < x2 ≤ {A}, "
            f"0 ≤ y1 < y2 ≤ {B}, and 0 ≤ z1 < z2 ≤ {C}. This operation sets all values in the sub-cube to 0. "
            "The cost of this operation is defined as min(x2 - x1, y2 - y1, z2 - z1).\n"
            "Please set all values in the cube to 0 using a set of such operations with the minimum total cost.\n\n"
            "Output Format: Output multiple lines. Each line should contain six integers "
            "x1 x2 y1 y2 z1 z2 (do NOT include quotes or backticks), separated by spaces, representing one operation. "
            "Place the entire set of lines inside a single \\boxed{...} block."
        )
        self.current_problem = problem_template

        obs = self._get_instructions() + problem_template
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the provided operations and assign reward."""
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, self.reward_format_error, True, False, {"error": "format_error"}

        # Parse operations from boxed content
        ops = self._process_operations(boxed)
        if ops is None:
            return TERMINAL_STATE, self.reward_format_error, True, False, {"error": "format_error"}

        assert self.A is not None and self.B is not None and self.C is not None
        assert self.one_coordinates is not None and self.gold_cost is not None

        # Validate operations and compute total cost
        A, B, C = self.A, self.B, self.C
        disinfected = [[[False] * C for _ in range(B)] for _ in range(A)]
        total_cost = 0

        for idx, (x1, x2, y1, y2, z1, z2) in enumerate(ops):
            # Range checks
            if not (0 <= x1 < x2 <= A):
                return TERMINAL_STATE, self.reward_wrong, True, False, {
                    "error": "invalid_solution",
                    "reason": "x_range_out_of_bounds",
                    "operation_index": idx,
                    "operation": (x1, x2, y1, y2, z1, z2),
                }
            if not (0 <= y1 < y2 <= B):
                return TERMINAL_STATE, self.reward_wrong, True, False, {
                    "error": "invalid_solution",
                    "reason": "y_range_out_of_bounds",
                    "operation_index": idx,
                    "operation": (x1, x2, y1, y2, z1, z2),
                }
            if not (0 <= z1 < z2 <= C):
                return TERMINAL_STATE, self.reward_wrong, True, False, {
                    "error": "invalid_solution",
                    "reason": "z_range_out_of_bounds",
                    "operation_index": idx,
                    "operation": (x1, x2, y1, y2, z1, z2),
                }

            for x in range(x1, x2):
                for y in range(y1, y2):
                    for z in range(z1, z2):
                        disinfected[x][y][z] = True
            total_cost += min(x2 - x1, y2 - y1, z2 - z1)

        # Check coverage of all original 1-cells
        for x, y, z in self.one_coordinates:
            if not disinfected[x][y][z]:
                return TERMINAL_STATE, self.reward_wrong, True, False, {
                    "error": "unsuccessful_solution",
                    "covered_all": False,
                    "gold_cost": self.gold_cost,
                    "user_cost": total_cost,
                }

        # Decide reward based on optimality
        is_optimal = (total_cost == self.gold_cost)
        reward = self.reward_correct if is_optimal else self.reward_wrong

        info = {
            "correct": is_optimal,
            "covered_all": True,
            "gold_cost": self.gold_cost,
            "user_cost": total_cost,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} occurrence."""
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process_operations(self, content: str) -> Optional[List[Tuple[int, int, int, int, int, int]]]:
        """Parse lines of six integers from the boxed content."""
        try:
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            if not lines:
                return []
            ops: List[Tuple[int, int, int, int, int, int]] = []
            for ln in lines:
                parts = ln.split()
                if len(parts) != 6:
                    return None
                nums = tuple(int(x) for x in parts)
                assert len(nums) == 6
                ops.append(nums)  # type: ignore[arg-type]
            return ops
        except Exception:
            return None

    def _compute_optimal_cost(
        self,
        A: int,
        B: int,
        C: int,
        one_coordinates: List[Tuple[int, int, int]],
    ) -> int:
        """Compute the optimal total cost using the original algorithm."""
        DIMS = [A, B, C]

        # Find the shortest axis
        pos = DIMS.index(min(DIMS))  # 0, 1, or 2
        SMALL = DIMS[pos]            # length of the short axis

        # Decide which of the remaining two axes is left (U) and right (V)
        if pos == 0:
            left_len, right_len = B, C  # U = j, V = k
        elif pos == 1:
            left_len, right_len = A, C  # U = i, V = k
        else:  # pos == 2
            left_len, right_len = A, B  # U = i, V = j

        CNT = max(left_len, right_len)

        # Build adjacency list with layer indices
        adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(CNT)]

        def add_edge(u: int, v: int, layer: int) -> None:
            adjacency[u].append((v, layer))

        for i, j, k in one_coordinates:
            if pos == 0:          # short axis = i
                u, v, layer = j, k, i
            elif pos == 1:        # short axis = j
                u, v, layer = i, k, j
            else:                 # short axis = k
                u, v, layer = i, j, k
            add_edge(u, v, layer)

        SEL = [False] * SMALL          # which layers of the short axis are chosen
        VIS = [0] * CNT                # visitation timestamps
        MATCH = [-1] * CNT             # right-side matches
        cur_time = 0                   # DFS clock
        best_answer = [10 ** 9]        # mutable holder for current best

        def dfs(u: int) -> bool:
            nonlocal cur_time
            for v, lay in adjacency[u]:
                if SEL[lay]:
                    continue
                if VIS[v] == cur_time:
                    continue
                VIS[v] = cur_time
                if MATCH[v] == -1 or dfs(MATCH[v]):
                    MATCH[v] = u
                    return True
            return False

        def run_matching(paid: int) -> int:
            """Return paid + |maximum matching| with pruning."""
            nonlocal cur_time, MATCH
            MATCH = [-1] * CNT
            matched = 0
            for u in range(CNT):
                cur_time += 1
                if dfs(u):
                    matched += 1
                    if paid + matched >= best_answer[0]:
                        return paid + matched  # prune
            return paid + matched

        def enumerate_layers(depth: int, paid: int) -> None:
            if depth == SMALL:
                cost = run_matching(paid)
                if cost < best_answer[0]:
                    best_answer[0] = cost
                return
            # Case 1: pay for this layer
            SEL[depth] = True
            enumerate_layers(depth + 1, paid + 1)
            # Case 2: do not pay for this layer
            SEL[depth] = False
            enumerate_layers(depth + 1, paid)

        enumerate_layers(0, 0)
        return best_answer[0]

    def sample_random_action(self) -> str:
        """Sample a random feasible action: one operation per 1-cell (cost may not be optimal)."""
        if self.one_coordinates is None:
            # Fallback: empty answer
            return "\\boxed{}"
        lines = []
        for (x, y, z) in self.one_coordinates:
            # Single-cell operation [x, x+1) × [y, y+1) × [z, z+1), cost = 1
            lines.append(f"{x} {x+1} {y} {y+1} {z} {z+1}")
        content = "\\n".join(lines)
        return f"\\boxed{{{content}}}"