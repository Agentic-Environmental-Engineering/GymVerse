import random
from collections import deque
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CirculatingGridEnv(Env):
    """Circulating Grid environment (single-turn Q&A).

    Task:
    - Given an R x C toroidal grid with directions L/R/U/D in each cell.
    - You may modify any number of cells.
    - The final grid must satisfy: starting from any cell and following directions, it is possible to eventually return to the same cell (not counting simply standing still).
      This is equivalent to each cell having in-degree exactly 1 (forming a disjoint union of directed cycles).
    - Minimize the number of modified cells (i.e., the number of cells whose character differs from the original grid).
    - Output the modified grid in exactly R lines, each with C characters, no separators.

    Answer format:
    - Return the final grid inside \\boxed{...} with newline separators between lines.
    - Example (for R=2, C=3):
      \\boxed{LRU
      DDL}

    Reward:
    - Correct and optimal (valid and uses the minimum number of changes): 1.0
    - Valid but not optimal, or invalid: 0.0
    - Format error: -0.1
    """

    def __init__(
        self,
        max_r_c: int = 10,
        force_r: Optional[int] = None,
        force_c: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.max_r_c = max_r_c
        self.force_r = force_r
        self.force_c = force_c

        # Internal state for the current instance
        self.R: Optional[int] = None
        self.C: Optional[int] = None
        self.original_grid: Optional[List[List[str]]] = None
        self.current_problem: Optional[str] = None
        self.reference_min_changes: Optional[int] = None

        # Validation
        assert self.max_r_c >= 3, "max_r_c must be at least 3"
        if self.force_r is not None:
            assert 2 <= self.force_r <= self.max_r_c, "force_r must be in [2, max_r_c]"
        if self.force_c is not None:
            assert 2 <= self.force_c <= self.max_r_c, "force_c must be in [2, max_r_c]"

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a toroidal grid graph problem.\n"
            "Goal: Modify the fewest number of cells so that the resulting grid consists only of directed cycles (every cell has in-degree exactly 1).\n"
            "Answer format: Return the final grid inside \\boxed{...}, with exactly R lines and each line having C characters (only L, R, U, D), no separators.\n"
            "Example: For R=2, C=3, a valid output might be:\n"
            "\\boxed{LRU\nDDL}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment, generate a new instance, and return the problem statement."""
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        # Determine R and C
        R = self.force_r if self.force_r is not None else random.randint(2, self.max_r_c)
        C = self.force_c if self.force_c is not None else random.randint(2, self.max_r_c)

        # Generate a random distribution for L/R/U/D and construct the grid
        LRUD_distribution = [random.randint(1, R * C) for _ in range(4)]
        def sample_char() -> str:
            return random.choices(['L', 'R', 'U', 'D'], weights=LRUD_distribution, k=1)[0]
        grid = [[sample_char() for _ in range(C)] for _ in range(R)]

        # Compute the minimum number of changes using a min-cost max-flow construction
        # The idea: bipartite assignment between cells and their 4 neighbors; cost 0 if we keep the original direction, else 1.
        DX = [0, 0, -1, 1]   # row delta for L, R, U, D
        DY = [-1, 1, 0, 0]   # col delta for L, R, U, D
        DIR_ID = {'L': 0, 'R': 1, 'U': 2, 'D': 3}

        class Edge:
            __slots__ = ('to', 'rev', 'cap', 'cost')
            def __init__(self, to: int, rev: int, cap: int, cost: int) -> None:
                self.to = to
                self.rev = rev
                self.cap = cap
                self.cost = cost

        def add_edge(graph: List[List[Edge]], u: int, v: int, cap: int, cost: int) -> None:
            graph[u].append(Edge(v, len(graph[v]), cap, cost))
            graph[v].append(Edge(u, len(graph[u]) - 1, 0, -cost))

        def min_cost_max_flow(graph: List[List[Edge]], N: int, s: int, t: int, INF: int) -> Tuple[int, int]:
            flow = 0
            cost = 0
            dist = [0] * N
            inq = [False] * N
            prev_node = [-1] * N
            prev_edge = [-1] * N

            while True:
                for i in range(N):
                    dist[i] = INF
                    inq[i] = False
                    prev_node[i] = -1
                    prev_edge[i] = -1
                dist[s] = 0
                q: deque[int] = deque([s])
                inq[s] = True

                # SPFA to find shortest augmenting path by cost
                while q:
                    u = q.popleft()
                    inq[u] = False
                    for ei, e in enumerate(graph[u]):
                        if e.cap > 0:
                            v = e.to
                            nd = dist[u] + e.cost
                            if nd < dist[v]:
                                dist[v] = nd
                                prev_node[v] = u
                                prev_edge[v] = ei
                                if not inq[v]:
                                    inq[v] = True
                                    q.append(v)

                if prev_node[t] == -1:
                    break  # no more augmenting paths

                # Find bottleneck
                addf = INF
                v = t
                while v != s:
                    u = prev_node[v]
                    ei = prev_edge[v]
                    e = graph[u][ei]
                    if e.cap < addf:
                        addf = e.cap
                    v = u

                # Augment
                v = t
                while v != s:
                    u = prev_node[v]
                    ei = prev_edge[v]
                    e = graph[u][ei]
                    e.cap -= addf
                    graph[v][e.rev].cap += addf
                    cost += addf * e.cost
                    v = u

                flow += addf

            return flow, cost

        def compute_min_changes() -> int:
            MP = [[DIR_ID[grid[i][j]] for j in range(C)] for i in range(R)]

            n_left = R * C
            offset = n_left
            s = 2 * n_left
            t = s + 1
            N = t + 1

            INF = R * C * 4 + 5
            graph: List[List[Edge]] = [[] for _ in range(N)]

            # Build edges from each cell (left partition) to its 4 neighbors (right partition)
            for i in range(R):
                for j in range(C):
                    u = i * C + j
                    for k in range(4):
                        ni = (i + DX[k]) % R
                        nj = (j + DY[k]) % C
                        v = offset + (ni * C + nj)
                        cost = 0 if k == MP[i][j] else 1
                        add_edge(graph, u, v, 1, cost)

            # Source to all left nodes; all right nodes to sink
            for u in range(n_left):
                add_edge(graph, s, u, 1, 0)
            for v in range(offset, offset + n_left):
                add_edge(graph, v, t, 1, 0)

            _, total_cost = min_cost_max_flow(graph, N, s, t, INF)
            return total_cost

        gold_min_changes = compute_min_changes()

        # Save state
        self.R = R
        self.C = C
        self.original_grid = grid
        self.reference_min_changes = gold_min_changes

        # Build the problem prompt
        grid_str = "\n".join("".join(row) for row in grid)
        problem_text = (
            f"Consider a {R} × {C} grid, where each cell has coordinates (i, j) (0 ≤ i < {R}, 0 ≤ j < {C}). "
            "Each cell contains one of the characters L, R, U, or D, meaning:\n"
            f"- L: moves to (i, (j - 1) MOD {C})\n"
            f"- R: moves to (i, (j + 1) MOD {C})\n"
            f"- U: moves to ((i - 1) MOD {R}, j)\n"
            f"- D: moves to ((i + 1) MOD {R}, j)\n"
            "Here, (-1 MOD N) = N - 1.\n\n"
            "You are given such a grid:\n"
            f"{grid_str}\n\n"
            "Modify any number of cells so that the resulting grid satisfies the following condition: "
            "Starting from any cell, it must be possible to eventually return to the same cell (simply standing there at the beginning does not count). "
            "Use as few changes (number of cells modified) as possible. "
            f"Output the modified grid in the same format — exactly {R} lines, each containing {C} characters (L, R, U, or D) with no separators.\n\n"
            "Output Format: Put your final grid inside \\boxed{...} with newline separators between rows.\n"
            "Example: \\boxed{LRU\\nDDL}"
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify a single answer and terminate."""
        # Parse boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure we have a current problem
        if self.R is None or self.C is None or self.original_grid is None or self.reference_min_changes is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "no_active_problem"}

        # Process the boxed grid
        lines = [line.strip() for line in boxed_content.splitlines() if line.strip() != ""]
        if len(lines) != self.R:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "wrong_number_of_rows"}

        if not all(len(row) == self.C for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "wrong_number_of_columns"}

        if not all(all(c in "LRUD" for c in row) for row in lines):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error", "detail": "invalid_characters"}

        # Check in-degree constraints (toroidal)
        R, C = self.R, self.C
        in_degree = [[0] * C for _ in range(R)]
        for i in range(R):
            for j in range(C):
                ch = lines[i][j]
                if ch == "L":
                    in_degree[i][(j - 1 + C) % C] += 1
                elif ch == "R":
                    in_degree[i][(j + 1) % C] += 1
                elif ch == "U":
                    in_degree[(i - 1 + R) % R][j] += 1
                elif ch == "D":
                    in_degree[(i + 1) % R][j] += 1
                else:
                    # Should not happen due to earlier check
                    return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        valid_solution = all(in_degree[i][j] == 1 for i in range(R) for j in range(C))
        if not valid_solution:
            info = {
                "valid_format": True,
                "valid_solution": False,
                "optimal": False,
                "reference_min_changes": self.reference_min_changes,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Count changes relative to original grid
        changes = sum(
            1 if lines[i][j] != self.original_grid[i][j] else 0
            for i in range(R)
            for j in range(C)
        )
        optimal = (changes == self.reference_min_changes)
        reward = 1.0 if optimal else 0.0

        info = {
            "valid_format": True,
            "valid_solution": True,
            "optimal": optimal,
            "changes": changes,
            "reference_min_changes": self.reference_min_changes,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Supports multiline."""
        import re

        pattern = r'\\boxed\{(.*?)\}'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random (not necessarily valid nor optimal) grid action."""
        if self.R is None or self.C is None:
            # Provide a simple fallback
            return "\\boxed{L}"

        chars = ['L', 'R', 'U', 'D']
        lines = []
        for _ in range(self.R):
            row = "".join(random.choice(chars) for _ in range(self.C))
            lines.append(row)
        content = "\n".join(lines)
        return f"\\boxed{{{content}}}"