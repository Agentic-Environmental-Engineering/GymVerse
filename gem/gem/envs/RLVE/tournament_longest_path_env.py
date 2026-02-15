import random
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class Tournament_LongestPathEnv(Env):
    """Tournament longest path environment - single-turn Q&A.

    This environment generates a directed tournament graph with N vertices.
    It asks for the longest simple path starting from a specific vertex S.
    The answer must be provided as a sequence of vertex labels separated by spaces,
    enclosed in \\boxed{...}.
    """

    def __init__(
        self,
        N: int,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = +1.0,
        rewarding_beta: float = 5.0,
        **kwargs
    ):
        """Initialize the environment with the given parameters.

        Parameters:
        - N: Number of vertices in the graph (must be >= 3).
        - wrong_format: Legacy parameter from the original environment (not used in GEM scoring).
        - invalid_solution: Legacy parameter from the original environment (not used in GEM scoring).
        - rewarding_strategy: Legacy parameter from the original environment (not used in GEM scoring).
        - rewarding_weight: Legacy parameter from the original environment (not used in GEM scoring).
        - rewarding_beta: Legacy parameter from the original environment (not used in GEM scoring).
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N

        # Legacy reward configuration retained for compatibility but not used in GEM scoring.
        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # Problem-specific state
        self.TO: Optional[List[List[bool]]] = None
        self.S: Optional[int] = None
        self.reference_answer: Optional[str] = None
        self.gold_length: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a directed tournament graph problem.\n"
            "Please output the longest path starting from the specified vertex S.\n"
            "The path must list vertex labels separated by single spaces, and must be enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        # Generate a random tournament graph (exactly one directed edge between every pair of distinct vertices)
        keep_probability = random.random()
        TO = [[False] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < keep_probability:
                    TO[i][j] = True
                else:
                    TO[j][i] = True
        self.TO = TO

        # Tarjan's algorithm for SCC
        dfn = [0] * N
        low = [0] * N
        on_stack = [False] * N
        stack: List[int] = []
        scc = [0] * N
        comp_nodes: List[List[int]] = []
        time_counter = 0
        scc_count = 0

        def tarjan(u: int) -> None:
            nonlocal time_counter, scc_count
            time_counter += 1
            dfn[u] = low[u] = time_counter
            stack.append(u)
            on_stack[u] = True
            for v in range(N):
                if TO[u][v]:
                    if dfn[v] == 0:
                        tarjan(v)
                        low[u] = min(low[u], low[v])
                    elif on_stack[v]:
                        low[u] = min(low[u], dfn[v])
            if dfn[u] == low[u]:
                comp_nodes.append([])
                cid = scc_count
                scc_count += 1
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc[w] = cid
                    comp_nodes[cid].append(w)
                    if w == u:
                        break

        for i in range(N):
            if dfn[i] == 0:
                tarjan(i)

        # Build a Hamiltonian cycle in each non-trivial SCC
        nxt: List[Optional[int]] = [None] * N

        def solve(cid: int) -> None:
            nodes = comp_nodes[cid]
            if len(nodes) <= 1:
                return
            s = t = nodes[0]
            for x in nodes[1:]:
                if TO[t][x]:
                    nxt[t] = x
                    t = x
                elif TO[x][s]:
                    nxt[x] = s
                    s = x
                else:
                    j = s
                    while j != t:
                        nj = nxt[j]
                        assert nj is not None
                        if TO[j][x] and TO[x][nj]:
                            nxt[x] = nj
                            nxt[j] = x
                            break
                        j = nj
            # close the cycle
            t2: Optional[int] = None
            i = nxt[s]
            while i is not None:
                if TO[i][s]:
                    t2 = i
                elif t2 is not None:
                    j = s
                    while j != t2:
                        nj = nxt[j]
                        assert nj is not None
                        if TO[i][nj]:
                            x = nj
                            nxt[j] = nxt[t2]
                            nxt[t2] = s
                            s = x
                            t2 = i
                            break
                        j = nj
                i = nxt[i]
            assert t2 is not None
            nxt[t2] = s

        for cid in range(scc_count):
            solve(cid)

        # Build answers for each starting vertex
        ans: List[List[int]] = [[] for _ in range(N)]
        for i in range(N):
            x = i
            cid = scc[i]
            while True:
                ans[i].append(x)
                nodes = comp_nodes[cid]
                if len(nodes) == 1:
                    if cid == 0:
                        break
                    cid -= 1
                    x = comp_nodes[cid][0]
                    continue
                j = nxt[x]
                assert j is not None
                while j != x:
                    ans[i].append(j)
                    j = nxt[j]
                    assert j is not None
                if cid == 0:
                    break
                cid -= 1
                x = comp_nodes[cid][0]

        # Select random starting vertex S and prepare reference answer
        S = random.randint(0, N - 1)
        self.S = S
        path = ans[S]
        self.gold_length = len(path)
        self.reference_answer = " ".join(map(str, path))

        # Build problem prompt
        edges_str = "\n".join(f"({s}, {t})" for s in range(N) for t in range(N) if TO[s][t])
        self.current_problem = (
            f"You are given a directed graph with {N} vertices labeled from 0 to {N - 1}. "
            f"The graph contains the following directed edges. Each edge is represented as a tuple (s, t), "
            f"meaning there is a directed edge from vertex s to vertex t:\n{edges_str}\n\n"
            f"It is guaranteed that there is exactly one directed edge between every pair of two distinct vertices.\n"
            f"Please find the longest path starting from vertex {S}, such that no vertex is visited more than once.\n"
            f"Output Format: The path should be a sequence of vertex labels, starting from {S}, separated by spaces, "
            f"and enclosed in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": N,
            "S": S,
            "gold_length": self.gold_length,
        }

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the submitted answer."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse path as space-separated integers
        try:
            path_str = boxed_content.strip()
            path_list = list(map(int, path_str.split())) if path_str else []
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate problem has been set
        if self.TO is None or self.S is None or self.gold_length is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        # Validate path constraints
        if len(path_list) == 0:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "empty_path"}
        if path_list[0] != self.S:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "wrong_start"}
        if not all(0 <= v < self.N for v in path_list):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "vertex_out_of_range"}
        if len(set(path_list)) != len(path_list):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "repeated_vertex"}
        if not all(self.TO[s][t] for s, t in zip(path_list, path_list[1:])):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "nonexistent_edge"}

        # Check correctness: path length must equal the gold longest length
        is_correct = (len(path_list) == self.gold_length)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "gold_length": self.gold_length,
            "user_answer": path_list,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: generates a random (possibly invalid) path starting at S."""
        if self.S is None or self.N is None:
            return "\\boxed{}"
        # Generate a random simple path starting from S
        remaining = list(range(self.N))
        random.shuffle(remaining)
        if self.S in remaining:
            remaining.remove(self.S)
        path_len = random.randint(1, self.N)
        path = [self.S]
        for v in remaining:
            if len(path) >= path_len:
                break
            path.append(v)
        return f"\\boxed{{{' '.join(map(str, path))}}}"