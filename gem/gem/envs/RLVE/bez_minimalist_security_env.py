import random
from typing import Any, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BEZMinimalistSecurityEnv(Env):
    """
    Minimalist Security (BEZ) environment converted to GEM format.

    Task:
    - Given an array P of length N (with values P[i] >= 0) and a set of constraints of the form
      P'[u] + P'[v] = w, construct an array P' such that 0 <= P'[i] <= P[i] for all i,
      all constraints are satisfied, and the sum sum(P') is minimized or maximized as specified.

    Answer format:
    - Return the entire array P' as a space-separated list of integers inside \\boxed{...},
      for example: \\boxed{0 1 2 3}
    """

    def __init__(
        self,
        # Problem size configuration
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 10,
        # Edge density configuration (edge count ~ int(edge_ratio * N), clamped to [1, N*(N-1)//2])
        edge_ratio: Optional[float] = None,
        edge_ratio_min: float = 0.5,
        edge_ratio_max: float = 1.5,
        **kwargs,
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, use this fixed N (must be >= 3). If None, a random N in [min_N, max_N] is used at each reset.
        - min_N, max_N: Range for N if N is not fixed. Must satisfy min_N >= 3 and min_N <= max_N.
        - edge_ratio: If provided, use this fixed edge_ratio (>= 0). If None, a random value in [edge_ratio_min, edge_ratio_max] is used at each reset.
        - edge_ratio_min, edge_ratio_max: Range for edge_ratio if not fixed. Must satisfy 0 <= edge_ratio_min <= edge_ratio_max.
        """
        super().__init__()

        # Validate configuration
        assert isinstance(min_N, int) and isinstance(max_N, int), "min_N and max_N must be integers"
        assert min_N >= 3, "min_N must be at least 3"
        assert min_N <= max_N, "min_N must be <= max_N"
        if N is not None:
            assert isinstance(N, int) and N >= 3, "N must be an integer >= 3"

        if edge_ratio is not None:
            assert edge_ratio >= 0, "edge_ratio must be >= 0"
        assert edge_ratio_min >= 0 and edge_ratio_max >= edge_ratio_min, "edge_ratio range must satisfy 0 <= min <= max"

        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N

        self.fixed_edge_ratio = edge_ratio
        self.edge_ratio_min = edge_ratio_min
        self.edge_ratio_max = edge_ratio_max

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_sum: Optional[int] = None
        self.N: Optional[int] = None
        self.P: Optional[List[int]] = None
        self.edges: Optional[List[Tuple[int, int, int]]] = None
        self.objective: Optional[str] = None  # "minimized" or "maximized"
        self.feasible_solution: Optional[List[int]] = None  # A known feasible P' (used for sampling)

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an array constraint optimization problem.\n"
            "Given P and constraints of the form P'[u] + P'[v] = w, find P' with 0 <= P'[i] <= P[i]\n"
            "such that all constraints are satisfied and the sum is minimized or maximized as specified.\n"
            "Output Format: Provide the space-separated array P' inside \\boxed{...}, e.g., \\boxed{v0 v1 ... vN-1}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be at least 3"

        # Generate a feasible solution P' first
        P_prime = [random.randint(0, N) for _ in range(N)]

        # Determine edge_ratio
        if self.fixed_edge_ratio is not None:
            edge_ratio = self.fixed_edge_ratio
        else:
            edge_ratio = random.uniform(self.edge_ratio_min, self.edge_ratio_max)

        # Build edges from pairs (u, v) with w = P_prime[u] + P_prime[v]
        all_pairs = [(u, v, P_prime[u] + P_prime[v]) for u in range(N) for v in range(u + 1, N)]
        max_edges = N * (N - 1) // 2
        k_edges = max(1, min(max_edges, int(edge_ratio * N)))
        edges = random.sample(all_pairs, k_edges)
        random.shuffle(edges)
        for u, v, w in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, _ in edges)), "edges should be unique"

        # Construct P with P[i] >= P_prime[i]
        P = [P_prime_u + random.randint(0, N) for P_prime_u in P_prime]

        # Compute optimal objective via component analysis
        adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
        for u, v, w in edges:
            adjacency[u].append((v, w))
            adjacency[v].append((u, w))

        vis = [False] * N
        sgn = [0] * N
        cons = [0] * N
        q = [0] * N
        mn = 0
        mx = 0

        def wa() -> None:
            # Generation should always be consistent; if not, raise error
            raise AssertionError("Invalid generated instance")

        def dfs(u: int) -> None:
            nonlocal fix
            vis[u] = True
            stc.append(u)
            if cons[u] > 10**6:
                wa()
            for v, w in adjacency[u]:
                if not vis[v]:
                    sgn[v] = -sgn[u]
                    cons[v] = w - cons[u]
                    dfs(v)
                else:
                    if sgn[u] == sgn[v]:
                        res = w - cons[u] - cons[v]
                        if res & 1:
                            wa()
                        denom = 2 * sgn[u]
                        res //= denom
                        if res < 0 or res > P[anc] or (fix is not None and fix != res):
                            wa()
                        fix = res
                    else:
                        if cons[u] + cons[v] != w:
                            wa()

        for i in range(N):
            if not vis[i]:
                stc: List[int] = []
                anc = i
                fix: Optional[int] = None
                sgn[i] = 1
                cons[i] = 0
                dfs(i)

                if fix is not None:
                    for u in stc:
                        q[u] = sgn[u] * fix + cons[u]
                        delta = P[u] - q[u]
                        mn += delta
                        mx += delta
                        if q[u] < 0 or q[u] > P[u]:
                            wa()
                    for u in stc:
                        for v, w in adjacency[u]:
                            if q[u] + q[v] != w:
                                wa()
                else:
                    l, r = 0, P[anc]
                    for u in stc:
                        if sgn[u] == 1:
                            l = max(l, -cons[u])
                            r = min(r, P[u] - cons[u])
                        else:
                            l = max(l, cons[u] - P[u])
                            r = min(r, cons[u])
                    if l > r:
                        wa()
                    base_sum = 0
                    tsign = 0
                    for u in stc:
                        base_sum += P[u] - (l * sgn[u] + cons[u])
                        tsign -= sgn[u]
                    if tsign > 0:
                        mx += base_sum + tsign * (r - l)
                        mn += base_sum
                    else:
                        mx += base_sum
                        mn += base_sum + tsign * (r - l)

        objective = random.choice(["minimized", "maximized"])
        sum_P = sum(P)
        if objective == "minimized":
            reference_sum = sum_P - mx
        elif objective == "maximized":
            reference_sum = sum_P - mn
        else:
            raise ValueError("Objective should be either 'minimized' or 'maximized'")

        # Save state
        self.N = N
        self.P = P
        self.edges = edges
        self.objective = objective
        self.reference_sum = int(reference_sum)
        self.feasible_solution = P_prime[:]  # known feasible solution

        # Build prompt
        P_str = " ".join(f"P[{i}]={val}" for i, val in enumerate(P))
        constraints_str = "\n".join(f"P'[{u}] + P'[{v}] = {w}" for u, v, w in edges)
        problem_text = (
            f"There is an array P of length {N}. Initially, P is: {P_str}\n\n"
            f"Now we want to construct a new array P' of length {N}, where 0 <= P'[i] <= P[i] for all i. "
            f"Additionally, there are some constraints of the form P'[u] + P'[v] = w, where u and v are indices and w is a constant "
            f"(it is guaranteed that P[u] + P[v] >= w). The constraints are:\n{constraints_str}\n\n"
            f"Please output P'[0], P'[1], ..., P'[{N - 1}], separated by spaces, such that they satisfy all the constraints "
            f"and their sum is {objective}.\n\n"
            f"Output Format: Put the space-separated list inside \\boxed{{...}}, e.g., \\boxed{{v0 v1 ... v{N-1}}}."
        )

        self.current_problem = self._get_instructions() + problem_text

        obs = self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer."""
        # Extract boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse into a list of integers
        try:
            # Allow both spaces and commas as separators
            content = boxed.replace(",", " ")
            tokens = [t for t in content.split() if t]
            answer_list = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate structure
        if self.N is None or self.P is None or self.edges is None or self.objective is None or self.reference_sum is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "internal_state_error"}

        if len(answer_list) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "wrong_length"}

        # Feasibility checks
        P_prime = answer_list
        # Bounds
        for val, bound in zip(P_prime, self.P):
            if not (0 <= val <= bound):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "out_of_bounds"}

        # Constraints
        for u, v, w in self.edges:
            if P_prime[u] + P_prime[v] != w:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "constraint_violation"}

        # Optimality: sum must match the reference optimal sum
        user_sum = sum(P_prime)
        is_correct = (user_sum == self.reference_sum)

        info = {
            "correct": is_correct,
            "objective": self.objective,
            "reference_sum": self.reference_sum,
            "user_sum": user_sum,
        }
        reward = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content from the text."""
        import re

        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """
        Sample a random action.
        Returns a known feasible solution (captured during generation) if available,
        otherwise a random guess of zeros.
        """
        if self.feasible_solution is not None:
            ans = " ".join(str(x) for x in self.feasible_solution)
        elif self.N is not None:
            ans = " ".join("0" for _ in range(self.N))
        else:
            ans = "0"
        return f"\\boxed{{{ans}}}"