from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TreeTopologicalSequenceCountingEnv(Env):
    """Environment for counting the number of permutations satisfying tree-based inequality constraints."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 12,
        max_MOD: int = 1000000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: Fixed number of elements in the permutation. If None, it will be sampled in [min_N, max_N].
            min_N: Minimum N if sampling.
            max_N: Maximum N if sampling.
            max_MOD: Upper bound for random modulo selection (inclusive of 2..max_MOD).
        """
        super().__init__()
        self.fixed_N = N
        self.min_N = min_N
        self.max_N = max_N
        self.max_MOD = max_MOD

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.N: Optional[int] = None
        self.MOD: Optional[int] = None
        self.edges: List[tuple[int, str, int]] = []

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are given a counting problem over permutations with inequality constraints forming a tree.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n"
            "If your answer is not enclosed in \\boxed{...}, it will be considered a format error.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new single-turn problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        assert N >= 3, "N should be greater than or equal to 3"
        self.N = N

        # Select MOD
        MOD = random.randint(2, self.max_MOD)
        self.MOD = MOD

        # Generate constraints on a random tree
        p = list(range(N))
        random.shuffle(p)

        edges: List[tuple[int, str, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            edges.append((u, "<" if p[u] < p[v] else ">", v))
        random.shuffle(edges)

        for u, w, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, w, v in edges)) == N - 1

        tree = networkx.Graph()
        tree.add_edges_from((u, v) for u, w, v in edges)
        assert networkx.is_tree(tree)

        # Compute the reference answer using the provided algorithm
        # Precompute binomial coefficients up to N
        C = [[0] * (N + 1) for _ in range(N + 1)]
        for i in range(N + 1):
            C[i][0] = 1
            for j in range(1, i + 1):
                C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % MOD

        def dfs(u: int, parent: int, h1: List[List[int]], h2: List[List[int]]) -> Tuple[List[int], int]:
            # f_raw[k]: number of ways (raw) to have exactly k nodes before u
            f_raw = [0, 1]   # only u itself => 1 way with k=1
            sz = 1           # size of subtree rooted at u

            # First, merge all children v where u < v (v must come after u)
            for v in h1[u]:
                if v == parent:
                    continue
                f_v, sz_v = dfs(v, u, h1, h2)
                g = f_raw[:]          # copy old
                new_sz = sz + sz_v
                new_f = [0] * (new_sz + 1)
                for j in range(1, sz + 1):
                    gj = g[j]
                    if gj == 0:
                        continue
                    for i_count in range(j, sz_v + j):
                        # Combine with child-subtree counts that place at least (i_count-j+1) before v
                        diff = f_v[sz_v] - f_v[i_count - j]
                        if diff < 0:
                            diff += MOD
                        term = gj
                        term = term * C[i_count - 1][j - 1] % MOD
                        term = term * C[sz + sz_v - i_count][sz - j] % MOD
                        term = term * diff % MOD
                        new_f[i_count] = (new_f[i_count] + term) % MOD
                f_raw = new_f
                sz = new_sz

            # Then, merge all children v where u > v (v must come before u)
            for v in h2[u]:
                if v == parent:
                    continue
                f_v, sz_v = dfs(v, u, h1, h2)
                g = f_raw[:]
                new_sz = sz + sz_v
                new_f = [0] * (new_sz + 1)
                for j in range(1, sz + 1):
                    gj = g[j]
                    if gj == 0:
                        continue
                    for i_count in range(j + 1, sz_v + j + 1):
                        # Combine with child-subtree counts that place exactly (i_count-j) before v
                        term = gj
                        term = term * C[i_count - 1][j - 1] % MOD
                        term = term * C[sz + sz_v - i_count][sz - j] % MOD
                        term = term * f_v[i_count - j] % MOD
                        new_f[i_count] = (new_f[i_count] + term) % MOD
                f_raw = new_f
                sz = new_sz

            # Turn raw counts into prefix-sums: f_pref[k] = sum_{t=1..k} f_raw[t]
            f_pref = [0] * (sz + 1)
            for i_count in range(1, sz + 1):
                s = f_pref[i_count - 1] + f_raw[i_count]
                if s >= MOD:
                    s -= MOD
                f_pref[i_count] = s

            return f_pref, sz

        # Build directed adjacency lists (1-indexed internally for the DP)
        h1 = [[] for _ in range(N + 1)]
        h2 = [[] for _ in range(N + 1)]
        for a, sign, b in edges:
            x, y = a + 1, b + 1
            if sign == '<':
                h1[x].append(y)
                h2[y].append(x)
            else:
                h1[y].append(x)
                h2[x].append(y)

        f_root, _ = dfs(1, 0, h1, h2)
        reference_answer = f_root[N] % MOD

        # Store for step validation
        self.edges = edges
        self.reference_answer = reference_answer

        # Build the problem statement
        constraints_str = "; ".join(f"p[{u}] {w} p[{v}]" for u, w, v in edges)
        self.current_problem = (
            f"Please count the number of permutations of the integers from 0 to {N - 1}, "
            f"denoted as p[0], p[1], ..., p[{N - 1}], such that the following {N - 1} constraints are satisfied: {constraints_str}\n"
            f"Note that each constraint above is of the form `p[i] < p[j]` or `p[i] > p[j]`, and collectively, these constraints correspond to a tree — "
            f"that is, a connected undirected graph with no cycles — on {N} vertices labeled from 0 to {N - 1}.\n"
            f"You should output the number of valid permutations modulo {MOD}.\n\n"
            f"Output Format: Your final answer should be a single integer in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": N,
            "MOD": MOD,
            "constraints": edges[:],
        }

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the single-shot answer."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if self.reference_answer is None or self.MOD is None or self.N is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_ready"}

        # Optional range check: original environment required 0 <= answer < MOD
        out_of_range = not (0 <= user_answer < self.MOD)

        is_correct = (not out_of_range) and (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "out_of_range": out_of_range,
            "N": self.N,
            "MOD": self.MOD,
            "constraints": self.edges[:],
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
        """Sample a random action formatted as \\boxed{...}."""
        mod = self.MOD if self.MOD is not None else max(2, self.max_MOD)
        random_answer = random.randint(0, max(1, mod - 1))
        return f"\\boxed{{{random_answer}}}"