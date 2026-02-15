from typing import Any, Optional, SupportsFloat, Tuple, List
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MoneyChargingGameEnv(Env):
    """Money Charging Game environment - single-turn Q&A.

    This environment generates a probabilistic process involving N nodes with associated values A[i][1], A[i][2], A[i][3].
    For each node i, W[i] is randomly assigned from {1, 2, 3} with probabilities proportional to A[i][j].
    A random process repeatedly selects nodes with probability proportional to W[i], and T[i] is the first time node i is added.
    Given constraints T[u] < T[v] corresponding to edges of an undirected tree, compute the total probability that all constraints hold, modulo MOD.

    The answer must be submitted in \\boxed{...} format.
    """

    def __init__(
        self,
        min_n: int = 2,
        max_n: int = 50,
        modulo: int = 998244353,
        fixed_n: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        assert min_n >= 2, "min_n should be at least 2"
        assert max_n >= min_n, "max_n should be >= min_n"
        self.min_n = min_n
        self.max_n = max_n
        self.modulo = modulo
        self.fixed_n = fixed_n

        self.N: Optional[int] = None
        self.A: Optional[List[int]] = None
        self.B: Optional[List[int]] = None
        self.C: Optional[List[int]] = None
        self.T_inequalities: Optional[List[Tuple[int, int]]] = None

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a probabilistic tree constraint problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem."""
        super().reset(seed)

        # Choose N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 2, "N should be greater than or equal to 2"
        self.N = N

        # Generate A, B, C arrays
        A = [random.randint(1, N) for _ in range(N)]
        B = [random.randint(1, N) for _ in range(N)]
        C = [random.randint(1, N) for _ in range(N)]
        self.A, self.B, self.C = A, B, C

        # Generate constraints T_inequalities (random tree edges with directions)
        T_inequalities: List[Tuple[int, int]] = []
        permutation = list(range(N))
        swap_probability = random.random()
        random.shuffle(permutation)
        for i in range(1, N):
            u = permutation[random.randint(0, i - 1)]
            v = permutation[i]
            if random.random() < swap_probability:
                u, v = v, u
            T_inequalities.append((u, v))
        random.shuffle(T_inequalities)

        assert len(T_inequalities) == N - 1, "T_inequalities should have exactly N-1 elements"
        assert len(T_inequalities) == len(set(T_inequalities)), "T_inequalities should not have duplicates"
        for u, v in T_inequalities:
            assert 0 <= u < N and 0 <= v < N, "T_inequalities should contain valid indices"
            assert u != v, "T_inequalities should not contain self-loops"
        tree = networkx.Graph()
        tree.add_edges_from(T_inequalities)
        assert networkx.is_tree(tree), "Generated constraints must form a tree"
        self.T_inequalities = T_inequalities

        # Compute reference answer using the original DP logic
        MOD = self.modulo

        S = []
        for a1, a2, a3 in zip(A, B, C):
            total = a1 + a2 + a3
            S.append(pow(total, MOD - 2, MOD))

        # Precompute modular inverses of 1..3N
        invs = [0] * (3 * N + 1)
        for k in range(1, 3 * N + 1):
            invs[k] = pow(k, MOD - 2, MOD)

        # Build the tree (0-indexed) with flags
        G: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
        for u, v in T_inequalities:
            G[v].append((u, 1))
            G[u].append((v, 0))

        # DP arrays
        f: List[Optional[List[int]]] = [None] * N
        size = [0] * N

        def dfs(x: int, parent: int) -> None:
            size[x] = 1
            fx = [0] * (3 * size[x] + 1)
            fx[1] = A[x] * S[x] % MOD
            fx[2] = B[x] * S[x] % MOD * 2 % MOD
            fx[3] = C[x] * S[x] % MOD * 3 % MOD

            for (v, t) in G[x]:
                if v == parent:
                    continue
                dfs(v, x)
                fy = f[v]
                assert fy is not None
                new_size = size[x] + size[v]
                tmp = [0] * (3 * new_size + 1)

                for i in range(1, size[x] * 3 + 1):
                    if fx[i] == 0:
                        continue
                    for j in range(1, size[v] * 3 + 1):
                        res = fx[i] * fy[j] % MOD
                        if t:
                            tmp[i + j] = (tmp[i + j] - res) % MOD
                            tmp[i] = (tmp[i] + res) % MOD
                        else:
                            tmp[i + j] = (tmp[i + j] + res) % MOD

                size[x] = new_size
                fx = tmp

            for k in range(1, size[x] * 3 + 1):
                fx[k] = fx[k] * invs[k] % MOD

            f[x] = fx

        dfs(0, -1)
        assert f[0] is not None
        self.reference_answer = sum(f[0][1:3 * size[0] + 1]) % MOD

        # Build problem prompt
        A_lines = "\n".join(
            f"A[{i}][1, 2, 3] = [{a}, {b}, {c}]"
            for i, (a, b, c) in enumerate(zip(A, B, C))
        )
        T_lines = "\n".join(f"T[{u}] < T[{v}]" for u, v in T_inequalities)

        self.current_problem = (
            f"There are {N} nodes, each associated with values A[i][1], A[i][2], and A[i][3]. "
            f"For each node i, define: P[i][j] = A[i][j] / (A[i][1] + A[i][2] + A[i][3]) for j = 1, 2, 3. "
            f"The values A are given as follows:\n{A_lines}\n\n"
            "We define the following random process:\n"
            "1. For each node i, randomly assign W[i] = j with probability P[i][j] for j = 1, 2, 3.\n"
            "2. Starting from an empty set, repeatedly select a node i with probability proportional to W[i], and add it to the set (duplicates are allowed). Continue until all nodes are in the set.\n"
            "3. Let T[i] denote the first time node i is added to the set.\n\n"
            "You are also given a set of constraints (each of the form T[u] < T[v]) that correspond to the edges of an undirected tree:\n"
            f"{T_lines}\n\n"
            f"Please compute the total probability that all the above T[u] < T[v] conditions hold during the random process. "
            f"Output the result modulo {MOD}.\n\n"
            "Output Format: Your final answer should be a single integer in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {
            "N": N,
            "modulo": MOD,
            "A": A,
            "B": B,
            "C": C,
            "constraints": T_inequalities,
            "reference_answer": self.reference_answer,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step and verify the answer."""
        answer_str = self._parse_answer(action)

        if answer_str is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        try:
            user_answer = int(answer_str)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Range check as in original logic
        if not (0 <= user_answer < (self.modulo if self.modulo is not None else 10**9 + 7)):
            info = {
                "correct": False,
                "reference_answer": self.reference_answer,
                "user_answer": user_answer,
                "error": "out_of_range",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0
        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer inside \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action."""
        random_answer = random.randint(0, self.modulo - 1)
        return f"\\boxed{{{random_answer}}}"