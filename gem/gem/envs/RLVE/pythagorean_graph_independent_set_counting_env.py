import math
import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PythagoreanGraph_IndependentSetCountingEnv(Env):
    """Environment for counting non-empty independent sets in a Pythagorean graph constructed from an array H."""

    def __init__(
        self,
        N: int,
        max_MOD: int = 1000000,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            N: Length of the array H (number of vertices). Must be >= 3.
            max_MOD: Upper bound for randomly selected modulo (inclusive lower bound is 2).
        """
        super().__init__()
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"
        assert isinstance(max_MOD, int) and max_MOD >= 2, "max_MOD must be an integer >= 2"

        self.N: int = N
        self.max_MOD: int = max_MOD

        # Problem state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.MOD: Optional[int] = None
        self.H: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "You are given an array H and must construct an undirected graph as specified, then count the number of "
            "non-empty independent sets modulo a given MOD.\n"
            "Please provide your answer in \\boxed{...} format, containing a single integer in the range [0, MOD-1].\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        # Randomly choose modulo
        MOD = random.randint(2, self.max_MOD)

        while True:
            # Generate H
            H = [random.randint(1, 2 * N) for _ in range(N)]

            # Count occurrences of each length
            maxH = max(H)
            num = [0] * (maxH + 1)
            for h in H:
                num[h] += 1

            # Precompute powers of 2 modulo MOD up to N
            PW2 = [1] * (N + 1)
            for i in range(1, N + 1):
                PW2[i] = (PW2[i - 1] * 2) % MOD

            # Build adjacency lists for primitive Pythagorean pairs among lengths present in H
            to = [[] for _ in range(maxH + 1)]
            limit_i = int(math.isqrt(maxH))
            two_max = 2 * maxH
            for i in range(1, limit_i + 1):
                j_max1 = maxH // (2 * i)
                j_max2 = int(math.isqrt(two_max))
                j_max = min(j_max1, j_max2)
                for j in range(i + 1, j_max + 1):
                    x = j * j - i * i
                    y = 2 * i * j
                    if x > maxH or y > maxH:
                        continue
                    if num[x] == 0 or num[y] == 0:
                        continue
                    if math.gcd(x, y) != 1:
                        continue
                    to[x].append(y)
                    to[y].append(x)

            # Arrays and helpers for DFS and DP
            vis = [False] * (maxH + 1)  # visited for initialization DFS
            ins = [False] * (maxH + 1)  # in-cycle indicator
            sat = [0] * (maxH + 1)      # forced status: 1 selected, -1 not selected, 0 free
            des = [0] * (maxH + 1)      # DP visit stamp
            dp0 = [0] * (maxH + 1)      # ways with none selected at this node
            dp1 = [0] * (maxH + 1)      # ways with at least one selected at this node
            QE: List[int] = []          # cycle nodes
            pnt = 0                     # DP traversal stamp

            def dfs_init(u: int, parent: int) -> None:
                """DFS to find back-edges and collect cycle nodes."""
                vis[u] = True
                for v in to[u]:
                    if v == parent:
                        continue
                    if not vis[v]:
                        dfs_init(v, u)
                    else:
                        if not ins[u]:
                            QE.append(u)
                        if not ins[v]:
                            QE.append(v)
                        ins[u] = True
                        ins[v] = True

            def check() -> bool:
                """Check that no two forced-selected cycle nodes are adjacent."""
                for u in QE:
                    if sat[u] == 1:
                        for v in to[u]:
                            if sat[v] == 1:
                                return False
                return True

            def dfs_dp(u: int) -> int:
                """Tree-like DP over the component with forced cycle-node statuses applied."""
                nonlocal pnt
                dp0[u] = 1
                dp1[u] = (PW2[num[u]] - 1) % MOD
                des[u] = pnt
                for v in to[u]:
                    if des[v] != pnt:
                        dfs_dp(v)
                        dp0[u] = (dp0[u] * ((dp0[v] + dp1[v]) % MOD)) % MOD
                        dp1[u] = (dp1[u] * dp0[v]) % MOD
                if sat[u] == 1:
                    dp0[u] = 0
                if sat[u] == -1:
                    dp1[u] = 0
                return (dp0[u] + dp1[u]) % MOD

            def query(root: int) -> int:
                """Solve one connected component by enumerating forced statuses for cycle nodes."""
                nonlocal pnt
                QE.clear()
                dfs_init(root, root)

                comp_ans = 0
                k = len(QE)
                for mask in range(1 << k):
                    for i in range(k):
                        u = QE[i]
                        sat[u] = 1 if ((mask >> i) & 1) else -1
                    if not check():
                        continue
                    pnt += 1
                    comp_ans = (comp_ans + dfs_dp(root)) % MOD

                # reset sat flags for cycle nodes
                for u in QE:
                    sat[u] = 0
                return comp_ans

            # Combine answers across all length-nodes
            answer = 1
            for length in range(1, maxH + 1):
                if num[length] > 0 and not vis[length]:
                    if not to[length]:
                        # Isolated node: any subset of sticks with this length
                        answer = (answer * PW2[num[length]]) % MOD
                        vis[length] = True
                    else:
                        answer = (answer * query(length)) % MOD

            # Avoid the trivial case where the entire graph contributes exactly 2^N modulo MOD
            if answer != PW2[N]:
                reference_answer = (answer - 1) % MOD
                # Set environment state
                self.H = H
                self.MOD = MOD
                self.reference_answer = reference_answer
                break

        # Build problem statement
        H_desc = " ".join(f"H[{i}]={h}" for i, h in enumerate(self.H))
        self.current_problem = (
            f"You are given an array H of length {self.N}: {H_desc}\n"
            f"Construct an undirected graph with vertices labeled from 0 to {self.N - 1}. There is an edge between "
            f"vertex i and vertex j (i ≠ j) if and only if:\n"
            f"- There exists an integer C such that H[i]^2 + H[j]^2 = C^2\n"
            f"- gcd(H[i], H[j]) = 1 (i.e., H[i] and H[j] are coprime)\n\n"
            f"Your task is to count the number of non-empty independent sets in this graph — subsets of vertices such "
            f"that no two vertices in the subset are connected by an edge.\n\n"
            f"Output Format: Output a single integer — the number of non-empty independent sets modulo {self.MOD}.\n"
            f"Your final answer must be provided in \\boxed{{...}} and be an integer in [0, {self.MOD - 1}]."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.N,
            "MOD": self.MOD,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the provided answer and return the outcome."""
        # Parse boxed answer
        answer_text = self._parse_answer(action)
        if answer_text is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate and compare
        try:
            user_answer = int(answer_text)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if self.MOD is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_initialized"}

        if not (0 <= user_answer < self.MOD):
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
            "MOD": self.MOD,
            "N": self.N,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random valid answer format)."""
        if self.MOD is None:
            # Fallback if reset has not been called
            mod = max(2, self.max_MOD)
        else:
            mod = self.MOD
        random_answer = random.randint(0, mod - 1)
        return f"\\boxed{{{random_answer}}}"