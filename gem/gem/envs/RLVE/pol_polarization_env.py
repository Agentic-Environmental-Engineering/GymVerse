from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class POLPolarizationEnv(Env):
    """
    POL Polarization problem environment (single-turn Q&A).

    Task:
    - You are given an undirected tree with N vertices labeled from 0 to N-1 and N-1 edges.
    - You should conceptually orient each edge to form a directed tree to maximize the number
      of ordered pairs (X, Y), X != Y, such that Y is reachable from X along directed edges.
    - Output the maximum number of such ordered pairs as a single integer in \\boxed{...}.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 4,
        max_n: int = 2000,
        **kwargs
    ):
        super().__init__()
        # Problem size control
        self.N = N
        self.min_n = min_n
        self.max_n = max_n

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.edges: List[tuple[int, int]] = []

    def _get_instructions(self) -> str:
        """Return the task instruction string."""
        return (
            "You are solving a tree orientation optimization problem.\n"
            "Please provide your final answer as a single integer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.N is not None:
            N = self.N
            if N < 4:
                raise ValueError("N should be greater than or equal to 4")
        else:
            if self.min_n < 4:
                raise ValueError("min_n should be greater than or equal to 4")
            if self.min_n > self.max_n:
                raise ValueError("min_n should be less than or equal to max_n")
            N = random.randint(self.min_n, self.max_n)

        # Generate a random tree using a shuffled permutation and random parent selection
        permutations = list(range(N))
        random.shuffle(permutations)
        edges: List[tuple[int, int]] = []
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            edges.append((u, v))
        random.shuffle(edges)

        # Validate edges form a proper tree specification
        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1

        # Build adjacency list
        adjacency: List[List[int]] = [[] for _ in range(N)]
        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)

        # First DFS: compute subtree sizes and "max part" sizes to find centroid
        siz = [0] * N
        msiz = [0] * N
        rt = 0
        best_ms = N

        def dfs(p: int, fa: int) -> None:
            nonlocal rt, best_ms
            siz[p] = 1
            max_sub = 0
            for w in adjacency[p]:
                if w == fa:
                    continue
                dfs(w, p)
                siz[p] += siz[w]
                if siz[w] > max_sub:
                    max_sub = siz[w]
            # consider the "upward" part when p is removed
            up = N - siz[p]
            if up > max_sub:
                max_sub = up
            msiz[p] = max_sub
            # update centroid if this node is better
            if max_sub < best_ms:
                best_ms = max_sub
                rt = p

        dfs(0, -1)

        # Second DFS from centroid: recompute subtree sizes and record parents
        siz = [0] * N
        parent = [-1] * N

        def dfs2(p: int, fa: int) -> None:
            siz[p] = 1
            parent[p] = fa
            for w in adjacency[p]:
                if w == fa:
                    continue
                dfs2(w, p)
                siz[p] += siz[w]

        dfs2(rt, -1)

        # Initial answer: sum of sizes of all subtrees except the centroid itself
        ans = sum(siz[i] for i in range(N) if i != rt)

        # Count how many child-subtrees of each size the centroid has
        cnt = [0] * (N + 1)
        for w in adjacency[rt]:
            if parent[w] == rt:
                cnt[siz[w]] += 1

        # Merge pairs of equal sizes greedily
        for i in range(1, N // 2 + 1):
            while cnt[i] > 2:
                cnt[i] -= 2
                cnt[2 * i] += 1

        # Subset-sum via bitset in an integer
        dp = 1
        for i in range(1, N + 1):
            for _ in range(cnt[i]):
                dp |= dp << i

        # Find the best split i ≤ N//2 that is reachable
        half = N // 2
        for i in range(half, -1, -1):
            if (dp >> i) & 1:
                ans += i * (N - i - 1)
                break

        # Store state
        self.reference_answer = ans
        self.edges = edges

        # Build problem text
        edges_str = "\n".join(f"{u} {v}" for u, v in edges)
        problem_text = (
            f"You are given a tree (i.e., a connected undirected graph with no cycles) with {N} "
            f"vertices labeled from 0 to {N - 1}. The tree contains the following {N - 1} undirected edges. "
            f"Each edge is represented as a tuple (u, v), meaning there is an undirected edge connecting "
            f"vertex u and vertex v:\n{edges_str}\n\n"
            "Your task is to assign a direction to each edge (i.e., for each edge (u, v), you may direct it either "
            "from u to v or from v to u) to form a directed tree. Try your best to maximize the number of ordered "
            "pairs (X, Y) such that X ≠ Y and vertex X can reach vertex Y along directed edges (i.e., Y is reachable "
            "from X in the directed tree). Output a single integer — the maximum number of such ordered pairs (X, Y).\n\n"
            "Output Format: Provide your answer in \\boxed{...}."
        )
        self.current_problem = problem_text

        obs = self._get_instructions() + problem_text
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer and return the result."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate numeric answer
        try:
            user_answer = int(parsed)
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        is_correct = (self.reference_answer is not None and user_answer == self.reference_answer)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content within \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action formatted as \\boxed{...}."""
        # A rough upper bound for answers could be N*(N-1) as maximum possible ordered pairs in a DAG,
        # but we keep it moderate for sampling purposes.
        random_answer = random.randint(0, 10**9)
        return f"\\boxed{{{random_answer}}}"