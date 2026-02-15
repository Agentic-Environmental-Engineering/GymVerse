from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TriumphalArchEnv(Env):
    """
    Triumphal Arch game on a tree - single-turn QA environment.

    The problem:
    - Given a tree with N vertices labeled 0..N-1, vertex 0 initially black (permanently).
    - Each turn: Alice marks K vertices black, then Bob moves to an adjacent vertex.
    - If Bob reaches any non-black vertex at any time, he wins; if all vertices become black, Alice wins.
    The task is to compute the minimum K such that Alice is guaranteed to win, assuming optimal play.

    Answer format:
    - The agent must output the answer in \\boxed{...}.
    """

    def __init__(
        self,
        min_n: int = 3,
        max_n: int = 100,
        fixed_n: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the environment.

        Args:
            min_n: Minimum number of vertices N to sample when fixed_n is None (must be >= 3).
            max_n: Maximum number of vertices N to sample when fixed_n is None.
            fixed_n: If provided, N will be fixed to this value (must be >= 3).
            **kwargs: Additional arguments reserved for compatibility.
        """
        super().__init__()
        assert min_n >= 3, "min_n should be greater than or equal to 3"
        assert max_n >= min_n, "max_n should be >= min_n"
        if fixed_n is not None:
            assert fixed_n >= 3, "fixed_n should be greater than or equal to 3"

        self.min_n: int = min_n
        self.max_n: int = max_n
        self.fixed_n: Optional[int] = fixed_n

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[int] = None
        self.current_n: Optional[int] = None
        self.current_edges: Optional[List[Tuple[int, int]]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a single-turn tree game problem.\n"
            "Task: Compute the minimum K such that Alice is guaranteed to win.\n"
            "Output Format: Provide a single integer in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 3, "N should be greater than or equal to 3"
        self.current_n = N

        # Generate a random tree using a randomized attachment process
        permutations = list(range(N))
        random.shuffle(permutations)
        edges: List[Tuple[int, int]] = []
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            parent = random.choice(permutations[:index])
            u, v = (vertex, parent)
            u, v = (u, v) if u < v else (v, u)
            edges.append((u, v))
        random.shuffle(edges)

        # Validate edges
        for u, v in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1

        self.current_edges = edges

        # Build adjacency list
        S: List[List[int]] = [[] for _ in range(N)]
        for u, v in edges:
            S[u].append(v)
            S[v].append(u)

        # Compute number of children in rooted tree at 0
        son: List[int] = [0] * N

        def dfs1(u: int, p: int) -> None:
            for w in S[u]:
                if w == p:
                    continue
                son[u] += 1
                dfs1(w, u)

        dfs1(0, -1)

        # Binary search between L and R
        L = son[0]
        R = max(son) if son else 0

        f: List[int] = [0] * N

        def dfs2(u: int, p: int, k: int) -> None:
            total = son[u] - k
            for w in S[u]:
                if w == p:
                    continue
                dfs2(w, u, k)
                if f[w] > 0:
                    total += f[w]
            f[u] = total

        ans = R
        while L <= R:
            mid = (L + R) // 2
            # Reset DP array before each evaluation
            for i in range(N):
                f[i] = 0
            dfs2(0, -1, mid)
            if f[0] <= 0:
                ans = mid
                R = mid - 1
            else:
                L = mid + 1

        self.reference_answer = ans

        # Build problem statement
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        problem = (
            f"You are given a tree with {N} vertices labeled from 0 to {N - 1}.\n"
            f"The edges of the tree are given as follows:\n{edges_str}\n\n"
            "Game rules:\n"
            "- Initially, Bob is at vertex 0. Vertex 0 is already (permanently) black; all other vertices are white.\n"
            "- In each turn:\n"
            "  - Alice first chooses any K vertices and marks them as (permanently) black.\n"
            "  - Then, Bob may move to any vertex adjacent to his current position.\n"
            "- If Bob ever reaches a non-black vertex on any turn, he wins. If all vertices eventually become black, Alice wins.\n\n"
            "Assuming both players play optimally, what is the minimum value of K such that Alice is guaranteed to win?\n\n"
            "Output Format: Provide your final answer as a single integer in \\boxed{...}."
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Parse and verify the user's answer; single-turn termination."""
        # Parse answer from \boxed{...}
        parsed = self._parse_answer(action)
        if parsed is None:
            return (
                TERMINAL_STATE,
                -0.1,
                True,
                False,
                {"error": "format_error", "message": "Answer must be in \\boxed{...} format."},
            )

        try:
            user_answer = int(parsed.strip())
        except ValueError:
            return (
                TERMINAL_STATE,
                0.0,
                True,
                False,
                {"error": "invalid_answer", "message": "Parsed content is not an integer."},
            )

        assert self.reference_answer is not None, "Environment not initialized properly; call reset() first."
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.current_n,
            "edges": self.current_edges,
        }

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
        """Sample a random action formatted in \\boxed{...}."""
        # A reasonable guess range: 0..N-1 (if N available), else 0..10
        if self.current_n is not None:
            guess = random.randint(0, max(0, self.current_n - 1))
        else:
            guess = random.randint(0, 10)
        return f"\\boxed{{{guess}}}"