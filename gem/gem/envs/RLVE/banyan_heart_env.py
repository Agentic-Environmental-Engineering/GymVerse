import random
from typing import Any, Optional, SupportsFloat, Tuple, List
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class BanyanHeartEnv(Env):
    """Banyan Heart Vertex Problem Environment - Single Turn Q&A

    This environment generates a random tree with N vertices and asks the agent
    to determine which vertices can be the heart vertex after a specific process.
    The agent should return a string of length N with 'T' or 'F' characters,
    boxed in \\boxed{...} format.
    """

    def __init__(
        self,
        min_n: int = 4,
        max_n: int = 50,
        fixed_n: Optional[int] = None,
        format_error_reward: float = -0.1,
        **kwargs
    ):
        """
        Initialize the BanyanHeartEnv instance.

        Parameters:
        - min_n: minimum number of vertices (must be >= 4)
        - max_n: maximum number of vertices
        - fixed_n: if provided, use this exact N (must be >= 4)
        - format_error_reward: reward for format errors (default -0.1)
        """
        super().__init__()
        assert min_n >= 4, "min_n should be greater than or equal to 4"
        if fixed_n is not None:
            assert fixed_n >= 4, "fixed_n should be greater than or equal to 4"
        assert max_n >= min_n, "max_n should be greater than or equal to min_n"

        self.min_n = min_n
        self.max_n = max_n
        self.fixed_n = fixed_n
        self.format_error_reward = format_error_reward

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.N: Optional[int] = None
        self.edges: Optional[List[Tuple[int, int]]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a tree process problem called the 'Banyan Heart'.\n"
            "Task:\n"
            "- A tree with N vertices labeled 1..N is built incrementally. Start with vertex 1 as the heart.\n"
            "- When a new vertex i (2 ≤ i ≤ N) is added and connected to an existing vertex, "
            "the heart moves one step toward i (to the neighbor closer to i).\n"
            "- After all N vertices are added, determine which vertices could be the final heart.\n\n"
            "Output Format:\n"
            "- Return a single line with N characters using 'T' or 'F' without separators.\n"
            "- The i-th character is 'T' if vertex i can be the final heart, and 'F' otherwise.\n"
            "- Your final answer must be enclosed in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Choose N
        if self.fixed_n is not None:
            N = self.fixed_n
        else:
            N = random.randint(self.min_n, self.max_n)
        assert N >= 4, "N should be greater than or equal to 4"
        self.N = N

        # Generate edges using the original randomized process
        edges: List[Tuple[int, int]] = []
        permutations = list(range(1, N + 1))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
        random.shuffle(edges)

        # Validate edges
        for u, v in edges:
            assert 1 <= u < v <= N
        assert len(edges) == len(set(edges)) == N - 1

        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree), "Generated graph must be a tree"

        self.edges = edges

        # Build adjacency list dynamically
        adjacency = [[] for _ in range(N + 1)]
        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)

        # Arrays (1..N); index 0 acts as a dummy node
        dep = [0] * (N + 1)
        siz = [0] * (N + 1)
        hson = [0] * (N + 1)
        hson2 = [0] * (N + 1)
        f = [0] * (N + 1)
        ans = [False] * (N + 1)

        # cmp function: return the index with larger siz
        def cmp(x: int, y: int) -> int:
            return x if siz[x] > siz[y] else y

        # Iterative dfs1: compute dep, siz, hson, hson2, f
        stack: List[Tuple[int, int, int]] = [(1, 0, 0)]  # (u, parent, state) state 0=enter, 1=exit
        dep[0] = 0
        while stack:
            u, fa, state = stack.pop()
            if state == 0:
                dep[u] = dep[fa] + 1
                stack.append((u, fa, 1))
                for v in adjacency[u]:
                    if v == fa:
                        continue
                    stack.append((v, u, 0))
            else:
                # post-order processing
                s = 1
                h1 = 0
                h2 = 0
                for v in adjacency[u]:
                    if v == fa:
                        continue
                    s += siz[v]
                    if siz[v] > siz[h1]:
                        h2 = h1
                        h1 = v
                    elif siz[v] > siz[h2]:
                        h2 = v
                siz[u] = s
                hson[u] = h1
                hson2[u] = h2

                if f[h1] <= (siz[u] - 1 - siz[h1]):
                    fv = (siz[u] - 1) % 2
                else:
                    fv = f[h1] - (siz[u] - 1 - siz[h1])
                f[u] = fv + 1

        # Iterative dfs2: compute ans
        stack2: List[Tuple[int, int, int]] = [(1, 0, 0)]  # (u, parent, h)
        while stack2:
            u, fa, h = stack2.pop()
            tmp = cmp(hson[u], h)
            if f[tmp] <= N - dep[u] - siz[tmp]:
                ans[u] = ((N & 1) == (dep[u] & 1))
            for v in adjacency[u]:
                if v == fa:
                    continue
                if v == hson[u]:
                    h_child = cmp(hson2[u], h)
                else:
                    h_child = cmp(hson[u], h)
                stack2.append((v, u, h_child))

        reference_answer = "".join("T" if ans[i] else "F" for i in range(1, N + 1))
        assert "T" in reference_answer, "At least one vertex should be able to be the heart vertex"
        self.reference_answer = reference_answer

        # Build prompt
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        problem_prompt = (
            f"We use the following process to generate a tree with {N} vertices labeled from 1 to {N}:\n"
            f"- Initially, the tree contains only vertex 1, and its heart vertex is also 1.\n"
            f"- At each step, we add a new vertex i (2 ≤ i ≤ {N}) and connect it to an existing vertex with an undirected edge. "
            f"Then, the heart vertex moves one step toward i (i.e., it moves to the neighbor that is closer to i).\n"
            f"- This process continues until all {N} vertices have been added.\n\n"
            f"The final tree has the following edges:\n"
            f"{edges_str}\n\n"
            f"Can you determine which vertices could be the heart vertex after the process is completed?\n"
            f"Output a single line with {N} characters (either 'T' or 'F') without separators, where the i-th character is 'T' "
            f"if vertex i can be the heart vertex, and 'F' otherwise.\n\n"
            f"Output Format: Your final answer should be enclosed in \\boxed{{...}}."
        )

        self.current_problem = problem_prompt

        obs = self._get_instructions() + problem_prompt
        info: dict[str, Any] = {
            "N": N,
            "edges": edges,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step by verifying the user's answer."""
        # Extract boxed content
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        # Validate format: length N and only 'T' or 'F'
        if self.N is None or self.reference_answer is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_initialized"}

        expected_len = self.N
        if not (len(parsed) == expected_len and all(c in "TF" for c in parsed)):
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        user_answer = parsed
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "edges": self.edges,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid-looking action."""
        if self.N is None:
            # Fallback: produce a small random string
            n = random.randint(self.min_n, self.max_n)
        else:
            n = self.N
        random_answer = "".join(random.choice("TF") for _ in range(n))
        return f"\\boxed{{{random_answer}}}"