from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE
import re


class OddVisitationEnv(Env):
    """Odd visitation on a connected undirected graph - single-turn Q&A environment."""

    def __init__(
        self,
        N: int = 5,
        edge_ratio: float = 1.5,
        **kwargs
    ):
        """
        Initialize the OddVisitationEnv.

        Parameters:
            N (int): Number of vertices in the graph. Must be >= 3.
            edge_ratio (float): Ratio to determine the number of edges. Final number of edges will be max(N-1, int(N * edge_ratio)).
        """
        super().__init__()
        assert N >= 3, "N should be greater than or equal to 3"
        self.N: int = N
        self.edge_ratio: float = edge_ratio

        self.edges: List[Tuple[int, int]] = []
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task description and answer format."""
        return (
            "You are given a connected undirected graph problem.\n"
            "Your task is to find a trajectory that visits each vertex an odd number of times.\n"
            "Please provide your answer as a single line of integers separated by spaces, wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new graph problem."""
        super().reset(seed)

        N = self.N
        edge_ratio = self.edge_ratio

        # Generate a random spanning tree using a Prufer-like sequence approach
        edges: List[Tuple[int, int]] = []

        prufer = [random.randint(0, N - 1) for _ in range(N - 2)]
        degree = [1] * N
        for v in prufer:
            degree[v] += 1
        leaves = [i for i in range(N) if degree[i] == 1]
        for v in prufer:
            u = leaves.pop(0)
            if u > v:
                edges.append((v, u))
            else:
                edges.append((u, v))
            degree[u] -= 1
            degree[v] -= 1
            if degree[u] == 1:
                leaves.append(u)
            if degree[v] == 1 and v not in leaves:
                leaves.append(v)
        u = leaves.pop(0)
        v = leaves.pop(0)
        if u > v:
            u, v = v, u
        edges.append((u, v))

        # Add additional edges based on edge_ratio
        num_edges = int(N * edge_ratio)
        if len(edges) < num_edges:
            remaining_edges = list(set((a, b) for a in range(N) for b in range(a + 1, N)) - set(edges))
            additional_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            edges += additional_edges
        random.shuffle(edges)

        # Validate edges
        for a, b in edges:
            assert 0 <= a < b < N
        assert len(edges) == len(set(edges)), "Edges should be unique"

        self.edges = edges

        # Build adjacency list
        adjacency: List[List[int]] = [[] for _ in range(N)]
        for a, b in self.edges:
            adjacency[a].append(b)
            adjacency[b].append(a)

        # Build DFS tree sons from root 0
        sons: List[List[int]] = [[] for _ in range(N)]
        visited = [False] * N

        def dfs1(x: int, parent: int) -> None:
            visited[x] = True
            for y in adjacency[x]:
                if y != parent and not visited[y]:
                    sons[x].append(y)
                    dfs1(y, x)

        dfs1(0, -1)

        # Construct a sequence visiting each vertex an odd number of times
        answer_seq: List[int] = []

        def dfs2(x: int) -> bool:
            x_visit = 1
            answer_seq.append(x)
            for y in sons[x]:
                finished = dfs2(y)
                x_visit += 1
                answer_seq.append(x)
                if not finished:
                    answer_seq.append(y)
                    x_visit += 1
                    answer_seq.append(x)
            return x_visit % 2 == 1

        dfs2(0)
        if sum(1 for vv in answer_seq if vv == 0) % 2 == 0:
            assert answer_seq[-1] == 0, "The last vertex should be 0 to ensure odd visitation."
            answer_seq = answer_seq[:-1]

        self.reference_answer = " ".join(map(str, answer_seq))

        # Build the problem string
        edges_str = "\n".join(f"({a}, {b})" for a, b in self.edges)
        self.current_problem = (
            f"You are given a connected undirected graph with {N} vertices labeled from 0 to {N - 1}. "
            f"The graph contains the following undirected edges:\n{edges_str}\n\n"
            "Your task is to find a trajectory that visits each vertex odd numbers of times, and the starting and ending vertices can be arbitrary.\n"
            "Formally, you should find a sequence of length K (which is decided by you), v_0, v_1, ..., v_{K-1}, such that:\n"
            "(1) v_i and v_{i+1} are connected by an edge for all 0 <= i < K - 1;\n"
            "(2) for each vertex with label v (0 <= v < N), the number of times it appears in the sequence is odd.\n\n"
            "Output Format: Your output should be one single line of K integers (you don't need to output K), separated by spaces, "
            "representing the sequence v_0, v_1, ..., v_{K-1}, and wrapped in \\boxed{...}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

        # Note: The reference answer is stored for verification in step().

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the provided answer."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            # Format error: missing or invalid \\boxed{...}
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse sequence of integers from boxed content
        parts = boxed_content.strip().split()
        if not parts:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        try:
            seq = list(map(int, parts))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate vertex indices
        for v in seq:
            if not (0 <= v < self.N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        # Validate adjacency for consecutive vertices
        edge_set = set(self.edges)  # edges stored as (u, v) with u < v
        for i in range(len(seq) - 1):
            u, v = seq[i], seq[i + 1]
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution"}

        # Count visits for odd visitation requirement
        cnt = [0] * self.N
        for v in seq:
            cnt[v] += 1

        is_correct = not any(c % 2 == 0 for c in cnt)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, seq)),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the provided text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action. Here we return the reference answer if available, otherwise a random valid vertex."""
        if self.reference_answer:
            return f"\\boxed{{{self.reference_answer}}}"
        else:
            # Fallback: a single vertex 0
            return "\\boxed{0}"