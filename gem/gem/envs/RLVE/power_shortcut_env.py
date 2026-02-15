from typing import Any, List, Optional, SupportsFloat, Tuple
import random
import re
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class PowerShortcutEnv(Env):
    """PowerShortcut graph problem environment - single-turn Q&A.

    The agent is given a directed graph with N vertices and must output a sequence of vertices
    starting at 0 and ending at N-1 such that each consecutive pair (u, v) is connected by a path
    whose length is exactly 2^k for some integer k with 0 <= k <= K. The goal is to minimize
    the length of the sequence (i.e., the number of vertices in the sequence).
    """

    def __init__(
        self,
        N: int = 8,
        K: int = 3,
        edge_density: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.N: int = N
        self.K: int = K
        self.edge_density: float = edge_density

        # Internal state for the current episode
        self.edges: List[Tuple[int, int]] = []
        self.achievable: List[List[bool]] = []
        self.reference_answer: str = ""
        self.gold_answer: int = 0  # minimal sequence length (number of vertices)
        self.current_problem: str = ""

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a directed graph task called PowerShortcut.\n"
            "Your goal is to output a sequence of vertices p[1], p[2], ..., p[m] such that:\n"
            "- p[1] = 0 and p[m] = N-1.\n"
            "- For each consecutive pair (p[i], p[i+1]), there exists a path from p[i] to p[i+1] whose length is exactly 2^k for some integer k with 0 <= k <= K.\n"
            "Your objective is to minimize m (the number of vertices in the sequence).\n\n"
            "Output Format:\n"
            "Return your final sequence inside \\boxed{...} with vertices separated by spaces.\n"
            "Example: \\boxed{0 5 3 7}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new PowerShortcut problem."""
        super().reset(seed)

        # Validate parameters
        assert isinstance(self.N, int), "N must be an integer"
        assert self.N >= 3, "N should be greater than or equal to 3"
        assert isinstance(self.K, int), "K must be an integer"
        assert self.K >= 0, "K should be greater than or equal to 0"
        assert isinstance(self.edge_density, float) or isinstance(self.edge_density, int), "edge_density must be a float"
        assert 0.0 <= float(self.edge_density) <= 1.0, "edge_density should be between 0.0 and 1.0"

        N = self.N
        K = self.K
        edge_density = float(self.edge_density)

        # Construct a guaranteed path from 0 to N-1 (random permutation of intermediate vertices)
        constructed_path = list(range(1, N - 1))
        random.shuffle(constructed_path)
        constructed_path = [0] + constructed_path + [N - 1]

        edges: List[Tuple[int, int]] = []
        for s, t in zip(constructed_path, constructed_path[1:]):
            edges.append((s, t))

        # Fill additional edges up to edge_density * N * (N - 1)
        num_edges = int(edge_density * N * (N - 1))
        if len(edges) < num_edges:
            all_possible = {(s, t) for s in range(N) for t in range(N) if s != t}
            remaining_edges = list(all_possible - set(edges))
            if remaining_edges:
                edges += random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
        random.shuffle(edges)

        # Sanity checks on edges
        assert len(edges) == len(set(edges)), "Edges should be unique"
        for s, t in edges:
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"

        # Compute achievability for path lengths being powers of two up to 2^K
        achievable = [[[False] * N for _ in range(N)] for _ in range(K + 1)]
        # path[s][t] will store list of intermediate vertices to realize minimal number of steps where each step is achievable
        path: List[List[Optional[List[int]]]] = [[None] * N for _ in range(N)]
        for s in range(N):
            path[s][s] = []
        for s, t in edges:
            achievable[0][s][t] = True
            path[s][t] = []

        # Doubling reachability: achievable[k] = paths of length exactly 2^k
        for k in range(1, K + 1):
            for s in range(N):
                for t in range(N):
                    # exists m such that s->m in 2^(k-1) and m->t in 2^(k-1)
                    flag = False
                    for m in range(N):
                        if achievable[k - 1][s][m] and achievable[k - 1][m][t]:
                            flag = True
                            break
                    achievable[k][s][t] = flag
                    if flag:
                        # Mark as directly achievable in one "PowerShortcut step"
                        path[s][t] = []

        # Build 2D achievable matrix (there exists k in [0..K] with achievable[k][s][t])
        achievable_any = [[any(achievable[k][s][t] for k in range(K + 1)) for t in range(N)] for s in range(N)]
        self.achievable = achievable_any

        # Floyd-Warshall-like relaxation over "PowerShortcut steps" to minimize number of steps
        for m in range(N):
            for s in range(N):
                for t in range(N):
                    if path[s][m] is not None and path[m][t] is not None:
                        candidate = path[s][m] + [m] + path[m][t]
                        if path[s][t] is None or (len(path[s][t]) > len(candidate)):
                            path[s][t] = candidate

        # Build reference answer (minimal sequence)
        minimal_intermediates = path[0][N - 1]
        assert minimal_intermediates is not None, "There must be a path from 0 to N-1"
        reference_list = [0] + minimal_intermediates + [N - 1]
        self.reference_answer = " ".join(map(str, reference_list))
        self.gold_answer = len(reference_list)

        # Build problem prompt
        edges_text = "\n".join(f"({s}, {t})" for s, t in edges)
        self.edges = edges
        self.current_problem = (
            f"You are given a directed graph with {N} vertices, labeled from 0 to {N - 1}.\n\n"
            f"The graph contains the following directed edges. Each edge is represented as a tuple (s, t), "
            f"meaning there is a directed edge from vertex s to vertex t:\n{edges_text}\n\n"
            f"Your task is to find a sequence of vertices p[1], p[2], ..., p[m] such that:\n"
            f"- p[1] = 0 (the sequence starts at vertex 0) and p[m] = {N - 1} (the sequence ends at vertex {N - 1})\n"
            f"- For each consecutive pair (p[i], p[i + 1]), there exists a path from p[i] to p[i + 1] whose length is exactly 2^k "
            f"for some integer k where 0 <= k <= {K}.\n\n"
            f"Your goal is to minimize the length m of the sequence — that is, the number of steps in the sequence.\n\n"
            f"Output Format:\n"
            f"Your final answer should be a single line containing the sequence p[1] p[2] ... p[m], separated by spaces, "
            f"and wrapped inside \\boxed{{...}}.\n"
            f"Example: \\boxed{{0 1 {N - 1}}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(
        self,
        action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted sequence and return the reward."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            # Format error: cannot find \boxed{...}
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse sequence of integers from boxed content
        try:
            tokens = boxed_content.strip().split()
            if not tokens:
                return TERMINAL_STATE, 0.0, True, False, {"error": "empty_answer"}
            path = list(map(int, tokens))
        except ValueError:
            # Not a valid sequence of integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.N

        # Validate vertices are in range
        for v in path:
            if not (0 <= v < N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "vertex_out_of_range", "user_answer": path}

        # Validate start and end vertices
        if not (path[0] == 0 and path[-1] == N - 1):
            return TERMINAL_STATE, 0.0, True, False, {"error": "wrong_endpoints", "user_answer": path}

        # Validate each step is achievable by a path of length 2^k (0 <= k <= K)
        for s, t in zip(path, path[1:]):
            if not self.achievable[s][t]:
                return TERMINAL_STATE, 0.0, True, False, {"error": "unachievable_step", "user_answer": path, "bad_step": (s, t)}

        # Check minimality: correct if sequence length equals gold_answer
        is_minimal = (len(path) == self.gold_answer)
        reward = 1.0 if is_minimal else 0.0

        info = {
            "correct": is_minimal,
            "reference_answer": self.reference_answer,
            "gold_length": self.gold_answer,
            "user_length": len(path),
            "user_answer": path
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Returns the last match if multiple are present."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random sequence from 0 to N-1 along the constructed path)."""
        # This is a naive sampler: either use the reference or a random subsequence of the reference
        if self.reference_answer:
            seq = self.reference_answer.split()
            # With some probability, drop a random internal node (likely to be invalid/minimality fails)
            if len(seq) > 2 and random.random() < 0.5:
                idx = random.randint(1, len(seq) - 2)
                seq = seq[:idx] + seq[idx + 1:]
            return f"\\boxed{{{' '.join(seq)}}}"
        else:
            # Fallback: trivial output
            return "\\boxed{0 " + str(max(1, self.N - 1)) + "}"