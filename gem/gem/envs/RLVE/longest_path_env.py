import random
import re
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class LongestPathEnv(Env):
    """Directed graph longest path environment - single-turn Q&A.

    The agent is given a directed weighted graph with N vertices and a list of edges.
    The goal is to output a simple path (no repeated vertices) that maximizes the sum
    of edge weights along the path. The answer must be provided inside \\boxed{...}.
    """

    def __init__(
        self,
        N: int = 8,
        edge_density: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__()
        # Parameter validation (preserved logic)
        assert isinstance(N, int), "N must be an integer"
        assert N >= 3, "N should be greater than or equal to 3"
        assert isinstance(edge_density, float) or isinstance(edge_density, int), "edge_density must be a float"
        assert 0.0 < float(edge_density) <= 1.0, "edge_density should be between 0.0 and 1.0"

        self.N: int = int(N)
        self.edge_density: float = float(edge_density)

        # Internal state
        self.current_problem: Optional[str] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.adjacent: List[List[Tuple[int, int]]] = []
        self.gold_weight: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a longest path problem on a directed weighted graph.\n"
            "Your answer must be a path as space-separated vertex indices wrapped in \\boxed{...}.\n"
            "Example: \\boxed{0 3 5}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        # Generate edges
        all_triples = [(s, t, random.randint(1, N)) for s in range(N) for t in range(N) if s != t]
        k = int(self.edge_density * N * (N - 1))
        self.edges = random.sample(all_triples, k)
        random.shuffle(self.edges)
        # Validation (preserved)
        assert len(self.edges), "No edges were generated; consider increasing edge_density"

        assert len(self.edges) == len(set((s, t) for s, t, w in self.edges)), "edges should be unique"
        for s, t, w in self.edges:
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"

        # Build adjacency list
        self.adjacent = [[] for _ in range(N)]
        for s, t, w in self.edges:
            self.adjacent[s].append((t, w))

        # Compute gold (maximum total weight of a simple path)
        self.gold_weight = 0
        dpF: Dict[Tuple[int, int], int] = {}

        def dp(s: int, visited: int) -> int:
            if visited == (1 << N) - 1:
                return 0
            key = (s, visited)
            if key in dpF:
                return dpF[key]
            ans = 0
            for t, w in self.adjacent[s]:
                if (visited & (1 << t)) == 0:
                    ans = max(ans, dp(t, visited | (1 << t)) + w)
            dpF[key] = ans
            return ans

        for s in range(N):
            self.gold_weight = max(self.gold_weight, dp(s, 1 << s))

        # Problem statement
        edges_str = "\n".join(f"({s}, {t}, {w})" for s, t, w in self.edges)
        self.current_problem = (
            f"You are given a directed graph with {N} vertices, labeled from 0 to {N - 1}.\n\n"
            "The graph contains the following directed edges. Each edge is represented as a tuple (s, t, w), "
            "meaning there is a directed edge from vertex s to vertex t with weight w:\n"
            f"{edges_str}\n\n"
            "Your task is to find a path p1, p2, ..., pk such that:\n"
            "- No vertex appears more than once in the path.\n"
            "- Maximize the total weight of the path (i.e., the sum of all edge weights used).\n\n"
            "Output Format: Your final answer should be the path in order as space-separated integers inside "
            "\\boxed{...}.\n"
            f"Example: \\boxed{{0 1 {N - 1}}}"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Verify the submitted path."""
        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.gold_weight is None or self.current_problem is None:
            return TERMINAL_STATE, 0.0, True, False, {"error": "environment_not_initialized"}

        # Attempt to parse path as list of integers
        try:
            tokens = boxed_content.strip().split()
            if not tokens:
                return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}
            path = [int(tok) for tok in tokens]
        except Exception:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        N = self.N
        # Validate path vertices are in range
        if not all(0 <= v < N for v in path):
            info = {
                "correct": False,
                "path_valid": False,
                "reason": "vertex_out_of_range",
                "reference_weight": self.gold_weight,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Validate uniqueness (simple path)
        if len(path) != len(set(path)):
            info = {
                "correct": False,
                "path_valid": False,
                "reason": "repeated_vertex",
                "reference_weight": self.gold_weight,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Build map for quick edge lookup
        edge2weight: Dict[Tuple[int, int], int] = {(s, t): w for s, t, w in self.edges}

        # Validate edges existence and compute total weight
        answer_weight = 0
        for s, t in zip(path, path[1:]):
            w = edge2weight.get((s, t))
            if w is None:
                info = {
                    "correct": False,
                    "path_valid": False,
                    "reason": "missing_edge",
                    "reference_weight": self.gold_weight,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            answer_weight += w

        gold = self.gold_weight
        # According to original logic, edges > 0 implies gold > 0
        assert gold is not None
        assert gold >= 0

        is_correct = (answer_weight == gold)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "path_valid": True,
            "user_weight": answer_weight,
            "reference_weight": gold,
            "N": self.N,
            "num_edges": len(self.edges),
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the answer from \\boxed{...}."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random path action wrapped in \\boxed{...}."""
        if not self.adjacent:
            # If the environment has not been reset, just return a single vertex
            s = random.randint(0, max(0, self.N - 1))
            return f"\\boxed{{{s}}}"

        N = self.N
        s = random.randint(0, N - 1)
        path = [s]
        visited = {s}
        # Randomly walk without revisiting vertices
        current = s
        for _ in range(N - 1):
            candidates = [(t, w) for (t, w) in self.adjacent[current] if t not in visited]
            if not candidates:
                break
            t, _ = random.choice(candidates)
            path.append(t)
            visited.add(t)
            current = t
        path_str = " ".join(map(str, path))
        return f"\\boxed{{{path_str}}}"