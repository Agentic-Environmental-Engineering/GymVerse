import heapq
import random
from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class HamiltonianPathEnv(Env):
    """Directed graph path visiting all vertices at least once with minimal total weight - single turn Q&A environment."""

    prompt_template = (
        "You are given a directed graph with {N} vertices, labeled from 0 to {N_minus_1}.\n\n"
        "The graph contains the following directed edges. Each edge is represented as a tuple (s, t, w), meaning there is a directed edge from vertex s to vertex t with weight w:\n"
        "{edges}\n\n"
        "Your task is to find a path p1, p2, ..., pk such that:\n"
        "- The path visits every vertex at least once (revisiting vertices is allowed).\n"
        "- Your goal is to minimize the total weight of the path. The total weight is the sum of the weights of all edges used in the path.\n\n"
        "Output Format:\n"
        "Your final answer should be the path as space-separated integers inside \\boxed{{...}}.\n"
        "Example: \\boxed{{0 1 0 2}} (do NOT include any quotes); this means the path starts at vertex 0, goes to 1, returns to 0, and then to 2 — thus visiting all three vertices at least once (assuming 3 vertices in total).\n"
    )

    def __init__(
        self,
        N: Optional[int] = None,
        edge_density: Optional[float] = None,
        min_N: int = 2,
        max_N: int = 12,
        **kwargs
    ):
        """
        Initialize the HamiltonianPathEnv instance.

        Parameters:
        - N: number of vertices. If None, will be randomly selected in [min_N, max_N].
        - edge_density: density of directed edges (excluding self-loops), in [0.0, 1.0]. If None, will be randomly selected in [0.4, 0.9].
        - min_N: minimum number of vertices for random generation when N is None.
        - max_N: maximum number of vertices for random generation when N is None.
        """
        super().__init__()
        self.fixed_N = N
        self.fixed_edge_density = edge_density
        self.min_N = min_N
        self.max_N = max_N

        # State placeholders
        self.N: int = 0
        self.edge_density: float = 0.0
        self.edges: List[Tuple[int, int, int]] = []
        self.reference_answer: Optional[str] = None
        self.reference_answer_weight: Optional[int] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return high-level task instructions."""
        return (
            "You are solving a directed graph path optimization problem.\n"
            "Find a path that visits every vertex at least once and minimizes total edge weight.\n"
            "Please provide your final path in \\boxed{...} format, where the content is a space-separated list of vertex IDs.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        # Generate parameters
        self.N = self.fixed_N if self.fixed_N is not None else random.randint(self.min_N, self.max_N)
        assert self.N >= 2, "N should be greater than or equal to 2"

        if self.fixed_edge_density is None:
            # Choose a reasonably dense random edge density if not provided
            self.edge_density = random.uniform(0.4, 0.9)
        else:
            self.edge_density = self.fixed_edge_density
        assert 0.0 <= self.edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        # Build a random path that visits all vertices to guarantee feasibility
        constructed_path = list(range(self.N))
        random.shuffle(constructed_path)

        # Initialize edges with the constructed path edges
        edges: List[Tuple[int, int, int]] = []
        ref_weight = 0
        for s, t in zip(constructed_path, constructed_path[1:]):
            w = random.randint(1, self.N)
            edges.append((s, t, w))
            ref_weight += w

        # Add additional random edges according to edge density
        num_edges = int(self.edge_density * self.N * (self.N - 1))  # directed edges without self-loops
        if len(edges) < num_edges:
            existing_pairs = set((s, t) for s, t, _ in edges)
            all_pairs = set((s, t) for s in range(self.N) for t in range(self.N) if s != t)
            remaining_edges = list(all_pairs - existing_pairs)
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for s, t in remaining_edges:
                edges.append((s, t, random.randint(1, max(1, self.N // 2))))
        random.shuffle(edges)

        # Validation
        assert len(edges) == len(set((s, t) for s, t, w in edges)), "edges should be unique"
        for s, t, w in edges:
            assert 0 <= s < self.N, "s should be in range"
            assert 0 <= t < self.N, "t should be in range"
            assert s != t, "s should not be equal to t"

        self.edges = edges
        self.reference_answer = " ".join(map(str, constructed_path))
        self.reference_answer_weight = ref_weight

        # Build adjacency list
        adjacent: List[List[Tuple[int, int]]] = [[] for _ in range(self.N)]
        for s, t, w in edges:
            adjacent[s].append((t, w))

        # Shortest path to visit all vertices at least once (bitmask DP with Dijkstra)
        priority_queue: List[Tuple[int, Tuple[int, int]]] = [(0, (1 << start, start)) for start in range(self.N)]
        visited_states: set[Tuple[int, int]] = set()
        dist: Dict[Tuple[int, int], int] = {(1 << start, start): 0 for start in range(self.N)}
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {(1 << start, start): (0, -1) for start in range(self.N)}

        full_mask = (1 << self.N) - 1

        while priority_queue:
            current_dist, (visited, s) = heapq.heappop(priority_queue)

            if visited == full_mask:
                # Update reference to the true minimal path
                assert current_dist < self.reference_answer_weight, "current_dist should be less than or equal to reference_answer_weight"
                self.reference_answer_weight = current_dist

                # Reconstruct path
                path_nodes: List[int] = []
                v, u = visited, s
                while True:
                    assert (v == 0) == (u == -1), "visited should be 0 if and only if s is -1"
                    if v == 0:
                        break
                    path_nodes.append(u)
                    v, u = prev[(v, u)]
                path_nodes.reverse()
                self.reference_answer = " ".join(map(str, path_nodes))
                break

            if (visited, s) in visited_states:
                continue
            visited_states.add((visited, s))

            for t, w in adjacent[s]:
                new_visited = visited | (1 << t)
                new_dist = current_dist + w
                if dist.get((new_visited, t), self.reference_answer_weight) > new_dist:
                    dist[(new_visited, t)] = new_dist
                    prev[(new_visited, t)] = (visited, s)
                    heapq.heappush(priority_queue, (new_dist, (new_visited, t)))

        # Build problem statement
        problem = self.prompt_template.format(
            N=self.N,
            N_minus_1=self.N - 1,
            edges="\n".join(f"({s}, {t}, {w})" for s, t, w in self.edges),
        )
        self.current_problem = problem

        obs = self._get_instructions() + self.current_problem
        return obs, {"N": self.N, "edge_density": self.edge_density}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Verify the submitted path and provide a reward."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse space-separated integers
        try:
            path = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate the path according to rules
        # 1) Each vertex id must be in range [0, N-1]
        for v in path:
            if not (0 <= v < self.N):
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "vertex_out_of_range"}

        # 2) Must cover all vertices at least once
        if len(set(path)) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "not_all_vertices_visited"}

        # 3) All consecutive pairs must be valid directed edges
        edge2weight = {(s, t): w for s, t, w in self.edges}
        answer_weight = 0
        for s, t in zip(path, path[1:]):
            if (s, t) not in edge2weight:
                return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "reason": "missing_edge"}
            answer_weight += edge2weight[(s, t)]

        # 4) Compare with reference minimal weight
        assert self.reference_answer_weight is not None
        assert self.reference_answer_weight <= answer_weight, "answer weight should be greater than or equal to reference_answer_weight"

        is_correct = (answer_weight == self.reference_answer_weight)
        reward = 1.0 if is_correct else 0.0

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "reference_answer_weight": self.reference_answer_weight,
            "user_path": path,
            "user_weight": answer_weight,
            "N": self.N,
            "edges": self.edges,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the agent's response."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a valid action by using the current reference answer."""
        # If reference answer is available, return it to demonstrate a correct submission.
        if self.reference_answer is not None:
            return f"\\boxed{{{self.reference_answer}}}"
        # Fallback: sample a random permutation (may be invalid for the current graph)
        path = list(range(self.N)) if self.N else [0, 1]
        random.shuffle(path)
        return f"\\boxed{{{' '.join(map(str, path))}}}"