from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinimumDirectedSpanningTreeEnv(Env):
    """Minimum Directed Spanning Tree (Arborescence) environment - single-turn Q&A.

    The task is to select exactly N-1 directed edges forming a spanning arborescence
    rooted at a specified root, minimizing the total weight. The answer should be
    provided as space-separated endpoints inside \\boxed{...}, e.g., \\boxed{0 1 0 2 2 3}.
    """

    def __init__(
        self,
        N: int = 5,
        edge_density: float = 0.5,
        wrong_format_reward: float = -0.1,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        **kwargs
    ):
        super().__init__()

        # Parameter validation as in the original environment
        assert N >= 3, "N should be greater than or equal to 3"
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        self.N: int = N
        self.edge_density: float = edge_density

        # Reward configuration per conversion requirements
        self.wrong_format_reward: float = wrong_format_reward
        self.correct_reward: float = correct_reward
        self.incorrect_reward: float = incorrect_reward

        # Internal state
        self.current_problem: Optional[str] = None
        self.edges: List[Tuple[int, int, int]] = []
        self.root: Optional[int] = None
        self.reference_answer: Optional[str] = None  # "s t s t ..."
        self.gold_answer_weight: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a directed graph and must select exactly N-1 edges to form a "
            "minimum-weight spanning arborescence rooted at a specified root.\n"
            "Please provide your answer as space-separated endpoints inside \\boxed{...}.\n"
            "Example: \\boxed{0 1 0 2 2 3}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and generate a new problem instance."""
        super().reset(seed)

        N = self.N
        edge_density = self.edge_density

        # Problem generation logic adapted from the original _generate()
        while True:
            edges: List[Tuple[int, int, int]] = []
            permutations = list(range(N))
            random.shuffle(permutations)

            # Build a base set of edges ensuring reachability (a random arborescence backbone)
            for index, vertex in enumerate(permutations):
                if index == 0:
                    continue
                t = vertex
                s = random.choice(permutations[:index])
                w = random.randint(1, max(1, int(edge_density * N * (N - 1))))
                edges.append((s, t, w))
            root = permutations[0]

            # Fill additional edges according to edge density
            num_edges = int(edge_density * N * (N - 1))
            if len(edges) < num_edges:
                remaining = list(
                    set((s, t) for s in range(N) for t in range(N) if s != t)
                    - set((s, t) for s, t, _ in edges)
                )
                remaining = random.sample(remaining, min(len(remaining), num_edges - len(edges)))
                for s, t in remaining:
                    w = random.randint(1, max(1, int(edge_density * N * (N - 1))))
                    edges.append((s, t, w))

            random.shuffle(edges)

            # Assertions and validity checks as in the original environment
            for s, t, w in edges:
                assert 0 <= s < N and 0 <= t < N, "s and t should be in range [0, N)"
                assert s != t
            assert len(edges) == len(set((s, t) for s, t, _ in edges)), "edges should be unique"

            # Compute reference answer using NetworkX
            try:
                G = networkx.DiGraph()
                G.add_weighted_edges_from(edges + [(N, root, 0)])
                msa = networkx.minimum_spanning_arborescence(G)
                reference_edges = [(s, t) for s, t in msa.edges() if (s, t) != (N, root)]
                reference_answer_str = " ".join("{} {}".format(s, t) for s, t in reference_edges)
                gold_weight = sum(msa[s][t]["weight"] for s, t in msa.edges())

                if gold_weight > 0:
                    # Save generated instance
                    self.edges = edges
                    self.root = root
                    self.reference_answer = reference_answer_str
                    self.gold_answer_weight = gold_weight
                    break
                else:
                    continue
            except Exception:
                # There might be a bug or rare failure in networkx.minimum_spanning_arborescence
                continue

        # Build problem prompt adapted from the original _prompt_generate()
        edges_str = "\n".join("({}, {}, {})".format(s, t, w) for s, t, w in self.edges)
        problem_text = (
            f"You are given a directed graph with {N} vertices, labeled from 0 to {N - 1}.\n\n"
            "The graph contains the following directed edges. Each edge is represented as a tuple (s, t, w), "
            "meaning a directed edge from vertex s to vertex t with weight w:\n"
            f"{edges_str}\n\n"
            "Your task is to select a subset of edges T = [(s_1, t_1, w_1), (s_2, t_2, w_2), ..., (s_k, t_k)] such that:\n"
            f"- k = {N} - 1 = {N - 1} (i.e., you select exactly {N - 1} edges).\n"
            f"- The selected edges form a spanning arborescence rooted at vertex {self.root} — meaning:\n"
            f"  - All vertices are reachable from vertex {self.root}.\n"
            "  - Each vertex other than the root has exactly one incoming edge.\n"
            "  - The selected edges form no cycles.\n"
            "Your goal is to minimize the total weight of the selected edges.\n\n"
            "Output Format:\n"
            "Your final answer should be a single line containing the endpoints of the selected edges in order:\n"
            "s_1 t_1 s_2 t_2 ... s_k t_k, separated by spaces, and placed inside \\boxed{...}.\n"
            "Example: \\boxed{0 1 0 2 2 3}\n"
        )

        self.current_problem = problem_text

        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {
            "N": self.N,
            "root": self.root,
            "edge_density": self.edge_density,
        }
        return obs, info

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute a step by verifying the provided answer."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Parse integers from boxed content
        try:
            tokens = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        if len(tokens) % 2 != 0:
            return TERMINAL_STATE, self.wrong_format_reward, True, False, {"error": "format_error"}

        # Build edge list from pairs
        user_edges: List[Tuple[int, int]] = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]

        # Validation logic adapted from original scorer()
        N = self.N

        # Must have exactly N - 1 edges
        if len(user_edges) != N - 1:
            info = {"error": "invalid_solution", "reason": "edge_count_mismatch"}
            return TERMINAL_STATE, self.incorrect_reward, True, False, info

        # Must cover all vertices
        if not ((set(s for s, t in user_edges) | set(t for s, t in user_edges)) == set(range(N))):
            info = {"error": "invalid_solution", "reason": "not_spanning"}
            return TERMINAL_STATE, self.incorrect_reward, True, False, info

        # Build adjacency and check for cycles using DFS as in original
        adjacent_list: List[List[int]] = [[] for _ in range(N)]
        for s, t in user_edges:
            if not (0 <= s < N and 0 <= t < N):
                info = {"error": "invalid_solution", "reason": "vertex_out_of_range"}
                return TERMINAL_STATE, self.incorrect_reward, True, False, info
            if s == t:
                info = {"error": "invalid_solution", "reason": "self_loop"}
                return TERMINAL_STATE, self.incorrect_reward, True, False, info
            adjacent_list[s].append(t)

        visited = [False] * N

        def dfs(vertex: int) -> bool:
            for neighbor in adjacent_list[vertex]:
                if visited[neighbor]:
                    return False
                visited[neighbor] = True
                if not dfs(neighbor):
                    return False
            return True

        if self.root is None:
            info = {"error": "internal_error", "reason": "root_not_set"}
            return TERMINAL_STATE, self.incorrect_reward, True, False, info

        visited[self.root] = True
        if not dfs(self.root):
            info = {"error": "invalid_solution", "reason": "cycle_detected"}
            return TERMINAL_STATE, self.incorrect_reward, True, False, info
        if not all(visited):
            info = {"error": "invalid_solution", "reason": "not_all_reachable"}
            return TERMINAL_STATE, self.incorrect_reward, True, False, info

        # Check arborescence using networkx
        G = networkx.DiGraph()
        G.add_nodes_from(range(N + 1))
        G.add_edges_from(user_edges + [(N, self.root)])
        if not networkx.is_arborescence(G):
            info = {"error": "invalid_solution", "reason": "not_arborescence"}
            return TERMINAL_STATE, self.incorrect_reward, True, False, info

        # Check that all edges exist and compute total weight
        edges_weight_map: Dict[Tuple[int, int], int] = {(s, t): w for s, t, w in self.edges}
        answer_weight = 0
        for s, t in user_edges:
            if (s, t) not in edges_weight_map:
                info = {"error": "invalid_solution", "reason": "edge_not_in_graph", "edge": (s, t)}
                return TERMINAL_STATE, self.incorrect_reward, True, False, info
            answer_weight += edges_weight_map[(s, t)]

        # Compare with gold weight
        is_correct = (self.gold_answer_weight is not None) and (answer_weight == self.gold_answer_weight)

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "reference_weight": self.gold_answer_weight,
            "user_answer_weight": answer_weight,
            "root": self.root,
            "N": self.N,
        }

        reward = self.correct_reward if is_correct else self.incorrect_reward
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract content within \\boxed{...} from the text."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: pick random pairs of endpoints inside \\boxed{...}."""
        # Attempt to sample N-1 edges from the existing graph (not guaranteed to be valid)
        num_pairs = self.N - 1
        available_pairs = [(s, t) for s, t, _ in self.edges]
        if len(available_pairs) >= num_pairs:
            sampled = random.sample(available_pairs, num_pairs)
        else:
            # Fallback: randomly generate pairs (may include invalid ones)
            nodes = list(range(self.N))
            sampled = []
            for _ in range(num_pairs):
                s = random.choice(nodes)
                t = random.choice([x for x in nodes if x != s])
                sampled.append((s, t))

        flat = " ".join(f"{s} {t}" for s, t in sampled)
        return f"\\boxed{{{flat}}}"