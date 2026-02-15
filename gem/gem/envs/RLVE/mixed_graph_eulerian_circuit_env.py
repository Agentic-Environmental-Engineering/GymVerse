from typing import Any, Optional, SupportsFloat, Tuple, List, Dict
import random
import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MixedGraphEulerianCircuitEnv(Env):
    """Environment for finding an Eulerian circuit in a mixed graph (with both undirected and directed edges).

    The undirected graph is guaranteed to be connected, have no repeated edges, and every vertex has an even degree.
    A subset of the edges from a valid Eulerian circuit is then marked as directed.
    The task is to output a closed walk that starts and ends at the same vertex and uses each listed edge exactly once,
    respecting direction for directed edges and treating undirected edges as usable in either direction.

    The answer must be provided in \\boxed{...} format, containing a space-separated sequence of vertex labels.
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 12,
        **kwargs
    ):
        """Initialize the environment.

        Parameters:
            N: Optional fixed number of vertices. If None, N will be randomly sampled in [min_N, max_N] on reset.
            min_N: Minimum number of vertices if N is not provided. Must be >= 3.
            max_N: Maximum number of vertices if N is not provided. Must be >= min_N.
        """
        super().__init__()
        if N is not None and N < 3:
            raise ValueError("N should be greater than or equal to 3")
        if min_N < 3:
            raise ValueError("min_N should be greater than or equal to 3")
        if max_N < min_N:
            raise ValueError("max_N should be greater than or equal to min_N")

        self.fixed_N: Optional[int] = N
        self.min_N: int = min_N
        self.max_N: int = max_N

        # Problem state
        self.N: Optional[int] = None
        self.undirected_edges: List[Tuple[int, int]] = []
        self.directed_edges: List[Tuple[int, int]] = []
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return general task instructions."""
        return (
            "You are given a graph problem involving Eulerian circuits on a mixed graph.\n"
            "Please provide your answer as a sequence of vertex labels separated by spaces inside \\boxed{...}.\n"
            "For example: \\boxed{0 1 2 1 0}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment by generating a new problem instance."""
        super().reset(seed)
        # Determine N
        if self.fixed_N is not None:
            N = self.fixed_N
        else:
            N = random.randint(self.min_N, self.max_N)
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        # Generate an undirected Eulerian connected graph with N vertices
        while True:
            degrees = [0] * N
            edges: List[Tuple[int, int]] = []

            # Add undirected edges ensuring no duplicate and potential connectivity
            for v in range(1, N - 1):
                neighbors = random.sample(range(v), random.randint(0, v))
                for u in neighbors:
                    if not (0 <= u < v < N):
                        raise AssertionError("Undirected edges should be within the range of vertex labels and u < v")
                    edges.append((u, v))
                    degrees[u] += 1
                    degrees[v] += 1

            # Adjust degrees to ensure they are even by connecting to the last vertex
            for u in range(N - 1):
                if degrees[u] % 2 == 1:
                    v = N - 1
                    edges.append((u, v))
                    degrees[u] += 1
                    degrees[v] += 1

            if not all(degree % 2 == 0 for degree in degrees):
                # Retry if not all even degrees (should not happen due to the adjustment)
                continue

            random.shuffle(edges)
            if len(edges) != len(set(edges)):
                # Ensure no repeated undirected edges
                continue
            for u, v in edges:
                if not (0 <= u < v < N):
                    # Ensure undirected edges are added with u < v and within range
                    continue

            # Check connectivity and Eulerian property
            undirected_graph = networkx.Graph()
            undirected_graph.add_nodes_from(range(N))
            undirected_graph.add_edges_from(edges)
            if networkx.is_connected(undirected_graph) and networkx.is_eulerian(undirected_graph):
                # Successful graph generation
                break

        # Compute an Eulerian circuit
        eulerian_circuit = list(networkx.eulerian_circuit(undirected_graph))
        if len(eulerian_circuit) != len(edges):
            raise AssertionError("The Eulerian circuit should visit each edge exactly once")

        # Randomly flag some edges in the Eulerian circuit as directed, at least one of each type
        directed_flags = [False] * len(eulerian_circuit)
        for idx in random.sample(range(len(eulerian_circuit)), random.randint(1, len(eulerian_circuit) - 1)):
            directed_flags[idx] = True

        undirected_edges: List[Tuple[int, int]] = []
        directed_edges: List[Tuple[int, int]] = []
        reference_path_vertices: List[int] = []

        for (u, v), directed_flag in zip(eulerian_circuit, directed_flags):
            reference_path_vertices.append(u)
            if directed_flag:
                directed_edges.append((u, v))
            else:
                undirected_edges.append((min(u, v), max(u, v)))
        reference_path_vertices.append(eulerian_circuit[-1][1])

        if reference_path_vertices[0] != reference_path_vertices[-1]:
            raise AssertionError("The Eulerian circuit should start and end at the same vertex")

        if len(undirected_edges) == 0 or len(directed_edges) == 0:
            # Ensure at least one undirected and one directed edge
            # If not satisfied, regenerate
            return self.reset(seed=seed)

        random.shuffle(undirected_edges)
        random.shuffle(directed_edges)

        # Save problem state
        self.N = N
        self.undirected_edges = undirected_edges
        self.directed_edges = directed_edges
        self.reference_answer = " ".join(map(str, reference_path_vertices))

        # Build problem description
        problem_description = (
            f"You are given a graph with {N} vertices labeled from 0 to {N - 1}.\n\n"
            "The graph contains the following undirected edges:\n"
            + "\n".join(f"({u}, {v})" for u, v in self.undirected_edges)
            + "\n\n"
            "It also contains the following directed edges (each <u, v> represents a directed edge from vertex u to vertex v):\n"
            + "\n".join(f"<{u}, {v}>" for u, v in self.directed_edges)
            + "\n\n"
            "It is guaranteed that if all directed edges are treated as undirected, the resulting graph is connected and has no repeated edges, and every vertex has an even degree.\n\n"
            "Please find an Eulerian circuit in this graph — a closed path that starts and ends at the same vertex and visits each edge exactly once.\n"
            "Output Format: Provide a single line containing the sequence of vertex labels visited in order, separated by spaces, inside \\boxed{...}."
        )

        self.current_problem = problem_description
        obs = self._get_instructions() + self.current_problem
        info: Dict[str, Any] = {
            "N": self.N,
            "undirected_edges": self.undirected_edges,
            "directed_edges": self.directed_edges,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by verifying the user's answer."""
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the sequence of vertices from the boxed content
        tokens = boxed_content.strip().split()
        try:
            path_vertices = [int(tok) for tok in tokens]
        except ValueError:
            # Content inside boxed is not all integers
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate solution
        is_correct = self._validate_path(path_vertices)

        info: Dict[str, Any] = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, path_vertices)),
            "N": self.N,
            "undirected_edges": self.undirected_edges,
            "directed_edges": self.directed_edges,
        }
        reward: float = 1.0 if is_correct else 0.0
        return TERMINAL_STATE, reward, True, False, info

    def _validate_path(self, path_vertices: List[int]) -> bool:
        """Validate whether the provided path is a correct Eulerian circuit for the given mixed graph."""
        if self.N is None or self.reference_answer is None:
            return False

        # Basic checks
        if len(path_vertices) == 0:
            return False
        if not all(isinstance(u, int) and 0 <= u < self.N for u in path_vertices):
            return False
        if path_vertices[0] != path_vertices[-1]:
            return False

        # Prepare edge counters
        undirected_counts: Dict[Tuple[int, int], int] = {(u, v): 0 for u, v in self.undirected_edges}
        directed_counts: Dict[Tuple[int, int], int] = {(u, v): 0 for u, v in self.directed_edges}

        # Traverse edges in the path and count usage
        for u, v in zip(path_vertices, path_vertices[1:]):
            is_directed = (u, v) in directed_counts
            is_undirected = (min(u, v), max(u, v)) in undirected_counts

            # Ensure the step uses exactly one edge type (directed or undirected) from the given sets
            if is_directed and is_undirected:
                return False
            if not is_directed and not is_undirected:
                return False

            if is_directed:
                directed_counts[(u, v)] += 1
            else:
                undirected_counts[(min(u, v), max(u, v))] += 1

        # Check each edge is used exactly once
        if not all(count == 1 for count in directed_counts.values()):
            return False
        if not all(count == 1 for count in undirected_counts.values()):
            return False

        # Ensure the number of traversed edges matches the total number of edges
        total_edges = len(self.undirected_edges) + len(self.directed_edges)
        if len(path_vertices) - 1 != total_edges:
            return False

        return True

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action (random path format), usually incorrect."""
        # Generates a random short sequence to fit the required format.
        # This is primarily for demonstration/testing of format handling.
        if self.N is None:
            # Provide a simple boxed content
            return "\\boxed{0 0}"
        length = random.randint(2, max(2, self.N))
        seq = [random.randint(0, self.N - 1) for _ in range(length - 1)]
        if not seq:
            seq = [0]
        # Make it a closed path by repeating the first vertex
        seq = [seq[0]] + seq + [seq[0]]
        return "\\boxed{" + " ".join(map(str, seq)) + "}"