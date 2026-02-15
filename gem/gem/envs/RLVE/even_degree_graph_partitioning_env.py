from typing import Any, Optional, SupportsFloat, Tuple, List
import random
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class EvenDegreeGraphPartitioningEnv(Env):
    """Undirected Graph Partitioning Environment - Single-turn Q&A.

    Task:
    Partition the vertices of an undirected graph into two groups (1 or 2)
    such that for each vertex, the number of edges connecting it to vertices
    in the same group is even.

    Answer format:
    Provide N integers (space-separated) in \\boxed{...}, where the i-th integer
    is the group (1 or 2) assigned to vertex i (0-indexed).
    """

    def __init__(
        self,
        N: Optional[int] = None,
        min_N: int = 3,
        max_N: int = 50,
        **kwargs
    ):
        super().__init__()
        self.N_fixed = N
        self.min_N = min_N
        self.max_N = max_N

        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.edges: List[Tuple[int, int]] = []
        self.N_current: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving an undirected graph partitioning problem.\n"
            "Please provide your answer in \\boxed{...} format.\n"
            "Inside the box, output N integers separated by spaces, where the i-th integer (for vertex i) is 1 or 2.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Determine N, ensuring N >= 3
        if self.N_fixed is not None:
            assert self.N_fixed >= 3, "N should be greater than or equal to 3"
            N = self.N_fixed
        else:
            N = random.randint(self.min_N, self.max_N)
            assert N >= 3, "N should be greater than or equal to 3"

        # Generate a graph and a valid partition
        while True:
            vertex_permutation = list(range(N))
            random.shuffle(vertex_permutation)
            group_1 = vertex_permutation[: random.randint(0, N)]
            group_2 = vertex_permutation[len(group_1):]

            edges: List[Tuple[int, int]] = []
            degrees = [0] * N

            def build(group: List[int]) -> None:
                # Build intra-group edges to ensure even degree within the group
                if len(group) <= 2:
                    return
                # Randomly connect vertices to earlier vertices in the group
                for i in range(1, len(group) - 1):
                    neighbors = random.sample(group[:i], random.randint(0, i))
                    for neighbor in neighbors:
                        u, v = min(group[i], neighbor), max(group[i], neighbor)
                        edges.append((u, v))
                        degrees[u] += 1
                        degrees[v] += 1
                # Ensure even degrees by connecting the last vertex to odd-degree vertices
                for vertex in group[:-1]:
                    if degrees[vertex] % 2 == 1:
                        u, v = min(group[-1], vertex), max(group[-1], vertex)
                        edges.append((u, v))
                        degrees[u] += 1
                        degrees[v] += 1
                # All vertices in the group should have even degree
                assert all(degrees[vertex] % 2 == 0 for vertex in group), "All vertices in the group should have even degree"

            build(group_1)
            build(group_2)

            # Optionally add inter-group edges (do not affect even-degree constraint within groups)
            if len(group_1) and len(group_2):
                all_cross = [(min(u, v), max(u, v)) for u in group_1 for v in group_2]
                if all_cross:
                    edges += random.sample(all_cross, random.randint(0, len(all_cross)))

            if len(edges) > 0:
                # Accept only non-empty edge sets
                self.edges = edges
                break

        random.shuffle(self.edges)

        # Validate edges and uniqueness
        for u, v in self.edges:
            assert 0 <= u < v < N, "Edge endpoints must satisfy 0 <= u < v < N"
        assert len(self.edges) == len(set(self.edges)), "Edges should be unique"

        # Construct reference partition labels
        labels = [0] * N
        for i in range(len(group_1)):
            labels[group_1[i]] = 1
        for i in range(len(group_2)):
            labels[group_2[i]] = 2
        self.reference_answer = " ".join(map(str, labels))

        self.N_current = N

        # Build problem description
        edges_text = "\n".join(f"({u}, {v})" for u, v in self.edges)
        problem_text = (
            f"You are given an undirected graph with {N} vertices, labeled from 0 to {N - 1}. "
            f"The graph contains the following undirected edges:\n{edges_text}\n\n"
            "Please partition the vertices into two groups (labeled 1 and 2) such that:\n"
            "1. Each vertex belongs to exactly one group.\n"
            "2. For each vertex, the number of edges connecting it to vertices in the same group is even.\n\n"
            f"Output Format: A single line containing {N} integers (separated by spaces), where the i-th integer "
            f"is the group number (1 or 2) assigned to vertex i (from 0 to {N - 1}). "
            "Your final answer should be enclosed in \\boxed{...}."
        )

        self.current_problem = problem_text
        obs = self._get_instructions() + self.current_problem
        return obs, {"N": N, "num_edges": len(self.edges)}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Validate the submitted answer."""
        # Extract the boxed content
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.N_current is None or self.reference_answer is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "not_initialized"}

        # Parse labels from boxed content
        try:
            labels = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        N = self.N_current
        if len(labels) != N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "detail": "wrong_length"}
        if not all(label in (1, 2) for label in labels):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer", "detail": "labels_out_of_range"}

        # Verify the even-degree constraint within the same group
        degrees = [0] * N
        for u, v in self.edges:
            same_group = labels[u] == labels[v]
            if same_group:
                degrees[u] += 1
                degrees[v] += 1

        satisfied = sum(deg % 2 == 0 for deg in degrees)
        is_correct = (satisfied == N)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "satisfied": satisfied,
            "N": N,
            "num_edges": len(self.edges),
            "reference_answer": self.reference_answer,
            "user_answer": labels,
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: random partition into groups 1 or 2."""
        if self.N_current is None:
            # Default to a reasonable size if reset hasn't been called
            N = max(self.min_N, 3)
        else:
            N = self.N_current
        random_labels = [str(random.choice([1, 2])) for _ in range(N)]
        return f"\\boxed{{{' '.join(random_labels)}}}"