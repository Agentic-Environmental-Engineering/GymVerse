import random
from typing import Any, Optional, SupportsFloat, Tuple, List
import networkx as nx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TreeEvenPartitioningEnv(Env):
    """Tree Even Partitioning environment - single-turn Q&A.

    Task:
    - Given a tree with NK vertices and (NK - 1) edges.
    - Partition all vertices into N disjoint sets (each size K, where K = NK / N).
    - Each set must induce a connected subgraph.
    - Output N lines, each containing exactly K vertex labels (integers).
    - The N lines (sets) can be in any order; vertices within a line can be in any order.
    - The entire multi-line answer must be placed inside a single \\boxed{...} block.
    """

    prompt_template = (
        "You have a tree (i.e., a connected undirected graph with no cycles) with {NK} vertices "
        "labeled from 1 to {NK}. The tree contains the following {NK} - 1 undirected edges. "
        "Each edge is represented as a tuple (u, v), meaning there is an undirected edge connecting "
        "vertex u to vertex v:\n"
        "{edges}\n\n"
        "Partition all vertices into {N} disjoint sets such that: (1) each set contains exactly {K} vertices "
        "(K = {NK} / {N}), AND (2) each set forms a connected subgraph of the tree.\n"
        "Output Format: Provide exactly {N} lines, each line containing {K} distinct integers between 1 and {NK}, "
        "separated by spaces. The sets and vertex orders may be arbitrary. Enclose your entire multi-line answer "
        "within a single \\boxed{{...}} block."
    )

    def __init__(
        self,
        max_n: int = 10,
        max_k: int = 10,
        correct_reward: float = 1.0,
        wrong_answer_reward: float = 0.0,
        format_error_reward: float = -0.1,
        **kwargs
    ):
        """
        Initialize the TreeEvenPartitioningEnv instance.

        Args:
            max_n: Maximum number of groups N. Must be >= 2.
            max_k: Maximum group size K. Must be >= 2.
            correct_reward: Reward for a fully correct partition.
            wrong_answer_reward: Reward for a wrong but well-formatted answer.
            format_error_reward: Reward for format errors (e.g., missing \\boxed{...} or wrong line/size counts).
        """
        super().__init__()
        assert max_n >= 2, "max_n should be greater than or equal to 2"
        assert max_k >= 2, "max_k should be greater than or equal to 2"
        self.max_n = max_n
        self.max_k = max_k

        self.correct_reward = correct_reward
        self.wrong_answer_reward = wrong_answer_reward
        self.format_error_reward = format_error_reward

        # Problem state
        self.N: Optional[int] = None
        self.K: Optional[int] = None
        self.NK: Optional[int] = None
        self.edges: Optional[List[tuple[int, int]]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return the task instructions."""
        return (
            "Task: Partition the vertices of the given tree into N connected sets of size K.\n"
            "Answer format:\n"
            "- Provide exactly N lines inside a single \\boxed{...} block.\n"
            "- Each line must list exactly K distinct integers (vertex labels), separated by spaces.\n"
            "- All vertex labels must be from 1 to NK, and every vertex must appear exactly once across all lines.\n"
            "- The order of lines and the order of vertices within a line do not matter.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem."""
        super().reset(seed)

        # Randomly generate N and K
        N = random.randint(2, self.max_n)
        K = random.randint(2, self.max_k)
        NK = N * K

        # Generate a random partition (reference partition) and a tree connecting them
        vertices = list(range(1, NK + 1))
        random.shuffle(vertices)
        groups = [vertices[i * K: (i + 1) * K] for i in range(N)]
        for i, g in enumerate(groups):
            assert len(g) == K, f"Group {i} should have exactly {K} vertices"

        edges: List[tuple[int, int]] = []

        # Build tree edges to ensure each group is connected internally, and groups are connected together
        for i, group in enumerate(groups):
            # Connect vertices within the group to form a tree (K-1 edges per group)
            for index, vertex in enumerate(group):
                if index == 0:
                    continue
                u, v = vertex, group[random.randint(0, index - 1)]
                u, v = (u, v) if u < v else (v, u)
                edges.append((u, v))
            # Connect this group to some previous group to ensure global connectivity
            if i > 0:
                u, v = random.choice(group), random.choice(groups[random.randint(0, i - 1)])
                u, v = (u, v) if u < v else (v, u)
                edges.append((u, v))

        random.shuffle(edges)

        # Validate edges
        for (u, v) in edges:
            assert 1 <= u < v <= NK
        assert len(edges) == len(set(edges)) == NK - 1

        # Validate the graph is a tree
        tree = nx.Graph()
        tree.add_edges_from(edges)
        assert nx.is_tree(tree)

        # Save state
        self.N = N
        self.K = K
        self.NK = NK
        self.edges = edges
        self.reference_answer = "\n".join(" ".join(map(str, group)) for group in groups)

        # Build problem text
        edges_text = "\n".join(f"({u}, {v})" for (u, v) in edges)
        self.current_problem = self.prompt_template.format(
            NK=NK, N=N, K=K, edges=edges_text
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step by validating the submitted partition."""
        if self.N is None or self.K is None or self.NK is None or self.edges is None:
            # Environment not properly reset
            return TERMINAL_STATE, self.wrong_answer_reward, True, False, {"error": "env_not_initialized"}

        boxed = self._parse_answer(action)
        if boxed is None:
            # Format error: missing or malformed \\boxed{...}
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "format_error"}

        # Parse group lines from boxed content
        groups: List[List[int]] = []
        try:
            lines = boxed.splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                # Must have exactly K numbers per line
                if len(tokens) != self.K:
                    return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "line_length_mismatch"}
                group = list(map(int, tokens))
                groups.append(group)
            # Must have exactly N lines
            if len(groups) != self.N:
                return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "line_count_mismatch"}
        except Exception:
            # Non-integer or parsing exception
            return TERMINAL_STATE, self.format_error_reward, True, False, {"error": "parse_error"}

        # Validate coverage and uniqueness of vertices
        flat = [v for g in groups for v in g]
        vertices_set = set(flat)
        expected_set = set(range(1, self.NK + 1))
        if len(flat) != self.NK or vertices_set != expected_set:
            # Wrong answer due to duplicates, missing, or out-of-range vertices
            info = {
                "correct": False,
                "reason": "coverage_or_range_error",
            }
            return TERMINAL_STATE, self.wrong_answer_reward, True, False, info

        # Build labels per vertex: label is the group index
        labels = [None] * (self.NK + 1)  # 1-based vertex indexing
        for label, group in enumerate(groups):
            for v in group:
                if labels[v] is not None:
                    # Duplicate vertex across groups
                    info = {
                        "correct": False,
                        "reason": "duplicate_vertex",
                        "vertex": v,
                    }
                    return TERMINAL_STATE, self.wrong_answer_reward, True, False, info
                labels[v] = label

        # Count internal edges within each group
        edge_numbers = [0] * self.N
        for (u, v) in self.edges:
            if labels[u] == labels[v]:
                edge_numbers[labels[u]] += 1

        # Check each group is connected: exactly K-1 internal edges
        connected_flags = [int(cnt == (self.K - 1)) for cnt in edge_numbers]
        all_connected = all(flag == 1 for flag in connected_flags)

        info = {
            "correct": all_connected,
            "connected_groups": sum(connected_flags),
            "total_groups": self.N,
            "edge_counts_per_group": edge_numbers,
            "reference_answer": self.reference_answer,
        }

        reward = self.correct_reward if all_connected else self.wrong_answer_reward
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside the last \\boxed{...} block, allowing multi-line content."""
        import re
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random valid action by returning the reference partition inside a box."""
        if self.reference_answer is None:
            # In case called before reset, provide a generic placeholder
            return r"\boxed{1 2\n3 4}"
        # Reference answer is already N lines with K integers per line
        return f"\\boxed{{\n{self.reference_answer}\n}}"