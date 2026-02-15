import random
from typing import Any, Optional, SupportsFloat, Tuple, List
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class CheckAllCycleXorZeroEnv(Env):
    """Environment for checking whether every cycle in an undirected weighted graph has XOR sum equal to 0."""

    def __init__(
        self,
        N: int = 5,
        edge_ratio: float = 2.0,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Number of vertices in the graph (must be >= 3).
        - edge_ratio: Controls the target number of edges as int(edge_ratio * N).
                      Must satisfy int(edge_ratio * N) > N - 1 to ensure additional edges beyond a spanning tree.
        """
        super().__init__()
        self.N: int = N
        self.edge_ratio: float = edge_ratio

        # Internal state for the current instance
        self.edges: List[Tuple[int, int, int]] = []
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

        # Validation of parameters
        if self.N < 3:
            raise ValueError("N should be greater than or equal to 3")

        if int(self.edge_ratio * self.N) <= (self.N - 1):
            raise ValueError(
                "edge_ratio is too small: int(edge_ratio * N) must be greater than N - 1 to allow non-tree edges."
            )

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "Task: Given an undirected weighted graph, determine whether every cycle in the graph has XOR sum 0.\n"
            "Output YES if the condition holds for every cycle, otherwise output NO.\n"
            "Please provide your final answer in \\boxed{YES} or \\boxed{NO} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new graph instance."""
        super().reset(seed)

        N = self.N
        # Compute an inclusive upper bound for weights
        weight_range = (1 << ((N * (N - 1) // 2).bit_length())) - 1

        edges: List[Tuple[int, int, int]] = []

        # Create a random spanning tree using a random permutation
        permutations = list(range(1, N + 1))
        random.shuffle(permutations)

        # XORs[v] holds XOR from chosen root to vertex v in the spanning tree
        XORs = [0] * (N + 1)
        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u = vertex
            v = random.choice(permutations[:index])
            w = random.randint(0, weight_range)
            XORs[u] = XORs[v] ^ w
            a, b = (u, v) if u < v else (v, u)
            edges.append((a, b, w))

        must_YES = random.choice(["YES", "NO"])
        reference_answer = "YES"

        # Determine total desired number of edges
        num_edges = int(self.edge_ratio * N)

        if len(edges) < num_edges:
            # All possible undirected pairs (u, v) with u < v
            all_pairs = set((u, v) for u in range(1, N + 1) for v in range(u + 1, N + 1))
            existing_pairs = set((u, v) for u, v, _ in edges)
            remaining_pairs = list(all_pairs - existing_pairs)

            # Sample up to the required number of remaining edges
            num_to_add = min(len(remaining_pairs), num_edges - len(edges))
            sampled_pairs = random.sample(remaining_pairs, num_to_add)

            for u, v in sampled_pairs:
                if must_YES == "YES":
                    # Force XOR condition to hold for every cycle
                    w = XORs[u] ^ XORs[v]
                else:
                    # Random weight; may or may not satisfy XOR condition
                    w = random.randint(0, weight_range)

                if (XORs[u] ^ XORs[v]) != w:
                    reference_answer = "NO"

                edges.append((u, v, w))
        else:
            raise ValueError("The number of edges in the initial spanning tree should be less than num_edges.")

        # If we purposely constructed a YES instance, ensure that it is indeed YES
        if must_YES == "YES" and reference_answer != "YES":
            # This should not happen under the construction above
            raise RuntimeError("The reference answer should be YES under must_YES == 'YES'.")

        # Shuffle edges
        random.shuffle(edges)

        # Sanity checks
        assert all(1 <= u < v <= N for u, v, _ in edges), "Edge endpoints must satisfy 1 <= u < v <= N"
        assert all(0 <= w <= weight_range for _, _, w in edges), "Edge weight out of range"
        assert len(edges) == len(set((u, v) for u, v, _ in edges)), "Edges must be unique"

        # Save to state
        self.edges = edges
        self.reference_answer = reference_answer

        # Build prompt
        edges_str = "\n".join(f"({u}, {v}, {w})" for u, v, w in edges)
        self.current_problem = (
            f"We have an undirected graph with {N} vertices labeled from 1 to {N}. "
            f"The graph contains the following undirected edges. Each edge is represented as a tuple (u, v, w), "
            f"meaning an undirected edge connects vertex u to vertex v with weight w:\n"
            f"{edges_str}\n\n"
            f"A cycle is defined as a path that starts and ends at the same vertex. "
            f"Determine whether every cycle in the graph has an XOR sum of its edge weights equal to 0.\n"
            f"Output Format: Please answer with \\boxed{{YES}} or \\boxed{{NO}}."
        )

        obs = self._get_instructions() + self.current_problem
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Verify the user's answer and return the result."""
        # Parse boxed answer
        parsed = self._parse_answer(action)
        if parsed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        user_answer = parsed.strip()
        if user_answer not in ("YES", "NO"):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        assert self.reference_answer is not None, "Environment not initialized properly. Call reset() first."
        is_correct = (user_answer == self.reference_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": user_answer,
            "N": self.N,
            "edges": list(self.edges),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action in the required format."""
        rand_ans = random.choice(["YES", "NO"])
        return f"\\boxed{{{rand_ans}}}"