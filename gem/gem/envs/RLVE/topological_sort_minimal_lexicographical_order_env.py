import heapq
import random
from typing import Any, List, Optional, SupportsFloat, Tuple
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TopologicalSort_MinimalLexicographicalOrderEnv(Env):
    """Environment for generating and verifying minimal lexicographical topological ordering problems."""

    def __init__(
        self,
        N: int,
        max_indeg: int = 3,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: Number of vertices (must be >= 3).
        - max_indeg: Maximum in-degree for each vertex when generating constraints.
        """
        super().__init__()
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")
        self.N = N
        self.max_indeg = max_indeg

        # Problem state
        self.current_problem: Optional[str] = None
        self.before_conditions: List[Tuple[int, int]] = []
        self.gold_answer: Optional[List[int]] = None
        self.reference_answer: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a minimal lexicographical topological sorting problem.\n"
            "Please provide your answer in \\boxed{...} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Generate random constraints
        permutation = list(range(self.N))
        random.shuffle(permutation)

        self.before_conditions = []
        while True:
            for i in range(self.N):
                if i == 0:
                    continue
                # Randomly select predecessors for vertex i, up to max_indeg
                count = random.randint(0, min(i, self.max_indeg))
                for j in random.sample(range(i), count):
                    self.before_conditions.append((permutation[j], permutation[i]))
            if self.before_conditions:
                break
        random.shuffle(self.before_conditions)

        # Build the reverse graph (Y → X)
        adjacency: List[List[int]] = [[] for _ in range(self.N)]  # adjacency[u] holds every v with edge u→v
        indeg: List[int] = [0] * self.N  # in-degree of each vertex

        for before, after in self.before_conditions:
            adjacency[after].append(before)
            indeg[before] += 1

        # Kahn’s algorithm with a max-heap to derive minimal lexicographical order after reversing
        pq: List[int] = []
        for i in range(self.N):
            if indeg[i] == 0:
                heapq.heappush(pq, -i)  # negate to turn min-heap into max-heap

        order: List[int] = []
        while pq:
            u = -heapq.heappop(pq)
            order.append(u)
            for v in adjacency[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(pq, -v)

        if len(order) < self.N:
            # This should not happen for the constructed constraints
            raise RuntimeError("Cycle detected in generated constraints, which should be impossible.")

        # The gold answer is the reversed extraction order
        self.gold_answer = list(reversed(order))
        self.reference_answer = " ".join(map(str, self.gold_answer))

        # Build the problem prompt
        before_lines = "\n".join(f"{j} must be before {i}" for j, i in self.before_conditions)
        self.current_problem = (
            f"Please find a permutation of 0 to {self.N - 1} ({self.N} integers in total) such that the following conditions are satisfied:\n"
            f"{before_lines}\n\n"
            "If multiple permutations satisfy the conditions, choose the one where:\n"
            "(1) 0 should appear as early as possible;\n"
            "(2) Subject to that, 1 should appear as early as possible;\n"
            "(3) Subject to that, 2 should appear as early as possible;\n"
            "(4) And so on...\n\n"
            "Output Format: Your final answer should be a single line containing the permutation "
            f"p(0), p(1), ..., p({self.N - 1}), separated by spaces, inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a single step to verify the provided answer."""
        boxed_content = self._parse_answer(action)

        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Parse the boxed content into a list of integers
        try:
            tokens = boxed_content.strip().split()
            permutation = list(map(int, tokens))
        except Exception:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate permutation format
        if len(permutation) != self.N:
            info = {
                "error": "invalid_length",
                "user_answer": permutation,
                "reference_answer": self.reference_answer,
                "correct": False,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        if len(set(permutation)) != self.N or not all(0 <= i < self.N for i in permutation):
            info = {
                "error": "invalid_elements",
                "user_answer": permutation,
                "reference_answer": self.reference_answer,
                "correct": False,
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Verify topological conditions
        positions = [None] * self.N
        for idx, p in enumerate(permutation):
            positions[p] = idx

        satisfied = sum(positions[j] < positions[i] for j, i in self.before_conditions)
        is_topologically_valid = satisfied == len(self.before_conditions)

        # Correctness requires exact match with the gold minimal lexicographical order
        is_correct = (self.gold_answer == permutation)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "topologically_valid": is_topologically_valid,
            "satisfied_constraints": satisfied,
            "total_constraints": len(self.before_conditions),
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, permutation)),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...} from the given text."""
        import re
        pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a random permutation inside \\boxed{...}."""
        perm = list(range(self.N))
        random.shuffle(perm)
        return "\\boxed{" + " ".join(map(str, perm)) + "}"