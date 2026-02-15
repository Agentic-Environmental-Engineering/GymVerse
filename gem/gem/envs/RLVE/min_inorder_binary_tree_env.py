import random
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MinInorderBinaryTreeEnv(Env):
    """Environment for selecting the binary tree with lexicographically smallest inorder traversal."""

    def __init__(
        self,
        N: Optional[int] = None,
        min_n: int = 4,
        max_n: int = 100,
        **kwargs
    ):
        """
        Initialize the environment.

        Parameters:
        - N: If provided, fixes the number of nodes. Must be >= 4.
        - min_n: Minimum number of nodes when N is not provided (default: 4).
        - max_n: Maximum number of nodes when N is not provided (default: 100).
        """
        super().__init__()
        self.fixed_N: Optional[int] = N
        self.min_n: int = min_n
        self.max_n: int = max_n

        # Internal state
        self.current_problem: Optional[str] = None
        self.reference_answer: Optional[str] = None
        self.gold_answer: Optional[List[int]] = None
        self.N: Optional[int] = None
        self.edges: Optional[List[Tuple[int, int]]] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given an undirected tree described by edges where parent-child direction is unspecified.\n"
            "Among all possible binary trees that can be formed using all edges, choose the one whose inorder traversal is lexicographically smallest.\n"
            "Output Format: Your final answer must be a single line of N space-separated integers wrapped in \\boxed{...}.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset environment and generate a new problem."""
        super().reset(seed)

        # Determine N
        if self.fixed_N is not None:
            assert self.fixed_N >= 4, "N should be greater than or equal to 4"
            N = self.fixed_N
        else:
            N = random.randint(self.min_n, self.max_n)
            assert N >= 4, "N should be greater than or equal to 4"

        self.N = N

        # Generate edges by constructing a random binary tree over nodes 1..N, then record undirected edges
        edges: List[Tuple[int, int]] = []

        def construct(nodes: List[int]) -> int:
            random.shuffle(nodes)
            root = nodes[0]
            left_size = random.randint(0, len(nodes) - 1)
            right_size = len(nodes) - 1 - left_size
            if left_size > 0:
                left_root = construct(nodes[1:1 + left_size])
                edges.append((min(root, left_root), max(root, left_root)))
            if right_size > 0:
                right_root = construct(nodes[1 + left_size:])
                edges.append((min(root, right_root), max(root, right_root)))
            return root

        construct(list(range(1, N + 1)))
        random.shuffle(edges)

        # Validate edges
        assert len(edges) == len(set(edges)) == N - 1, "edges should be unique and of size N-1"
        assert all(1 <= u < v <= N for u, v in edges), "edges should be between 1 and N"

        self.edges = edges

        # Build adjacency
        G: List[List[int]] = [[] for _ in range(N + 1)]
        SON: List[List[int]] = [[] for _ in range(N + 1)]
        FA: List[int] = [0] * (N + 1)
        HEAD: List[int] = [0] * (N + 1)

        for u, v in edges:
            G[u].append(v)
            G[v].append(u)

        # Choose a start node FIR: the smallest index (scanning from N down to 1) whose degree != 3
        FIR = 0
        for i in range(N, 0, -1):
            if (len(G[i]) ^ 3) != 0:
                FIR = i

        def build(start: int) -> None:
            """Build SON and HEAD given a root 'start' using FA as parent array."""
            # Clear SON
            for idx in range(1, N + 1):
                SON[idx] = []
            order: List[int] = []
            FA[start] = 0
            stack: List[int] = [start]
            while stack:
                u = stack.pop()
                order.append(u)
                for v in G[u]:
                    if v != FA[u]:
                        SON[u].append(v)
                        FA[v] = u
                        stack.append(v)
            # Post-order compute HEAD
            for u in reversed(order):
                if len(SON[u]) == 0:
                    HEAD[u] = u
                elif len(SON[u]) == 1:
                    c = SON[u][0]
                    HEAD[u] = u if u < HEAD[c] else HEAD[c]
                else:
                    a, b = SON[u][0], SON[u][1]
                    HEAD[u] = HEAD[a] if HEAD[a] < HEAD[b] else HEAD[b]

        # First build from FIR
        build(FIR)

        # dfs1(u): determine the root rt
        u = FIR
        while True:
            if len(SON[u]) == 0:
                rt = u
                break
            elif len(SON[u]) == 1:
                c = SON[u][0]
                if HEAD[c] < c:
                    rt = u
                    break
                else:
                    u = c
            else:  # len == 2
                a, b = SON[u][0], SON[u][1]
                if HEAD[a] < HEAD[b]:
                    u = b
                else:
                    u = a

        # Rebuild with chosen root
        FA[rt] = 0
        build(rt)

        # dfs2(u): inorder traversal with tie-breaking rules to get lexicographically smallest sequence
        ans: List[int] = []
        stack_lr: List[Tuple[int, str]] = [(rt, 'go')]
        while stack_lr:
            node, typ = stack_lr.pop()
            if typ == 'emit':
                ans.append(node)
                continue
            # typ == 'go'
            if len(SON[node]) == 0:
                ans.append(node)
            elif len(SON[node]) == 1:
                c = SON[node][0]
                if node < HEAD[c]:
                    # output node, then child
                    stack_lr.append((c, 'go'))
                    stack_lr.append((node, 'emit'))
                else:
                    # child, then node
                    stack_lr.append((node, 'emit'))
                    stack_lr.append((c, 'go'))
            else:
                a, b = SON[node][0], SON[node][1]
                # choose left/right based on HEAD comparison
                if HEAD[a] < HEAD[b]:
                    left, right = a, b
                else:
                    left, right = b, a
                # inorder: left, node, right => push in reverse
                stack_lr.append((right, 'go'))
                stack_lr.append((node, 'emit'))
                stack_lr.append((left, 'go'))

        self.gold_answer = ans
        self.reference_answer = " ".join(map(str, ans))

        # Build problem statement
        edges_str = "\n".join(f"({u}, {v})" for u, v in edges)
        self.current_problem = (
            f"You are given {N} nodes numbered from 1 to {N}, along with the following edges "
            f"(for each edge, the parent–child direction is not specified):\n{edges_str}\n\n"
            f"Please construct a valid binary tree using all these edges. Among all possible binary trees that can be formed, "
            f"choose the one whose inorder traversal is lexicographically smallest. "
            f"Output a single line containing {N} space-separated integers — the inorder traversal of the chosen binary tree.\n\n"
            f"Output Format: Provide your final answer as N space-separated integers in \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {
            "N": N,
            "edges": edges,
            "reference_answer": self.reference_answer
        }

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute a step, verify the provided answer."""
        if self.gold_answer is None or self.N is None:
            # Environment not properly reset
            return TERMINAL_STATE, 0.0, True, False, {"error": "env_not_ready"}

        # Parse boxed answer
        boxed_content = self._parse_answer(action)
        if boxed_content is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Attempt to parse space-separated integers
        try:
            user_list = list(map(int, boxed_content.strip().split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        # Validate solution structure
        N = self.N
        if len(user_list) != N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "wrong_length"}
        if set(user_list) != set(range(1, N + 1)):
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_solution", "detail": "not_a_permutation"}

        is_correct = (user_list == self.gold_answer)
        reward: float = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer,
            "user_answer": " ".join(map(str, user_list)),
            "N": N,
            "edges": self.edges
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
        """Sample a random action in the required boxed format."""
        if self.N is None:
            # Default fallback
            return "\\boxed{}"
        vals = list(range(1, self.N + 1))
        random.shuffle(vals)
        return f"\\boxed{{{' '.join(map(str, vals))}}}"