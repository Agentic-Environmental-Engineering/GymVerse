import random
import re
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple

import networkx
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class TreeDynamic_XORZeroPathEnv(Env):
    """
    GEM environment for the Tree Dynamic XOR Zero Path problem.

    Task:
    - Given a tree with N vertices labeled 0..N-1 and weighted edges.
    - An order of edge removals is specified (by edge indices).
    - After removing the first 0, 1, ..., N-1 edges in that order, compute
      the number of unordered vertex pairs (u, v), u < v, such that the XOR
      of edge weights along the unique simple path between u and v equals 0
      in the remaining graph.
    - Output N integers (separated by spaces), corresponding to the count at
      the beginning and after each successive removal.

    Single-turn environment:
    - reset() generates a random instance and provides the full problem text.
    - step(action) expects the answer in \\boxed{...} format and terminates.

    Rewards:
    - Correct answer: 1.0
    - Wrong answer: 0.0
    - Format error (no valid \\boxed{...}): -0.1
    """

    def __init__(
        self,
        n: Optional[int] = None,
        min_n: int = 3,
        max_n: int = 100,
        max_weight: Optional[int] = None,
        **kwargs: Any
    ):
        """
        Initialize the environment.

        Args:
            n: If provided, use this fixed number of vertices (N >= 3).
            min_n: Minimum number of vertices when sampling N (inclusive).
            max_n: Maximum number of vertices when sampling N (inclusive).
            max_weight: If provided, edge weights are sampled uniformly from [0, max_weight].
                        If None, edge weights are sampled uniformly from [0, N].
            **kwargs: Reserved for compatibility.
        """
        super().__init__()
        self.fixed_n = n
        self.min_n = min_n
        self.max_n = max_n
        self.max_weight = max_weight

        self.current_problem: Optional[str] = None
        self.reference_answer_list: Optional[List[int]] = None
        self.reference_answer_str: Optional[str] = None

        self.N: Optional[int] = None
        self.edges: Optional[List[Tuple[int, int, int]]] = None
        self.removes_order: Optional[List[int]] = None

    def _get_instructions(self) -> str:
        """Return generic task instructions."""
        return (
            "You are solving a tree XOR-zero path counting problem under dynamic edge removals.\n"
            "Please provide your final answer as N space-separated integers wrapped in \\boxed{...}.\n"
            "For example: \\boxed{a0 a1 a2 ... a_{N-1}}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the environment and generate a new problem instance."""
        super().reset(seed)

        # Determine N
        if self.fixed_n is not None:
            self.N = int(self.fixed_n)
        else:
            self.N = random.randint(self.min_n, self.max_n)
        assert self.N >= 3, "N should be greater than or equal to 3"

        N = self.N

        # Generate a random tree with weighted edges
        edges: List[Tuple[int, int, int]] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        # Weight upper bound: if max_weight is None, use N, else use max_weight
        wmax = N if self.max_weight is None else int(self.max_weight)

        for index, vertex in enumerate(permutations):
            if index == 0:
                continue
            u, v = vertex, random.choice(permutations[:index])
            u, v = (u, v) if u < v else (v, u)
            edges.append((u, v, random.randint(0, wmax)))
        random.shuffle(edges)

        # Sanity checks: unique edges, correct count, and that it forms a tree
        for u, v, _ in edges:
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, _ in edges)) == N - 1

        tree = networkx.Graph()
        tree.add_weighted_edges_from(edges)
        assert networkx.is_tree(tree)

        # Generate the removal order (by edge indices)
        removes_order = list(range(N - 1))
        random.shuffle(removes_order)

        # Compute the XOR-from-root values and the answers after removals
        answer_list = self._compute_answers(N, edges, removes_order)

        self.edges = edges
        self.removes_order = removes_order
        self.reference_answer_list = answer_list
        self.reference_answer_str = " ".join(map(str, answer_list))

        # Build problem prompt
        prompt = self._build_prompt(N, edges, removes_order)

        self.current_problem = self._get_instructions() + prompt
        return self.current_problem, {}

    def _build_prompt(
        self,
        N: int,
        edges: List[Tuple[int, int, int]],
        removes_order: List[int],
    ) -> str:
        """Construct the problem text."""
        edges_str = "\n".join(
            f"edge {i} : ({u} {v} {w})" for i, (u, v, w) in enumerate(edges)
        )
        removes_str = " ".join(map(str, removes_order))
        prompt = (
            f"You are given a tree (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from 0 to {N - 1}.\n\n"
            f"The tree has the following {N - 1} undirected edges. Each edge is represented as a tuple (u, v, w), "
            f"meaning there is an undirected edge between vertex u and vertex v with weight w:\n"
            f"{edges_str}\n\n"
            f"You will remove edges one by one in the following order: {removes_str}\n"
            f"After removing the first 0, 1, ..., {N - 1} edges (in the given order above), please compute the number of paths such that the XOR of the weights along the path is equal to 0. "
            f"There are C({N}, 2) paths in total, where C is the binomial coefficient.\n\n"
            f"Output Format: A single line containing {N} integers — the number of such paths at the beginning and after each removal, separated by spaces. "
            f"Wrap your answer in \\boxed{{...}}."
        )
        return prompt

    def _compute_answers(
        self,
        N: int,
        edges: List[Tuple[int, int, int]],
        removes_order: List[int],
    ) -> List[int]:
        """
        Compute the required answer list of length N:
        - answer[0] is the number of XOR-zero paths in the initial tree (0 edges removed),
        - answer[i] is the count after removing the first i edges in removes_order, for i from 1 to N-1.

        The method constructs answers by reversing the removals: start from an empty forest and add edges back.
        """
        # Build adjacency for XOR-from-root computation
        adj: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        # Compute XOR-from-root (root at 0)
        xor_from_0 = [0] * N

        def dfs_compute_xor(start: int = 0) -> None:
            stack = [(start, -1)]
            while stack:
                u, parent = stack.pop()
                for v, w in adj[u]:
                    if v == parent:
                        continue
                    xor_from_0[v] = xor_from_0[u] ^ w
                    stack.append((v, u))

        dfs_compute_xor(0)

        # DSU-like structure using explicit node lists and xor-count maps
        parent = list(range(N))
        xor2num: List[Dict[int, int]] = [{xor_from_0[u]: 1} for u in range(N)]
        nodes_list: List[List[int]] = [[u] for u in range(N)]

        # Reverse the removal process: start from empty forest, add edges back
        answer: List[int] = [0]
        for idx in reversed(removes_order):
            # Current number of XOR-zero pairs carried forward
            answer.append(answer[-1])

            u0, v0 = edges[idx][0], edges[idx][1]
            u = parent[u0]
            v = parent[v0]
            if u == v:
                # Should not happen for a tree, but keep a guard
                continue

            # Union by size: ensure u is the leader of the larger component
            if len(nodes_list[u]) < len(nodes_list[v]):
                u, v = v, u

            # Merge component v into u
            nodes_u = nodes_list[u]
            nodes_v = nodes_list[v]
            nodes_u.extend(nodes_v)

            # Count pairs with equal xor_from_0 values across the two components
            for node in nodes_v:
                answer[-1] += xor2num[u].get(xor_from_0[node], 0)
                parent[node] = u

            # Update xor frequency map for the merged component
            for node in nodes_v:
                x = xor_from_0[node]
                xor2num[u][x] = xor2num[u].get(x, 0) + 1

        # Reverse to match the required order: initial graph to fully removed
        answer.reverse()
        return answer

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Evaluate the submitted answer."""
        # Parse boxed content
        boxed = self._parse_answer(action)
        if boxed is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        if self.N is None or self.reference_answer_list is None:
            # Environment not properly initialized
            return TERMINAL_STATE, 0.0, True, False, {"error": "uninitialized_env"}

        # Parse integers from boxed content
        try:
            user_list = list(map(int, boxed.strip().split()))
        except ValueError:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_answer"}

        if len(user_list) != self.N:
            return TERMINAL_STATE, 0.0, True, False, {"error": "invalid_length", "expected_length": self.N}

        is_correct = user_list == self.reference_answer_list
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "reference_answer": self.reference_answer_str,
            "user_answer": " ".join(map(str, user_list)),
        }

        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def sample_random_action(self) -> str:
        """Sample a random action: a boxed string of N random integers."""
        if self.N is None:
            # Reasonable default if called before reset
            return "\\boxed{0}"
        # Produce a random guess of N integers
        guess = [str(random.randint(0, self.N)) for _ in range(self.N)]
        return f"\\boxed{{{' '.join(guess)}}}"