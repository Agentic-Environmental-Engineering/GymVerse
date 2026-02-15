# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One Touch Drawing environment - Eulerian path puzzle."""

import random
from collections import Counter, defaultdict
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class OneTouchDrawingEnv(Env):
    """
    One Touch Drawing (Eulerian Path) environment.

    Players must find a path that traverses each edge exactly once in an
    undirected graph. The graph is guaranteed to have an Eulerian path
    (0 or 2 nodes with odd degree).

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_nodes: int = 10,
        max_nodes: int = 40,
        extra_edge_ratio: float = 0.5,
        **_,
    ):
        """
        Initialize One Touch Drawing environment.

        Args:
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            extra_edge_ratio: Ratio of extra edges to add beyond the cycle
        """
        super().__init__()
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.extra_edge_ratio = extra_edge_ratio
        self.nodes = None
        self.edges = None

    def _generate_graph(self, seed: int, num_nodes: int) -> Tuple[List[str], List[List[str]]]:
        """
        Generate a graph with Eulerian path.

        Args:
            seed: Random seed
            num_nodes: Number of nodes

        Returns:
            Tuple of (nodes, edges)
        """
        random.seed(seed)
        nodes = [f"node {i+1}" for i in range(num_nodes)]

        # Create basic cycle to ensure connectivity
        cycle_edges = []
        for i in range(num_nodes):
            cycle_edges.append([nodes[i], nodes[(i+1) % num_nodes]])

        # Add extra edges
        extra_count = max(1, int(num_nodes * self.extra_edge_ratio))

        # Candidate edges (not in cycle)
        candidate_pool = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if not (j == i+1 or (i == 0 and j == num_nodes-1)):
                    candidate_pool.append((i, j))

        extra_count = min(extra_count, len(candidate_pool))
        selected = random.sample(candidate_pool, extra_count)
        extra_edges = [[nodes[i], nodes[j]] for i, j in selected]
        edges = cycle_edges + extra_edges

        # Ensure Eulerian path exists (0 or 2 odd-degree nodes)
        max_iterations = 100
        iteration = 0
        while iteration < max_iterations:
            degrees = defaultdict(int)
            for u, v in edges:
                degrees[u] += 1
                degrees[v] += 1

            odd_nodes = [node for node, d in degrees.items() if d % 2 != 0]
            if len(odd_nodes) in (0, 2):
                break

            # If more than 2 odd nodes, add edge between two odd nodes
            if len(odd_nodes) >= 2:
                u, v = random.sample(odd_nodes, 2)
                edges.append([u, v])

            iteration += 1

        return nodes, edges

    def _find_eulerian_path(self, nodes: List[str], edges: List[List[str]]) -> List[str]:
        """
        Find an Eulerian path using Hierholzer's algorithm.

        Args:
            nodes: List of node names
            edges: List of edges

        Returns:
            Path as list of nodes
        """
        # Build adjacency list
        graph = defaultdict(list)
        edge_count = defaultdict(int)

        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
            key = tuple(sorted([u, v]))
            edge_count[key] += 1

        # Find start node (odd degree or any node)
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1

        start = nodes[0]
        for node in nodes:
            if degrees[node] % 2 == 1:
                start = node
                break

        # Hierholzer's algorithm
        stack = [start]
        path = []
        current_edges = defaultdict(list)

        for u, v in edges:
            current_edges[u].append(v)
            current_edges[v].append(u)

        while stack:
            curr = stack[-1]
            if current_edges[curr]:
                next_node = current_edges[curr].pop()
                current_edges[next_node].remove(curr)
                stack.append(next_node)
            else:
                path.append(stack.pop())

        return path[::-1]

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: node 1, node 3, ...'\n\n"
            "Alternatively, you can use \\boxed{node 1, node 3, ...} format.\n\n"
            "You are a graph theory expert. Given the following nodes and edges, provide an Eulerian path that traverses each edge exactly once.\n"
            "Your answer should be a comma-separated list of node names.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new One Touch Drawing puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate graph
        num_nodes = random.randint(self.min_nodes, self.max_nodes)
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        self.nodes, self.edges = self._generate_graph(puzzle_seed, num_nodes)

        # Build observation
        edge_strs = [f"<{u}, {v}>" for u, v in self.edges]
        problem = (
            f"Nodes: {', '.join(self.nodes)}\n"
            f"Edges: {', '.join(edge_strs)}\n"
        )

        observation = f"{self._get_instructions()}{problem}"

        return observation, {
            "suffix": f"{len(self.nodes)} nodes, {len(self.edges)} edges."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Verify the Eulerian path.

        Args:
            action: Comma-separated list of nodes

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        parsed_action = extract_last_boxed_answer(action)

        if parsed_action is None:
            import re
            match = re.search(r'Answer:\s*(.+)', action, re.IGNORECASE)
            if match:
                parsed_action = match.group(1).strip()

        if parsed_action is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: node 1, node 2, ...' or \\boxed{node 1, node 2, ...} format."
            )
            return obs, 0.0, True, False, {}

        # Parse node list
        action_nodes = [n.strip() for n in parsed_action.split(",") if n.strip()]

        if not action_nodes:
            obs = "Empty path provided."
            return obs, 0.0, True, False, {}

        # Check path length
        if len(action_nodes) != len(self.edges) + 1:
            obs = (
                f"Path length mismatch. Expected {len(self.edges) + 1} nodes "
                f"(number of edges + 1), got {len(action_nodes)}."
            )
            return obs, 0.0, True, False, {}

        # Verify path
        edge_counter = Counter()
        for edge in self.edges:
            key = frozenset(edge)
            edge_counter[key] += 1

        # Check each consecutive pair
        for i in range(len(action_nodes) - 1):
            key = frozenset((action_nodes[i], action_nodes[i+1]))
            if edge_counter[key] <= 0:
                obs = (
                    f"Invalid path. Edge from '{action_nodes[i]}' to '{action_nodes[i+1]}' "
                    f"is not available or already used."
                )
                return obs, 0.0, True, False, {}
            edge_counter[key] -= 1

        # Check all edges used
        if sum(edge_counter.values()) == 0:
            obs = (
                f"Correct! You found a valid Eulerian path.\n"
                f"Path: {' -> '.join(action_nodes)}"
            )
            reward = 1.0
        else:
            obs = "Some edges were not traversed."
            reward = 0.0

        return obs, reward, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a valid Eulerian path.

        Returns:
            Valid Eulerian path
        """
        path = self._find_eulerian_path(self.nodes, self.edges)
        path_str = ", ".join(path)
        return f"\\boxed{{{path_str}}}"
