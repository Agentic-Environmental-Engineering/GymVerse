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

"""Diagram Coloring environment - Graph coloring puzzle."""

import ast
import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class DiagramColoringEnv(Env):
    """
    Graph coloring puzzle environment.

    The agent must provide a valid coloring scheme for a graph using
    the minimum number of colors (chromatic number).

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_nodes: int = 10,
        max_nodes: int = 50,
        **_,
    ):
        """
        Initialize Diagram Coloring environment.

        Args:
            min_nodes: Minimum number of nodes in the graph
            max_nodes: Maximum number of nodes in the graph
        """
        super().__init__()
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.adj = None
        self.chromatic_number = None
        self.num_nodes = None
        self.edges = None

    def _is_k_colorable(self, adj: List[List[int]], k: int) -> bool:
        """Check if a graph is k-colorable using backtracking."""
        n = len(adj)
        # Prioritize nodes with more constraints (by degree, descending)
        nodes = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        colors = [-1] * n

        def backtrack(index: int) -> bool:
            if index == n:
                return True
            node = nodes[index]
            used = set()
            for neighbor in adj[node]:
                if colors[neighbor] != -1:
                    used.add(colors[neighbor])
            for color in range(k):
                if color not in used:
                    colors[node] = color
                    if backtrack(index + 1):
                        return True
                    colors[node] = -1
            return False

        return backtrack(0)

    def _calculate_chromatic_number(self, adj: List[List[int]], e: int) -> int:
        """Calculate the chromatic number (minimum colors needed)."""
        n = len(adj)

        if e == 0:
            return 1

        # Check if bipartite (2-colorable)
        is_bipartite = True
        color = [-1] * n
        for i in range(n):
            if color[i] == -1:
                queue = [i]
                color[i] = 0
                while queue:
                    node = queue.pop(0)
                    for neighbor in adj[node]:
                        if color[neighbor] == -1:
                            color[neighbor] = color[node] ^ 1
                            queue.append(neighbor)
                        elif color[neighbor] == color[node]:
                            is_bipartite = False
                            break
                    if not is_bipartite:
                        break
            if not is_bipartite:
                break

        if is_bipartite:
            return 2

        # Try k-coloring from 3 onwards
        max_degree = max(len(neighbors) for neighbors in adj) if adj else 0
        for k in range(3, max_degree + 2):
            if self._is_k_colorable(adj, k):
                return k

        return max_degree + 1

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are an expert in graph theory and coloring. Below is the information about a graph.\n"
            "Your task:\n"
            "- Read the graph information carefully.\n"
            "- Provide a valid coloring scheme for the graph using the exact number of colors specified.\n"
            "- The coloring scheme should be a list of pairs [node, color] for each node.\n"
            "- Output format: 'Answer: [[0, 1], [1, 0], [2, 1],...]'.\n\n"
            "Alternatively, you can use \\boxed{[[0, 1], [1, 0], ...]} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new graph coloring puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate random graph
        self.num_nodes = random.randint(self.min_nodes, self.max_nodes)
        n = self.num_nodes

        # Generate edges (at least n-1 for connectivity, at most n+5)
        max_edges = n * (n - 1) // 2
        e = random.randint(n - 1, min(max_edges, n + 5))

        edges_set = set()
        nodes = list(range(n))
        while len(edges_set) < e:
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u == v:
                continue
            edge = tuple(sorted((u, v)))
            if edge in edges_set:
                continue
            edges_set.add(edge)

        self.edges = list(edges_set)

        # Build adjacency list
        self.adj = [[] for _ in range(n)]
        for u, v in self.edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        # Calculate chromatic number
        self.chromatic_number = self._calculate_chromatic_number(self.adj, e)
        m = self.chromatic_number

        # Build question text
        board = "Graph Coloring Problem:\n"
        board += "Nodes: " + ", ".join(str(i) for i in nodes) + "\n"
        board += "Edges (format: NodeA - NodeB):\n"
        for u, v in self.edges:
            board += f"{u} - {v}, "
        board = board.rstrip(", ") + "\n"
        board += (
            f"Question: Provide a valid coloring scheme for the graph using exactly {m} "
            f"colors (colors are numbered from 0 to {m-1}).\n"
            "The coloring scheme should be a list of pairs [node, color] for each node.\n"
            "Output format: 'Answer: [[0, 1], [1, 0], [2, 1],...]'"
        )

        observation = f"{self._get_instructions()}{board}"
        return observation, {"suffix": f"Provide a valid {m}-coloring."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's coloring scheme.

        Args:
            action: Agent's response containing the coloring scheme

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer using \boxed{} format first
        parsed_answer = extract_last_boxed_answer(action)

        # If not found, try "Answer:" format
        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(\[.+?\])(?:\n|$)', action, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: [[node, color], ...]' or \\boxed{[[node, color], ...]} format."
            )
            return obs, 0.0, True, False, {}

        # Try to parse as list
        try:
            coloring = ast.literal_eval(parsed_answer)
        except (ValueError, SyntaxError):
            obs = f"Failed to parse '{parsed_answer}' as a list."
            return obs, 0.0, True, False, {}

        # Validate format
        if not isinstance(coloring, list) or not all(isinstance(row, list) for row in coloring):
            obs = "Invalid format. Expected a list of [node, color] pairs."
            return obs, 0.0, True, False, {}

        # Check if chromatic number is minimal
        # (If we can color with m-1 colors, then m is not minimal)
        m = self.chromatic_number
        if m > 1 and self._is_k_colorable(self.adj, m - 1):
            obs = f"The chromatic number {m} is not minimal."
            return obs, 0.0, True, False, {}

        # Build color map and validate
        color_map = {}
        nodes_seen = set()
        n = self.num_nodes

        for entry in coloring:
            if not isinstance(entry, list) or len(entry) != 2:
                obs = f"Invalid entry format: {entry}. Expected [node, color]."
                return obs, 0.0, True, False, {}

            node, color = entry
            if not isinstance(node, int) or not isinstance(color, int):
                obs = f"Node and color must be integers. Got: {entry}"
                return obs, 0.0, True, False, {}

            if node < 0 or node >= n:
                obs = f"Invalid node {node}. Must be in range [0, {n-1}]."
                return obs, 0.0, True, False, {}

            if node in nodes_seen:
                obs = f"Duplicate node {node} in coloring."
                return obs, 0.0, True, False, {}

            nodes_seen.add(node)

            if color < 0 or color >= m:
                obs = f"Invalid color {color}. Must be in range [0, {m-1}]."
                return obs, 0.0, True, False, {}

            color_map[node] = color

        # Check all nodes are colored
        if len(nodes_seen) != n:
            obs = f"Not all nodes are colored. Expected {n} nodes, got {len(nodes_seen)}."
            return obs, 0.0, True, False, {}

        # Check adjacent nodes have different colors
        for u in range(n):
            for v in self.adj[u]:
                if u < v and color_map.get(u) == color_map.get(v):
                    obs = f"Adjacent nodes {u} and {v} have the same color {color_map[u]}."
                    return obs, 0.0, True, False, {}

        # Success!
        obs = f"Correct! Valid {m}-coloring provided."
        return obs, 1.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random valid coloring (for testing)
        """
        if self.adj is None or self.chromatic_number is None:
            return "\\boxed{[[0, 0]]}"

        # Use greedy coloring algorithm
        n = self.num_nodes
        m = self.chromatic_number
        colors = [-1] * n

        for node in range(n):
            used = set()
            for neighbor in self.adj[node]:
                if colors[neighbor] != -1:
                    used.add(colors[neighbor])
            # Assign smallest available color
            for color in range(m):
                if color not in used:
                    colors[node] = color
                    break

        coloring = [[i, colors[i]] for i in range(n)]
        return f"\\boxed{{{coloring}}}"
