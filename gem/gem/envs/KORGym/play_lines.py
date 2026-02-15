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

"""PlayLines environment - Flow Free puzzle game."""

import ast
import copy
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class PlayLinesEnv(Env):
    """
    PlayLines (Flow Free) environment.

    Players are given a grid with numbered endpoints and must connect
    matching numbers with non-branching paths that fill the entire grid.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 10,
        min_colors: int = 3,
        max_colors: int = 5,
        min_walls: int = 1,
        max_walls: int = 3,
        **_,
    ):
        """
        Initialize PlayLines environment.

        Args:
            min_size: Minimum grid size (n×n)
            max_size: Maximum grid size (n×n)
            min_colors: Minimum number of color pairs
            max_colors: Maximum number of color pairs
            min_walls: Minimum number of wall cells
            max_walls: Maximum number of wall cells
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.min_colors = min_colors
        self.max_colors = max_colors
        self.min_walls = min_walls
        self.max_walls = max_walls
        self.puzzle_grid = None
        self.endpoints = None
        self.grid_size = None
        self.solution_grid = None

    def _generate_endpoints(
        self,
        grid_size: int,
        num_colors: int,
        num_walls: int
    ) -> Tuple[Optional[List[List]], Optional[Dict]]:
        """
        Generate puzzle endpoints and wall placements.

        Args:
            grid_size: Size of grid
            num_colors: Number of color pairs
            num_walls: Number of wall cells

        Returns:
            Tuple of (grid, endpoints) or (None, None) if failed
        """
        grid = [['E' for _ in range(grid_size)] for _ in range(grid_size)]
        endpoints = {}

        # Shuffle all positions
        all_positions = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        random.shuffle(all_positions)

        # Place endpoints for each color
        for color in range(1, num_colors + 1):
            if len(all_positions) < 2:
                return None, None

            pos1 = all_positions.pop()

            # Find a non-adjacent position for second endpoint
            pos2 = None
            for i in range(len(all_positions)):
                candidate = all_positions[i]
                # Check if not adjacent
                if abs(candidate[0] - pos1[0]) + abs(candidate[1] - pos1[1]) > 1:
                    pos2 = candidate
                    all_positions.pop(i)
                    break

            if pos2 is None:
                return None, None

            grid[pos1[0]][pos1[1]] = color
            grid[pos2[0]][pos2[1]] = color
            endpoints[color] = (pos1, pos2)

        # Place walls
        wall_positions = random.sample(all_positions, min(num_walls, len(all_positions)))
        for (x, y) in wall_positions:
            grid[x][y] = 'X'

        return grid, endpoints

    def _bfs_path(
        self,
        grid: List[List],
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: int,
        allow_empty: bool = True
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find a path between two endpoints using BFS.

        Args:
            grid: Current grid state
            start: Start position
            end: End position
            color: Color to connect
            allow_empty: Whether to allow traversal through empty cells

        Returns:
            List of coordinates forming path, or None if no path exists
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        grid_size = len(grid)

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path

            x, y = current
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    neighbor = (nx, ny)
                    if neighbor not in visited:
                        cell = grid[nx][ny]
                        if cell != 'X' and (cell == color or (allow_empty and cell == 'E')):
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))

        return None

    def _compute_solution(self) -> Optional[List[List]]:
        """
        Compute a complete solution for the puzzle.

        Returns:
            Solution grid or None if no solution exists
        """
        if self.puzzle_grid is None or self.endpoints is None:
            return None

        # Copy puzzle grid
        sol_grid = copy.deepcopy(self.puzzle_grid)

        # Sort colors by Manhattan distance (longest first)
        def manhattan(color):
            (x1, y1), (x2, y2) = self.endpoints[color]
            return abs(x1 - x2) + abs(y1 - y2)

        colors = sorted(self.endpoints.keys(), key=manhattan, reverse=True)

        # Connect each color
        for color in colors:
            start, end = self.endpoints[color]
            path = self._bfs_path(sol_grid, start, end, color, allow_empty=True)
            if path is None:
                return None

            # Fill path with color
            for (x, y) in path:
                sol_grid[x][y] = color

        # Extend paths to fill empty cells
        sol_grid = self._extend_paths(sol_grid)

        # Check if fully filled
        if any('E' in row for row in sol_grid):
            return None

        # Check no branching
        if not self._check_no_branching(sol_grid):
            return None

        return sol_grid

    def _extend_paths(self, grid: List[List]) -> List[List]:
        """
        Extend color paths into empty cells where extension is deterministic.

        Args:
            grid: Partially filled grid

        Returns:
            Updated grid
        """
        grid_size = len(grid)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        changed = True

        while changed:
            changed = False

            # Find current endpoints for each color
            current_endpoints = {}
            for color in self.endpoints.keys():
                current_endpoints[color] = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        if grid[i][j] == color:
                            # Count same-color neighbors
                            count = sum(
                                1 for dx, dy in directions
                                if 0 <= i + dx < grid_size and 0 <= j + dy < grid_size
                                and grid[i + dx][j + dy] == color
                            )
                            if count == 1:
                                current_endpoints[color].append((i, j))

            # Try to extend each endpoint
            for color, eps in current_endpoints.items():
                for ep in eps:
                    i, j = ep
                    candidates = []

                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            if grid[ni][nj] == 'E':
                                # Check if filling with color would create valid connection
                                count_same = sum(
                                    1 for ddx, ddy in directions
                                    if 0 <= ni + ddx < grid_size and 0 <= nj + ddy < grid_size
                                    and grid[ni + ddx][nj + ddy] == color
                                )
                                if count_same == 1:
                                    candidates.append((ni, nj))

                    # Only extend if there's exactly one candidate
                    if len(candidates) == 1:
                        grid[candidates[0][0]][candidates[0][1]] = color
                        changed = True

        return grid

    def _check_no_branching(self, grid: List[List]) -> bool:
        """
        Check that all color paths form non-branching single lines.

        Args:
            grid: Grid to validate

        Returns:
            True if all paths are valid
        """
        grid_size = len(grid)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for color, (start, end) in self.endpoints.items():
            # Get all cells of this color
            cells = [
                (i, j) for i in range(grid_size) for j in range(grid_size)
                if grid[i][j] == color
            ]

            endpoint_count = 0
            for i, j in cells:
                # Count same-color neighbors
                count = sum(
                    1 for dx, dy in directions
                    if 0 <= i + dx < grid_size and 0 <= j + dy < grid_size
                    and grid[i + dx][j + dy] == color
                )

                # Endpoints should have exactly 1 neighbor
                if (i, j) == start or (i, j) == end:
                    if count != 1:
                        return False
                    endpoint_count += 1
                # Internal cells should have exactly 2 neighbors
                else:
                    if count != 2:
                        return False

            # Must have exactly 2 endpoints
            if endpoint_count != 2:
                return False

        return True

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., \"Answer: [['E','X','E',...],['E','1','1',...]...]\"\n\n"
            "Alternatively, you can use \\boxed{[[...]]} format.\n\n"
            "Next, I will provide an n×n chessboard. On the chessboard, 'E' indicates that the "
            "element is an empty space, 'X' indicates a node that cannot be passed through, and "
            "numbers indicate nodes that need to be connected. You need to fill in the numbers on "
            "the empty spaces of the chessboard so that all identical numbers on the chessboard are "
            "connected. Moreover, the final chessboard must not have any empty spaces; every cell "
            "must be filled with a number (or remain 'X' if it's an impassable cell). Importantly, "
            "the connection for each color must form a single continuous line without branching.\n\n"
            "For example, if the initial chessboard is:\n"
            "E E E E E\n"
            "E X E 3 E\n"
            "E 3 E 1 E\n"
            "E 2 E E E\n"
            "1 E E 2 E\n\n"
            "The filled chessboard could be:\n"
            "2 2 2 2 2\n"
            "2 X 3 3 2\n"
            "2 3 3 1 2\n"
            "2 2 1 1 2\n"
            "1 1 1 2 2\n\n"
            "When all the numbers on the chessboard are connected, it is considered a game victory.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new PlayLines puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle parameters
        max_attempts = 10000

        for attempt in range(max_attempts):
            self.grid_size = random.randint(self.min_size, self.max_size)
            num_colors = random.randint(self.min_colors, self.max_colors)
            num_walls = random.randint(self.min_walls, self.max_walls)

            # Generate endpoints
            self.puzzle_grid, self.endpoints = self._generate_endpoints(
                self.grid_size, num_colors, num_walls
            )

            if self.puzzle_grid is None:
                continue

            # Try to solve
            self.solution_grid = self._compute_solution()

            if self.solution_grid is not None:
                break

        if self.solution_grid is None:
            raise RuntimeError("Failed to generate valid puzzle after many attempts")

        # Build observation
        board_str = "\n".join(
            ["".join(str(cell) for cell in row) for row in self.puzzle_grid]
        )
        observation = f"{self._get_instructions()}Board:\n{board_str}"

        return observation, {
            "suffix": f"Connect all {len(self.endpoints)} color pairs ({self.grid_size}×{self.grid_size} grid)."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's solution.

        Args:
            action: Agent's response containing the filled grid

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer
        parsed_answer = extract_last_boxed_answer(action)

        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(\[\[.+?\]\])', action, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: [[...]]' or \\boxed{[[...]]} format."
            )
            return obs, 0.0, True, False, {}

        # Parse grid
        try:
            sol_grid = ast.literal_eval(parsed_answer)
        except Exception as e:
            obs = f"Failed to parse answer. Error: {e}"
            return obs, 0.0, True, False, {}

        if not isinstance(sol_grid, list) or not all(isinstance(row, list) for row in sol_grid):
            obs = "Answer must be a 2D list (grid)."
            return obs, 0.0, True, False, {}

        # Check each color is connected
        for color, (start, end) in self.endpoints.items():
            path = self._bfs_path(sol_grid, start, end, color, allow_empty=False)
            if path is None:
                obs = f"Color {color} is not connected from {start} to {end}."
                return obs, 0.0, True, False, {}

        # Check no empty cells
        for row in sol_grid:
            if 'E' in row:
                obs = "Grid still contains empty cells 'E'. All cells must be filled."
                return obs, 0.0, True, False, {}

        # Check no branching
        if not self._check_no_branching(sol_grid):
            obs = "Some color paths have branching. Each path must be a single line."
            return obs, 0.0, True, False, {}

        obs = f"Correct! All {len(self.endpoints)} colors connected without branching."
        return obs, 1.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            The computed solution
        """
        if self.solution_grid is not None:
            return f"\\boxed{{{self.solution_grid}}}"
        else:
            return "\\boxed{[[1]]}"
