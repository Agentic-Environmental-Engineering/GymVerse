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

"""Maze environment - Pathfinding puzzle game."""

import ast
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class MazeEnv(Env):
    """
    Maze pathfinding environment.

    Players are given a maze with a start point 'I' and end point 'X'.
    They must provide a sequence of moves (up, down, left, right) to
    navigate from start to end without hitting walls.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_size: int = 10,
        max_size: int = 30,
        **_,
    ):
        """
        Initialize Maze environment.

        Args:
            min_size: Minimum maze size (n×n grid)
            max_size: Maximum maze size (n×n grid)
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.char_maze = None
        self.start = None
        self.end = None

    def _prim_maze(self, size: Tuple[int, int]) -> np.ndarray:
        """
        Generate a maze using Prim's algorithm.

        Args:
            size: Base maze size

        Returns:
            Binary maze map (0 = path, 1 = wall)
        """
        # Reduce base size by half
        size = (size[0] // 2, size[1] // 2)

        # Initialize maze: [visited, up_wall, right_wall, down_wall, left_wall]
        maze = np.zeros((size[0], size[1], 5), dtype=np.uint8)
        maze[:, :, 1:] = 1  # All walls initially closed
        maze[0, 0, 0] = 1  # Mark start as visited

        # Frontier list
        memory = [[0, 0]]
        directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        while memory:
            # Choose random cell from frontier
            index = random.choice(memory)
            legal_directions = []

            # Check all directions
            for i, d in enumerate(directions):
                new_pos = [index[0] + d[0], index[1] + d[1]]
                if not (0 <= new_pos[0] < size[0] and 0 <= new_pos[1] < size[1]):
                    continue
                if maze[new_pos[0], new_pos[1], 0] == 1:
                    continue
                legal_directions.append(i)

            if legal_directions:
                # Pick random legal direction
                dire = random.choice(legal_directions)
                new_pos = [index[0] + directions[dire][0], index[1] + directions[dire][1]]

                # Check if new position is not in frontier
                is_new = all(
                    m[0] != new_pos[0] or m[1] != new_pos[1]
                    for m in memory
                )

                if is_new:
                    memory.append(new_pos)
                    # Remove walls between current and new cell
                    maze[index[0], index[1], dire + 1] = 0
                    maze[new_pos[0], new_pos[1], (dire + 2) % 4 + 1] = 0
                    maze[new_pos[0], new_pos[1], 0] = 1
                else:
                    memory.remove(index)
            else:
                memory.remove(index)

        # Convert to 2D map
        return self._prim_to_map(maze)

    def _prim_to_map(self, maze: np.ndarray) -> np.ndarray:
        """
        Convert 3D Prim maze to 2D binary map.

        Args:
            maze: 3D maze representation

        Returns:
            2D binary maze (0 = path, 1 = wall)
        """
        shape = maze.shape[:2]
        maze_map = np.ones((shape[0] * 2 - 1, shape[1] * 2 - 1), dtype=np.uint8)

        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if i % 2 == 0 and j % 2 == 0:
                    # Cell positions
                    maze_map[i, j] = 0
                elif i % 2 == 0 and j % 2 == 1:
                    # Horizontal walls
                    maze_map[i, j] = maze[i // 2, j // 2, 1] + maze[i // 2, j // 2 + 1, 3]
                elif i % 2 == 1 and j % 2 == 0:
                    # Vertical walls
                    maze_map[i, j] = maze[i // 2, j // 2, 2] + maze[i // 2 + 1, j // 2, 4]
                # Corners remain as walls (value 1)

        return maze_map

    def _init_maze(self, numeric_maze: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Select start and end positions for the maze.

        Args:
            numeric_maze: Binary maze matrix

        Returns:
            (start_position, end_position)
        """
        start = (0, 0)
        road = np.argwhere(numeric_maze == 0)
        # Choose farthest point as end
        end = tuple(road[np.argmax(np.sum(road * 2, axis=1))])
        return start, end

    def _convert_to_char_matrix(
        self,
        numeric_maze: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> List[List[str]]:
        """
        Convert binary maze to character matrix.

        Args:
            numeric_maze: Binary maze
            start: Start position
            end: End position

        Returns:
            Character maze with 'I' (start), 'X' (end), 'o' (path), '*' (wall)
        """
        char_maze = []
        for i in range(numeric_maze.shape[0]):
            row = []
            for j in range(numeric_maze.shape[1]):
                if (i, j) == start:
                    row.append('I')
                elif (i, j) == end:
                    row.append('X')
                else:
                    row.append('o' if numeric_maze[i, j] == 0 else '*')
            char_maze.append(row)
        return char_maze

    def _find_path(self) -> Optional[List[str]]:
        """
        Find a path from start to end using BFS.

        Returns:
            List of moves or None if no path exists
        """
        if self.char_maze is None or self.start is None or self.end is None:
            return None

        rows = len(self.char_maze)
        cols = len(self.char_maze[0])
        directions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        queue = deque([(self.start, [])])
        visited = {self.start}

        while queue:
            (r, c), path = queue.popleft()

            if (r, c) == self.end:
                return path

            for move_name, (dr, dc) in directions.items():
                nr, nc = r + dr, c + dc

                if (0 <= nr < rows and 0 <= nc < cols and
                    (nr, nc) not in visited and
                    self.char_maze[nr][nc] != '*'):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [move_name]))

        return None

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You need to provide a path from the start point to the end point "
            "based on an n×n maze map that I provide. Output your answer in the "
            "form of a list, where:\n"
            "'I' represents the starting point\n"
            "'X' represents the destination point\n"
            "'o' represents empty space (passable)\n"
            "'*' represents a wall (impassable)\n\n"
            "Your available moves are:\n"
            "'up': move one cell upwards\n"
            "'down': move one cell downwards\n"
            "'left': move one cell to the left\n"
            "'right': move one cell to the right\n\n"
            "You need to output your answer as a list of these strings.\n"
            "Use format: Answer: ['up','down','down',...] or \\boxed{['up','down',...]}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new maze.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate maze size
        n = random.randint(self.min_size, self.max_size)
        base_size = ((n + 1) // 2, (n + 1) // 2)

        # Generate maze using Prim's algorithm
        numeric_maze = self._prim_maze(base_size)

        # Set start and end
        self.start, self.end = self._init_maze(numeric_maze)

        # Convert to character maze
        self.char_maze = self._convert_to_char_matrix(numeric_maze, self.start, self.end)

        # Build question
        maze_str = "\n".join(["".join(row) for row in self.char_maze])
        observation = f"{self._get_instructions()}Maze Board:\n{maze_str}"

        return observation, {"suffix": f"Find path from start to goal ({n}×{n} maze)."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's path.

        Args:
            action: Agent's response containing the sequence of moves

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer using \boxed{} format first
        parsed_answer = extract_last_boxed_answer(action)

        # If not found, try "Answer:" format
        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(\[.+?\])', action, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: ['up','down',...]' or \\boxed{['up','down',...]} format."
            )
            return obs, 0.0, True, False, {}

        # Parse the move sequence
        try:
            moves = ast.literal_eval(parsed_answer)
        except Exception as e:
            obs = f"Failed to parse answer. Error: {e}"
            return obs, 0.0, True, False, {}

        if not isinstance(moves, list):
            obs = "Answer must be a list of direction strings."
            return obs, 0.0, True, False, {}

        # Simulate moves
        dir_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        rows = len(self.char_maze)
        cols = len(self.char_maze[0])
        current = self.start

        for i, move in enumerate(moves):
            if move not in dir_map:
                obs = f"Invalid move '{move}'. Must be 'up', 'down', 'left', or 'right'."
                return obs, 0.0, True, False, {}

            dr, dc = dir_map[move]
            nr, nc = current[0] + dr, current[1] + dc

            # Check bounds
            if not (0 <= nr < rows and 0 <= nc < cols):
                obs = f"Move #{i+1} '{move}' goes out of bounds from {current}."
                return obs, 0.0, True, False, {}

            # Check wall
            if self.char_maze[nr][nc] == '*':
                obs = f"Move #{i+1} '{move}' hits a wall at ({nr}, {nc})."
                return obs, 0.0, True, False, {}

            # Check if reached end early
            if (nr, nc) == self.end and i < len(moves) - 1:
                obs = f"Reached end at move #{i+1}, but there are {len(moves) - i - 1} extra moves."
                return obs, 0.0, True, False, {}

            current = (nr, nc)

        # Check if reached end
        if current == self.end:
            obs = f"Correct! Reached the end in {len(moves)} moves."
            return obs, 1.0, True, False, {}
        else:
            obs = f"Path ends at {current}, but goal is at {self.end}."
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            A valid path from start to end
        """
        path = self._find_path()
        if path:
            return f"\\boxed{{{path}}}"
        else:
            return "\\boxed{['right']}"
