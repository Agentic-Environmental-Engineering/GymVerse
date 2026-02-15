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

"""Pipe Game environment - Connect pipes from start to end."""

import random
import ast
from collections import deque, defaultdict
from typing import Optional, Tuple, Dict, Any, List, Set

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


# Direction constants
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIR_VEC = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


class PipeGameEnv(Env):
    """
    Pipe Game environment.

    Players must rotate pipes on a grid to create a path from the start
    (left of position (0,0)) to the end (right of position (n-1,n-1)).

    Three pipe types:
    - 'L': connects top and right sides
    - '|': connects top and bottom sides
    - '┏': connects top, left, and right sides

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_size: int = 4,
        max_size: int = 6,
        **_,
    ):
        """
        Initialize Pipe Game environment.

        Args:
            min_size: Minimum grid size
            max_size: Maximum grid size
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.puzzle_grid = None
        self.solution = None
        self.grid_size = None

    def _generate_puzzle(self, n: int, seed: int) -> Tuple[List[List[str]], List[List[int]]]:
        """
        Generate pipe puzzle using DFS.

        Args:
            n: Grid size
            seed: Random seed

        Returns:
            Tuple of (puzzle_grid, solution)
        """
        random.seed(seed)
        start = (0, 0)
        end = (n - 1, n - 1)

        # Initialize connectivity
        connected = {(i, j): set() for i in range(n) for j in range(n)}
        deg = {(i, j): 0 for i in range(n) for j in range(n)}
        visited = [[False] * n for _ in range(n)]

        # Mark end as visited initially to prevent early connection
        visited[end[0]][end[1]] = True

        def dfs(cell, parent=None):
            i, j = cell
            visited[i][j] = True
            dirs = [UP, RIGHT, DOWN, LEFT]
            random.shuffle(dirs)

            for d in dirs:
                ni, nj = i + DIR_VEC[d][0], j + DIR_VEC[d][1]
                if 0 <= ni < n and 0 <= nj < n and not visited[ni][nj]:
                    connected[cell].add((ni, nj))
                    connected[(ni, nj)].add(cell)
                    deg[cell] += 1
                    deg[(ni, nj)] += 1
                    dfs((ni, nj), cell)

            # For non-start/end leaf nodes, try to connect to visited neighbor
            if cell not in (start, end) and parent is not None and deg[cell] == 1:
                dirs2 = [UP, RIGHT, DOWN, LEFT]
                random.shuffle(dirs2)
                for d in dirs2:
                    ni, nj = i + DIR_VEC[d][0], j + DIR_VEC[d][1]
                    if (0 <= ni < n and 0 <= nj < n and visited[ni][nj] and
                        (ni, nj) != parent and (ni, nj) not in connected[cell] and
                        deg[(ni, nj)] < 3):
                        connected[cell].add((ni, nj))
                        connected[(ni, nj)].add(cell)
                        deg[cell] += 1
                        deg[(ni, nj)] += 1
                        break

        dfs(start)

        # Connect end
        visited[end[0]][end[1]] = False
        for d in [UP, RIGHT, DOWN, LEFT]:
            ni, nj = end[0] + DIR_VEC[d][0], end[1] + DIR_VEC[d][1]
            if 0 <= ni < n and 0 <= nj < n and visited[ni][nj] and deg[(ni, nj)] < 3:
                connected[end].add((ni, nj))
                connected[(ni, nj)].add(end)
                deg[end] += 1
                deg[(ni, nj)] += 1
                break
        visited[end[0]][end[1]] = True

        # Connect isolated cells
        for i in range(n):
            for j in range(n):
                if not visited[i][j]:
                    for d in [UP, RIGHT, DOWN, LEFT]:
                        ni, nj = i + DIR_VEC[d][0], j + DIR_VEC[d][1]
                        if 0 <= ni < n and 0 <= nj < n and visited[ni][nj]:
                            connected[(i, j)].add((ni, nj))
                            connected[(ni, nj)].add((i, j))
                            deg[(i, j)] += 1
                            deg[(ni, nj)] += 1
                            break
                    visited[i][j] = True

        # Determine pipe type and rotation for each cell
        board = [[None] * n for _ in range(n)]
        solution = [[0 for _ in range(n)] for _ in range(n)]

        # Default pipe openings
        default_opens = {
            '|': {UP, DOWN},
            'L': {UP, RIGHT},
            '┏': {UP, RIGHT, LEFT}
        }

        for i in range(n):
            for j in range(n):
                cell = (i, j)
                opens = set()

                # Start connects to left
                if cell == start:
                    opens.add(LEFT)
                # End connects to right
                if cell == end:
                    opens.add(RIGHT)

                for (ni, nj) in connected[cell]:
                    if ni == i - 1 and nj == j:
                        opens.add(UP)
                    elif ni == i + 1 and nj == j:
                        opens.add(DOWN)
                    elif ni == i and nj == j - 1:
                        opens.add(LEFT)
                    elif ni == i and nj == j + 1:
                        opens.add(RIGHT)

                # Determine pipe type based on connections
                if len(opens) == 3:
                    piece_type = '┏'
                elif len(opens) == 2:
                    if opens == {UP, DOWN} or opens == {LEFT, RIGHT}:
                        piece_type = '|'
                    else:
                        piece_type = 'L'
                elif len(opens) == 1:
                    piece_type = 'L'
                    if cell == start and opens == {RIGHT}:
                        piece_type = '|'
                    if cell == end and opens == {LEFT}:
                        piece_type = '|'
                else:
                    piece_type = ' '

                board[i][j] = piece_type

                # Calculate rotation needed
                if piece_type in default_opens:
                    base_opens = default_opens[piece_type]
                    # Try each rotation (0, 1, 2, 3) to find match
                    for rot in range(4):
                        rotated_opens = {(d + rot) % 4 for d in base_opens}
                        if rotated_opens == opens:
                            solution[i][j] = rot
                            break

        return board, solution

    def _verify_solution(self, board: List[List[str]], answer: List[List[int]]) -> bool:
        """Verify if pipe configuration creates valid path using BFS."""
        n = len(board)

        # Default openings for each pipe type
        default_opens = {
            '|': {UP, DOWN},
            'L': {UP, RIGHT},
            '┏': {UP, RIGHT, LEFT}
        }

        def get_opens(i: int, j: int, rot: int) -> Set[int]:
            """Get pipe openings after rotation."""
            piece = board[i][j]
            if piece not in default_opens:
                return set()
            # Each 90° clockwise rotation adds 1 to all directions (mod 4)
            return {(d + rot) % 4 for d in default_opens[piece]}

        queue = deque()
        visited = set()

        # Start if pipe at (0,0) opens to left
        start_opens = get_opens(0, 0, answer[0][0])
        if LEFT in start_opens:
            queue.append((0, 0, LEFT))
            visited.add((0, 0, LEFT))

        while queue:
            i, j, src_dir = queue.popleft()
            opens = get_opens(i, j, answer[i][j])

            # Remove source direction to prevent backflow
            if src_dir in opens:
                opens = opens - {src_dir}

            for d in opens:
                # Check if reached end
                if i == n - 1 and j == n - 1 and d == RIGHT:
                    return True

                # Calculate next cell
                if d == UP and i - 1 >= 0:
                    ni, nj, incoming = i - 1, j, DOWN
                elif d == RIGHT and j + 1 < n:
                    ni, nj, incoming = i, j + 1, LEFT
                elif d == DOWN and i + 1 < n:
                    ni, nj, incoming = i + 1, j, UP
                elif d == LEFT and j - 1 >= 0:
                    ni, nj, incoming = i, j - 1, RIGHT
                else:
                    continue

                neighbor_opens = get_opens(ni, nj, answer[ni][nj])
                if incoming in neighbor_opens:
                    state = (ni, nj, incoming)
                    if state not in visited:
                        visited.add(state)
                        queue.append((ni, nj, incoming))

        return False

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: [[0,1,1,3...],[1,3,2,3...],...]'.\n\n"
            "Alternatively, you can use \\boxed{[[0,1,1,3...],[1,3,2,3...],...]]} format.\n\n"
            "Given three types of pipes with the following initial connections:\n"
            "- L connects the top and right sides\n"
            "- | connects the top and bottom sides\n"
            "- ┏ connects the top, left, and right sides\n\n"
            "You are provided with an n x n grid, where each cell contains one type of pipe. "
            "The starting point is to the left of position (0,0), and the goal is to reach the "
            "right side of position (n-1,n-1). Players need to rotate the pipes in the grid to "
            "ensure a valid connection from the start to the end.\n\n"
            "Your task is to output an n x n list in one line, where each element indicates the "
            "number of 90° clockwise rotations (0, 1, 2, or 3) applied to the pipe at that position.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new Pipe Game puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        self.grid_size = random.randint(self.min_size, self.max_size)
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        self.puzzle_grid, self.solution = self._generate_puzzle(self.grid_size, puzzle_seed)

        # Build observation
        board_str = "\n".join(" ".join(str(cell) for cell in row) for row in self.puzzle_grid)
        observation = f"{self._get_instructions()}Board:\n{board_str}"

        return observation, {
            "suffix": f"Grid size: {self.grid_size}x{self.grid_size}."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Verify the pipe rotation solution.

        Args:
            action: Rotation matrix as 2D list string

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        parsed_action = extract_last_boxed_answer(action)

        if parsed_action is None:
            import re
            match = re.search(r'Answer:\s*(.+)', action, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_action = match.group(1).strip()

        if parsed_action is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: [[...], [...], ...]' or \\boxed{[[...], [...], ...]} format."
            )
            return obs, 0.0, True, False, {}

        # Parse rotation matrix
        try:
            if isinstance(parsed_action, str):
                user_answer = ast.literal_eval(parsed_action)
            else:
                user_answer = parsed_action
        except Exception as e:
            obs = f"Failed to parse answer: {parsed_action}. Error: {e}"
            return obs, 0.0, True, False, {}

        # Check format
        n = self.grid_size
        if not (isinstance(user_answer, list) and len(user_answer) == n and
                all(isinstance(row, list) and len(row) == n for row in user_answer)):
            obs = (
                f"Answer size mismatch. Expected {n}x{n}, "
                f"got {len(user_answer)}x{len(user_answer[0]) if user_answer else 0}."
            )
            return obs, 0.0, True, False, {}

        # Verify solution
        if self._verify_solution(self.puzzle_grid, user_answer):
            obs = (
                "Correct! The pipes are connected from start to end.\n"
                f"Your solution:\n{user_answer}"
            )
            reward = 1.0
        else:
            obs = (
                "Incorrect. The pipes do not form a valid path from start to end.\n"
                f"Your answer:\n{user_answer}"
            )
            reward = 0.0

        return obs, reward, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample the correct solution.

        Returns:
            Correct solution as string
        """
        return f"\\boxed{{{self.solution}}}"
