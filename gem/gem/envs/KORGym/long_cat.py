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

"""LongCat environment - Sliding cat puzzle game."""

import ast
import copy
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any, List, Set

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class LongCatEnv(Env):
    """
    LongCat sliding puzzle environment.

    The cat starts at position 'C', and slides in one direction until hitting a wall ('X').
    All empty spaces ('E') traversed along the path turn into walls. The goal is to fill
    all empty spaces by controlling the cat's movement.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 10,
        **_,
    ):
        """
        Initialize LongCat environment.

        Args:
            min_size: Minimum grid size (5-10)
            max_size: Maximum grid size (5-10)
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.game_map = None
        self.rows = None
        self.cols = None

    def _init_map(self, rows: int, cols: int) -> List[List[str]]:
        """Initialize map with walls on borders and empty cells inside."""
        return [['X' if r == 0 or r == rows - 1 or c == 0 or c == cols - 1 else 'E'
                 for c in range(cols)] for r in range(rows)]

    def _get_neighbors(self, game_map: List[List[str]], r: int, c: int, cell_type: str) -> List[Tuple[int, int]]:
        """Get all 4-directional neighbors of a given cell that match a specific type."""
        rows = len(game_map)
        cols = len(game_map[0])
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and game_map[nr][nc] == cell_type:
                neighbors.append((nr, nc))
        return neighbors

    def _bfs_connectivity(self, game_map: List[List[str]], start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Use BFS to determine the connected 'E' area from a starting cell."""
        visited = set()
        q = deque([start])
        visited.add(start)

        while q:
            r, c = q.popleft()
            for nr, nc in self._get_neighbors(game_map, r, c, 'E'):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return visited

    def _add_random_walls(self, game_map: List[List[str]], rows: int, cols: int) -> None:
        """Randomly add a few internal walls while maintaining connectivity."""
        internal_cells = [(r, c) for r in range(1, rows-1)
                          for c in range(1, cols-1) if game_map[r][c] == 'E']
        num_walls = len(internal_cells) // 5

        for _ in range(num_walls):
            if not internal_cells:
                break
            r, c = random.choice(internal_cells)
            original = game_map[r][c]
            game_map[r][c] = 'X'

            # Check connectivity
            e_cells = [(r, c) for r in range(rows) for c in range(cols) if game_map[r][c] == 'E']
            if e_cells:
                visited = self._bfs_connectivity(game_map, e_cells[0])
                if len(visited) != len(e_cells):
                    game_map[r][c] = original

    def _place_cat(self, game_map: List[List[str]], rows: int, cols: int) -> None:
        """Place the cat on a cell, preferably one with only one adjacent 'E'."""
        leaves = []
        all_e = []
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                if game_map[r][c] == 'E':
                    all_e.append((r, c))
                    if len(self._get_neighbors(game_map, r, c, 'E')) == 1:
                        leaves.append((r, c))

        if leaves:
            cat_r, cat_c = random.choice(leaves)
        elif all_e:
            cat_r, cat_c = random.choice(all_e)
        else:
            raise RuntimeError("No empty cells available to place cat")

        game_map[cat_r][cat_c] = 'C'

    def _is_solvable(self, game_map: List[List[str]]) -> bool:
        """Determine if the board is solvable using DFS."""
        rows = len(game_map)
        cols = len(game_map[0])

        # Find cat position
        cat_pos = None
        board = [row.copy() for row in game_map]
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'C':
                    cat_pos = (r, c)
                    board[r][c] = 'X'
                    break
            if cat_pos:
                break

        if not cat_pos:
            return False

        directions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        def board_to_key(board, cat_pos):
            return (cat_pos, tuple(tuple(row) for row in board))

        visited = {}

        def dfs(board, cat_pos):
            key = board_to_key(board, cat_pos)
            if key in visited:
                return visited[key]

            # Check if all cells are filled
            if all(cell != 'E' for row in board for cell in row):
                visited[key] = True
                return True

            for dr, dc in directions.values():
                r, c = cat_pos
                path = []
                while True:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        break
                    if board[nr][nc] == 'X':
                        break
                    if board[nr][nc] == 'E':
                        path.append((nr, nc))
                    r, c = nr, nc

                if not path:
                    continue

                new_board = [list(row) for row in board]
                for pr, pc in path:
                    new_board[pr][pc] = 'X'
                new_cat_pos = path[-1]

                if dfs(new_board, new_cat_pos):
                    visited[key] = True
                    return True

            visited[key] = False
            return False

        return dfs(board, cat_pos)

    def _format_board(self, game_map: List[List[str]]) -> str:
        """Format board as string."""
        return "\n".join([" ".join(row) for row in game_map])

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: ['left', 'down', 'right', 'up', 'left']'\n\n"
            "Alternatively, you can use \\boxed{['left', 'down', 'right']} format.\n\n"
            "Next, I will provide an n × n board containing a cat ('C'), empty spaces ('E'), and walls ('X'). "
            "You need to control the cat's movement by entering directions: up, down, left, or right. "
            "The cat moves from its initial position, sliding continuously in the chosen direction until hitting a wall. "
            "All empty spaces ('E') traversed along the path will turn into walls ('X'). "
            "The game is won when all empty spaces have been filled. "
            "Please output your solution as a list containing directions ('up', 'left', 'right', 'down').\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new LongCat puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate random grid size
        self.rows = random.randint(self.min_size, self.max_size)
        self.cols = random.randint(self.min_size, self.max_size)

        # Generate solvable puzzle
        max_attempts = 100
        for attempt in range(max_attempts):
            self.game_map = self._init_map(self.rows, self.cols)
            self._add_random_walls(self.game_map, self.rows, self.cols)
            self._place_cat(self.game_map, self.rows, self.cols)

            if self._is_solvable(self.game_map):
                break

        # Build question
        board_str = self._format_board(self.game_map)
        question = f"Board:\n{board_str}"

        observation = f"{self._get_instructions()}{question}"
        return observation, {"suffix": f"Fill all empty spaces ({self.rows}x{self.cols} grid)."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's solution.

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
                "Please use 'Answer: ['direction', ...]' or \\boxed{['direction', ...]} format."
            )
            return obs, 0.0, True, False, {}

        # Parse the sequence of moves
        try:
            moves = ast.literal_eval(parsed_answer)
        except Exception as e:
            obs = f"Failed to parse answer. Error: {e}"
            return obs, 0.0, True, False, {}

        if not isinstance(moves, list):
            obs = f"Answer must be a list of directions."
            return obs, 0.0, True, False, {}

        # Simulate the moves
        current_map = [row.copy() for row in self.game_map]

        # Find cat position
        cat_pos = None
        for r in range(self.rows):
            for c in range(self.cols):
                if current_map[r][c] == 'C':
                    cat_pos = (r, c)
                    current_map[r][c] = 'X'
                    break
            if cat_pos:
                break

        if not cat_pos:
            obs = "Error: Cat not found on board."
            return obs, 0.0, True, False, {}

        directions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        for move in moves:
            move_lower = move.lower() if isinstance(move, str) else str(move).lower()
            if move_lower not in directions:
                continue

            dr, dc = directions[move_lower]
            current_r, current_c = cat_pos
            path = []

            while True:
                next_r = current_r + dr
                next_c = current_c + dc
                if not (0 <= next_r < self.rows and 0 <= next_c < self.cols):
                    break
                if current_map[next_r][next_c] == 'X':
                    break
                if current_map[next_r][next_c] == 'E':
                    path.append((next_r, next_c))
                current_r, current_c = next_r, next_c

            if not path:
                continue

            for r, c in path:
                current_map[r][c] = 'X'
            cat_pos = path[-1]

        # Check if all empty spaces are filled
        all_filled = all(cell != 'E' for row in current_map for cell in row)

        if all_filled:
            obs = f"Correct! All empty spaces are filled using {len(moves)} moves: {moves}"
            return obs, 1.0, True, False, {}
        else:
            remaining = sum(1 for row in current_map for cell in row if cell == 'E')
            obs = f"Incorrect. After your {len(moves)} moves, {remaining} empty spaces remain."
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            A valid solution using DFS solver
        """
        if self.game_map is None:
            return f"\\boxed{{['right']}}"

        # Use DFS to find a solution
        solution = self._find_solution()
        if solution:
            return f"\\boxed{{{solution}}}"
        else:
            return f"\\boxed{{['right', 'down', 'left', 'up']}}"

    def _find_solution(self) -> Optional[List[str]]:
        """Find a solution using DFS."""
        rows = len(self.game_map)
        cols = len(self.game_map[0])

        # Find cat position
        cat_pos = None
        board = [row.copy() for row in self.game_map]
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'C':
                    cat_pos = (r, c)
                    board[r][c] = 'X'
                    break
            if cat_pos:
                break

        if not cat_pos:
            return None

        directions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        direction_names = ['up', 'down', 'left', 'right']

        def board_to_key(board, cat_pos):
            return (cat_pos, tuple(tuple(row) for row in board))

        visited = set()

        def dfs(board, cat_pos, moves):
            key = board_to_key(board, cat_pos)
            if key in visited:
                return None
            visited.add(key)

            # Check if all cells are filled
            if all(cell != 'E' for row in board for cell in row):
                return moves

            for i, (dr, dc) in enumerate(directions.values()):
                r, c = cat_pos
                path = []
                while True:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        break
                    if board[nr][nc] == 'X':
                        break
                    if board[nr][nc] == 'E':
                        path.append((nr, nc))
                    r, c = nr, nc

                if not path:
                    continue

                new_board = [list(row) for row in board]
                for pr, pc in path:
                    new_board[pr][pc] = 'X'
                new_cat_pos = path[-1]

                result = dfs(new_board, new_cat_pos, moves + [direction_names[i]])
                if result is not None:
                    return result

            return None

        solution = dfs(board, cat_pos, [])
        return solution
