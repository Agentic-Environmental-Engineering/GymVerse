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

"""CircleTheCat environment - Hexagonal board cat trapping game."""

import random
from copy import deepcopy
from collections import deque
from typing import Optional, Tuple, Dict, Any, List, Set

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class CircleTheCatEnv(Env):
    """
    CircleTheCat environment.

    A hexagonal board game where the player must trap the cat by placing walls.
    The cat starts at the center and tries to reach any boundary cell (exit).
    Players and the cat take turns alternately.

    This is a multi-turn environment with sparse terminal reward.
    """

    def __init__(
        self,
        board_size: int = 11,
        wall_density: float = 0.15,
        max_turns: int = 50,
        **_,
    ):
        """
        Initialize CircleTheCat environment.

        Args:
            board_size: Size of the hexagonal board
            wall_density: Initial random wall density
            max_turns: Maximum number of turns
        """
        super().__init__()
        self.board_size = board_size
        self.wall_density = wall_density
        self.max_turns = max_turns
        self.board = None
        self.cat = None
        self.walls = None
        self.turn = None

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get neighbors in hexagonal grid."""
        x, y = pos
        neighbors = []

        if x % 2 == 1:  # Odd row
            offsets = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:  # Even row
            offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                neighbors.append((nx, ny))
        return neighbors

    def _is_boundary(self, pos: Tuple[int, int]) -> bool:
        """Check if position is on boundary (exit)."""
        x, y = pos
        return x == 0 or x == self.board_size - 1 or y == 0 or y == self.board_size - 1

    def _calculate_distance(self, pos: Tuple[int, int], temp_walls: Set[Tuple[int, int]]) -> float:
        """Calculate shortest distance to boundary using BFS."""
        visited = set()
        q = deque([(pos[0], pos[1], 0)])
        visited.add(pos)

        while q:
            x, y, dist = q.popleft()
            if self._is_boundary((x, y)):
                return dist + 1

            for nx, ny in self._get_neighbors((x, y)):
                if (nx, ny) not in visited and (nx, ny) not in temp_walls and (nx, ny) != self.cat:
                    visited.add((nx, ny))
                    q.append((nx, ny, dist + 1))

        return float('inf')

    def _find_best_cat_move(self) -> Optional[Tuple[int, int]]:
        """Find best move for cat using minimax strategy."""
        current_pos = self.cat
        best_move = None
        min_max_distance = float('inf')

        neighbors = self._get_neighbors(current_pos)
        valid_moves = [n for n in neighbors if self.board[n[0]][n[1]] == '0']

        if not valid_moves:
            return None  # Cat is trapped

        for move in valid_moves:
            temp_walls = set(self.walls)
            temp_cat = move

            # If can reach boundary immediately
            if self._is_boundary(move):
                return move

            # Evaluate all possible player responses
            max_distance = 0
            possible_walls = [
                (i, j) for i in range(self.board_size) for j in range(self.board_size)
                if self.board[i][j] == '0' and (i, j) != temp_cat
            ]

            if not possible_walls:
                curr_dist = self._calculate_distance(temp_cat, temp_walls)
                if curr_dist < min_max_distance:
                    min_max_distance = curr_dist
                    best_move = move
                continue

            for wall in possible_walls:
                new_walls = temp_walls | {wall}
                dist = self._calculate_distance(temp_cat, new_walls)
                if dist > max_distance:
                    max_distance = dist

            if max_distance < min_max_distance:
                min_max_distance = max_distance
                best_move = move

        return best_move if best_move is not None else (valid_moves[0] if valid_moves else None)

    def _generate_puzzle(self, seed: int) -> Tuple[List[List[str]], Tuple[int, int], Set[Tuple[int, int]]]:
        """Generate initial game board."""
        random.seed(seed)
        board = [['0' for _ in range(self.board_size)] for _ in range(self.board_size)]
        center = self.board_size // 2
        cat = (center, center)
        board[center][center] = 'C'

        # Generate random walls
        walls = set()
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i, j) == cat:
                    continue
                if random.random() < self.wall_density:
                    board[i][j] = '1'
                    walls.add((i, j))

        return board, cat, walls

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: 5 3'.\n\n"
            "Alternatively, you can use \\boxed{5 3} format.\n\n"
            "Below is a hexagonal board represented in a textual grid. Each cell is labeled with a "
            "character. 'E' stands for an exit, '1' stands for a wall, 'C' stands for the cat, "
            "and '0' stands for an empty space. Although shown as a grid, each row in a hex grid "
            "is slightly offset from its neighbors, and each cell has up to six neighbors.\n\n"
            "Specifically, for a cell at coordinates (r, c): if r is even, the adjacent cells are "
            "(r-1, c), (r-1, c+1), (r, c-1), (r, c+1), (r+1, c), and (r+1, c+1). If r is odd, "
            "the adjacent cells are (r-1, c-1), (r-1, c), (r, c-1), (r, c+1), (r+1, c-1), and (r+1, c).\n\n"
            "In this game, the cat ('C') aims to reach any exit ('E') on the boundary. You aim to "
            "trap the cat by placing walls ('1') so that it can no longer move to an exit. You and "
            "the cat take turns. On the cat's turn, it moves to an adjacent empty cell ('0') if "
            "possible. On your turn, you place a wall on a currently empty cell ('0'), but not on "
            "an exit cell ('E'). If the cat reaches an exit ('E'), you lose. If the cat cannot "
            "move and is not on an exit, you win.\n\n"
            "Your task is to first give your move according to the current board and rules. Then, "
            "output the move in the required format. The last line of your response should be "
            "'Answer: X Y', where (X, Y) is the coordinate where you choose to place a wall.\n\n"
        )

    def _format_board(self) -> str:
        """Format the game board as string."""
        output = ""
        for i in range(self.board_size):
            for j in range(self.board_size):
                # Show exits on boundary
                if self.board[i][j] == '0' and self._is_boundary((i, j)):
                    output += 'E'
                else:
                    output += self.board[i][j]
            output += '\n'
        return output

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new CircleTheCat game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        self.board, self.cat, self.walls = self._generate_puzzle(puzzle_seed)
        self.turn = 0

        # Build observation
        board_str = self._format_board()
        observation = f"{self._get_instructions()}The board layout:\n{board_str}\n"

        return observation, {
            "suffix": f"Board size: {self.board_size}x{self.board_size}, Max turns: {self.max_turns}."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process player's action.

        Args:
            action: Wall coordinates as "X Y"

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
                "Please use 'Answer: X Y' or \\boxed{X Y} format."
            )
            return obs, 0.0, True, False, {}

        # Parse coordinates
        try:
            x, y = map(int, str(parsed_action).split())
        except Exception as e:
            obs = f"Failed to parse action: {parsed_action}. Error: {e}"
            return obs, 0.0, True, False, {}

        # Validate move
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            obs = f"Invalid coordinates: ({x}, {y}) out of bounds."
            return obs, 0.0, True, False, {}

        if self.board[x][y] != '0':
            obs = f"Invalid move: cell ({x}, {y}) is not empty."
            return obs, 0.0, True, False, {}

        # Place wall
        self.board[x][y] = '1'
        self.walls.add((x, y))
        self.turn += 1

        # Cat's turn
        cat_move = self._find_best_cat_move()

        if cat_move is None:
            # Cat is trapped - player wins
            obs = (
                f"Congratulations! You trapped the cat!\n"
                f"Turns used: {self.turn}/{self.max_turns}\n"
                f"{self._format_board()}"
            )
            return obs, 1.0, True, False, {}

        # Check if cat reached boundary
        if self._is_boundary(cat_move):
            obs = (
                f"Game over! The cat reached the exit at ({cat_move[0]}, {cat_move[1]}).\n"
                f"Turns used: {self.turn}/{self.max_turns}\n"
                f"{self._format_board()}"
            )
            return obs, 0.0, True, False, {}

        # Move cat
        cx, cy = self.cat
        self.board[cx][cy] = '0'
        self.board[cat_move[0]][cat_move[1]] = 'C'
        self.cat = cat_move

        # Check turn limit
        if self.turn >= self.max_turns:
            obs = (
                f"Maximum turns reached ({self.max_turns}). Game over.\n"
                f"{self._format_board()}"
            )
            return obs, 0.0, True, False, {}

        # Continue game
        board_str = self._format_board()
        observation = f"{self._get_instructions()}The board layout:\n{board_str}\n"

        return observation, 0.0, False, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random valid action.

        Returns:
            Random action as string
        """
        # Find empty cells
        empty_cells = [
            (i, j) for i in range(self.board_size) for j in range(self.board_size)
            if self.board[i][j] == '0' and not self._is_boundary((i, j))
        ]

        if empty_cells:
            x, y = random.choice(empty_cells)
            return f"\\boxed{{{x} {y}}}"
        return "\\boxed{0 0}"
