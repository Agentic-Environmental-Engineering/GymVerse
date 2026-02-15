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

"""PuzzleGame environment - Sliding puzzle game."""

import random
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class PuzzleGameEnv(Env):
    """
    PuzzleGame (Sliding Puzzle) environment.

    Players must move tiles in an n×n grid to move a target tile to a
    target position. Only tiles adjacent to the empty space can be moved.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_size: int = 3,
        max_size: int = 5,
        **_,
    ):
        """
        Initialize PuzzleGame environment.

        Args:
            min_size: Minimum grid size (n×n)
            max_size: Maximum grid size (n×n)
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.board = None
        self.target_num = None
        self.target_grid = None
        self.n = None

    def _generate_solvable_puzzle(self, n: int) -> List[int]:
        """
        Generate a solvable sliding puzzle.

        Args:
            n: Grid size

        Returns:
            Flattened puzzle array
        """
        total_tiles = n * n - 1
        arr = list(range(1, total_tiles + 1))
        parity = False

        # Shuffle first total_tiles-2 elements
        for i in range(total_tiles - 2):
            t = random.randint(i, total_tiles - 1)
            if i != t:
                parity = not parity
            arr[i], arr[t] = arr[t], arr[i]

        # Ensure solvability by swapping last two if needed
        if parity:
            arr[total_tiles - 2], arr[total_tiles - 1] = arr[total_tiles - 1], arr[total_tiles - 2]

        # Add empty space
        arr.append(0)

        # Random adjustment of empty space position
        blank_index = total_tiles
        d, r = random.randint(0, n - 1), random.randint(0, n - 1)
        for _ in range(d):
            arr[blank_index], arr[blank_index - n] = arr[blank_index - n], arr[blank_index]
            blank_index = blank_index - n
        for _ in range(r):
            arr[blank_index], arr[blank_index - 1] = arr[blank_index - 1], arr[blank_index]
            blank_index = blank_index - 1

        return arr

    def _find_empty(self, board: List[List[int]]) -> Tuple[int, int]:
        """Find the position of empty space (0)."""
        n = len(board)
        for r in range(n):
            for c in range(n):
                if board[r][c] == 0:
                    return r, c
        return -1, -1

    def _find_tile(self, board: List[List[int]], num: int) -> Tuple[int, int]:
        """Find the position of a specific tile."""
        n = len(board)
        for r in range(n):
            for c in range(n):
                if board[r][c] == num:
                    return r, c
        return -1, -1

    def _solve_bfs(self, board: List[List[int]], target_num: int, target_pos: Tuple[int, int]) -> List[int]:
        """
        Find a solution using BFS to move target_num to target_pos.

        Args:
            board: Initial board state
            target_num: Tile to move
            target_pos: Target position (row, col) 0-indexed

        Returns:
            List of moves (tile numbers to move into empty space)
        """
        n = len(board)
        initial_state = tuple(tuple(row) for row in board)

        # BFS
        queue = deque([(initial_state, [])])
        visited = {initial_state}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        max_iterations = 100000
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            state, moves = queue.popleft()

            # Convert to list for easier manipulation
            current_board = [list(row) for row in state]

            # Check if target tile is at target position
            tile_pos = self._find_tile(current_board, target_num)
            if tile_pos == target_pos:
                return moves

            # Find empty position
            empty_r, empty_c = self._find_empty(current_board)

            # Try all possible moves
            for dr, dc in directions:
                new_r, new_c = empty_r + dr, empty_c + dc

                if 0 <= new_r < n and 0 <= new_c < n:
                    # Create new state
                    new_board = [row[:] for row in current_board]
                    tile_to_move = new_board[new_r][new_c]
                    new_board[empty_r][empty_c], new_board[new_r][new_c] = new_board[new_r][new_c], new_board[empty_r][empty_c]

                    new_state = tuple(tuple(row) for row in new_board)

                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, moves + [tile_to_move]))

        # If no solution found, return empty list
        return []

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: 15 3 4 2'\n\n"
            "Alternatively, you can use \\boxed{15 3 4 2} format.\n\n"
            "The goal of this game is to rearrange tiles into an n×n grid by moving a specified "
            "target tile to a target position. For example, if the input target is \"4 (2,1)\", "
            "you must move tile \"4\" to row 2, column 1 according to specific rules.\n\n"
            "Rules:\n"
            "- Grid positions are indexed starting from 1, with the top-left coordinate as (1,1).\n"
            "- The puzzle consists of an n×n grid: n×n-1 numbered tiles and one empty space (represented by 0).\n"
            "- A tile can only be moved into the empty space if it is directly adjacent to it (left, right, above, or below).\n"
            "- Tiles cannot move diagonally. For example, in the following state, you cannot move tile \"7\" into the empty space:\n"
            "  5 6 8\n"
            "  7 2 3\n"
            "  1 0 4\n\n"
            "- Only one tile can be moved at each step.\n"
            "- The puzzle is completed when the target tile reaches the target position.\n"
            "- You must output your solution as a sequence of numbers separated by spaces. Each number "
            "indicates the tile moved into the empty space, following the above rules. For example:\n"
            "'Answer: 15 3 4 2'\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new sliding puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle size
        self.n = random.randint(self.min_size, self.max_size)

        # Generate solvable puzzle
        arr = self._generate_solvable_puzzle(self.n)

        # Convert to 2D board
        self.board = [arr[i:i+self.n] for i in range(0, len(arr), self.n)]

        # Generate target tile and position
        self.target_num = random.randint(1, 3)
        self.target_grid = [random.randint(1, self.n), random.randint(1, self.n)]

        # Build board string
        board_str = "\n".join(" ".join(map(str, row)) for row in self.board)
        target_info = f"Target: move {self.target_num} to {self.target_grid}"

        observation = f"{self._get_instructions()}{board_str}\n{target_info}"

        return observation, {
            "suffix": f"Move tile {self.target_num} to position {self.target_grid}."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's solution.

        Args:
            action: Agent's response containing the move sequence

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer
        parsed_answer = extract_last_boxed_answer(action)

        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(.+)', action, re.IGNORECASE)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: MOVES' or \\boxed{MOVES} format."
            )
            return obs, 0.0, True, False, {}

        # Parse move sequence
        user_moves = parsed_answer.strip()
        if not user_moves:
            obs = "Empty move sequence provided."
            return obs, 0.0, True, False, {}

        try:
            # Simulate moves
            board = [row[:] for row in self.board]
            n = len(board)
            target_row, target_col = self.target_grid

            # Find empty position
            empty_r, empty_c = self._find_empty(board)

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            moves = user_moves.split()

            for move in moves:
                move_num = int(move)

                # Find tile position
                tile_r, tile_c = self._find_tile(board, move_num)
                if tile_r == -1:
                    obs = f"Tile {move_num} not found on board."
                    return obs, 0.0, True, False, {}

                # Check if tile is adjacent to empty
                legal_move = False
                for dr, dc in directions:
                    new_r, new_c = tile_r + dr, tile_c + dc
                    if 0 <= new_r < n and 0 <= new_c < n and new_r == empty_r and new_c == empty_c:
                        # Swap
                        board[empty_r][empty_c], board[tile_r][tile_c] = board[tile_r][tile_c], board[empty_r][empty_c]
                        empty_r, empty_c = tile_r, tile_c
                        legal_move = True
                        break

                if not legal_move:
                    obs = f"Illegal move: tile {move_num} is not adjacent to empty space."
                    return obs, 0.0, True, False, {}

            # Check if target tile is at target position
            final_r, final_c = self._find_tile(board, self.target_num)
            if (final_r, final_c) == (target_row - 1, target_col - 1):
                obs = f"Correct! Tile {self.target_num} is now at position {self.target_grid}."
                return obs, 1.0, True, False, {}
            else:
                obs = (f"Incorrect. Tile {self.target_num} is at ({final_r+1}, {final_c+1}), "
                      f"but should be at {self.target_grid}.")
                return obs, 0.0, True, False, {}

        except Exception as e:
            obs = f"Error processing moves: {e}"
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            A valid solution sequence
        """
        if self.board is None:
            return "\\boxed{1}"

        # Find solution using BFS
        target_pos = (self.target_grid[0] - 1, self.target_grid[1] - 1)
        moves = self._solve_bfs(self.board, self.target_num, target_pos)

        if moves:
            move_str = " ".join(map(str, moves))
            return f"\\boxed{{{move_str}}}"
        else:
            return "\\boxed{1}"
