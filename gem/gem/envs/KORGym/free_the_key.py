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

"""FreeTheKey environment - Sliding block puzzle to free the key."""

import random
import copy
from collections import deque
from typing import Optional, Tuple, Dict, Any, List, Set

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class FreeTheKeyEnv(Env):
    """
    FreeTheKey environment.

    A sliding block puzzle where the player must move blocks to create a path
    for the key to reach the exit. Horizontal blocks can only move left/right,
    vertical blocks can only move up/down. The key is always horizontal and can
    only move left/right.

    This is a multi-turn environment with sparse terminal reward.
    """

    def __init__(
        self,
        board_size: int = 6,
        min_blocks: int = 8,
        max_blocks: int = 15,
        max_turns: int = 100,
        **_,
    ):
        """
        Initialize FreeTheKey environment.

        Args:
            board_size: Size of the square board
            min_blocks: Minimum number of blocks
            max_blocks: Maximum number of blocks
            max_turns: Maximum number of turns
        """
        super().__init__()
        self.board_size = board_size
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.max_turns = max_turns
        self.board = None
        self.blocks = None  # Dict of block_id -> {positions, direction}
        self.key_positions = None
        self.exit_position = None
        self.turn = None

    def _get_block_letter(self, idx: int) -> str:
        """Get letter for block index."""
        if idx < 26:
            return chr(ord('A') + idx)
        else:
            return chr(ord('a') + (idx - 26))

    def _can_place_block(self, board: List[List[str]], positions: List[Tuple[int, int]]) -> bool:
        """Check if block can be placed at positions."""
        for x, y in positions:
            if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
                return False
            if board[x][y] != '0':
                return False
        return True

    def _place_block(self, board: List[List[str]], block_id: str, positions: List[Tuple[int, int]]):
        """Place block on board."""
        for x, y in positions:
            board[x][y] = block_id

    def _remove_block(self, board: List[List[str]], positions: List[Tuple[int, int]]):
        """Remove block from board."""
        for x, y in positions:
            board[x][y] = '0'

    def _generate_puzzle(self, seed: int) -> Tuple[List[List[str]], Dict, List[Tuple[int, int]], Tuple[int, int]]:
        """Generate puzzle using backtracking."""
        random.seed(seed)

        # Initialize board
        board = [['0' for _ in range(self.board_size)] for _ in range(self.board_size)]

        # Place exit at fixed position (middle row, rightmost column)
        exit_row = self.board_size // 2
        exit_col = self.board_size - 1
        self.exit_position = (exit_row, exit_col)
        board[exit_row][exit_col] = '2'

        # Place key (horizontal, length 2) somewhere in the same row as exit
        # Start with key not directly at exit to make puzzle interesting
        key_col = random.randint(0, self.board_size - 3)
        key_positions = [(exit_row, key_col), (exit_row, key_col + 1)]
        for x, y in key_positions:
            board[x][y] = '1'

        # Generate random blocks using backtracking
        blocks = {}
        num_blocks = random.randint(self.min_blocks, self.max_blocks)

        for i in range(num_blocks):
            block_id = self._get_block_letter(i)

            # Try to place block
            max_attempts = 100
            placed = False

            for _ in range(max_attempts):
                # Random orientation and length
                is_horizontal = random.choice([True, False])
                length = random.randint(2, 3)

                if is_horizontal:
                    x = random.randint(0, self.board_size - 1)
                    y = random.randint(0, self.board_size - length)
                    positions = [(x, y + j) for j in range(length)]
                    direction = 'horizontal'
                else:
                    x = random.randint(0, self.board_size - length)
                    y = random.randint(0, self.board_size - 1)
                    positions = [(x + j, y) for j in range(length)]
                    direction = 'vertical'

                if self._can_place_block(board, positions):
                    self._place_block(board, block_id, positions)
                    blocks[block_id] = {
                        'positions': positions,
                        'direction': direction
                    }
                    placed = True
                    break

            if not placed:
                # Can't place more blocks, continue with what we have
                break

        return board, blocks, key_positions, self.exit_position

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: A right'.\n\n"
            "Alternatively, you can use \\boxed{A right} format.\n\n"
            "This is a sliding block puzzle. The board contains:\n"
            "- Blocks labeled with letters (A, B, C, etc.)\n"
            "- A key labeled with '1' (horizontal block, length 2)\n"
            "- An exit labeled with '2'\n"
            "- Empty spaces labeled with '0'\n\n"
            "Rules:\n"
            "1. Horizontal blocks (including the key) can only move left or right.\n"
            "2. Vertical blocks can only move up or down.\n"
            "3. Blocks cannot overlap or move outside the board.\n"
            "4. Your goal is to move the key '1' to reach the exit '2'.\n\n"
            "On each turn, specify which block to move and in which direction.\n"
            "Format: 'BLOCK DIRECTION' where BLOCK is the letter/number and DIRECTION is "
            "up/down/left/right.\n"
            "Example: 'Answer: A right' or 'Answer: 1 left'\n\n"
        )

    def _format_board(self) -> str:
        """Format the game board."""
        output = "Current Board:\n"
        for row in self.board:
            output += " ".join(row) + "\n"
        return output

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new FreeTheKey puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        self.board, self.blocks, self.key_positions, self.exit_position = self._generate_puzzle(puzzle_seed)
        self.turn = 0

        # Build observation
        board_str = self._format_board()
        observation = f"{self._get_instructions()}{board_str}\n"

        return observation, {
            "suffix": f"Board size: {self.board_size}x{self.board_size}, Max turns: {self.max_turns}."
        }

    def _move_block(self, block_id: str, direction: str) -> bool:
        """
        Try to move a block in the specified direction.

        Returns:
            True if move was successful, False otherwise
        """
        # Get block info
        if block_id == '1':
            # Key
            positions = self.key_positions
            block_direction = 'horizontal'
        elif block_id in self.blocks:
            positions = self.blocks[block_id]['positions']
            block_direction = self.blocks[block_id]['direction']
        else:
            return False

        # Check if direction is valid for block orientation
        if block_direction == 'horizontal' and direction not in ['left', 'right']:
            return False
        if block_direction == 'vertical' and direction not in ['up', 'down']:
            return False

        # Calculate new positions
        dx, dy = 0, 0
        if direction == 'up':
            dx = -1
        elif direction == 'down':
            dx = 1
        elif direction == 'left':
            dy = -1
        elif direction == 'right':
            dy = 1
        else:
            return False

        new_positions = [(x + dx, y + dy) for x, y in positions]

        # Check if new positions are valid
        for x, y in new_positions:
            if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
                return False
            # Check if cell is empty or is part of current block or is exit
            cell = self.board[x][y]
            if cell != '0' and cell != block_id and cell != '2':
                return False

        # Remove block from old positions
        self._remove_block(self.board, positions)

        # Place block at new positions
        for x, y in new_positions:
            if self.board[x][y] != '2':  # Don't overwrite exit
                self.board[x][y] = block_id

        # Update positions
        if block_id == '1':
            self.key_positions = new_positions
        else:
            self.blocks[block_id]['positions'] = new_positions

        return True

    def _check_win(self) -> bool:
        """Check if key reached the exit."""
        exit_row, exit_col = self.exit_position
        for x, y in self.key_positions:
            if (x, y) == (exit_row, exit_col):
                return True
        return False

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process player's action.

        Args:
            action: Block and direction as "BLOCK DIRECTION"

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
                "Please use 'Answer: A right' or \\boxed{A right} format."
            )
            return obs, 0.0, True, False, {}

        # Parse block and direction
        try:
            parts = str(parsed_action).strip().split()
            if len(parts) != 2:
                raise ValueError("Invalid format")
            block_id = parts[0].upper() if parts[0] != '1' else '1'
            direction = parts[1].lower()
        except Exception as e:
            obs = f"Failed to parse action: {parsed_action}. Error: {e}"
            return obs, 0.0, True, False, {}

        # Try to move block
        success = self._move_block(block_id, direction)

        if not success:
            obs = f"Invalid move: Cannot move block {block_id} {direction}."
            return obs, 0.0, True, False, {}

        self.turn += 1

        # Check win condition
        if self._check_win():
            obs = (
                f"Congratulations! You freed the key!\n"
                f"Turns used: {self.turn}/{self.max_turns}\n"
                f"{self._format_board()}"
            )
            return obs, 1.0, True, False, {}

        # Check turn limit
        if self.turn >= self.max_turns:
            obs = (
                f"Maximum turns reached ({self.max_turns}). Game over.\n"
                f"{self._format_board()}"
            )
            return obs, 0.0, True, False, {}

        # Continue game
        board_str = self._format_board()
        observation = f"{self._get_instructions()}{board_str}\n"

        return observation, 0.0, False, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random valid action.

        Returns:
            Random action as string
        """
        # Try random moves
        for _ in range(100):
            # Select random block (including key)
            all_blocks = ['1'] + list(self.blocks.keys())
            block_id = random.choice(all_blocks)

            # Get block direction
            if block_id == '1':
                block_direction = 'horizontal'
            else:
                block_direction = self.blocks[block_id]['direction']

            # Select valid direction for this block
            if block_direction == 'horizontal':
                direction = random.choice(['left', 'right'])
            else:
                direction = random.choice(['up', 'down'])

            # Test if move is valid
            temp_env = FreeTheKeyEnv()
            temp_env.board = copy.deepcopy(self.board)
            temp_env.blocks = copy.deepcopy(self.blocks)
            temp_env.key_positions = self.key_positions.copy()
            temp_env.exit_position = self.exit_position
            temp_env.board_size = self.board_size

            if temp_env._move_block(block_id, direction):
                return f"\\boxed{{{block_id} {direction}}}"

        # Fallback
        return "\\boxed{1 right}"
