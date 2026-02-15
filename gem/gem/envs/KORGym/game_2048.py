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

"""2048 game environment."""

import math
import random
import re
from typing import List, Optional, Tuple, Dict, Any

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class KORGym2048Env(Env):
    """
    2048 game environment.

    The agent must combine tiles to reach a target tile value (e.g., 2048).
    This is a multi-turn environment with dense rewards for progress.

    Actions: 'up', 'down', 'left', 'right' (wrapped in \\boxed{})
    """

    def __init__(
        self,
        target_tile: int = 2048,
        max_turns: int = 100,
        board_size: int = 4,
        **_,
    ):
        """
        Initialize 2048 environment.

        Args:
            target_tile: Target tile value to reach (e.g., 64, 512, 2048)
            max_turns: Maximum number of turns allowed
            board_size: Size of the board (default 4x4)
        """
        super().__init__()
        self.target_tile = target_tile
        self.max_turns = max_turns
        self.board_size = board_size
        self.board = None
        self.turn_count = 0
        self.max_tile_achieved = 0
        self.init_max_tile_achieved = 0

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game problem-solver. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), "
            f"where YOUR_ANSWER is your final answer to the question, e.g., 'Answer: LEFT'\n\n"
            f"Rules: The game is played on a {self.board_size}x{self.board_size} grid, with each tile containing "
            f"a number that is a power of 2 (e.g., 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048). "
            f"Your goal is to combine the tiles to reach {self.target_tile}. The game ends when there are no more "
            f"valid moves, or when you achieve the {self.target_tile} tile.\n\n"
            f"In the game board, 0 means empty tile and | means the delimiter between tiles. "
            f"At the beginning of the game, two tiles with the number 2 or 4 will appear randomly on the grid.\n\n"
            f"You can swipe LEFT, RIGHT, UP, or DOWN to move all tiles in that direction. "
            f"All tiles will shift to the edge of the grid, and any empty spaces will be filled by a new tile (2 or 4). "
            f"When two tiles of the same number touch, they will merge into one tile with the sum of those numbers "
            f"and you will get the score of the new tiles. For example, two tiles with the number 2 will merge to form a 4.\n\n"
            f"After merging, the new tile will not combine again in the same move. You lose the game if the grid is full, "
            f"and no valid moves are left. A valid move is when two adjacent tiles are the same or there is an empty space "
            f"to move a tile into. Keep in mind that combining tiles strategically is key. Try to keep the larger tiles "
            f"in a corner and work towards merging smaller tiles to get higher scores.\n\n"
            f"Remember, the game will end after the {self.max_turns}th epoch.\n\n"
            f"The answer you give should be one of 'LEFT', 'RIGHT', 'UP' and 'DOWN'"
        )

    def _get_board_str(self) -> str:
        """Format the board as a string."""
        output = ""
        for i in range(self.board_size):
            for j in range(self.board_size):
                output += str(self.board[i][j])
                if j != self.board_size - 1:
                    output += '|'
                else:
                    output += '\n'
        return output.rstrip()

    def get_task_suffix(self) -> str:
        """Get current game state as prompt suffix."""
        board_str = self._get_board_str()
        return f"Game board:\n{board_str}\nCurrent epoch: {self.turn_count}"

    def _get_new_tile_value(self) -> int:
        """Determine the value of a new tile based on current max tile."""
        max_tile = max(max(row) for row in self.board)
        if max_tile < 4:
            return 2
        allowed = []
        v = 2
        while v <= max_tile // 2:
            allowed.append(v)
            v *= 2
        if not allowed:
            allowed = [2]
        return random.choice(allowed)

    def _add_random_tile(self):
        """Add a new tile to a random empty position."""
        empty_positions = [
            (i, j)
            for i in range(self.board_size)
            for j in range(self.board_size)
            if self.board[i][j] == 0
        ]
        if empty_positions:
            i, j = random.choice(empty_positions)
            self.board[i][j] = self._get_new_tile_value()

    def _compress(self, board: List[List[int]]) -> Tuple[List[List[int]], int]:
        """Compress board to the left and calculate score."""
        new_board = [[0] * self.board_size for _ in range(self.board_size)]
        score = 0

        for i in range(self.board_size):
            filtered = [num for num in board[i] if num != 0]
            new_line = []
            skip = False

            for j in range(len(filtered)):
                if skip:
                    skip = False
                    continue
                if j < len(filtered) - 1 and filtered[j] == filtered[j + 1]:
                    new_line.append(filtered[j] * 2)
                    score += filtered[j] * 2
                    skip = True
                else:
                    new_line.append(filtered[j])

            # Fill remaining with zeros
            new_line.extend([0] * (self.board_size - len(new_line)))
            new_board[i] = new_line

        return new_board, score

    def _reverse_board(self, board: List[List[int]]) -> List[List[int]]:
        """Reverse each row of the board."""
        return [row[::-1] for row in board]

    def _transpose_board(self, board: List[List[int]]) -> List[List[int]]:
        """Transpose the board."""
        return [[board[j][i] for j in range(self.board_size)] for i in range(self.board_size)]

    def _move(self, direction: str) -> Tuple[List[List[int]], int, bool]:
        """
        Execute a move in the given direction.

        Returns:
            Tuple of (new_board, score_gained, move_was_valid)
        """
        original_board = [row[:] for row in self.board]

        if direction == 'left':
            new_board, score = self._compress(self.board)
        elif direction == 'right':
            reversed_board = self._reverse_board(self.board)
            compressed, score = self._compress(reversed_board)
            new_board = self._reverse_board(compressed)
        elif direction == 'up':
            transposed = self._transpose_board(self.board)
            compressed, score = self._compress(transposed)
            new_board = self._transpose_board(compressed)
        elif direction == 'down':
            transposed = self._transpose_board(self.board)
            reversed_board = self._reverse_board(transposed)
            compressed, score = self._compress(reversed_board)
            unreversed = self._reverse_board(compressed)
            new_board = self._transpose_board(unreversed)
        else:
            return self.board, 0, False

        # Check if board changed
        move_was_valid = new_board != original_board
        return new_board, score, move_was_valid

    def _is_game_over(self) -> bool:
        """Check if any valid moves remain."""
        for direction in ['left', 'right', 'up', 'down']:
            _, _, is_valid = self._move(direction)
            if is_valid:
                return False
        return True

    def _get_progress_reward(self) -> float:
        """Calculate reward for tile progression."""
        current_max = max(max(row) for row in self.board)
        if current_max > self.target_tile:
            current_max = self.target_tile

        if current_max > self.max_tile_achieved:
            # Progressive reward based on log scale
            reward = (math.log2(current_max) - math.log2(self.max_tile_achieved)) / (
                math.log2(self.target_tile) - math.log2(self.init_max_tile_achieved)
            )
            self.max_tile_achieved = current_max
            return reward
        return 0.0

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset the game to initial state."""
        super().reset(seed)

        # Initialize board
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.turn_count = 0

        # Add initial tiles
        positions = random.sample(range(self.board_size * self.board_size), 2)
        for pos in positions:
            self.board[pos // self.board_size][pos % self.board_size] = self._get_new_tile_value()

        self.max_tile_achieved = max(max(row) for row in self.board)
        self.init_max_tile_achieved = self.max_tile_achieved

        observation = f"{self._get_instructions()}\n\n{self.get_task_suffix()}"
        return observation, {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """Execute one step of the game."""
        self.turn_count += 1

        # Parse action
        clean_action = extract_last_boxed_answer(action)
        if clean_action is None:
            # Try "Answer:" format
            match = re.search(r'Answer:\s*(\w+)', action, re.IGNORECASE)
            if match:
                clean_action = match.group(1)

        if clean_action is None:
            obs = "Your response did not contain a valid move. Please use \\boxed{direction} format."
            return obs, -0.1, True, self.turn_count >= self.max_turns, {"suffix": self.get_task_suffix()}

        clean_action = clean_action.lower()

        # Execute move
        new_board, score_gained, is_valid = self._move(clean_action)

        if not is_valid:
            obs = f"The move '{clean_action}' is invalid (doesn't change the board). Try a different direction."
            return obs, -0.05, False, False, {"suffix": self.get_task_suffix()}

        # Update board
        self.board = new_board

        # Calculate progress reward
        reward = self._get_progress_reward()

        # Add new tile
        self._add_random_tile()

        # Check win condition
        if self.max_tile_achieved >= self.target_tile:
            obs = f"Congratulations! You reached the {self.target_tile} tile in {self.turn_count} turns!"
            return obs, reward + 1.0, True, False, {"suffix": self.get_task_suffix()}

        # Check game over
        if self._is_game_over():
            obs = f"Game over! No more moves possible. Maximum tile achieved: {self.max_tile_achieved}"
            return obs, reward, True, False, {"suffix": self.get_task_suffix()}

        # Check max turns
        if self.turn_count >= self.max_turns:
            obs = f"Maximum turns reached. Maximum tile achieved: {self.max_tile_achieved}"
            return obs, reward, True, True, {"suffix": self.get_task_suffix()}

        # Continue game
        obs = f"You moved {clean_action.upper()}."
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def sample_random_action(self) -> str:
        """Sample a random action."""
        direction = random.choice(['up', 'down', 'left', 'right'])
        return f"\\boxed{{{direction}}}"
