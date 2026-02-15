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

"""Tetris environment - Classic Tetris game."""

import random
from copy import deepcopy
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


# Tetris block shapes
SHAPES = [
    [[1, 1, 1, 1]],          # I
    [[1, 1], [1, 1]],        # O
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[1, 0, 0], [1, 1, 1]],  # L
    [[0, 0, 1], [1, 1, 1]],  # J
    [[0, 1, 0], [1, 1, 1]]   # T
]


class TetrisEnv(Env):
    """
    Tetris game environment.

    Players must place falling blocks by choosing rotation angle and drop position.
    Lines are cleared when complete, earning points. Game ends when blocks exceed
    the top boundary or max turns are reached.

    This is a multi-turn environment with dense rewards.
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        max_turns: int = 100,
        **_,
    ):
        """
        Initialize Tetris environment.

        Args:
            width: Board width
            height: Board height
            max_turns: Maximum number of turns
        """
        super().__init__()
        self.width = width
        self.height = height
        self.max_turns = max_turns
        self.board = None
        self.current_block = None
        self.score = 0
        self.epoch = 0

    def _get_board_str(self, matrix: List[List[int]], zero_flag: str, one_flag: str) -> str:
        """Convert matrix to string representation."""
        m, n = len(matrix), len(matrix[0])
        res = [[zero_flag] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j]:
                    res[i][j] = one_flag
        return "\n".join("".join(row) for row in res)

    def _get_shape_input(self, shape: List[List[int]]) -> str:
        """Get all rotation representations of a shape."""
        res = "0°:\n" + self._get_board_str(shape, "0", "*")
        rotated = shape
        for i in range(3):
            rotated = [list(row) for row in zip(*rotated[::-1])]
            res += f"\n{(i+1)*90}°:\n" + self._get_board_str(rotated, "0", "*")
        return res

    def _get_board_input(self) -> str:
        """Get board string representation."""
        return self._get_board_str(self.board, ".", "*")

    def _rotate_block(self, block: List[List[int]], times: int) -> List[List[int]]:
        """Rotate block 90 degrees clockwise `times` times."""
        result = block
        for _ in range(times):
            result = [list(row) for row in zip(*result[::-1])]
        return result

    def _place_block(self, block: List[List[int]], location: int) -> bool:
        """
        Place block on board at given location.

        Args:
            block: Block shape
            location: Drop column (1-indexed, leftmost valid position)

        Returns:
            True if placement successful, False otherwise
        """
        block_height, block_width = len(block), len(block[0])

        # Find leftmost column with block content
        left_offset = None
        for j in range(block_width):
            for i in range(block_height):
                if block[i][j] == 1:
                    left_offset = j
                    break
            if left_offset is not None:
                break
        if left_offset is None:
            left_offset = 0

        # Calculate x position (0-indexed)
        x = location - 1 - left_offset
        x = max(0, min(x, self.width - block_width))

        # Find lowest valid y position
        final_y = None
        for y in range(self.height - block_height + 1):
            valid = True
            for i in range(block_height):
                for j in range(block_width):
                    if block[i][j] == 1 and self.board[y+i][x+j] == 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                final_y = y
            else:
                break

        if final_y is None:
            return False

        # Place block
        for i in range(block_height):
            for j in range(block_width):
                if block[i][j] == 1:
                    self.board[final_y+i][x+j] = 1

        return True

    def _clear_lines(self) -> int:
        """Clear complete lines and return number of lines cleared."""
        complete_lines = [i for i in range(self.height) if sum(self.board[i]) == self.width]
        n = len(complete_lines)

        if n == 0:
            return 0

        # Create new board with cleared lines
        new_board = [[0] * self.width for _ in range(n)]
        for i, line in enumerate(self.board):
            if i not in complete_lines:
                new_board.append(line)

        self.board = new_board
        return n

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a skilled game player. I'll provide you with a game board and rules.\n\n"
            "Your tasks are as follows:\n\n"
            "- First, provide your solution based on the given game board and rules.\n"
            "- Second, present your solution strictly in the required format. The final line of your response must follow this format exactly:\n"
            "  'Answer: $YOUR_ANSWER'\n"
            "(without quotes), where 'YOUR_ANSWER' is your final response. For example, 'Answer: 4 90'.\n\n"
            "Alternatively, you can use \\boxed{4 90} format.\n\n"
            f"You are playing Tetris on a {self.width}×{self.height} board.\n"
            f"The game ends after {self.max_turns} turns or if a block exceeds the top boundary.\n"
            "When an entire row is filled, it is cleared and the score increases by 1.\n"
            "Blocks fall until they hit another block or the bottom.\n\n"
            "Board notation:\n"
            "- '*' represents an occupied square\n"
            "- '.' represents an empty square\n\n"
            "Block notation:\n"
            "- '*' represents block squares\n"
            "- '0' represents empty spaces\n\n"
            "You must provide your action in the format:\n"
            "'Answer: [drop_coordinate] [rotation_angle]' (e.g., 'Answer: 4 90')\n"
            "where:\n"
            f"- [drop_coordinate] is the column position (1 to {self.width}) for the leftmost square of the block\n"
            "- [rotation_angle] is the rotation (0, 90, 180, or 270 degrees)\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new Tetris game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Initialize empty board
        self.board = [[0] * self.width for _ in range(self.height)]

        # Generate first block
        self.current_block = random.choice(SHAPES)
        self.score = 0
        self.epoch = 1

        # Build observation
        board_str = self._get_board_input()
        block_str = self._get_shape_input(self.current_block)

        observation = (
            f"{self._get_instructions()}"
            f"Epoch: {self.epoch}\n"
            f"Score: {self.score}\n"
            f"Board:\n{board_str}\n"
            f"Current Block (rotations):\n{block_str}\n"
        )

        return observation, {
            "suffix": f"Place block (Turn {self.epoch}/{self.max_turns}, Score: {self.score})."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute one turn of Tetris.

        Args:
            action: Drop position and rotation angle (e.g., "4 90")

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
                "Please use 'Answer: POSITION ANGLE' or \\boxed{POSITION ANGLE} format."
            )
            return obs, 0.0, True, False, {}

        # Parse position and angle
        try:
            parts = parsed_action.split()
            if len(parts) != 2:
                obs = f"Invalid action format: {parsed_action}. Expected 'POSITION ANGLE'."
                return obs, 0.0, True, False, {}

            location = int(parts[0])
            angle = int(parts[1].replace("°", ""))

            if angle not in [0, 90, 180, 270]:
                obs = f"Invalid rotation angle: {angle}. Must be 0, 90, 180, or 270."
                return obs, 0.0, True, False, {}

        except ValueError as e:
            obs = f"Failed to parse action: {parsed_action}. Error: {e}"
            return obs, 0.0, True, False, {}

        # Rotate block
        rotated_block = self._rotate_block(self.current_block, angle // 90)

        # Place block
        can_place = self._place_block(rotated_block, location)

        if not can_place:
            obs = (
                f"Game Over! Block cannot be placed.\n\n"
                f"Final board:\n{self._get_board_input()}\n"
                f"Final score: {self.score}\n"
                f"Turns: {self.epoch}"
            )
            return obs, 0.0, True, False, {"score": self.score, "epoch": self.epoch}

        # Clear lines and calculate reward
        lines_cleared = self._clear_lines()
        reward = float(lines_cleared)
        self.score += lines_cleared

        # Generate next block
        self.current_block = random.choice(SHAPES)
        self.epoch += 1

        # Check truncation
        truncated = self.epoch > self.max_turns

        if truncated:
            obs = (
                f"Game finished! Maximum turns reached.\n\n"
                f"Final board:\n{self._get_board_input()}\n"
                f"Final score: {self.score}\n"
                f"Turns: {self.epoch - 1}/{self.max_turns}"
            )
            return obs, reward, False, True, {"score": self.score, "epoch": self.epoch}

        # Build next observation
        board_str = self._get_board_input()
        block_str = self._get_shape_input(self.current_block)

        observation = (
            f"Epoch: {self.epoch}\n"
            f"Score: {self.score}\n"
            f"Board:\n{board_str}\n"
            f"Current Block (rotations):\n{block_str}\n"
        )

        return observation, reward, False, False, {"score": self.score, "epoch": self.epoch}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random position and rotation
        """
        position = random.randint(1, self.width)
        angle = random.choice([0, 90, 180, 270])
        return f"\\boxed{{{position} {angle}}}"
