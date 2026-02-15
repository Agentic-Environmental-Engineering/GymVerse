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

"""Lights Out environment - Toggle light puzzle game."""

import random
import re
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class LightsOutEnv(Env):
    """
    Lights Out puzzle environment.

    The game consists of a grid of lights (3x3 or 4x4). Pressing any light
    toggles it and its adjacent lights (up, down, left, right). The goal is to
    turn off all lights.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_level: int = 1,
        max_level: int = 15,
        **_,
    ):
        """
        Initialize Lights Out environment.

        Args:
            min_level: Minimum difficulty level (1-15)
            max_level: Maximum difficulty level (1-15)
                      Level 1-5: 3x3 grid
                      Level 6-15: 4x4 grid
        """
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.board = None
        self.level = None
        self.solution = None

    def _toggle(self, board: List[List[int]], i: int, j: int) -> None:
        """Toggle light at (i, j) and its adjacent lights."""
        n = len(board)
        board[i][j] ^= 1
        if i > 0:
            board[i-1][j] ^= 1
        if i < n - 1:
            board[i+1][j] ^= 1
        if j > 0:
            board[i][j-1] ^= 1
        if j < n - 1:
            board[i][j+1] ^= 1

    def _format_board(self, board: List[List[int]]) -> str:
        """Format board as string."""
        output = ""
        for i in range(len(board)):
            for j in range(len(board)):
                output += str(board[i][j])
            output += "\n"
        return output

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game problem-solver. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: (0,2), (2,1)'\n\n"
            "Alternatively, you can use \\boxed{(0,2), (2,1)} format.\n\n"
            "The game consists of a grid of lights. '1' means the light is on and '0' means it's off. "
            "Pressing any light will toggle it and the adjacent lights (up, left, right, and down).\n\n"
            "For example, if the board is:\n"
            "000\n000\n000\n"
            "and you press (1,1), the board becomes:\n"
            "010\n111\n010\n\n"
            "If a light is at the boundary, it only affects adjacent lights within the board.\n\n"
            "The goal is to switch all lights off, preferably in as few presses as possible. "
            "Give your answer as a series of (row, col) positions separated by commas.\n"
            "If multiple solutions exist, provide any one correct answer.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new Lights Out puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate random level
        self.level = random.randint(self.min_level, self.max_level)

        # Determine grid size and number of presses
        if self.level <= 5:
            n = 3
            k = self.level  # Number of random presses to generate puzzle
        else:
            n = 4
            k = self.level - 4

        # Start with all lights off
        self.board = [[0 for _ in range(n)] for _ in range(n)]

        # Generate puzzle by randomly pressing k positions
        all_positions = [(i, j) for i in range(n) for j in range(n)]
        self.solution = random.sample(all_positions, k)

        for i, j in self.solution:
            self._toggle(self.board, i, j)

        # Build question
        board_str = self._format_board(self.board)
        question = f"Board:\n{board_str}"

        observation = f"{self._get_instructions()}{question}"
        return observation, {"suffix": f"Turn off all lights (level {self.level}, {n}x{n} grid)."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's solution.

        Args:
            action: Agent's response containing the sequence of presses

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        import re

        # Try to extract answer using \boxed{} format first
        parsed_answer = extract_last_boxed_answer(action)

        # If not found, try "Answer:" format
        if parsed_answer is None:
            match = re.search(r'Answer:\s*(.+?)(?:\n|$)', action, re.IGNORECASE | re.MULTILINE)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: (row,col), (row,col), ...' or \\boxed{(row,col), (row,col), ...} format."
            )
            return obs, 0.0, True, False, {}

        # Parse the sequence of positions
        try:
            # Extract all (row, col) tuples from the answer
            answer = [
                tuple(map(int, re.findall(r'\d+', item.strip())))
                for item in parsed_answer.split('),') if item.strip()
            ]
        except Exception as e:
            obs = f"Failed to parse answer. Error: {e}"
            return obs, 0.0, True, False, {}

        if not answer:
            obs = "No valid positions found in your answer."
            return obs, 0.0, True, False, {}

        # Simulate the presses
        n = len(self.board)
        current = [row.copy() for row in self.board]

        for step in answer:
            if len(step) != 2:
                obs = f"Invalid position format: {step}. Expected (row, col)."
                return obs, 0.0, True, False, {}

            i, j = step
            if i < 0 or i >= n or j < 0 or j >= n:
                obs = f"Position ({i}, {j}) is out of bounds for {n}x{n} grid."
                return obs, 0.0, True, False, {}

            self._toggle(current, i, j)

        # Check if all lights are off
        all_off = all(cell == 0 for row in current for cell in row)

        if all_off:
            obs = f"Correct! All lights are off using {len(answer)} presses: {answer}"
            return obs, 1.0, True, False, {}
        else:
            obs = f"Incorrect. After your {len(answer)} presses, some lights are still on:\n{self._format_board(current)}"
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            The solution that was used to generate the puzzle
        """
        if self.solution is not None:
            # Return the solution that generated the puzzle
            answer_str = ", ".join([f"({i},{j})" for i, j in self.solution])
            return f"\\boxed{{{answer_str}}}"
        else:
            return f"\\boxed{{(0,0)}}"
