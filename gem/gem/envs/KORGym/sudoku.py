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

"""Sudoku environment - Classic Sudoku puzzle game."""

import ast
import copy
import math
import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class SudokuEnv(Env):
    """
    Sudoku puzzle environment.

    The agent must fill in a 9x9 grid so that each row, column, and 3x3 block
    contains all numbers from 1 to 9 without repetition.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        size: int = 9,
        difficulty: str = "moderate",
        **_,
    ):
        """
        Initialize Sudoku environment.

        Args:
            size: Size of the Sudoku grid (default 9x9)
            difficulty: Difficulty level ('easy', 'moderate', 'difficult')
        """
        super().__init__()
        self.size = size
        self.sqrt_size = int(math.sqrt(size))
        self.difficulty = difficulty

        # Difficulty levels: (logical_cutoff, random_cutoff)
        self.difficulty_levels = {
            "easy": ((int(size * size / 2) - int(size * size / 10)), 0),
            "moderate": (int(size * size), int(size * size / 15)),
            "difficult": (int(size * size), int(size * size / 10))
        }

        self.solution = None
        self.current_board = None

    def _make_board(self) -> List[List[int]]:
        """Construct a complete, valid Sudoku solution."""
        board = [[0 for _ in range(self.size)] for _ in range(self.size)]

        # Create base pattern
        for i in range(self.size):
            for j in range(self.size):
                board[i][j] = int((i * self.sqrt_size + i // self.sqrt_size + j) % self.size) + 1

        # Shuffle to create variation
        for _ in range(random.randint(8, 15)):
            board = self._shuffle_board(board)

        return board

    def _shuffle_board(self, board: List[List[int]]) -> List[List[int]]:
        """Shuffle board by swapping values and rows within blocks."""
        # Swap two random numbers
        num1 = random.randint(1, self.size)
        num2 = random.randint(1, self.size)
        while num2 == num1:
            num2 = random.randint(1, self.size)

        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == num1:
                    board[i][j] = num2
                elif board[i][j] == num2:
                    board[i][j] = num1

        # Swap two rows within a block
        if self.sqrt_size > 1:
            block = random.randint(0, self.sqrt_size - 1)
            row1 = random.randint(0, self.sqrt_size - 1) + block * self.sqrt_size
            row2 = random.randint(0, self.sqrt_size - 1) + block * self.sqrt_size
            while row2 == row1:
                row2 = random.randint(0, self.sqrt_size - 1) + block * self.sqrt_size
            board[row1], board[row2] = board[row2], board[row1]

        return board

    def _get_possible_numbers(self, board: List[List[int]], row: int, col: int) -> List[int]:
        """Get all valid numbers for a given cell."""
        possible = set(range(1, self.size + 1))

        # Remove numbers in same row
        possible -= set(board[row])

        # Remove numbers in same column
        possible -= set(board[i][col] for i in range(self.size))

        # Remove numbers in same block
        start_row = (row // self.sqrt_size) * self.sqrt_size
        start_col = (col // self.sqrt_size) * self.sqrt_size
        for i in range(start_row, start_row + self.sqrt_size):
            for j in range(start_col, start_col + self.sqrt_size):
                possible.discard(board[i][j])

        return list(possible)

    def _remove_numbers_logically(self, board: List[List[int]], cutoff: int) -> List[List[int]]:
        """Remove numbers using logical constraint checks."""
        removed_items = 0
        for _ in range(self.size * 500):
            if removed_items >= cutoff:
                break

            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)

            if board[i][j] == 0:
                continue

            temp = board[i][j]
            board[i][j] = 0

            # Only remove if cell has unique solution
            if len(self._get_possible_numbers(board, i, j)) == 1:
                removed_items += 1
            else:
                board[i][j] = temp

        return board

    def _is_safe(self, board: List[List[int]], row: int, col: int, num: int) -> bool:
        """Check if placing a number is valid."""
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[i][col] for i in range(self.size)]:
            return False

        # Check block
        start_row = (row // self.sqrt_size) * self.sqrt_size
        start_col = (col // self.sqrt_size) * self.sqrt_size
        for i in range(start_row, start_row + self.sqrt_size):
            for j in range(start_col, start_col + self.sqrt_size):
                if board[i][j] == num:
                    return False

        return True

    def _find_empty(self, board: List[List[int]]) -> Optional[Tuple[int, int]]:
        """Find first empty cell (with value 0)."""
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _solve_board(self, board: List[List[int]]) -> bool:
        """Solve Sudoku using backtracking."""
        empty = self._find_empty(board)
        if not empty:
            return True

        row, col = empty

        for num in range(1, self.size + 1):
            if self._is_safe(board, row, col, num):
                board[row][col] = num
                if self._solve_board(board):
                    return True
                board[row][col] = 0

        return False

    def _remove_numbers_randomly(self, board: List[List[int]], cutoff: int) -> List[List[int]]:
        """Remove numbers randomly while ensuring puzzle is solvable."""
        removed_items = 0
        for i in range(self.size):
            for j in range(self.size):
                if removed_items >= cutoff:
                    return board
                if board[i][j] == 0:
                    continue

                temp = board[i][j]
                board[i][j] = 0
                test_board = [row[:] for row in board]

                if self._solve_board(test_board):
                    removed_items += 1
                else:
                    board[i][j] = temp

        return board

    def _make_puzzle(self, board: List[List[int]], difficulty: str) -> List[List[int]]:
        """Remove numbers based on difficulty level."""
        logical_cutoff, random_cutoff = self.difficulty_levels[difficulty]
        board = self._remove_numbers_logically(board, logical_cutoff)
        if random_cutoff > 0:
            board = self._remove_numbers_randomly(board, random_cutoff)
        return board

    def _is_valid_unit(self, unit: List[int]) -> bool:
        """Check if a unit (row/column/block) is valid."""
        return sorted(unit) == list(range(1, self.size + 1))

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes)\n\n"
            "Alternatively, you can use \\boxed{[[...]]} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new Sudoku puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate solution
        self.solution = self._make_board()

        # Create puzzle by removing numbers
        self.current_board = [row[:] for row in self.solution]
        self.current_board = self._make_puzzle(self.current_board, self.difficulty)

        # Build question text
        prompt = "Please solve this Sudoku puzzle. Fill in the empty cells (marked as 0) "
        prompt += "with numbers 1-9 so that each row, column, and 3x3 block contains "
        prompt += "all numbers from 1 to 9 without repetition.\n\n"
        prompt += "Please provide your solution in exactly the same format as below, "
        prompt += "i.e., a 9x9 grid where each row is a list of numbers.\n"
        prompt += "Example format: Answer: [[1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 1, 2, 3], ...]\n\n"
        prompt += "Current Sudoku board:\n"
        for row in self.current_board:
            prompt += str(row) + "\n"

        observation = f"{self._get_instructions()}{prompt}"
        return observation, {"suffix": f"Solve the Sudoku puzzle (difficulty: {self.difficulty})."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's Sudoku solution.

        Args:
            action: Agent's response containing the solution board

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer using \boxed{} format first
        parsed_answer = extract_last_boxed_answer(action)

        # If not found, try "Answer:" format
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

        # Try to parse as 2D list
        try:
            board = ast.literal_eval(parsed_answer)
        except (ValueError, SyntaxError):
            obs = f"Failed to parse '{parsed_answer[:100]}...' as a 2D list."
            return obs, 0.0, True, False, {}

        # Validate format
        if not isinstance(board, list) or len(board) != self.size:
            obs = f"Invalid board format. Expected {self.size}x{self.size} grid, got {len(board)} rows."
            return obs, 0.0, True, False, {}

        for i, row in enumerate(board):
            if not isinstance(row, list) or len(row) != self.size:
                obs = f"Invalid row {i}. Expected {self.size} elements."
                return obs, 0.0, True, False, {}

        # Check that original clues are preserved
        for i in range(self.size):
            for j in range(self.size):
                if self.current_board[i][j] != 0 and self.current_board[i][j] != board[i][j]:
                    obs = f"Error: Cell ({i}, {j}) was {self.current_board[i][j]} but you changed it to {board[i][j]}."
                    return obs, 0.0, True, False, {}

        # Validate rows
        for i, row in enumerate(board):
            if not self._is_valid_unit(row):
                obs = f"Invalid row {i}: {row}. Must contain all numbers 1-9."
                return obs, 0.0, True, False, {}

        # Validate columns
        for j in range(self.size):
            column = [board[i][j] for i in range(self.size)]
            if not self._is_valid_unit(column):
                obs = f"Invalid column {j}. Must contain all numbers 1-9."
                return obs, 0.0, True, False, {}

        # Validate blocks
        for block_row in range(0, self.size, self.sqrt_size):
            for block_col in range(0, self.size, self.sqrt_size):
                block = []
                for i in range(block_row, block_row + self.sqrt_size):
                    for j in range(block_col, block_col + self.sqrt_size):
                        block.append(board[i][j])
                if not self._is_valid_unit(block):
                    obs = f"Invalid 3x3 block at ({block_row}, {block_col}). Must contain all numbers 1-9."
                    return obs, 0.0, True, False, {}

        # Success!
        obs = "Correct! You solved the Sudoku puzzle!"
        return obs, 1.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            The correct solution (for testing)
        """
        if self.solution is not None:
            return f"\\boxed{{{self.solution}}}"
        else:
            # Fallback
            board = [[i + 1 for i in range(9)] for _ in range(9)]
            return f"\\boxed{{{board}}}"
