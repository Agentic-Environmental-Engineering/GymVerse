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

"""Jiafa environment - Symbol-based arithmetic puzzle (text version)."""

import ast
import random
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class JiafaEnv(Env):
    """
    Symbol-based arithmetic puzzle environment (text version).

    The agent is shown a grid of symbols where each symbol represents a number.
    Row sums are provided, and the agent must calculate column sums.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_rows: int = 5,
        max_rows: int = 10,
        min_cols: int = 5,
        max_cols: int = 10,
        min_value: int = -100,
        max_value: int = 100,
        **_,
    ):
        """
        Initialize Jiafa environment.

        Args:
            min_rows: Minimum number of rows
            max_rows: Maximum number of rows
            min_cols: Minimum number of columns
            max_cols: Maximum number of columns
            min_value: Minimum value for symbols
            max_value: Maximum value for symbols
        """
        super().__init__()
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.min_cols = min_cols
        self.max_cols = max_cols
        self.min_value = min_value
        self.max_value = max_value

        # Game state
        self.grid = None
        self.row_sums = None
        self.col_sums = None
        self.col_count = None
        self.symbol_values = None

    def _generate_symbols(self, num_symbols: int) -> Dict[str, int]:
        """Generate unique symbols with random numerical values."""
        symbols = [
            '*', '@', '√', 'x', '+', '-', '*', '/', '%', '^', '&', '#', '$', '!', '?',
            '∑', '∆', '∏', '∫', '≈', '≠', '≥', '≤', '⊕', '⊗', '⊙', '∩', '∈', '∉',
            '⇒', '⇔', '←', '→', '↑', '↓', '∇', '∞', '∂', '∃', '∀', '¬', '∝', '⊥', '∥',
            '∅', '∴', '∵', '♠', '♣', '♥', '♦'
        ]
        chosen_symbols = random.sample(symbols, num_symbols)
        symbol_values = {symbol: random.randint(self.min_value, self.max_value)
                        for symbol in chosen_symbols}
        return symbol_values

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, analyze the symbol grid and calculate column sums.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (e.g., \"Answer: [12, 3, 12, -15]\").\n\n"
            "Alternatively, you can use \\boxed{[12, 3, 12, -15]} format.\n\n"
            "Given a rectangular grid that contains several symbols, where each symbol represents "
            "a numerical value, and provided with the sum of the elements in each row, you need to "
            "compute the sum of the elements in each column and output the result as a list.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new puzzle instance.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Keep generating until we get a full-rank matrix (unique solution)
        max_attempts = 100
        for attempt in range(max_attempts):
            row_count = random.randint(self.min_rows, self.max_rows)
            col_count = random.randint(self.min_cols, self.max_cols)

            # Number of symbols = min(row_count, col_count) - 1
            symbol_num = min(row_count, col_count) - 1
            symbol_values = self._generate_symbols(symbol_num)

            # Generate grid with random symbols
            grid = []
            for i in range(row_count):
                row = [random.choice(list(symbol_values.keys())) for _ in range(col_count)]
                grid.append(row)

            # Ensure each symbol appears at least once
            used_symbols = {symbol for row in grid for symbol in row}
            missing_symbols = set(symbol_values.keys()) - used_symbols
            if missing_symbols:
                positions = [(i, j) for i in range(row_count) for j in range(col_count)]
                random.shuffle(positions)
                for symbol in missing_symbols:
                    if positions:
                        i, j = positions.pop()
                        grid[i][j] = symbol

            # Build coefficient matrix: count occurrences of each symbol per row
            symbols = list(symbol_values.keys())
            A = np.array([[row.count(sym) for sym in symbols] for row in grid])

            # Check if matrix has full rank (ensures unique solution)
            if np.linalg.matrix_rank(A) == len(symbols):
                break
        else:
            # Fallback: use last attempt even if not full rank
            pass

        # Calculate row and column sums
        row_sums = []
        col_sums = [0] * col_count
        for i, row in enumerate(grid):
            current_row_sum = 0
            for j, symbol in enumerate(row):
                current_row_sum += symbol_values[symbol]
                col_sums[j] += symbol_values[symbol]
            row_sums.append(current_row_sum)

        # Store state
        self.grid = grid
        self.row_sums = row_sums
        self.col_sums = col_sums
        self.col_count = col_count
        self.symbol_values = symbol_values

        # Format the grid display
        grid_display = ""
        for i, row in enumerate(grid):
            grid_display += ''.join(row) + f"  {row_sums[i]}\n"

        observation = f"{self._get_instructions()}Grid:\n{grid_display}"
        return observation, {"suffix": f"Calculate column sums for this {row_count}x{col_count} grid."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's answer.

        Args:
            action: Agent's response containing column sums

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer using \boxed{} format first
        parsed_answer = extract_last_boxed_answer(action)

        # If not found, try "Answer:" format
        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(.+?)(?:\n|$)', action, re.IGNORECASE | re.MULTILINE)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: [col1, col2, ...]' or \\boxed{[col1, col2, ...]} format."
            )
            return obs, 0.0, True, False, {}

        # Try to parse as Python list
        try:
            user_answer = ast.literal_eval(parsed_answer)
            if not isinstance(user_answer, list):
                raise ValueError("Answer must be a list")
            if not all(isinstance(x, (int, float)) for x in user_answer):
                raise ValueError("All elements must be numbers")
            if len(user_answer) != self.col_count:
                obs = f"Your answer has {len(user_answer)} values, but the grid has {self.col_count} columns."
                return obs, 0.0, True, False, {}
        except (ValueError, SyntaxError) as e:
            obs = f"Failed to parse your answer as a list: {e}"
            return obs, 0.0, True, False, {}

        # Check if answer is correct (allow small floating point errors)
        is_correct = all(abs(user_answer[i] - self.col_sums[i]) < 0.001
                        for i in range(self.col_count))
        reward = 1.0 if is_correct else 0.0

        if is_correct:
            obs = f"Correct! The column sums are {self.col_sums}."
        else:
            obs = f"Incorrect. Your answer was {user_answer}, but the correct column sums are {self.col_sums}."

        return obs, reward, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random column sums (for testing, returns correct answer)
        """
        if self.col_sums is not None:
            return f"\\boxed{{{self.col_sums}}}"
        else:
            # Return a random list
            return f"\\boxed{{[{', '.join(str(random.randint(self.min_value, self.max_value)) for _ in range(5))}]}}"
