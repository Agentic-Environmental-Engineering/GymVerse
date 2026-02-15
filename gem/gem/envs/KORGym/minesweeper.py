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

"""Minesweeper environment - Classic minesweeper game."""

import random
from typing import Optional, Tuple, Dict, Any, List
from collections import deque

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class MinesweeperEnv(Env):
    """
    Minesweeper environment.

    A classic puzzle game where players must uncover all safe cells without
    hitting mines, while using number clues to locate and flag mines.

    This is a multi-turn environment with partial reward based on correctly flagged mines.
    """

    def __init__(
        self,
        min_size: int = 10,
        max_size: int = 20,
        mine_ratio: float = 0.15,
        max_turns: int = 100,
        **_,
    ):
        """
        Initialize Minesweeper environment.

        Args:
            min_size: Minimum board size
            max_size: Maximum board size
            mine_ratio: Ratio of mines to total cells (approximate)
            max_turns: Maximum number of turns allowed
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.mine_ratio = mine_ratio
        self.max_turns = max_turns
        self.actual = None
        self.mask = None
        self.rows = None
        self.cols = None
        self.mines = None
        self.turn = None
        self.score = None
        self.unflags = None

    def _generate_board(self, seed: int):
        """Generate minesweeper board."""
        random.seed(seed)
        n = random.randint(self.min_size, self.max_size)
        self.rows = n
        self.cols = n
        self.mines = random.randint(int(n * n * self.mine_ratio * 0.5),
                                     int(n * n * self.mine_ratio * 1.5))
        self.mines = min(self.mines, self.rows * self.cols - 1)

        # Generate mine positions
        positions = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        mine_pos = random.sample(positions, self.mines)

        # Initialize actual board (-1 = mine)
        self.actual = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i, j in mine_pos:
            self.actual[i][j] = -1

        # Calculate numbers
        directions = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

        for i in range(self.rows):
            for j in range(self.cols):
                if self.actual[i][j] == -1:
                    continue
                count = sum(1 for dx, dy in directions
                           if 0 <= i+dx < self.rows and 0 <= j+dy < self.cols
                           and self.actual[i+dx][j+dy] == -1)
                self.actual[i][j] = count

        # Initialize mask (what player sees)
        self.mask = [['?' for _ in range(self.cols)] for _ in range(self.rows)]
        self.turn = 0
        self.score = 0.0
        self.unflags = 0

    def _reveal_empty(self, start_r: int, start_c: int):
        """Reveal empty cells using flood fill."""
        visited = set()
        queue = deque([(start_r, start_c)])

        # If starting cell is not 0, also check adjacent 0 cells
        if self.actual[start_r][start_c] != 0:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = start_r + dr, start_c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and self.actual[nr][nc] == 0:
                        queue.append((nr, nc))

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            if self.mask[r][c] != '?':
                continue

            # Reveal cell
            self.mask[r][c] = str(self.actual[r][c]) if self.actual[r][c] > 0 else '0'

            # If cell is empty, add neighbors
            if self.actual[r][c] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in visited:
                            queue.append((nr, nc))

    def _check_victory(self) -> bool:
        """Check if player has won."""
        total_flags = sum(row.count('F') for row in self.mask)
        correct_flags = sum(1 for i in range(self.rows) for j in range(self.cols)
                          if self.mask[i][j] == 'F' and self.actual[i][j] == -1)
        return total_flags == self.mines and correct_flags == self.mines

    def _get_board_string(self) -> str:
        """Get current board state as string."""
        flags = sum(row.count('F') for row in self.mask)
        board_lines = [
            f"Score: {self.score:.2f}, Flags: {flags}/{self.mines}, Unflags: {self.unflags}",
            f"Turn: {self.turn}/{self.max_turns}",
            "Current Board:"
        ]
        for row in self.mask:
            board_lines.append(" ".join(f"{cell:2}" for cell in row))
        return "\n".join(board_lines)

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "Minesweeper Game Rules:\n"
            "1. The board is a grid with hidden mines. The coordinate of the top-leftmost grid is (0, 0).\n"
            "2. Input Format:\n"
            "   - Uncover a cell: 'uncover (row, col)' e.g., 'uncover (3,4)'\n"
            "   - Flag a mine: 'flag (row, col)' e.g., 'flag (0,0)'\n"
            "   - Unflag a cell: 'unflag (row, col)' e.g., 'unflag (0,0)'\n"
            "3. Win Condition: Correctly flag all mines or uncover all safe cells.\n"
            "4. The meanings of the blocks are as follows:\n"
            "    - ?: Unknown block\n"
            "    - Number: The total number of mines in the eight adjacent cells\n"
            "    - F: Flagged block\n"
            f"5. The game will end at the {self.max_turns}th turn or you uncover a mine.\n"
            "6. The final score is calculated as follows: the mines you flag correctly / total mines.\n\n"
            f"{self._get_board_string()}\n\n"
            "Please output your action in the following format: 'Answer: uncover (3,4)'\n"
            "Alternatively, you can use \\boxed{uncover (3,4)} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new Minesweeper game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate board
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        self._generate_board(puzzle_seed)

        observation = self._get_instructions()

        return observation, {
            "suffix": f"Board: {self.rows}x{self.cols}, Mines: {self.mines}."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process player's action.

        Args:
            action: Player's action (uncover/flag/unflag coordinates)

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
                "Your response did not contain a valid action. "
                "Please use 'Answer: uncover (x,y)' or \\boxed{uncover (x,y)} format."
            )
            return obs, self.score, True, False, {}

        # Parse command and coordinates
        try:
            action_str = str(parsed_action).strip().lower()
            cmd, pos = action_str.split(' ', 1)
            if pos.startswith('(') and pos.endswith(')'):
                pos = pos[1:-1]
            row, col = map(int, pos.split(','))
        except Exception as e:
            obs = f"Failed to parse action: {parsed_action}. Error: {e}"
            return obs, self.score, True, False, {}

        # Validate coordinates
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            obs = f"Invalid coordinates: ({row}, {col}). Must be within board bounds."
            return obs, self.score, True, False, {}

        current_flags = sum(r.count('F') for r in self.mask)

        # Process command
        if cmd == 'uncover':
            if self.mask[row][col] != '?':
                obs = f"Cell ({row}, {col}) is already revealed or flagged."
                return obs, self.score, True, False, {}

            if self.actual[row][col] == -1:
                # Hit a mine - game over
                for i in range(self.rows):
                    for j in range(self.cols):
                        if self.actual[i][j] == -1:
                            self.mask[i][j] = 'X'

                obs = (
                    f"Game Over! You hit a mine at ({row}, {col}).\n\n"
                    f"{self._get_board_string()}\n\n"
                    f"Final Score: {self.score:.2f}"
                )
                return obs, self.score, True, False, {}
            else:
                # Safe cell - reveal
                self._reveal_empty(row, col)

                # Check if all safe cells revealed
                safe_cells = self.rows * self.cols - self.mines
                revealed = sum(1 for i in range(self.rows) for j in range(self.cols)
                             if self.mask[i][j] not in ['?', 'F'])

                if revealed == safe_cells:
                    self.score = 1.0
                    obs = (
                        f"Congratulations! You uncovered all safe cells!\n\n"
                        f"{self._get_board_string()}\n\n"
                        f"Final Score: {self.score:.2f}"
                    )
                    return obs, self.score, True, False, {}

        elif cmd == 'flag':
            if self.mask[row][col] != '?':
                obs = f"Cell ({row}, {col}) cannot be flagged (already revealed or flagged)."
                return obs, self.score, True, False, {}

            if current_flags >= self.mines:
                obs = f"Cannot place more flags. Already placed {current_flags}/{self.mines} flags."
                return obs, self.score, True, False, {}

            self.mask[row][col] = 'F'
            if self.actual[row][col] == -1:
                self.score += 1.0 / self.mines

        elif cmd == 'unflag':
            if self.mask[row][col] != 'F':
                obs = f"Cell ({row}, {col}) is not flagged."
                return obs, self.score, True, False, {}

            self.mask[row][col] = '?'
            self.unflags += 1
            if self.actual[row][col] == -1:
                self.score -= 1.0 / self.mines

        else:
            obs = f"Unknown command: {cmd}. Use 'uncover', 'flag', or 'unflag'."
            return obs, self.score, True, False, {}

        # Check victory
        if self._check_victory():
            self.score = 1.0
            obs = (
                f"Congratulations! You correctly flagged all mines!\n\n"
                f"{self._get_board_string()}\n\n"
                f"Final Score: {self.score:.2f}"
            )
            return obs, self.score, True, False, {}

        self.turn += 1

        # Check max turns
        if self.turn >= self.max_turns:
            obs = (
                f"Maximum turns ({self.max_turns}) reached.\n\n"
                f"{self._get_board_string()}\n\n"
                f"Final Score: {self.score:.2f}"
            )
            return obs, self.score, True, False, {}

        # Continue game
        observation = self._get_instructions()
        return observation, self.score, False, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a safe action (uncover a safe cell or flag a mine).

        Returns:
            Action as string
        """
        # Strategy: Find a safe cell to uncover or a mine to flag
        # First try to find a cell with 0 adjacent mines
        for i in range(self.rows):
            for j in range(self.cols):
                if self.mask[i][j] == '?' and self.actual[i][j] == 0:
                    return f"\\boxed{{uncover ({i},{j})}}"

        # Find any safe cell
        for i in range(self.rows):
            for j in range(self.cols):
                if self.mask[i][j] == '?' and self.actual[i][j] != -1:
                    return f"\\boxed{{uncover ({i},{j})}}"

        # Find a mine to flag
        for i in range(self.rows):
            for j in range(self.cols):
                if self.mask[i][j] == '?' and self.actual[i][j] == -1:
                    return f"\\boxed{{flag ({i},{j})}}"

        # Fallback
        return "\\boxed{uncover (0,0)}"
