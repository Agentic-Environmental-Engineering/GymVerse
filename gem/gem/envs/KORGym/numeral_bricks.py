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

"""Numeral Bricks environment - Cross-shaped region tiling puzzle."""

import random
import ast
from typing import Optional, Tuple, Dict, Any, List, Set

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class NumeralBricksEnv(Env):
    """
    Numeral Bricks environment.

    Players must fill a grid by coloring cells based on numeric clues. Each
    letter indicates a region, and the number specifies how many additional
    cells should be colored from that position in cross (plus sign) pattern.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        rows: int = 5,
        cols: int = 6,
        **_,
    ):
        """
        Initialize Numeral Bricks environment.

        Args:
            rows: Grid height
            cols: Grid width
        """
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.sol_board = None
        self.puzzle_board = None
        self.clues = None

    def _generate_candidates(self, max_region_size: Optional[int] = None) -> List[Dict]:
        """Generate all candidate cross-shaped regions."""
        candidates = []
        for r in range(self.rows):
            for c in range(self.cols):
                max_left = c
                max_right = self.cols - 1 - c
                max_up = r
                max_down = self.rows - 1 - r

                for a in range(max_left + 1):  # Extend left
                    for b in range(max_right + 1):  # Extend right
                        for u in range(max_up + 1):  # Extend up
                            for d in range(max_down + 1):  # Extend down
                                if a == 0 and b == 0 and u == 0 and d == 0:
                                    continue  # Skip size-1 regions

                                cells = set()
                                # Horizontal direction
                                for j in range(c - a, c + b + 1):
                                    cells.add((r, j))
                                # Vertical direction
                                for i in range(r - u, r + d + 1):
                                    cells.add((i, c))

                                region_size = len(cells)
                                if max_region_size is not None and region_size > max_region_size:
                                    continue

                                candidate = {
                                    'clue': (r, c),
                                    'left': a,
                                    'right': b,
                                    'up': u,
                                    'down': d,
                                    'cells': cells,
                                    'digit': region_size - 1
                                }
                                candidates.append(candidate)
        return candidates

    def _solve_tiling(self, candidates: List[Dict]) -> Optional[List[Dict]]:
        """Solve tiling problem using backtracking with MRV heuristic."""
        board_cells = {(i, j) for i in range(self.rows) for j in range(self.cols)}

        # Build mapping from cell to candidates
        cell_to_candidates: Dict[tuple, List[Dict]] = {
            (i, j): [] for i in range(self.rows) for j in range(self.cols)
        }
        for cand in candidates:
            for cell in cand['cells']:
                if cell in cell_to_candidates:
                    cell_to_candidates[cell].append(cand)

        def backtrack(covered: Set[tuple], solution: List[Dict]) -> Optional[List[Dict]]:
            if covered == board_cells:
                return solution

            # MRV heuristic: choose uncovered cell with fewest candidates
            uncovered = board_cells - covered
            best_cell = None
            best_cands = None
            best_count = float('inf')

            for cell in uncovered:
                valid_cands = [cand for cand in cell_to_candidates[cell]
                               if cand['cells'].isdisjoint(covered)]
                if len(valid_cands) < best_count:
                    best_count = len(valid_cands)
                    best_cell = cell
                    best_cands = valid_cands
                if best_count == 0:
                    return None  # Prune: no candidates for this cell

            random.shuffle(best_cands)
            for cand in best_cands:
                new_covered = covered.union(cand['cells'])
                solution.append(cand)
                res = backtrack(new_covered, solution)
                if res is not None:
                    return res
                solution.pop()
            return None

        return backtrack(set(), [])

    def _solve_tiling_with_retries(self, seed: int, max_attempts: int = 50) -> Optional[List[Dict]]:
        """Attempt to solve tiling with multiple retries."""
        random.seed(seed)
        for _ in range(max_attempts):
            candidates = self._generate_candidates()
            solution = self._solve_tiling(candidates)
            if solution is not None:
                return solution
        return None

    def _create_boards(self, solution: List[Dict]) -> Tuple[List[List[str]], List[List[str]], Dict[str, int]]:
        """Create solution board, puzzle board, and clues from solution."""
        sol_board = [['' for _ in range(self.cols)] for _ in range(self.rows)]
        puzzle_board = [['0' for _ in range(self.cols)] for _ in range(self.rows)]
        clues = {}

        for index, region in enumerate(solution):
            letter = chr(97 + index)  # a, b, c, ...
            r, c = region['clue']
            clues[letter] = region['digit']

            for (i, j) in region['cells']:
                sol_board[i][j] = letter

            puzzle_board[r][c] = letter

        return sol_board, puzzle_board, clues

    def _board_to_string(self, board: List[List[str]]) -> str:
        """Convert 2D board to string."""
        return "\n".join("".join(row) for row in board)

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            'answer, e.g., "Answer: [[\'c\',\'a\',\'a\',...],[\'c\',\'a\',\'b\',...]...]"\n\n'
            'Alternatively, you can use \\boxed{[[\'c\',\'a\',\'a\'],[\'c\',\'a\',\'b\'],...]} format.\n\n'
            "You need to fill the grid by coloring cells based on the numeric mappings. The letter indicates the color of the cell, and each cell containing a digit indicates the total number of adjacent cells (including diagonal, horizontal, or vertical directions) that should be colored starting from that cell. For example, if a cell contains the number 5, you must color exactly 5 cells adjacent to it, forming a path in any combination of directions. For instance, you might move 3 steps to the right, 1 step up, and 1 step down from the starting cell, totaling 5 colored cells. You are free to choose the direction of each segment, but the total number of colored cells must exactly match the number in the starting cell. Each segment must be a straight line — once a direction is chosen, you must continue in that direction without turning. The number specified in the mapping refers to the number of additional cells to be colored, excluding the starting cell itself.\n"
            "For example, if the board is:\n"
            "0a000\n"
            "00b00\n"
            "c0000\n"
            "00de0\n"
            "0f00g\n"
            "and a:4 b:2 c:3 d:3 e:1 f:1 g:5\n"
            "The game answer can be:\n"
            "caaaa\n"
            "cabbb\n"
            "ccdeg\n"
            "dddeg\n"
            "ffggg\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new Numeral Bricks puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        solution = self._solve_tiling_with_retries(puzzle_seed)

        if solution is None:
            raise RuntimeError("Failed to generate valid puzzle. Please try again.")

        self.sol_board, self.puzzle_board, self.clues = self._create_boards(solution)

        # Build observation
        board_str = "Board:\n" + self._board_to_string(self.puzzle_board) + "\n\n"
        if self.clues:
            board_str += "Mapping (Letter : Number, where Number = Expansion Grid Count):\n"
            for letter in sorted(self.clues.keys()):
                board_str += f"{letter} : {self.clues[letter]}\n"

        observation = f"{self._get_instructions()}{board_str}"

        return observation, {
            "suffix": f"Grid size: {self.rows}x{self.cols}, {len(self.clues)} regions."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Verify the solution.

        Args:
            action: Player's filled grid as 2D list string

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
                "Please use 'Answer: [[...], [...], ...]' or \\boxed{[[...], [...], ...]} format."
            )
            return obs, 0.0, True, False, {}

        # Parse 2D list
        try:
            if isinstance(parsed_action, str):
                generated_answer = ast.literal_eval(parsed_action)
            else:
                generated_answer = parsed_action
        except Exception as e:
            obs = f"Failed to parse answer: {parsed_action}. Error: {e}"
            return obs, 0.0, True, False, {}

        # Check format
        if not (isinstance(generated_answer, list) and len(generated_answer) == self.rows and
                all(isinstance(row, list) and len(row) == self.cols for row in generated_answer)):
            obs = f"Answer size mismatch. Expected {self.rows}x{self.cols}, got {len(generated_answer)}x{len(generated_answer[0]) if generated_answer else 0}."
            return obs, 0.0, True, False, {}

        # Verify solution
        valid = self._verify_solution(generated_answer)

        if valid:
            obs = (
                "Correct! You successfully filled the grid.\n"
                f"Your solution:\n{self._board_to_string(generated_answer)}"
            )
            reward = 1.0
        else:
            obs = (
                "Incorrect solution. The filled grid does not match the rules.\n"
                f"Your answer:\n{self._board_to_string(generated_answer)}"
            )
            reward = 0.0

        return obs, reward, True, False, {}

    def _verify_solution(self, generated_answer: List[List[str]]) -> bool:
        """Verify if the generated answer is valid."""
        valid_letters = set(self.clues.keys())

        # Check all cells contain valid letters
        for r in range(self.rows):
            for c in range(self.cols):
                if generated_answer[r][c] not in valid_letters:
                    return False

        for letter, clue_val in self.clues.items():
            # Find clue position in puzzle_board
            clue_positions = [(r, c) for r in range(self.rows) for c in range(self.cols)
                              if self.puzzle_board[r][c] == letter]
            if len(clue_positions) != 1:
                return False

            clue_r, clue_c = clue_positions[0]

            # Check generated_answer has letter at clue position
            if generated_answer[clue_r][clue_c] != letter:
                return False

            # Get all cells with this letter
            region_cells = {(r, c) for r in range(self.rows) for c in range(self.cols)
                            if generated_answer[r][c] == letter}

            # All cells must be in same row or column as clue
            for (r, c) in region_cells:
                if r != clue_r and c != clue_c:
                    return False

            # Scan horizontal segment
            row_segment = set()
            c_left = clue_c
            while c_left >= 0 and generated_answer[clue_r][c_left] == letter:
                row_segment.add((clue_r, c_left))
                c_left -= 1
            c_right = clue_c + 1
            while c_right < self.cols and generated_answer[clue_r][c_right] == letter:
                row_segment.add((clue_r, c_right))
                c_right += 1

            # Scan vertical segment
            col_segment = set()
            r_up = clue_r - 1
            while r_up >= 0 and generated_answer[r_up][clue_c] == letter:
                col_segment.add((r_up, clue_c))
                r_up -= 1
            r_down = clue_r + 1
            while r_down < self.rows and generated_answer[r_down][clue_c] == letter:
                col_segment.add((r_down, clue_c))
                r_down += 1

            expected_region = row_segment | col_segment
            if expected_region != region_cells:
                return False

            horizontal_extension = len(row_segment) - 1
            vertical_extension = len(col_segment) - 1
            expected_clue = horizontal_extension + vertical_extension
            if abs(expected_clue - clue_val) > 1:
                return False

        return True

    def sample_random_action(self) -> str:
        """
        Sample the correct solution.

        Returns:
            Correct solution as string
        """
        return f"\\boxed{{{repr(self.sol_board)}}}"
