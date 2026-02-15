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

"""MapPositionSimulation environment - Map navigation puzzle with special elements."""

import random
import math
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class MapPositionSimulationEnv(Env):
    """
    Map Position Simulation environment.

    A puzzle game where players must predict the final position of a player
    navigating a map with special elements (portals, jumpers, traps, etc.)
    after executing a sequence of moves.

    This is a single-turn environment with sparse reward.
    """

    def __init__(
        self,
        min_size: int = 10,
        max_size: int = 50,
        min_steps: int = 10,
        max_steps: int = 50,
        **_,
    ):
        """
        Initialize MapPositionSimulation environment.

        Args:
            min_size: Minimum map size
            max_size: Maximum map size
            min_steps: Minimum number of steps
            max_steps: Maximum number of steps
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.answer = None

    def _generate_map(self, seed: int, rows: int, cols: int, num_steps: int) -> Tuple[List[List[str]], List[str]]:
        """Generate map and move sequence."""
        random.seed(seed)
        area = (rows - 2) * (cols - 2)
        portal_num_max = math.ceil(area * 0.05)
        jatr_num_max = math.ceil(area * 0.4) // 4

        if area <= 1 + portal_num_max * 2 + jatr_num_max:
            return None, None

        # Initialize map: interior is E, boundaries are W
        game_map = [['E' for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    game_map[i][j] = 'W'

        # Place player P
        possible_positions = [(i, j) for i in range(1, rows - 1) for j in range(1, cols - 1)]
        if not possible_positions:
            return None, None
        p_pos = random.choice(possible_positions)
        possible_positions.remove(p_pos)
        game_map[p_pos[0]][p_pos[1]] = 'P'

        # Place portals (paired numbers)
        portal_num = random.randint(1, portal_num_max) if portal_num_max >= 1 else 1
        portal_id = 1
        for _ in range(portal_num):
            if len(possible_positions) >= 2:
                pos1 = random.choice(possible_positions)
                possible_positions.remove(pos1)
                pos2 = random.choice(possible_positions)
                possible_positions.remove(pos2)
                game_map[pos1[0]][pos1[1]] = str(portal_id)
                game_map[pos2[0]][pos2[1]] = str(portal_id)
                portal_id += 1

        # Place other elements: J (jumper), A (reverser), T (trap), R (repeater)
        elements = ['J', 'A', 'T', 'R']
        for elem in elements:
            count = random.randint(0, jatr_num_max)
            for _ in range(count):
                if possible_positions:
                    pos = random.choice(possible_positions)
                    possible_positions.remove(pos)
                    game_map[pos[0]][pos[1]] = elem

        # Generate move sequence
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        task = [random.choice(directions) for _ in range(num_steps)]
        return game_map, task

    def _simulate(self, game_map: List[List[str]], task: List[str]) -> Optional[Tuple[int, int]]:
        """Simulate player movement and return final position."""
        rows = len(game_map)
        cols = len(game_map[0]) if rows > 0 else 0

        # Find player start position
        start_pos = None
        for i in range(rows):
            for j in range(cols):
                if game_map[i][j] == 'P':
                    start_pos = (i, j)
                    break
            if start_pos:
                break
        if not start_pos:
            return None

        current_pos = start_pos
        action_idx = 0
        trapped = 0
        repeated_action = None
        outer_loop_count = 0

        while action_idx < len(task):
            outer_loop_count += 1
            if outer_loop_count > 200:
                return None

            if trapped > 0:
                trapped -= 1
                action_idx += 1
                continue

            if repeated_action is not None:
                current_action = repeated_action
                repeated_action = None
            else:
                current_action = task[action_idx]
                action_idx += 1

            dx, dy = 0, 0
            if current_action == 'UP':
                dx = -1
            elif current_action == 'DOWN':
                dx = 1
            elif current_action == 'LEFT':
                dy = -1
            elif current_action == 'RIGHT':
                dy = 1

            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy

            if not (0 <= new_x < rows and 0 <= new_y < cols):
                new_x, new_y = current_pos
                element = 'W'
            else:
                element = game_map[new_x][new_y]

            inner_loop_count = 0
            while True:
                inner_loop_count += 1
                if inner_loop_count > 200:
                    return None

                if element == 'W':
                    new_x, new_y = current_pos
                    break

                if element.isdigit():
                    # Portal: find matching number
                    other = None
                    for i in range(rows):
                        for j in range(cols):
                            if game_map[i][j] == element and (i, j) != (new_x, new_y):
                                other = (i, j)
                                break
                        if other:
                            break
                    if other:
                        new_x, new_y = other
                    break

                elif element == 'J':
                    # Jumper: leap 2 steps in current direction
                    jump_x = new_x + dx * 2
                    jump_y = new_y + dy * 2
                    if 0 <= jump_x < rows and 0 <= jump_y < cols and game_map[jump_x][jump_y] != 'W':
                        new_x, new_y = jump_x, jump_y
                        element = game_map[new_x][new_y]
                    else:
                        element = 'E'
                        break

                elif element == 'A':
                    # Reverser: reverse direction
                    dx, dy = -dx, -dy
                    rev_x = current_pos[0] + dx
                    rev_y = current_pos[1] + dy
                    if 0 <= rev_x < rows and 0 <= rev_y < cols and game_map[rev_x][rev_y] != 'W':
                        new_x, new_y = rev_x, rev_y
                        element = game_map[new_x][new_y]
                    else:
                        new_x, new_y = current_pos
                        element = 'E'
                        break

                elif element == 'T':
                    # Trap: next move is skipped
                    trapped = 1
                    break

                elif element == 'R':
                    # Repeater: extra move in same direction
                    repeated_action = current_action
                    break

                else:
                    break

            current_pos = (new_x, new_y)

        return current_pos

    def _generate_puzzle(self, seed: int) -> Tuple[str, Tuple[int, int]]:
        """Generate complete puzzle with validation."""
        random.seed(seed)
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            rows = random.randint(self.min_size, self.max_size)
            cols = random.randint(self.min_size, self.max_size)
            num_steps = random.randint(self.min_steps, self.max_steps)

            game_map, task = self._generate_map(seed + attempts, rows, cols, num_steps)
            if game_map is None or task is None:
                attempts += 1
                continue

            result = self._simulate(game_map, task)
            if result is None:
                attempts += 1
                continue

            # Generate question text
            board_str = "\n".join([" ".join(row) for row in game_map])
            task_str = ", ".join(task)

            question = (
                "You are a good game player, I'll give you a game board and rules.\n"
                "Your task is:\n"
                "- First, give your answer according to the game board and rules.\n"
                "- Second, output the answer in the required format. The last line of your response "
                "should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where "
                "YOUR_ANSWER is your final answer to the question, e.g. 'Answer: (3, 12)'.\n\n"
                "Alternatively, you can use \\boxed{(3, 12)} format.\n\n"
                "You will be given an n*n map containing the following elements:\n"
                "  - Player (P)\n"
                "  - Empty cell (E)\n"
                "  - Portal (paired with matching numbers): Stepping onto one portal will teleport the "
                "player to the other portal with the same number.\n"
                "  - Jumper (J): Stepping onto a jumper will cause the player to leap two steps in the "
                "current direction, skipping the cell in between.\n"
                "  - Wall (W): A wall blocks the player's movement.\n"
                "  - Reverser (A): The direction of movement will be reversed when passing through a reverser.\n"
                "  - Trap (T): Stepping into a trap will trap the player for one turn, making the next move ineffective.\n"
                "  - Repeater (R): Stepping onto a repeater causes the player to move an extra step in the same direction.\n\n"
                "Additional Rules:\n"
                "  - Map elements can be combined.\n"
                "  - Elements that have already been triggered during the current turn will not trigger again (except for walls).\n"
                "  - The map boundaries are all walls to prevent going out of bounds.\n"
                "  - Map coordinates start from (0,0), i.e., the top-left corner is (0,0).\n\n"
                "Based on the given map and the move sequence, determine the player's final position after "
                "executing all moves.\n\n"
                f"Map:\n{board_str}\n\n"
                f"Move sequence:\n{task_str}\n\n"
                "Please output the final player coordinate in the following format: 'Answer: (row, col)', "
                "e.g. 'Answer: (3, 12)'"
            )

            return question, result

        # Fallback with simple map
        return "Failed to generate valid puzzle", (0, 0)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new MapPositionSimulation puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        question, self.answer = self._generate_puzzle(puzzle_seed)

        return question, {
            "suffix": f"Map size: {self.min_size}-{self.max_size}, Steps: {self.min_steps}-{self.max_steps}."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process player's answer.

        Args:
            action: Player's coordinate answer as "(row, col)"

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
                "Please use 'Answer: (row, col)' or \\boxed{(row, col)} format."
            )
            return obs, 0.0, True, False, {}

        # Parse coordinate
        try:
            import ast
            user_answer = ast.literal_eval(str(parsed_action).strip())
            if not (isinstance(user_answer, tuple) or isinstance(user_answer, list)):
                raise ValueError("Answer must be tuple or list")
            user_answer = tuple(user_answer)
        except Exception as e:
            obs = f"Failed to parse coordinate answer: {parsed_action}. Error: {e}"
            return obs, 0.0, True, False, {}

        # Check answer
        if user_answer == self.answer:
            obs = f"Correct! The final position is {self.answer}."
            return obs, 1.0, True, False, {}
        else:
            obs = f"Incorrect. Your answer: {user_answer}, Correct answer: {self.answer}."
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample the correct answer.

        Returns:
            Correct answer as string
        """
        return f"\\boxed{{{self.answer}}}"
