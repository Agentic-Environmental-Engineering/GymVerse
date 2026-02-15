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

"""Tower of Hanoi environment - Classic puzzle game."""

import random
import ast
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class TowerOfHanoiEnv(Env):
    """
    Tower of Hanoi environment.

    Classic puzzle with 4 columns and 5 disks. Move all disks to the target
    column (column 4) following the rules:
    - Only move one disk at a time
    - Only take the top disk from a column
    - Never place a larger disk on a smaller disk

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_level: int = 1,
        max_level: int = 13,
        **_,
    ):
        """
        Initialize Tower of Hanoi environment.

        Args:
            min_level: Minimum difficulty level (minimum steps needed)
            max_level: Maximum difficulty level (maximum steps needed)
        """
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.current_state = None
        self.difficulty = None

    def _generate_state(self, seed: int, level: int) -> List[str]:
        """
        Generate initial state for given difficulty level using BFS.

        Args:
            seed: Random seed
            level: Difficulty level (number of steps needed to solve)

        Returns:
            Initial state as list of column strings
        """
        # Target state: all disks on column 4 in ascending order
        target_state = (tuple([]), tuple([]), tuple([]), tuple(['a', 'b', 'c', 'd', 'e']))

        # BFS to pre-compute all states and their minimum steps
        visited = {}
        queue = deque([(target_state, 0)])
        visited[target_state] = 0

        def can_place(column: Tuple[str, ...], disk: str) -> bool:
            """Check if disk can be placed on top of column."""
            return not column or disk < column[0]

        def get_prev_states(state: Tuple[Tuple[str, ...], ...]) -> List[Tuple[Tuple[Tuple[str, ...], ...], Tuple[str, int]]]:
            """Get all previous states (reverse moves from current state)."""
            prev_states = []
            # Try all possible reverse moves
            for dest in range(4):
                if not state[dest]:
                    continue
                # Take top disk from destination
                disk_to_move = state[dest][0]
                # Try moving to other columns (source)
                for src in range(4):
                    if src == dest:
                        continue
                    # Check if reverse move is valid
                    if can_place(state[src], disk_to_move):
                        # Create new state
                        new_dest = list(state[dest][1:])
                        new_src = [disk_to_move] + list(state[src])
                        new_state = list(state)
                        new_state = list(map(list, new_state))
                        new_state[dest] = new_dest
                        new_state[src] = new_src
                        new_state = tuple(map(tuple, new_state))
                        prev_states.append((new_state, (disk_to_move, src + 1)))
            return prev_states

        # BFS to find all reachable states
        while queue:
            current, steps = queue.popleft()
            for next_state, move in get_prev_states(current):
                if next_state not in visited:
                    visited[next_state] = steps + 1
                    queue.append((next_state, steps + 1))

        # Group states by steps
        steps_dict = {}
        for state, s in visited.items():
            if s not in steps_dict:
                steps_dict[s] = []
            steps_dict[s].append(state)

        # Select state based on level
        random.seed(seed)
        if level not in steps_dict:
            # If requested level not available, use closest available level
            level = min(steps_dict.keys(), key=lambda x: abs(x - level))

        chosen_state = random.choice(steps_dict[level])

        # Convert tuples to formatted strings
        formatted_state = []
        for col in chosen_state:
            if not col:
                formatted_state.append('null')
            else:
                formatted_state.append(','.join(col))

        return formatted_state

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            'answer, e.g., \'Answer: [("a", 2), ("b", 4)]\'\n\n'
            'Alternatively, you can use \\boxed{[("a", 2), ("b", 4)]} format.\n\n'
            "The Tower of Hanoi problem consists of four columns and five disks. The objective is to move all the disks to the target column. "
            "The rules are as follows: each move can only move one disk; you can only take off the top disk from a column; "
            "and you can never place a larger disk on top of a smaller one. The disks are labeled as a, b, c, d, e in ascending order of size.\n"
            "The initial state of the Hanoi Tower is similar to: 1: null, 2: a, b, 3: c, d, 4: e. This means column 1 has no disks; "
            "column 2 has disks a and b; column 3 has disks c and d; and column 4 has disk e. Note that column 4 is the target column.\n"
            "Your answer should be a list of moves in the format [(disk, target_column), ...], e.g., [('a', 2), ('b', 4)]. "
            "Here, ('a', 2) means moving disk a to column 2.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new Tower of Hanoi puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Select random difficulty level
        self.difficulty = random.randint(self.min_level, self.max_level)

        # Generate initial state
        self.current_state = self._generate_state(seed if seed else random.randint(0, 1000000), self.difficulty)

        # Build observation
        columns_desc = []
        for idx, col in enumerate(self.current_state):
            column_number = idx + 1
            if col == 'null':
                disks_desc = 'null'
            else:
                disks_desc = col
            columns_desc.append(f"{column_number}: {disks_desc}")
        current_state_str = ", ".join(columns_desc)

        observation = (
            f"{self._get_instructions()}"
            f"Current state of columns: {current_state_str}"
        )

        return observation, {
            "suffix": f"Difficulty level: {self.difficulty} steps needed."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Verify the solution.

        Args:
            action: List of moves as string, e.g., "[('a', 2), ('b', 4)]"

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        parsed_action = extract_last_boxed_answer(action)

        if parsed_action is None:
            import re
            match = re.search(r'Answer:\\s*(.+)', action, re.IGNORECASE)
            if match:
                parsed_action = match.group(1).strip()

        if parsed_action is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: [(disk, column), ...]' or \\boxed{[(disk, column), ...]} format."
            )
            return obs, 0.0, True, False, {}

        # Try to parse the move list
        try:
            if isinstance(parsed_action, str):
                moves = ast.literal_eval(parsed_action)
            else:
                moves = parsed_action
        except Exception as e:
            obs = f"Failed to parse moves: {parsed_action}. Error: {e}"
            return obs, 0.0, True, False, {}

        # Verify solution
        reward = self._verify_solution(self.current_state, moves)

        if reward == 1.0:
            obs = (
                f"Correct! You successfully solved the Tower of Hanoi puzzle in {len(moves)} moves.\n"
                f"Your solution: {moves}"
            )
        else:
            obs = (
                f"Incorrect solution. Your moves: {moves}\n"
                "Either the moves are invalid or the final state is not correct."
            )

        return obs, reward, True, False, {"difficulty": self.difficulty, "num_moves": len(moves) if isinstance(moves, list) else 0}

    def _verify_solution(self, state: List[str], moves: List[Tuple[str, int]]) -> float:
        """
        Verify if the move sequence solves the puzzle.

        Args:
            state: Initial state
            moves: List of (disk, target_column) tuples

        Returns:
            1.0 if solution is correct, 0.0 otherwise
        """
        # Convert state to internal representation
        current_state = []
        for col in state:
            if col == 'null':
                current_state.append([])
            else:
                current_state.append(col.split(','))

        target_column = 3  # Target is column 4 (index 3)

        # Validate each move
        for act in moves:
            # Parse move
            if not isinstance(act, tuple) or len(act) != 2:
                return 0.0

            disk, dest_col = act
            dest_col_idx = dest_col - 1  # Convert to 0-based index

            if dest_col_idx < 0 or dest_col_idx >= 4:
                return 0.0

            # Find source column
            src_col_idx = None
            for i in range(4):
                if current_state[i] and current_state[i][0] == disk:
                    src_col_idx = i
                    break

            if src_col_idx is None:
                return 0.0  # Disk not on top of any column

            # Check if destination allows placement
            dest_disks = current_state[dest_col_idx]
            if dest_disks:
                dest_top = dest_disks[0]
                if disk >= dest_top:
                    return 0.0  # Cannot place larger disk on smaller disk

            # Execute move
            current_state[src_col_idx].pop(0)
            current_state[dest_col_idx].insert(0, disk)

        # Verify final state
        target_disks = current_state[target_column]
        if len(target_disks) != 5:
            return 0.0

        expected = ['a', 'b', 'c', 'd', 'e']
        for i in range(5):
            if target_disks[i] != expected[i]:
                return 0.0

        return 1.0

    def sample_random_action(self) -> str:
        """
        Sample a random valid solution using BFS.

        Returns:
            Valid solution as string
        """
        # Convert current state to tuple format for BFS
        state_tuples = []
        for col in self.current_state:
            if col == 'null':
                state_tuples.append(tuple([]))
            else:
                state_tuples.append(tuple(col.split(',')))
        initial_state = tuple(state_tuples)

        # Target state
        target_state = (tuple([]), tuple([]), tuple([]), tuple(['a', 'b', 'c', 'd', 'e']))

        # BFS to find solution
        queue = deque([(initial_state, [])])
        visited = {initial_state}

        def can_place(column: Tuple[str, ...], disk: str) -> bool:
            return not column or disk < column[0]

        def get_next_states(state: Tuple[Tuple[str, ...], ...]) -> List[Tuple[Tuple[Tuple[str, ...], ...], Tuple[str, int]]]:
            next_states = []
            for src in range(4):
                if not state[src]:
                    continue
                disk = state[src][0]
                for dest in range(4):
                    if src == dest:
                        continue
                    if can_place(state[dest], disk):
                        new_src = list(state[src][1:])
                        new_dest = [disk] + list(state[dest])
                        new_state = list(state)
                        new_state = list(map(list, new_state))
                        new_state[src] = new_src
                        new_state[dest] = new_dest
                        new_state = tuple(map(tuple, new_state))
                        next_states.append((new_state, (disk, dest + 1)))
            return next_states

        while queue:
            current, moves = queue.popleft()
            if current == target_state:
                return f"\\boxed{{{moves}}}"

            for next_state, move in get_next_states(current):
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, moves + [move]))

        # Fallback (should not happen)
        return "\\boxed{[('a', 4)]}"
