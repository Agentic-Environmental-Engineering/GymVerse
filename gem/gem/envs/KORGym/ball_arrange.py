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

"""Ball Arrange environment - Ball sort puzzle game."""

import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class BallArrangeEnv(Env):
    """
    Ball Arrange (Ball Sort Puzzle) environment.

    Players must sort colored balls into tubes so that each tube contains
    only one color. Each tube can hold 4 balls. Balls can only be moved if
    the target tube is empty or the top ball matches the color being moved.

    This is a multi-turn environment with sparse terminal rewards.
    """

    CAPACITY = 4  # Each tube holds 4 balls

    def __init__(
        self,
        min_level: int = 1,
        max_level: int = 4,
        **_,
    ):
        """
        Initialize Ball Arrange environment.

        Args:
            min_level: Minimum difficulty level (number of colors - 2)
            max_level: Maximum difficulty level (number of colors - 2)
        """
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.level = None
        self.state = None
        self.epoch = 0

    def _num_colors_for_level(self, level: int) -> int:
        """Return number of colors for given level."""
        return level + 2

    def _generate_puzzle(self, level: int, seed: int) -> List[List[int]]:
        """
        Generate initial ball arrangement.

        Args:
            level: Difficulty level
            seed: Random seed

        Returns:
            Initial state with shuffled balls
        """
        random.seed(seed)
        n = self._num_colors_for_level(level)
        total_tubes = n + 2

        # Generate 4 balls of each color
        balls = []
        for color in range(1, n + 1):
            balls.extend([color] * self.CAPACITY)
        random.shuffle(balls)

        state = []
        idx = 0
        # First n tubes filled with balls
        for _ in range(n):
            tube = balls[idx: idx + self.CAPACITY]
            idx += self.CAPACITY
            state.append(tube)
        # Last 2 tubes are empty
        for _ in range(2):
            state.append([0] * self.CAPACITY)

        return state

    def _move_ball(self, state: List[List[int]], src: str, dst: str) -> bool:
        """
        Attempt to move top ball from src tube to dst tube.

        Args:
            state: Current game state
            src: Source tube label (e.g., 'A')
            dst: Destination tube label (e.g., 'B')

        Returns:
            True if move successful, False otherwise
        """
        label_map = {chr(65 + i): i for i in range(len(state))}

        src_idx = label_map.get(src.upper(), -1)
        dst_idx = label_map.get(dst.upper(), -1)

        if not (0 <= src_idx < len(state) and 0 <= dst_idx < len(state)):
            return False

        src_tube = state[src_idx]
        dst_tube = state[dst_idx]

        # Find top ball in source tube
        src_top = -1
        for i in range(self.CAPACITY - 1, -1, -1):
            if src_tube[i] != 0:
                src_top = i
                break
        if src_top == -1:
            return False  # Source tube is empty

        ball = src_tube[src_top]
        dst_count = sum(1 for x in dst_tube if x != 0)

        if dst_count >= self.CAPACITY:
            return False  # Destination tube is full

        # If destination not empty, top ball must match color
        if dst_count > 0:
            dst_top = -1
            for i in range(self.CAPACITY - 1, -1, -1):
                if dst_tube[i] != 0:
                    dst_top = i
                    break
            if dst_top == -1 or dst_tube[dst_top] != ball:
                return False
            place_index = dst_top + 1
        else:
            place_index = 0

        # Execute move
        src_tube[src_top] = 0
        dst_tube[place_index] = ball
        return True

    def _is_solved(self, state: List[List[int]]) -> bool:
        """Check if puzzle is solved (all non-empty tubes have same color)."""
        for tube in state:
            if all(x == 0 for x in tube):
                continue
            if any(x == 0 for x in tube):
                return False
            if len(set(tube)) != 1:
                return False
        return True

    def _is_stuck(self, state: List[List[int]]) -> bool:
        """Check if no legal moves are possible."""
        for i, tube in enumerate(state):
            top_idx = -1
            for j in range(self.CAPACITY - 1, -1, -1):
                if tube[j] != 0:
                    top_idx = j
                    break
            if top_idx == -1:
                continue  # Empty tube

            ball = tube[top_idx]
            for k, dst_tube in enumerate(state):
                if k == i:
                    continue
                dst_count = sum(1 for x in dst_tube if x != 0)
                if dst_count < self.CAPACITY:
                    if dst_count == 0:
                        return False
                    dst_top_idx = -1
                    for z in range(self.CAPACITY - 1, -1, -1):
                        if dst_tube[z] != 0:
                            dst_top_idx = z
                            break
                    if dst_top_idx != -1 and dst_tube[dst_top_idx] == ball:
                        return False
        return True

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: DB'\n\n"
            "Alternatively, you can use \\boxed{DB} format.\n\n"
            "Next, I'll provide a game board where each letter represents a test tube, numbers represent the colors of the balls inside the tubes, and '0' indicates empty space. The leftmost digit represents the bottom of the tube, and the rightmost digit represents the top. Your goal is to move the balls among the tubes so that three tubes each contain exactly four balls of the same color. Additionally, the ball being moved must either match the color of the ball at the top of the target tube or the target tube must be empty. You need to provide two letters to indicate moving the top ball from one tube onto the top of another tube. For example, 'Answer: DC' means moving the top ball from tube D onto the top of tube C.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new Ball Arrange puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Select random difficulty level
        self.level = random.randint(self.min_level, self.max_level)

        # Generate initial state
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        self.state = self._generate_puzzle(self.level, puzzle_seed)
        self.epoch = 1

        # Build observation
        observation = self._build_observation()

        return observation, {
            "suffix": f"Level {self.level}, Epoch {self.epoch}."
        }

    def _build_observation(self) -> str:
        """Build observation string from current state."""
        labels = [chr(65 + i) for i in range(len(self.state))]
        lines = []
        lines.append(f"Level: {self.level}    Epoch: {self.epoch}")
        lines.append("Note: Each tube is displayed in the format [bottom, ..., top].")
        for label, tube in zip(labels, self.state):
            lines.append(f"{label}: {tube}")

        board = "\n".join(lines)
        observation = f"{self._get_instructions()}{board}"
        return observation

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute one move.

        Args:
            action: Move command (e.g., "AD" to move from A to D)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        parsed_action = extract_last_boxed_answer(action)

        if parsed_action is None:
            import re
            match = re.search(r'Answer:\s*(\w+)', action, re.IGNORECASE)
            if match:
                parsed_action = match.group(1).strip()

        if parsed_action is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: AB' or \\boxed{AB} format."
            )
            return obs, 0.0, True, False, {}

        # Parse move
        move_str = parsed_action.replace(" ", "").upper()
        if len(move_str) != 2:
            obs = f"Invalid move format: {parsed_action}. Must be 2 letters (e.g., 'AD')."
            return obs, 0.0, True, False, {}

        src, dst = move_str[0], move_str[1]

        # Execute move
        if not self._move_ball(self.state, src, dst):
            obs = (
                f"Invalid move: {move_str}. Cannot move from {src} to {dst}.\n"
                f"Current state:\n{self._build_observation()}"
            )
            return obs, 0.0, True, False, {"epoch": self.epoch}

        self.epoch += 1

        # Check if puzzle is solved or stuck
        if self._is_solved(self.state):
            obs = (
                f"Congratulations! Puzzle solved in {self.epoch - 1} moves.\n"
                f"Final state:\n{self._build_observation()}"
            )
            return obs, 1.0, True, False, {"epoch": self.epoch, "level": self.level}
        elif self._is_stuck(self.state):
            obs = (
                f"Game over! No more valid moves available.\n"
                f"Final state:\n{self._build_observation()}"
            )
            return obs, 0.0, True, False, {"epoch": self.epoch, "level": self.level}
        else:
            # Continue game
            observation = self._build_observation()
            return observation, 0.0, False, False, {"epoch": self.epoch, "level": self.level}

    def sample_random_action(self) -> str:
        """
        Sample a random valid action.

        Returns:
            Random valid move
        """
        # Try to find any valid move
        for i, tube in enumerate(self.state):
            # Find top ball
            top_idx = -1
            for j in range(self.CAPACITY - 1, -1, -1):
                if tube[j] != 0:
                    top_idx = j
                    break
            if top_idx == -1:
                continue

            ball = tube[top_idx]
            src_label = chr(65 + i)

            # Try to move to any other tube
            for k, dst_tube in enumerate(self.state):
                if k == i:
                    continue
                dst_count = sum(1 for x in dst_tube if x != 0)
                if dst_count >= self.CAPACITY:
                    continue

                # Check if move is valid
                if dst_count == 0:
                    # Can move to empty tube
                    dst_label = chr(65 + k)
                    return f"\\boxed{{{src_label}{dst_label}}}"
                else:
                    # Check if top ball matches
                    dst_top_idx = -1
                    for z in range(self.CAPACITY - 1, -1, -1):
                        if dst_tube[z] != 0:
                            dst_top_idx = z
                            break
                    if dst_top_idx != -1 and dst_tube[dst_top_idx] == ball:
                        dst_label = chr(65 + k)
                        return f"\\boxed{{{src_label}{dst_label}}}"

        # Fallback (should not happen if not stuck)
        return "\\boxed{AB}"
