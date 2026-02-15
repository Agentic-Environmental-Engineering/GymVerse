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

"""Snake environment - Classic snake game."""

import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class SnakeEnv(Env):
    """
    Snake game environment.

    Players control a snake to eat apples and grow longer. The game ends
    when the snake hits a wall, itself, or reaches the maximum number of turns.

    This is a multi-turn environment with dense rewards.
    """

    def __init__(
        self,
        board_size: int = 8,
        max_turns: int = 100,
        initial_apples: int = 3,
        **_,
    ):
        """
        Initialize Snake environment.

        Args:
            board_size: Size of the game board (n×n)
            max_turns: Maximum number of turns before game ends
            initial_apples: Number of apples to keep on the board
        """
        super().__init__()
        self.board_size = board_size
        self.max_turns = max_turns
        self.initial_apples = initial_apples
        self.snake = None
        self.food = None
        self.direction = None
        self.score = 0
        self.epoch = 0
        self.wall = []  # No internal walls by default

    def _print_board(self) -> str:
        """Generate string representation of the current board."""
        output = ""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i, j) == self.snake[-1]:  # Snake head
                    output += 'H'
                elif (i, j) in self.snake:
                    output += 'S'
                elif (i, j) in self.food:
                    output += 'A'
                elif (i == 0 or i == self.board_size - 1 or
                      j == 0 or j == self.board_size - 1 or
                      (i, j) in self.wall):
                    output += '#'
                else:
                    output += ' '
            output += '\n'
        return output

    def _change_direction(self, current: str, new: str) -> str:
        """Prevent reverse movement."""
        opposite = {
            'RIGHT': 'LEFT', 'LEFT': 'RIGHT',
            'UP': 'DOWN', 'DOWN': 'UP'
        }
        if opposite.get(current) == new:
            return current
        return new

    def _move_snake(self, direction: str) -> Tuple[bool, int]:
        """
        Move the snake in the given direction.

        Returns:
            Tuple of (is_valid, score_increment)
        """
        head_x, head_y = self.snake[-1]

        # Calculate new head position
        if direction == 'LEFT':
            new_head = (head_x, head_y - 1)
        elif direction == 'RIGHT':
            new_head = (head_x, head_y + 1)
        elif direction == 'UP':
            new_head = (head_x - 1, head_y)
        elif direction == 'DOWN':
            new_head = (head_x + 1, head_y)
        else:
            return False, 0

        # Check collision with walls, boundaries, or self
        if (new_head in self.snake or new_head in self.wall or
            new_head[0] == 0 or new_head[1] == 0 or
            new_head[0] == self.board_size - 1 or
            new_head[1] == self.board_size - 1):
            return False, 0

        self.snake.append(new_head)
        score_increment = 0

        # Check if ate food
        if new_head in self.food:
            score_increment = 1
            self.food.remove(new_head)

            # Spawn new food
            while len(self.food) < self.initial_apples:
                new_food = (
                    random.randint(1, self.board_size - 2),
                    random.randint(1, self.board_size - 2)
                )
                if (new_food not in self.snake and
                    new_food not in self.wall and
                    new_food not in self.food):
                    self.food.append(new_food)
        else:
            # Remove tail if no food eaten
            self.snake.pop(0)

        return True, score_increment

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: LEFT'\n\n"
            "Alternatively, you can use \\boxed{LEFT} format.\n\n"
            f"You are controlling a snake in a Snake game. The board size is {self.board_size}×{self.board_size}.\n"
            "The goal is to eat as many apples as possible and grow the snake in length.\n"
            "Each time the snake eats an apple, the score increases by one.\n"
            "The game ends when the snake collides with itself, the walls, or reaches the maximum turns.\n\n"
            "Game rules:\n"
            "- The board is a grid with walls ('#') around the edges.\n"
            "- The snake head is represented by 'H' and the body by 'S'.\n"
            f"- There are {self.initial_apples} apples ('A') on the board at all times.\n"
            "- The snake moves one square at a time: 'UP', 'DOWN', 'LEFT', or 'RIGHT'.\n"
            "- The snake cannot reverse direction (e.g., cannot go from 'UP' to 'DOWN' directly).\n"
            "- When the snake eats an apple, it grows longer and a new apple appears.\n"
            f"- The game ends after {self.max_turns} turns.\n\n"
            "The direction you give should be one of 'LEFT', 'RIGHT', 'UP', or 'DOWN'.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new snake game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Initialize snake at center
        self.snake = [(self.board_size // 2, self.board_size // 2 - 1)]

        # Generate initial food
        self.food = []
        while len(self.food) < self.initial_apples:
            new_food = (
                random.randint(1, self.board_size - 2),
                random.randint(1, self.board_size - 2)
            )
            if (new_food not in self.snake and
                new_food not in self.wall and
                new_food not in self.food):
                self.food.append(new_food)

        self.direction = 'UP'
        self.score = 0
        self.epoch = 1

        # Build observation
        board_str = self._print_board()
        observation = (
            f"{self._get_instructions()}"
            f"Game board:\n\n{board_str}\n"
            f"Current direction: {self.direction}\n"
            f"Current epoch: {self.epoch}\n"
            f"Current score: {self.score}"
        )

        return observation, {
            "suffix": f"Control snake (Turn {self.epoch}/{self.max_turns}, Score: {self.score})."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the game.

        Args:
            action: Direction to move ('UP', 'DOWN', 'LEFT', 'RIGHT')

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        parsed_action = extract_last_boxed_answer(action)

        if parsed_action is None:
            import re
            match = re.search(r'Answer:\s*(\w+)', action, re.IGNORECASE)
            if match:
                parsed_action = match.group(1).strip().upper()

        if parsed_action is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: DIRECTION' or \\boxed{DIRECTION} format."
            )
            return obs, 0.0, True, False, {}

        new_direction = parsed_action.strip().upper()

        # Validate direction
        if new_direction not in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            obs = f"Invalid direction: {new_direction}. Must be UP, DOWN, LEFT, or RIGHT."
            return obs, 0.0, True, False, {}

        # Update direction (preventing reverse)
        self.direction = self._change_direction(self.direction, new_direction)

        # Move snake
        valid_move, score_increment = self._move_snake(self.direction)
        self.score += score_increment
        self.epoch += 1

        # Check termination
        terminated = not valid_move
        truncated = self.epoch > self.max_turns

        # Build observation
        board_str = self._print_board()

        if terminated:
            obs = (
                f"Game Over! The snake hit a wall or itself.\n\n"
                f"Final board:\n{board_str}\n"
                f"Final score: {self.score}\n"
                f"Turns survived: {self.epoch - 1}"
            )
        elif truncated:
            obs = (
                f"Game finished! Maximum turns reached.\n\n"
                f"Final board:\n{board_str}\n"
                f"Final score: {self.score}\n"
                f"Turns: {self.epoch - 1}/{self.max_turns}"
            )
        else:
            obs = (
                f"Game board:\n\n{board_str}\n"
                f"Current direction: {self.direction}\n"
                f"Current epoch: {self.epoch}\n"
                f"Current score: {self.score}"
            )

        reward = float(score_increment)

        return obs, reward, terminated, truncated, {
            "score": self.score,
            "epoch": self.epoch
        }

    def sample_random_action(self) -> str:
        """
        Sample a random valid action.

        Returns:
            Random direction
        """
        # Choose a valid direction (not reverse)
        valid_directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        opposite = {
            'RIGHT': 'LEFT', 'LEFT': 'RIGHT',
            'UP': 'DOWN', 'DOWN': 'UP'
        }

        if self.direction:
            valid_directions = [d for d in valid_directions if d != opposite.get(self.direction)]

        action = random.choice(valid_directions)
        return f"\\boxed{{{action}}}"
