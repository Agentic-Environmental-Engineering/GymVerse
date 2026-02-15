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

"""EmojiConnect environment - Count lines of matching emojis."""

import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class EmojiConnectEnv(Env):
    """
    EmojiConnect environment.

    Players are given a grid filled with emojis and must count the total
    number of horizontal and vertical lines where the same emoji appears
    consecutively (with length >= 2).

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 8,
        num_emoji_types: int = 5,
        **_,
    ):
        """
        Initialize EmojiConnect environment.

        Args:
            min_size: Minimum board size (n×n)
            max_size: Maximum board size (n×n)
            num_emoji_types: Number of different emoji types to use
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.num_emoji_types = num_emoji_types
        self.board = None
        self.board_size = None
        self.answer = None

    def _calculate_lines(self, board: List[List[str]]) -> int:
        """
        Calculate total number of horizontal and vertical lines.

        A line is defined as 2 or more consecutive same emojis in a row or column.

        Args:
            board: 2D grid of emojis

        Returns:
            Total number of lines
        """
        total = 0
        n = len(board)

        # Count horizontal lines
        for row in board:
            current_emoji = row[0]
            current_len = 1

            for j in range(1, n):
                if row[j] == current_emoji:
                    current_len += 1
                else:
                    if current_len >= 2:
                        total += 1
                    current_emoji = row[j]
                    current_len = 1

            # Check last segment
            if current_len >= 2:
                total += 1

        # Count vertical lines
        for col_idx in range(n):
            current_emoji = board[0][col_idx]
            current_len = 1

            for row_idx in range(1, n):
                if board[row_idx][col_idx] == current_emoji:
                    current_len += 1
                else:
                    if current_len >= 2:
                        total += 1
                    current_emoji = board[row_idx][col_idx]
                    current_len = 1

            # Check last segment
            if current_len >= 2:
                total += 1

        return total

    def _generate_board(self, size: int, num_types: int) -> List[List[str]]:
        """
        Generate a random emoji board.

        Args:
            size: Board size (n×n)
            num_types: Number of emoji types

        Returns:
            2D grid of emojis
        """
        # Common emoji set
        all_emojis = [
            "😀", "😃", "😄", "😁", "😆", "😅", "🤣", "😂",
            "🙂", "🙃", "😉", "😊", "😇", "🥰", "😍", "🤩",
            "😘", "😗", "😚", "😙", "🥲", "😋", "😛", "😜",
            "🤪", "😝", "🤑", "🤗", "🤭", "🤫", "🤔", "🤐"
        ]

        # Select subset of emojis
        selected_emojis = random.sample(all_emojis, min(num_types, len(all_emojis)))

        # Generate random board
        board = []
        for _ in range(size):
            row = [random.choice(selected_emojis) for _ in range(size)]
            board.append(row)

        return board

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., \"Answer: 15\"\n\n"
            "Alternatively, you can use \\boxed{15} format.\n\n"
            "I will provide you with an n×n board filled with emojis. Your task is to count "
            "the total number of 'lines' on the board.\n\n"
            "A 'line' is defined as:\n"
            "- 2 or more consecutive identical emojis in the same row (horizontal line), OR\n"
            "- 2 or more consecutive identical emojis in the same column (vertical line)\n\n"
            "For example, if a row contains: 😀 😀 😃 😃 😃 😁\n"
            "- There is 1 horizontal line of 😀 (length 2)\n"
            "- There is 1 horizontal line of 😃 (length 3)\n"
            "- Total: 2 lines in this row\n\n"
            "You need to count all such lines across all rows and columns, then output the "
            "total count as a single number.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new EmojiConnect puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate board
        self.board_size = random.randint(self.min_size, self.max_size)
        self.board = self._generate_board(self.board_size, self.num_emoji_types)

        # Calculate answer
        self.answer = self._calculate_lines(self.board)

        # Build observation
        board_str = "\n".join([" ".join(row) for row in self.board])
        observation = f"{self._get_instructions()}Board:\n{board_str}"

        return observation, {
            "suffix": f"Count emoji lines on {self.board_size}×{self.board_size} board."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's answer.

        Args:
            action: Agent's response containing the count

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer
        parsed_answer = extract_last_boxed_answer(action)

        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(\d+)', action, re.IGNORECASE)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: NUMBER' or \\boxed{NUMBER} format."
            )
            return obs, 0.0, True, False, {}

        # Parse number
        try:
            user_answer = int(parsed_answer)
        except ValueError:
            obs = f"Failed to parse answer as integer: '{parsed_answer}'"
            return obs, 0.0, True, False, {}

        # Check answer
        if user_answer == self.answer:
            obs = f"Correct! There are {self.answer} emoji lines on the board."
            return obs, 1.0, True, False, {}
        else:
            obs = f"Incorrect. You counted {user_answer} lines, but the correct answer is {self.answer}."
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            The correct answer
        """
        if self.answer is not None:
            return f"\\boxed{{{self.answer}}}"
        else:
            return "\\boxed{0}"
