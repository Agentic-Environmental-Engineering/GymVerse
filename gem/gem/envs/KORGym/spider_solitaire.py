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

"""SpiderSolitaire environment - Classic Spider Solitaire card game."""

import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class SpiderSolitaireEnv(Env):
    """
    Spider Solitaire environment.

    A classic solitaire card game where the player must build 8 complete
    K→A same-suit sequences by moving cards between 10 columns.

    This is a multi-turn environment with dense reward (score-based).
    """

    def __init__(
        self,
        max_turns: int = 100,
        **_,
    ):
        """
        Initialize SpiderSolitaire environment.

        Args:
            max_turns: Maximum number of turns (moves or hits)
        """
        super().__init__()
        self.max_turns = max_turns
        self.board = None
        self.deck = None
        self.visibility = None
        self.completed_sets = None
        self.score = None
        self.turn = None

    def _get_card_value(self, card: Tuple[str, str]) -> int:
        """Get numeric value of card."""
        rank_values = {
            'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13
        }
        return rank_values[card[1]]

    def _is_sequence(self, cards: List[Tuple[str, str]]) -> bool:
        """Check if cards form same-suit descending sequence."""
        if not cards:
            return False

        suit = cards[0][0]
        if any(card[0] != suit for card in cards):
            return False

        values = [self._get_card_value(card) for card in cards]
        for i in range(1, len(values)):
            if values[i] != values[i-1] - 1:
                return False

        return True

    def _is_complete_sequence(self, cards: List[Tuple[str, str]]) -> bool:
        """Check if cards form complete K→A sequence."""
        if len(cards) != 13:
            return False

        if self._get_card_value(cards[0]) != 13 or self._get_card_value(cards[-1]) != 1:
            return False

        return self._is_sequence(cards)

    def _setup_game(self, seed: int):
        """Initialize game board with shuffled deck."""
        random.seed(seed)

        # Create two decks (104 cards total)
        suits = ['♥', '♦', '♣', '♠']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

        self.deck = []
        for _ in range(2):
            for suit in suits:
                for rank in ranks:
                    self.deck.append((suit, rank))

        # Shuffle
        random.shuffle(self.deck)
        mid_point = len(self.deck) // 2
        cut_point = random.randint(mid_point - 10, mid_point + 10)
        temp = self.deck[cut_point:]
        random.shuffle(temp)
        self.deck = self.deck[:cut_point] + temp

        # Initialize 10 columns
        self.board = [[] for _ in range(10)]
        self.visibility = [[] for _ in range(10)]

        # First 4 columns get 6 cards each
        for i in range(4):
            for j in range(6):
                card = self.deck.pop(0)
                self.board[i].append(card)
                # Only last card is initially visible
                self.visibility[i].append(j == 5)

        # Remaining 6 columns get 5 cards each
        for i in range(4, 10):
            for j in range(5):
                card = self.deck.pop(0)
                self.board[i].append(card)
                # Only last card is initially visible
                self.visibility[i].append(j == 4)

        self.completed_sets = 0
        self.score = 0
        self.turn = 0

    def _check_completed_sequences(self) -> int:
        """Check and remove completed sequences. Returns count of completed sequences."""
        completed = 0

        for column_idx in range(len(self.board)):
            column = self.board[column_idx]
            i = len(column) - 1
            while i >= 12:
                all_visible = all(self.visibility[column_idx][j] for j in range(i-12, i+1))
                if all_visible and self._is_complete_sequence(column[i-12:i+1]):
                    # Remove completed sequence
                    self.board[column_idx] = column[:i-12] + column[i+1:]
                    self.visibility[column_idx] = self.visibility[column_idx][:i-12] + self.visibility[column_idx][i+1:]
                    column = self.board[column_idx]
                    completed += 1
                    i = len(column) - 1
                else:
                    i -= 1

        # Make sure last card in each column is visible
        for i in range(len(self.board)):
            if self.board[i] and not self.visibility[i][-1]:
                self.visibility[i][-1] = True

        self.completed_sets += completed
        return completed

    def _get_visible_board(self) -> List[List[Tuple]]:
        """Get board with hidden cards marked as unknown."""
        visible_board = []
        for col_idx, column in enumerate(self.board):
            visible_column = []
            for card_idx, card in enumerate(column):
                if self.visibility[col_idx][card_idx]:
                    visible_column.append(card)
                else:
                    visible_column.append(('unknown', 'unknown'))
            visible_board.append(visible_column)
        return visible_board

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a Spider Solitaire expert. After I show you the current board, choose the "
            "best next action and reply with:\n\n"
            "1. (Optional) Your reasoning.\n"
            "2. A final line in exactly this format: Answer: $YOUR_ANSWER\n\n"
            "Where '$YOUR_ANSWER' is one of:\n"
            "- A move '(FromColumn,StartIndex,ToColumn)', e.g. '(A,4,B)'\n"
            "- 'hit' to deal new cards\n\n"
            "Alternatively, you can use \\boxed{(A,4,B)} or \\boxed{hit} format.\n\n"
            "### Rules\n"
            "- **Goal**: Build 8 complete K→A sequences.\n"
            "- **Move**: You may relocate any descending, same-suit run onto a column whose top "
            "card is exactly one rank higher (or onto an empty column).\n"
            "- **Deal**: If no legal moves remain and the deck has ≥10 cards and every column is "
            "non-empty, use 'hit' (deals one card to each of the 10 columns).\n"
            "- **Score**: Start at 0; +1 for each K→A sequence removed.\n"
            "- **Turn Limit**: The game ends after 100 turns (moves or hits) or when 8 sequences "
            "are completed.\n\n"
            "### Visibility & Completion\n"
            "1. Only visible cards can be moved; hidden cards are shown as 'XX'.\n"
            "2. After you move a face-up run away, the new bottom card flips face-up automatically.\n"
            "3. Completing a full K→A same-suit sequence in any column removes those 13 cards "
            "immediately and awards +1 point.\n\n"
            "### Columns & Format\n"
            "- Columns are labeled A–J (indices 0–9).\n"
            "- Always output exactly: 'Answer: $YOUR_ANSWER', e.g. 'Answer: (A,4,B)' means move "
            "cards from column A starting at index 4 to column B.\n\n"
        )

    def _format_board(self) -> str:
        """Format the game board as string."""
        visible_board = self._get_visible_board()
        column_labels = "ABCDEFGHIJ"

        output = "  " + " ".join(column_labels[:len(visible_board)]) + "\n"
        output += "  " + "-" * (2 * len(visible_board) - 1) + "\n"

        max_length = max(len(col) for col in visible_board) if visible_board else 0
        for i in range(max_length):
            row = []
            for j, column in enumerate(visible_board):
                if i < len(column):
                    card = column[i]
                    if card[0] == 'unknown':
                        row.append("XX")
                    else:
                        row.append(f"{card[1]}{card[0][0]}")
                else:
                    row.append("  ")
            output += f"{i} {' '.join(row)}\n"

        remaining_hits = len(self.deck) // 10
        output += f"Turn: {self.turn}/{self.max_turns}\n"
        output += f"Score: {self.score}/8\n"
        output += f"Remaining hit chances: {remaining_hits}\n"

        return output

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new Spider Solitaire game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate game
        game_seed = seed if seed is not None else random.randint(0, 1000000)
        self._setup_game(game_seed)

        # Build observation
        board_str = self._format_board()
        observation = f"{self._get_instructions()}Current Game Board:\n{board_str}\n"

        return observation, {
            "suffix": f"Max turns: {self.max_turns}, Goal: 8 sequences."
        }

    def _can_move_cards(self, from_column: int, start_idx: int, to_column: int) -> bool:
        """Check if cards can be moved."""
        if from_column < 0 or from_column >= len(self.board):
            return False
        if start_idx < 0 or start_idx >= len(self.board[from_column]):
            return False
        if to_column < 0 or to_column >= len(self.board):
            return False
        if not self.visibility[from_column][start_idx]:
            return False

        cards_to_move = self.board[from_column][start_idx:]
        if len(cards_to_move) > 1 and not self._is_sequence(cards_to_move):
            return False

        if self.board[to_column]:
            top_card = self.board[to_column][-1]
            if not self.visibility[to_column][-1]:
                return False
            if self._get_card_value(cards_to_move[0]) != self._get_card_value(top_card) - 1:
                return False

        return True

    def _move_cards(self, from_column: int, start_idx: int, to_column: int) -> bool:
        """Move cards from one column to another."""
        if not self._can_move_cards(from_column, start_idx, to_column):
            return False

        # Move cards
        cards_to_move = self.board[from_column][start_idx:]
        visibility_to_move = self.visibility[from_column][start_idx:]

        self.board[from_column] = self.board[from_column][:start_idx]
        self.visibility[from_column] = self.visibility[from_column][:start_idx]

        self.board[to_column].extend(cards_to_move)
        self.visibility[to_column].extend(visibility_to_move)

        # Reveal new bottom card in source column
        if self.board[from_column] and not self.visibility[from_column][-1]:
            self.visibility[from_column][-1] = True

        # Check for completed sequences
        completed = self._check_completed_sequences()
        if completed > 0:
            self.score += completed

        return True

    def _deal_cards(self) -> bool:
        """Deal cards from deck."""
        if len(self.deck) < 10:
            return False
        if any(not column for column in self.board):
            return False

        for i in range(10):
            card = self.deck.pop(0)
            self.board[i].append(card)
            self.visibility[i].append(True)

        # Check for completed sequences
        completed = self._check_completed_sequences()
        if completed > 0:
            self.score += completed

        return True

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process player's action.

        Args:
            action: Move or hit command

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
                "Please use 'Answer: (A,4,B)' or 'Answer: hit' format."
            )
            return obs, 0.0, True, False, {}

        parsed_action = str(parsed_action).strip().lower()

        # Process action
        prev_score = self.score
        success = False

        if parsed_action == "hit":
            success = self._deal_cards()
            if not success:
                obs = "Cannot deal cards. Either deck has < 10 cards or some column is empty."
                return obs, 0.0, True, False, {}
        else:
            # Parse move
            try:
                # Remove parentheses and split
                parts = parsed_action.strip("()").split(",")
                if len(parts) != 3:
                    raise ValueError("Invalid format")

                from_col = ord(parts[0].strip().upper()) - ord('A')
                start_idx = int(parts[1].strip())
                to_col = ord(parts[2].strip().upper()) - ord('A')

                success = self._move_cards(from_col, start_idx, to_col)
                if not success:
                    obs = f"Invalid move: Cannot move cards from column {parts[0]} at index {start_idx} to column {parts[2]}."
                    return obs, 0.0, True, False, {}
            except Exception as e:
                obs = f"Failed to parse action: {parsed_action}. Error: {e}"
                return obs, 0.0, True, False, {}

        self.turn += 1

        # Calculate reward (score increase)
        reward = float(self.score - prev_score)

        # Check win condition
        if self.completed_sets >= 8:
            obs = (
                f"Congratulations! You completed all 8 sequences!\n"
                f"Final score: {self.score}\n"
                f"Turns used: {self.turn}/{self.max_turns}\n"
                f"{self._format_board()}"
            )
            return obs, reward, True, False, {}

        # Check turn limit
        if self.turn >= self.max_turns:
            obs = (
                f"Maximum turns reached ({self.max_turns}).\n"
                f"Final score: {self.score}/8\n"
                f"{self._format_board()}"
            )
            return obs, reward, True, False, {}

        # Continue game
        board_str = self._format_board()
        observation = f"{self._get_instructions()}Current Game Board:\n{board_str}\n"

        return observation, reward, False, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random valid action.

        Returns:
            Random action as string
        """
        # Try to find a valid move
        column_labels = "ABCDEFGHIJ"

        for _ in range(100):
            # Try random move
            from_col = random.randint(0, 9)
            if not self.board[from_col]:
                continue

            # Find visible cards
            visible_indices = [i for i, vis in enumerate(self.visibility[from_col]) if vis]
            if not visible_indices:
                continue

            start_idx = random.choice(visible_indices)
            to_col = random.randint(0, 9)

            if from_col != to_col and self._can_move_cards(from_col, start_idx, to_col):
                return f"\\boxed{{({column_labels[from_col]},{start_idx},{column_labels[to_col]})}}"

        # Try hit
        if len(self.deck) >= 10 and all(column for column in self.board):
            return "\\boxed{hit}"

        # Fallback to first valid move
        for from_col in range(10):
            if not self.board[from_col]:
                continue
            for i, vis in enumerate(self.visibility[from_col]):
                if vis:
                    for to_col in range(10):
                        if from_col != to_col and self._can_move_cards(from_col, i, to_col):
                            return f"\\boxed{{({column_labels[from_col]},{i},{column_labels[to_col]})}}"

        return "\\boxed{hit}"
