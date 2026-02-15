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

"""NpointPlus environment - Blackjack-like card game with variable threshold."""

import random
from copy import deepcopy
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


# Card values
POKER_VALUES = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13
}

CARD_NAMES = list(POKER_VALUES.keys())


class NpointPlusEnv(Env):
    """
    N-point Plus environment (Blackjack variant).

    Players compete against an opponent over multiple rounds. Each round has a
    random threshold N (24-50). Players try to get as close to N as possible
    without exceeding it by choosing to hit (draw card) or stand (stop).

    This is a multi-turn environment with dense rewards.
    """

    def __init__(
        self,
        total_rounds: int = 10,
        min_threshold: int = 24,
        max_threshold: int = 50,
        **_,
    ):
        """
        Initialize NpointPlus environment.

        Args:
            total_rounds: Total number of rounds to play
            min_threshold: Minimum threshold value
            max_threshold: Maximum threshold value
        """
        super().__init__()
        self.total_rounds = total_rounds
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.opponent_strategy = None
        self.n = None
        self.player_cards = None
        self.opponent_cards = None
        self.player_stand = False
        self.opponent_stand = False
        self.player_bust = False
        self.opponent_bust = False
        self.turn = 1
        self.moves = []
        self.last_player_action = None
        self.opponent_first_move = True
        self.current_round = 1
        self.score = 0.0
        self.history = []

    def _get_card(self) -> str:
        """Draw a random card from infinite deck."""
        return random.choice(CARD_NAMES)

    def _compute_points(self, cards: List[str]) -> int:
        """Compute total points of a hand."""
        return sum(POKER_VALUES[card] for card in cards)

    def _opponent_action(self) -> str:
        """Determine opponent's action based on strategy."""
        if self.opponent_strategy == "stubborn":
            return "hit"
        elif self.opponent_strategy == "careful":
            return "stand"
        elif self.opponent_strategy == "normal":
            if self._compute_points(self.opponent_cards) < self.n - 6:
                return "hit"
            else:
                return "stand"
        elif self.opponent_strategy == "repeat":
            if self.opponent_first_move:
                return "hit"
            else:
                return self.last_player_action if self.last_player_action in ["hit", "stand"] else "stand"
        else:
            return "stand"

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game problem-solver. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: Hit'\n\n"
            "Alternatively, you can use \\boxed{Hit} format.\n\n"
            "This is an N-point game (similar to Blackjack):\n\n"
            "Card values:\n"
            "- Aces are worth 1 point\n"
            "- Number cards (2-10) are worth their numerical value\n"
            "- Face cards (J, Q, K) are worth 11, 12, and 13 points respectively\n\n"
            "Rules:\n"
            "- At the start of each round, the dealer has one face-up card and one face-down card\n"
            "- You have two face-up cards\n"
            "- Each round has a threshold N (randomly chosen between 24-50)\n"
            "- You can request additional cards (Hit) until you decide to stop (Stand) or exceed N\n"
            "- If one side exceeds N and the other doesn't, the side that didn't exceed N wins\n"
            "- If both exceed N, it's a draw\n"
            "- If neither exceeds N, the side closest to N wins\n"
            "- If both have the same total, it's a draw\n"
            "- Winning earns 1 point, draw earns 0.5 points, losing earns 0 points\n\n"
            f"You will play {self.total_rounds} rounds against an opponent with a fixed strategy.\n"
            "You have access to complete records of all previous rounds.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new N-point Plus game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Choose random opponent strategy
        strategies = ["stubborn", "careful", "normal", "repeat"]
        self.opponent_strategy = random.choice(strategies)

        # Initialize first round
        self.n = random.randint(self.min_threshold, self.max_threshold)
        self.player_cards = [self._get_card(), self._get_card()]
        self.opponent_cards = [self._get_card(), self._get_card()]
        self.player_stand = False
        self.opponent_stand = False
        self.player_bust = False
        self.opponent_bust = False
        self.turn = 1
        self.moves = []
        self.last_player_action = None
        self.opponent_first_move = True
        self.current_round = 1
        self.score = 0.0
        self.history = []

        # Build observation
        observation = self._build_observation(reveal=False)

        return observation, {
            "suffix": f"Round {self.current_round}/{self.total_rounds}, Score: {self.score}."
        }

    def _build_observation(self, reveal: bool = False) -> str:
        """Build observation string."""
        s = ""

        # Show history
        s += "=== History ===\n"
        if self.history:
            for record in self.history:
                s += f"Round {record['round']}:\n"
                s += f"  Threshold (N): {record['n']}\n"
                s += f"  Moves: {' | '.join(record['moves'])}\n"
                s += f"  Your cards: {record['player_cards']} (Total: {record['player_total']})\n"
                s += f"  Opponent's cards: {record['opponent_cards']} (Total: {record['opponent_total']})\n"
                s += f"  Outcome: {record['outcome']}, Round Score: {record['round_score']}\n"
                s += "--------------------\n"
        else:
            s += "No previous rounds.\n"

        # Show current round
        s += "=== Current Round ===\n"
        s += f"Round: {self.current_round} / {self.total_rounds}\n"
        s += f"Score: {self.score}\n"
        s += f"Threshold (N): {self.n}\n"

        # Player cards
        player_total = self._compute_points(self.player_cards)
        s += f"Your cards: {self.player_cards} (Total: {player_total}).\n"

        # Opponent cards
        if not reveal:
            if len(self.opponent_cards) > 0:
                opponent_view = [self.opponent_cards[0]] + ["unknown card"] * (len(self.opponent_cards) - 1)
            else:
                opponent_view = []
            s += f"Opponent's cards: {opponent_view}.\n"
        else:
            opponent_total = self._compute_points(self.opponent_cards)
            s += f"Opponent's cards: {self.opponent_cards} (Total: {opponent_total}).\n"

        s += f"Turn: {self.turn}\n"
        s += "Move history:\n"
        if self.moves:
            for move in self.moves:
                s += f"  {move}\n"
        else:
            s += "  No moves yet.\n"

        observation = f"{self._get_instructions()}Now the current game situation is as follows:\n\n{s}"

        return observation

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute one turn.

        Args:
            action: Player's choice ('hit' or 'stand')

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        parsed_action = extract_last_boxed_answer(action)

        if parsed_action is None:
            import re
            match = re.search(r'Answer:\s*(\w+)', action, re.IGNORECASE)
            if match:
                parsed_action = match.group(1).strip().lower()

        if parsed_action is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: ACTION' or \\boxed{ACTION} format."
            )
            return obs, 0.0, True, False, {}

        player_action = parsed_action.strip().lower()

        if player_action not in ["hit", "stand"]:
            obs = f"Invalid action: {player_action}. Must be 'hit' or 'stand'."
            return obs, 0.0, True, False, {}

        # Player action
        if not self.player_stand and not self.player_bust:
            if player_action == "hit":
                card = self._get_card()
                self.player_cards.append(card)
                if self._compute_points(self.player_cards) > self.n:
                    self.player_bust = True
                    self.player_stand = True
            elif player_action == "stand":
                self.player_stand = True
            self.last_player_action = player_action

        # Opponent action
        if not self.opponent_stand and not self.opponent_bust:
            opponent_action = self._opponent_action()
            if opponent_action == "hit":
                card = self._get_card()
                self.opponent_cards.append(card)
                if self._compute_points(self.opponent_cards) > self.n:
                    self.opponent_bust = True
                    self.opponent_stand = True
                if self.opponent_strategy == "repeat":
                    self.opponent_first_move = False
            elif opponent_action == "stand":
                self.opponent_stand = True
        else:
            opponent_action = "stand"

        # Record move
        move_log = f"Turn {self.turn}: You: {player_action}; Opponent: {opponent_action}"
        self.moves.append(move_log)
        self.turn += 1

        # Check if round ended
        reward = 0.0
        if (self.player_stand or self.player_bust) and (self.opponent_stand or self.opponent_bust):
            # Round ended, calculate score
            player_total = self._compute_points(self.player_cards)
            opponent_total = self._compute_points(self.opponent_cards)

            if player_total > self.n and opponent_total <= self.n:
                outcome_str = "You lose!"
                round_score = 0.0
            elif opponent_total > self.n and player_total <= self.n:
                outcome_str = "You win!"
                round_score = 1.0
            elif player_total > self.n and opponent_total > self.n:
                outcome_str = "Draw game!"
                round_score = 0.5
            else:
                if player_total > opponent_total:
                    outcome_str = "You win!"
                    round_score = 1.0
                elif player_total == opponent_total:
                    outcome_str = "Draw game!"
                    round_score = 0.5
                else:
                    outcome_str = "You lose!"
                    round_score = 0.0

            reward = round_score
            self.score += round_score

            # Save round record
            round_record = {
                "round": self.current_round,
                "n": self.n,
                "moves": deepcopy(self.moves),
                "player_cards": self.player_cards,
                "player_total": player_total,
                "opponent_cards": self.opponent_cards,
                "opponent_total": opponent_total,
                "outcome": outcome_str,
                "round_score": round_score
            }
            self.history.append(round_record)

            # Check if more rounds
            if self.current_round < self.total_rounds:
                # Start new round
                self.current_round += 1
                self.n = random.randint(self.min_threshold, self.max_threshold)
                self.player_cards = [self._get_card(), self._get_card()]
                self.opponent_cards = [self._get_card(), self._get_card()]
                self.player_stand = False
                self.opponent_stand = False
                self.player_bust = False
                self.opponent_bust = False
                self.turn = 1
                self.moves = []
                self.last_player_action = None
                self.opponent_first_move = True

                observation = self._build_observation(reveal=False)
                return observation, reward, False, False, {
                    "score": self.score,
                    "round": self.current_round
                }
            else:
                # Game finished
                observation = self._build_observation(reveal=True)
                obs = (
                    f"{observation}\n\n"
                    f"Game finished! All {self.total_rounds} rounds completed.\n"
                    f"Final Score: {self.score}\n"
                    f"Opponent Strategy: {self.opponent_strategy}\n"
                )
                return obs, reward, False, True, {
                    "score": self.score,
                    "opponent": self.opponent_strategy
                }
        else:
            # Round continues
            observation = self._build_observation(reveal=False)
            return observation, reward, False, False, {
                "score": self.score,
                "round": self.current_round,
                "turn": self.turn
            }

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random action ('hit' or 'stand')
        """
        action = random.choice(["hit", "stand"])
        return f"\\boxed{{{action}}}"
