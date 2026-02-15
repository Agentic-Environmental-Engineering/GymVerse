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

"""TrustEvolution environment - Iterated prisoner's dilemma game."""

import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class TrustEvolutionEnv(Env):
    """
    Trust Evolution environment (Iterated Prisoner's Dilemma).

    Players compete against an opponent with a fixed strategy over 5 major rounds,
    each consisting of 8 minor rounds. Players can choose to "collaborate" or "cheat"
    in each round, earning points based on both players' choices.

    This is a multi-turn environment with dense rewards.
    """

    def __init__(
        self,
        major_rounds: int = 5,
        minor_rounds: int = 8,
        **_,
    ):
        """
        Initialize TrustEvolution environment.

        Args:
            major_rounds: Number of major rounds
            minor_rounds: Number of minor rounds per major round
        """
        super().__init__()
        self.major_rounds = major_rounds
        self.minor_rounds = minor_rounds
        self.opponent_type = None
        self.score = 0
        self.major_round = 1
        self.minor_round = 1
        self.history = []  # Completed major rounds history
        self.current_history = []  # Current major round history

    def _opponent_action(self, round_num: int, history: List[Tuple[str, str]]) -> str:
        """
        Determine opponent's action based on strategy and history.

        Args:
            round_num: Current minor round number (1-8)
            history: Current major round history

        Returns:
            Opponent's action ('collaborate' or 'cheat')
        """
        if self.opponent_type == "copycat":
            return history[-1][0] if history else "collaborate"
        elif self.opponent_type == "little pink":
            return "collaborate"
        elif self.opponent_type == "cheater":
            return "cheat"
        elif self.opponent_type == "grudger":
            if not history:
                return "collaborate"
            return "cheat" if "cheat" in [h[0] for h in history] else "collaborate"
        elif self.opponent_type == "detective":
            if round_num == 1:
                return "collaborate"
            elif round_num == 2:
                return "cheat"
            elif round_num in [3, 4]:
                return "collaborate"
            else:
                if any(h[0] == "cheat" for h in history):
                    return history[-1][0]
                else:
                    return "cheat"
        elif self.opponent_type == "copykitten":
            if len(history) >= 2 and history[-1][0] == history[-2][0] == "cheat":
                return "cheat"
            return history[-1][0] if history else "collaborate"
        elif self.opponent_type == "stubborn":
            if not history:
                return "collaborate"
            if history[-1][0] == history[-1][1]:
                return history[-1][1]
            else:
                return "cheat" if history[-1][1] == "collaborate" else "collaborate"
        elif self.opponent_type == "random":
            return random.choice(["cheat", "collaborate"])
        else:
            return "collaborate"

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: cheat'\n\n"
            "Alternatively, you can use \\boxed{cheat} format.\n\n"
            "This is an iterated prisoner's dilemma game:\n"
            "- When you put in one coin, your opponent receives three coins, and vice versa.\n"
            "- Both players can choose to 'collaborate' (put in one coin) or 'cheat' (put in no coins).\n"
            "- If you collaborate but your opponent cheats: you lose 1 coin, opponent gains 3 (-1 : +3)\n"
            "- If both collaborate: both gain 2 coins (+2 : +2)\n"
            "- If you cheat but your opponent collaborates: you gain 3 coins, opponent loses 1 (+3 : -1)\n"
            "- If both cheat: neither gains nor loses (0 : 0)\n\n"
            f"You will play {self.major_rounds} major rounds against an opponent with a fixed strategy.\n"
            f"Each major round consists of {self.minor_rounds} minor rounds.\n"
            "You have access to complete records of all previous major rounds and the current major round.\n"
            "Your opponent only has access to records of the current major round.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new Trust Evolution game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Choose random opponent strategy
        strategies = [
            "copycat", "little pink", "cheater", "grudger",
            "detective", "copykitten", "stubborn", "random"
        ]
        self.opponent_type = random.choice(strategies)

        self.score = 0
        self.major_round = 1
        self.minor_round = 1
        self.history = []
        self.current_history = []

        # Build observation
        board_info = (
            f"Major Round: {self.major_round} / {self.major_rounds}\n"
            f"Minor Round: {self.minor_round} / {self.minor_rounds}\n"
            f"Score: {self.score}\n"
            "Completed Major Rounds History:\n  None\n"
            "\nCurrent Major Round History:\n  No rounds played yet in this major round.\n"
            "\nPlease input your action for the next minor round in the format:\n"
            "Answer: cheat   (or)   Answer: collaborate\n"
        )

        observation = f"{self._get_instructions()}{board_info}"

        return observation, {
            "suffix": f"Round {self.major_round}.{self.minor_round}, Score: {self.score}."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute one minor round.

        Args:
            action: Player's choice ('collaborate' or 'cheat')

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

        if player_action not in ["cheat", "collaborate"]:
            obs = f"Invalid action: {player_action}. Must be 'cheat' or 'collaborate'."
            return obs, 0.0, True, False, {}

        # Get opponent's action
        opp_action = self._opponent_action(self.minor_round, self.current_history)

        # Calculate reward
        reward = 0.0
        if player_action == "collaborate" and opp_action == "collaborate":
            reward = 2.0
        elif player_action == "collaborate" and opp_action == "cheat":
            reward = -1.0
        elif player_action == "cheat" and opp_action == "collaborate":
            reward = 3.0
        # Both cheat: reward = 0

        self.score += int(reward)

        # Record this round
        self.current_history.append((player_action, opp_action))

        # Update rounds
        if self.minor_round < self.minor_rounds:
            self.minor_round += 1
        else:
            # Complete current major round
            self.history.append(self.current_history)
            self.current_history = []
            if self.major_round < self.major_rounds:
                self.major_round += 1
                self.minor_round = 1
            else:
                # Game finished
                obs = (
                    f"Game finished! All {self.major_rounds} major rounds completed.\n\n"
                    f"Final Score: {self.score}\n"
                    f"Opponent Strategy: {self.opponent_type}\n"
                )
                return obs, reward, False, True, {
                    "score": self.score,
                    "opponent": self.opponent_type
                }

        # Build observation
        board_info = (
            f"Major Round: {self.major_round} / {self.major_rounds}\n"
            f"Minor Round: {self.minor_round} / {self.minor_rounds}\n"
            f"Score: {self.score}\n"
            "Completed Major Rounds History:\n"
        )

        if self.history:
            for idx, major in enumerate(self.history, start=1):
                board_info += f"  Major Round {idx}:\n"
                for i, (p, o) in enumerate(major, start=1):
                    board_info += f"    Minor {i}: You: {p}, Opponent: {o}\n"
        else:
            board_info += "  None\n"

        board_info += "\nCurrent Major Round History:\n"
        if self.current_history:
            for i, (p, o) in enumerate(self.current_history, start=1):
                board_info += f"  Minor {i}: You: {p}, Opponent: {o}\n"
        else:
            board_info += "  No rounds played yet in this major round.\n"

        board_info += (
            "\nPlease input your action for the next minor round in the format:\n"
            "Answer: cheat   (or)   Answer: collaborate\n"
        )

        observation = board_info

        return observation, reward, False, False, {
            "score": self.score,
            "major_round": self.major_round,
            "minor_round": self.minor_round
        }

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random action ('cheat' or 'collaborate')
        """
        action = random.choice(["cheat", "collaborate"])
        return f"\\boxed{{{action}}}"
