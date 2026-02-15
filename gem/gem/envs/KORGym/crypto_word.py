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

"""CryptoWord environment - Emoji-encoded sentence decryption game."""

import random
import string
import collections
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class CryptoWordEnv(Env):
    """
    CryptoWord environment.

    Players must decode a sentence where letters have been replaced with emojis.
    A hint is provided for the most frequent emoji. Players have multiple
    attempts to guess the emoji-to-letter mapping.

    This is a multi-turn environment with sparse terminal reward.
    """

    def __init__(
        self,
        sentences: Optional[List[str]] = None,
        replacement_ratio_range: Tuple[float, float] = (0.3, 1.0),
        max_attempts: int = 10,
        **_,
    ):
        """
        Initialize CryptoWord environment.

        Args:
            sentences: List of sentences to encode
            replacement_ratio_range: Range of replacement ratios (min, max)
            max_attempts: Maximum number of attempts
        """
        super().__init__()
        self.sentences = sentences or [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
            "Sphinx of black quartz, judge my vow.",
            "Waltz, nymph, for quick jigs vex Bud.",
            "Glib jocks quiz nymph to vex dwarf.",
            "Bright vixens jump; dozy fowl quack.",
        ]
        self.replacement_ratio_range = replacement_ratio_range
        self.max_attempts = max_attempts
        self.emojis = [
            "😀", "😂", "😍", "🤔", "😎", "🥳", "😴", "🤩", "🥺", "😱",
            "🙄", "😇", "🤗", "🤫", "🤭", "🤥", "🤮", "🤧", "🥶", "🥵",
            "🤠", "🥴", "🤑", "🤓", "🧐", "😈", "👻", "👽", "🤖", "💩",
            "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯",
            "🦁", "🐮", "🐷", "🐸", "🐵", "🐔", "🐧", "🐦", "🦆", "🦉"
        ]
        self.encoded_sentence = None
        self.hint = None
        self.answer = None
        self.epoch = None
        self.history = None
        self.correct_guesses = None

    def _generate_puzzle(self, seed: int) -> Tuple[str, str, Dict[str, str]]:
        """
        Generate encoded sentence with hint.

        Args:
            seed: Random seed

        Returns:
            Tuple of (encoded_sentence, hint, answer)
        """
        random.seed(seed)

        # Select random sentence
        original_sentence = random.choice(self.sentences)
        original_sentence_lower = original_sentence.lower()

        # Count letter frequencies
        letter_counts = collections.Counter([c for c in original_sentence_lower if c in string.ascii_lowercase])
        unique_letters = list(letter_counts.keys())

        # Determine replacement ratio
        replacement_ratio = random.uniform(*self.replacement_ratio_range)
        num_unique_letters = len(unique_letters)
        num_to_replace = max(1, min(int(num_unique_letters * replacement_ratio), num_unique_letters))

        # Select letters to replace
        random.shuffle(unique_letters)
        letters_to_replace = unique_letters[:num_to_replace]

        # Prepare emojis
        available_emojis = self.emojis.copy()
        random.shuffle(available_emojis)

        # Build encoding table
        encoding_table = {}
        for i, letter in enumerate(letters_to_replace):
            if i < len(available_emojis):
                encoding_table[letter] = available_emojis[i]

        # Reverse mapping (emoji -> letter)
        reverse_mapping = {v: k for k, v in encoding_table.items()}

        # Encode sentence
        encoded_sentence = ""
        for char in original_sentence_lower:
            if char in encoding_table:
                encoded_sentence += encoding_table[char]
            else:
                encoded_sentence += char

        # Find most frequent emoji
        emoji_counts = {}
        for char in encoded_sentence:
            if char in reverse_mapping:
                emoji_counts[char] = emoji_counts.get(char, 0) + 1

        hint = ""
        if emoji_counts:
            most_frequent_emoji = max(emoji_counts.items(), key=lambda x: x[1])[0]
            hint = f"{most_frequent_emoji}={reverse_mapping[most_frequent_emoji]}"

        return encoded_sentence, hint, reverse_mapping

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: 😀=c,😂=d...'.\n\n"
            "Alternatively, you can use \\boxed{😀=c,😂=d...} format.\n\n"
            "Next, I will provide a sentence encoded by replacing each letter with a unique emoji. "
            "Then, I will reveal the letter corresponding to the most frequently occurring emoji. "
            "You'll have several attempts to guess the words, and each guess will be recorded in "
            "History for future reference. Based on the provided information, please submit your "
            "guesses for this round in the format 'emoji=word', separated by commas.\n"
            "Note that the emoji provided as a hint, as well as previously correctly answered "
            "emojis, must also be included in your answer.\n\n"
        )

    def _format_board(self) -> str:
        """Format the current game board."""
        lines = []
        lines.append("Crypto Word Game")
        current_attempt = min(self.epoch, self.max_attempts)
        lines.append(f"Attempt: {current_attempt} of {self.max_attempts}")
        lines.append(f"Encoded Sentence: {self.encoded_sentence}")
        lines.append(f"Hint: {self.hint}")
        lines.append("History:")
        if self.history:
            for idx, entry in enumerate(self.history, start=1):
                lines.append(f"{idx}. Guess: {entry['guess']}")
                lines.append(f"   Feedback: {entry['feedback']}")
        else:
            lines.append("No guesses yet.")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new CryptoWord puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        self.encoded_sentence, self.hint, self.answer = self._generate_puzzle(puzzle_seed)

        # Initialize game state
        self.epoch = 1
        self.history = []
        self.correct_guesses = {}

        # Build observation
        board_str = self._format_board()
        observation = f"{self._get_instructions()}{board_str}\n"

        return observation, {
            "suffix": f"Emojis to decode: {len(self.answer)}, Max attempts: {self.max_attempts}."
        }

    def _verify_guess(self, guess: str) -> Dict[str, bool]:
        """Verify player's guess and return feedback."""
        feedback = {}
        for pair in guess.split(','):
            pair = pair.strip()
            if '=' in pair:
                emoji_guess, letter_guess = pair.split('=', 1)
                emoji_guess = emoji_guess.strip()
                letter_guess = letter_guess.strip().lower()
                if emoji_guess in self.answer:
                    feedback[emoji_guess] = (letter_guess == self.answer[emoji_guess])
                else:
                    feedback[emoji_guess] = False
        return feedback

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process player's guess.

        Args:
            action: Guess in format "emoji=letter,emoji=letter,..."

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
                "Please use 'Answer: 😀=c,😂=d...' or \\boxed{😀=c,😂=d...} format."
            )
            return obs, 0.0, True, False, {}

        guess = str(parsed_action).strip()

        # Verify guess
        feedback = self._verify_guess(guess)

        # Record history
        self.history.append({"guess": guess, "feedback": feedback})

        # Update correct guesses
        for emoji, is_correct in feedback.items():
            if is_correct:
                letter = [pair.split('=')[1].strip().lower() for pair in guess.split(',')
                          if pair.strip().startswith(emoji + '=')][0]
                self.correct_guesses[emoji] = letter

        # Check if game is complete
        if len(self.correct_guesses) == len(self.answer):
            obs = (
                f"Congratulations! You decoded all emojis!\n"
                f"Attempts used: {self.epoch}/{self.max_attempts}\n"
                f"Correct mapping: {self.correct_guesses}"
            )
            return obs, 1.0, True, False, {}

        if self.epoch >= self.max_attempts:
            obs = (
                f"Game over! You've reached the maximum number of attempts.\n"
                f"Correct answer: {self.answer}\n"
                f"You decoded: {len(self.correct_guesses)}/{len(self.answer)} emojis"
            )
            return obs, 0.0, True, False, {}

        # Continue game
        self.epoch += 1
        board_str = self._format_board()
        observation = f"{self._get_instructions()}{board_str}\n"

        return observation, 0.0, False, False, {}

    def sample_random_action(self) -> str:
        """
        Sample the correct answer.

        Returns:
            Correct answer as string
        """
        pairs = [f"{emoji}={letter}" for emoji, letter in self.answer.items()]
        answer_str = ",".join(pairs)
        return f"\\boxed{{{answer_str}}}"
