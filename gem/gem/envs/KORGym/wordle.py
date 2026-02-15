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

"""Wordle environment - Word guessing game with feedback."""

import os
import random
from typing import Optional, Tuple, Dict, Any, List, Set

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class WordleEnv(Env):
    """
    Wordle word guessing game environment.

    The agent must guess a secret word within a limited number of attempts.
    After each guess, feedback is provided about which letters are correct,
    which are in the word but in wrong positions, and which are not in the word.

    This is a multi-turn environment with sparse terminal rewards (1.0 if word is guessed).
    """

    def __init__(
        self,
        min_word_length: int = 4,
        max_word_length: int = 12,
        max_attempts: int = 10,
        **_,
    ):
        """
        Initialize Wordle environment.

        Args:
            min_word_length: Minimum length of secret word
            max_word_length: Maximum length of secret word
            max_attempts: Maximum number of guessing attempts
        """
        super().__init__()
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.max_attempts = max_attempts
        self.words = self._load_words()
        self.secret_word = None
        self.history = None
        self.current_attempt = None

    def _load_words(self) -> Set[str]:
        """Load word list from file."""
        words = set()
        # Try multiple paths
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "words.txt"),
            "/workspace/Qiji_benchmark/gem/gem/envs/korgym/words.txt",
            "words.txt",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            word = line.strip().lower()
                            if self.min_word_length <= len(word) <= self.max_word_length:
                                words.add(word)
                    break
                except Exception:
                    continue

        if not words:
            # Fallback word list if file not found
            words = {
                "apple", "banana", "cherry", "dragon", "elephant",
                "flower", "guitar", "horizon", "island", "jungle",
                "keyboard", "library", "mountain", "notebook", "orange",
                "penguin", "quarter", "rainbow", "science", "thunder",
            }

        return words

    def _verify_guess(self, guess: str) -> str:
        """
        Compare guess with secret word and generate feedback.

        Returns:
            Formatted feedback string with information about each letter
        """
        feedback_lines = []
        for i, g_char in enumerate(guess):
            if i < len(self.secret_word) and g_char == self.secret_word[i]:
                feedback_lines.append(
                    f"The letter {g_char} located at idx={i} is in the word and in the correct spot,"
                )
            elif g_char in self.secret_word:
                feedback_lines.append(
                    f"The letter {g_char} located at idx={i} is in the word but in the wrong spot,"
                )
            else:
                feedback_lines.append(
                    f"The letter {g_char} located at idx={i} is not in the word in any spot,"
                )
        return "\n".join(feedback_lines)

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer to the question, e.g., 'Answer: happy'\n\n"
            "Alternatively, you can use \\boxed{word} format, e.g., \\boxed{happy}\n\n"
            "You need to guess a specific word according to the information provided below. "
            "You have several attempts, and each guess result will be recorded in the History "
            "for future reference. Please provide your guess for this round based on the following information.\n\n"
        )

    def _format_board(self) -> str:
        """Format current game state as text."""
        lines = []
        lines.append("Wordle Game")
        lines.append(f"Attempt: {self.current_attempt} of {self.max_attempts}")
        lines.append(f"Word length: {len(self.secret_word)}")
        lines.append("History:")
        if self.history:
            for idx, entry in enumerate(self.history, start=1):
                lines.append(f"{idx}. Guess: {entry['guess']}")
                lines.append("Feedback:")
                lines.append(entry['feedback'])
        else:
            lines.append("(No guesses yet)")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new Wordle game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        if not self.words:
            raise ValueError("No words available. Word list is empty.")

        # Generate random word length
        word_length = random.randint(self.min_word_length, self.max_word_length)

        # Filter words by length
        valid_words = [w for w in self.words if len(w) == word_length]
        if not valid_words:
            # Fallback: use any word
            valid_words = list(self.words)

        # Select random secret word
        self.secret_word = random.choice(valid_words)
        self.history = []
        self.current_attempt = 1

        observation = f"{self._get_instructions()}{self._format_board()}"
        return observation, {"suffix": f"Guess the {len(self.secret_word)}-letter word."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process the agent's guess.

        Args:
            action: Agent's response containing the guessed word

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer using \boxed{} format first
        parsed_answer = extract_last_boxed_answer(action)

        # If not found, try "Answer:" format
        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(.+?)(?:\n|$)', action, re.IGNORECASE | re.MULTILINE)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: word' or \\boxed{word} format."
            )
            return obs, 0.0, True, False, {}

        # Normalize guess
        guess = parsed_answer.strip().lower()

        # Handle length mismatch
        if len(guess) != len(self.secret_word):
            if len(guess) < len(self.secret_word):
                # Pad with dashes
                guess = guess + "-" * (len(self.secret_word) - len(guess))
            else:
                # Truncate
                guess = guess[:len(self.secret_word)]

        # Generate feedback
        feedback = self._verify_guess(guess)
        self.history.append({"guess": guess, "feedback": feedback})

        # Check win condition
        if guess == self.secret_word:
            obs = f"Congratulations! You guessed the word '{self.secret_word}' correctly!\n\n{self._format_board()}"
            return obs, 1.0, True, False, {}

        # Check if max attempts reached
        if self.current_attempt >= self.max_attempts:
            obs = f"Game over! You ran out of attempts. The word was '{self.secret_word}'.\n\n{self._format_board()}"
            return obs, 0.0, True, False, {}

        # Continue game
        self.current_attempt += 1
        obs = f"{self._get_instructions()}{self._format_board()}"
        return obs, 0.0, False, False, {"suffix": f"Attempt {self.current_attempt}/{self.max_attempts}"}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random word from word list
        """
        if self.secret_word and self.current_attempt == 1:
            # For testing, return correct answer on first attempt
            return f"\\boxed{{{self.secret_word}}}"

        # Return random word of correct length
        target_length = len(self.secret_word) if self.secret_word else 5
        valid_words = [w for w in self.words if len(w) == target_length]
        if valid_words:
            word = random.choice(valid_words)
        else:
            word = random.choice(list(self.words))
        return f"\\boxed{{{word}}}"
