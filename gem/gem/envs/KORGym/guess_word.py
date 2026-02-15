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

"""GuessWord environment - Word puzzle with letter-position constraints."""

import os
import random
import string
from typing import Optional, Tuple, Dict, Any, List, Set

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class GuessWordEnv(Env):
    """
    GuessWord puzzle environment.

    The agent must provide an English word that satisfies specific
    letter-position constraints (e.g., position 2 is 'a', position 5 is 't').

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 10,
        min_rules: int = 3,
        max_rules: int = 4,
        **_,
    ):
        """
        Initialize GuessWord environment.

        Args:
            min_length: Minimum word length
            max_length: Maximum word length
            min_rules: Minimum number of letter-position rules
            max_rules: Maximum number of letter-position rules
        """
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        self.min_rules = min_rules
        self.max_rules = max_rules
        self.words = self._load_words()
        self.word_length = None
        self.rules = None
        self.valid_words = None

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
                            if len(word) > 4:  # Only words longer than 4 letters
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

    def _get_valid_words(self, length: int, rules: List[Tuple[int, str]]) -> Set[str]:
        """Get all valid words matching the length and position rules."""
        valid_words = set()
        for word in self.words:
            if len(word) != length:
                continue
            match = True
            for pos, letter in rules:
                if pos >= length or word[pos] != letter:
                    match = False
                    break
            if match:
                valid_words.add(word)
        return valid_words

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer to the question, e.g., 'Answer: apple'\n\n"
            "Alternatively, you can use \\boxed{word} format, e.g., \\boxed{apple}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new word puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        if not self.words:
            raise ValueError("No words available. Word list is empty.")

        # Generate puzzle with valid solution
        max_attempts = 100
        for _ in range(max_attempts):
            self.word_length = random.randint(self.min_length, self.max_length)
            rules_num = random.randint(self.min_rules, self.max_rules)
            self.rules = []
            attempts = 0

            while len(self.rules) < rules_num and attempts < max_attempts:
                pos = random.randint(0, self.word_length - 1)
                letter = random.choice(string.ascii_lowercase)

                # Check if this position is already used
                if not any(pos == p for p, _ in self.rules):
                    temp_rules = self.rules + [(pos, letter)]
                    valid_words = self._get_valid_words(self.word_length, temp_rules)
                    if valid_words:
                        self.rules = temp_rules
                attempts += 1

            if len(self.rules) == rules_num:
                self.valid_words = self._get_valid_words(self.word_length, self.rules)
                if self.valid_words:
                    break

        if not self.valid_words:
            # Fallback: simpler puzzle
            self.word_length = 5
            self.rules = [(0, 'a')]
            self.valid_words = self._get_valid_words(self.word_length, self.rules)

        # Build question text
        rules_desc = []
        for pos, letter in self.rules:
            rules_desc.append(f"the letter at position {pos+1} is '{letter}'")
        rules_text = " and ".join(rules_desc)

        question = (
            f"Please provide an English word that meets the following requirements:\n"
            f"1. The word must be {self.word_length} letters long\n"
            f"2. {rules_text}\n"
        )

        observation = f"{self._get_instructions()}{question}"
        return observation, {"suffix": "Provide a valid word."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's word.

        Args:
            action: Agent's response containing the word

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

        # Normalize answer
        user_answer = parsed_answer.strip().lower()

        # Check length
        if len(user_answer) != self.word_length:
            obs = (
                f"Incorrect. The word must be {self.word_length} letters long. "
                f"Your answer '{user_answer}' has {len(user_answer)} letters."
            )
            return obs, 0.0, True, False, {}

        # Check if word is in dictionary
        if user_answer not in self.words:
            obs = f"Incorrect. '{user_answer}' is not a valid English word in our dictionary."
            return obs, 0.0, True, False, {}

        # Check position rules
        for pos, letter in self.rules:
            if user_answer[pos] != letter:
                obs = (
                    f"Incorrect. Your answer '{user_answer}' does not satisfy the constraint: "
                    f"position {pos+1} should be '{letter}' but is '{user_answer[pos]}'."
                )
                return obs, 0.0, True, False, {}

        # Success!
        obs = f"Correct! '{user_answer}' is a valid answer."
        return obs, 1.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random valid word (for testing)
        """
        if self.valid_words:
            word = random.choice(list(self.valid_words))
            return f"\\boxed{{{word}}}"
        else:
            return f"\\boxed{{apple}}"
