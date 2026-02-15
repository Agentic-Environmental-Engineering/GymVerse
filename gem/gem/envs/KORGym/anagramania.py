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

"""Anagramania environment - Rearrange shuffled letters to form the original word."""

import os
import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class AnagramaniaEnv(Env):
    """
    Anagramania word puzzle environment.

    The agent is shown a word with shuffled letters (first letter fixed).
    The agent must rearrange the letters to find the original word.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_word_length: int = 5,
        max_word_length: int = 10,
        **_,
    ):
        """
        Initialize Anagramania environment.

        Args:
            min_word_length: Minimum length of words to use
            max_word_length: Maximum length of words to use
        """
        super().__init__()
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.words = self._load_words()
        self.correct_word = None
        self.anagram = None

    def _load_words(self) -> List[str]:
        """Load word list from file."""
        words = []
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
                            line = line.strip().lower()
                            # Filter by length
                            if self.min_word_length <= len(line) <= self.max_word_length:
                                words.append(line)
                    break
                except Exception as e:
                    continue

        if not words:
            # Fallback word list if file not found
            words = [
                "apple", "banana", "cherry", "dragon", "elephant",
                "flower", "guitar", "horizon", "island", "jungle",
                "keyboard", "library", "mountain", "notebook", "orange",
                "penguin", "quarter", "rainbow", "science", "thunder",
                "umbrella", "volcano", "whisper", "xylophone", "yellow", "zebra"
            ]

        return words

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game problem-solver. I'll give you a question.\n"
            "Your task is:\n"
            "- First, answer the question.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer to the question, e.g., 'Answer: happy'\n\n"
            "Alternatively, you can use \\boxed{word} format, e.g., \\boxed{happy}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new anagram puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        if not self.words:
            raise ValueError("No words available. Word list is empty.")

        # Randomly select a word as the correct answer
        self.correct_word = random.choice(self.words)

        # Fix the first letter and shuffle the rest
        chars = list(self.correct_word[1:])
        random.shuffle(chars)
        anagram_chars = [self.correct_word[0]] + chars
        self.anagram = " ".join(anagram_chars)

        # Construct the question
        question = (
            "Please rearrange the letters to form the original word for this anagram. "
            "The first letter is already in the correct position.\n"
            f"{self.anagram}"
        )

        observation = f"{self._get_instructions()}{question}"
        return observation, {"suffix": "Rearrange the letters to find the word."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's answer.

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

        # Normalize answer (lowercase, strip spaces)
        user_answer = parsed_answer.strip().lower()
        correct_normalized = self.correct_word.lower()

        # Check if answer is correct
        is_correct = user_answer == correct_normalized
        reward = 1.0 if is_correct else 0.0

        if is_correct:
            obs = f"Correct! The word is '{self.correct_word}'."
        else:
            obs = f"Incorrect. Your answer was '{user_answer}', but the correct word is '{self.correct_word}'."

        return obs, reward, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random word (for testing, returns correct answer)
        """
        if self.correct_word is not None:
            return f"\\boxed{{{self.correct_word}}}"
        else:
            # Return a random word
            return f"\\boxed{{{random.choice(self.words)}}}"
