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

"""Word Encryption environment - Decrypt transformed words."""

import os
import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class WordEncryptionEnv(Env):
    """
    Word Encryption puzzle environment.

    A word is transformed using a series of transformations (shift, reverse, emoji, etc.).
    The agent must recover the original word.

    This is a single-turn environment with sparse rewards.
    """

    # Emoji mapping for Transform_8
    EMOJI_MAPPING = {
        'a': '😀🍎🚗', 'b': '🐶🌟📚', 'c': '🌈🍀🚀', 'd': '🐱🍉🏀',
        'e': '🍔🎉🎈', 'f': '🌸🍩🏰', 'g': '🦋🍇⚽', 'h': '🍕🎂🏝️',
        'i': '🍦🎁🎧', 'j': '🐸🍒🏆', 'k': '🦄🍓🎮', 'l': '🐰🍍📷',
        'm': '🌹🍌🎨', 'n': '🐼🍎🎤', 'o': '🍉🎵📚', 'p': '🌼🍇🎬',
        'q': '🐢🍓🎯', 'r': '🍒🎸📱', 's': '🌻🍍🎲', 't': '🐯🍌🎮',
        'u': '🍓🎹📖', 'v': '🌺🍉🎥', 'w': '🐳🍎🎭', 'x': '🍍🎤📡',
        'y': '🐥🍇🎨', 'z': '🌵🍒🎮'
    }

    def __init__(
        self,
        min_rules: int = 2,
        max_rules: int = 10,
        **_,
    ):
        """
        Initialize Word Encryption environment.

        Args:
            min_rules: Minimum number of transformation rules
            max_rules: Maximum number of transformation rules
        """
        super().__init__()
        self.min_rules = min_rules
        self.max_rules = max_rules
        self.words = self._load_words()
        self.original_word = None
        self.rules = None
        self.transformed_word = None

    def _load_words(self) -> List[str]:
        """Load word list from file."""
        words = []
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
                            if len(word) > 2:
                                words.append(word)
                    break
                except Exception:
                    continue

        if not words:
            words = [
                "happy", "world", "python", "coding", "computer",
                "science", "learning", "adventure", "beautiful", "challenge"
            ]

        return words

    def _char_drift(self, c: str) -> str:
        """Shift letter by one position (a->b, z->a)."""
        return chr((ord(c) - ord('a') + 1) % 26 + ord('a'))

    def _char2emoji(self, c: str) -> str:
        """Convert letter to emoji string."""
        return self.EMOJI_MAPPING.get(c, c)

    def _transform_1(self, word: str) -> str:
        """Transform_1: Repeat each letter twice (e.g., 'happy' -> 'hhaappppyy')."""
        return "".join([c * 2 for c in word])

    def _transform_2(self, word: str) -> str:
        """Transform_2: Shift each letter by one (e.g., 'happy' -> 'ibqqz')."""
        return "".join([self._char_drift(c) for c in word])

    def _transform_3(self, word: str) -> str:
        """Transform_3: Cyclic right shift by one (e.g., 'happy' -> 'yhapp')."""
        return word[-1] + word[:-1] if word else word

    def _transform_4(self, word: str) -> str:
        """Transform_4: Reverse the word (e.g., 'happy' -> 'yppah')."""
        return word[::-1]

    def _transform_5(self, word: str) -> str:
        """Transform_5: Cyclic left shift by two (e.g., 'happy' -> 'ppyha')."""
        return word[2:] + word[:2] if len(word) > 2 else word

    def _transform_6(self, word: str) -> str:
        """Transform_6: Shift even-indexed letters (e.g., 'happy' -> 'hbpqy')."""
        return "".join([self._char_drift(c) if i % 2 == 0 else c for i, c in enumerate(word)])

    def _transform_7(self, word: str) -> str:
        """Transform_7: Shift odd-indexed letters (e.g., 'happy' -> 'iaqpz')."""
        return "".join([self._char_drift(c) if i % 2 == 1 else c for i, c in enumerate(word)])

    def _transform_8(self, word: str) -> str:
        """Transform_8: Convert letters to emojis."""
        return "".join([self._char2emoji(c) for c in word])

    def _apply_transforms(self, word: str, rules: List[str]) -> str:
        """Apply a series of transformations to a word."""
        rule_functions = {
            "Transform_1": self._transform_1,
            "Transform_2": self._transform_2,
            "Transform_3": self._transform_3,
            "Transform_4": self._transform_4,
            "Transform_5": self._transform_5,
            "Transform_6": self._transform_6,
            "Transform_7": self._transform_7,
            "Transform_8": self._transform_8,
        }

        for rule in rules:
            word = rule_functions[rule](word)
        return word

    def _get_instructions(self) -> str:
        """Return game instructions."""
        mapping_str = str(self.EMOJI_MAPPING)
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer to the question, e.g., 'Answer: happy'\n\n"
            "Alternatively, you can use \\boxed{word} format, e.g., \\boxed{happy}\n\n"
            "This game provides you with a transformed word generated by applying a series of "
            "transformations on an original word. Your task is to recover the original word.\n\n"
            "Transformations:\n"
            "Transform_1: Repeat each letter (e.g., 'happy' -> 'hhaappppyy').\n"
            "Transform_2: Shift each letter to the next letter (e.g., 'happy' -> 'ibqqz').\n"
            "Transform_3: Cyclic shift right by one (e.g., 'happy' -> 'yhapp').\n"
            "Transform_4: Reverse the word (e.g., 'happy' -> 'yppah').\n"
            "Transform_5: Cyclic shift left by two (e.g., 'happy' -> 'ppyha').\n"
            "Transform_6: Shift even-indexed letters (e.g., 'happy' -> 'hbpqy').\n"
            "Transform_7: Shift odd-indexed letters (e.g., 'happy' -> 'iaqpz').\n"
            f"Transform_8: Convert letters to emojis. Mapping table: {mapping_str}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new puzzle instance.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        if not self.words:
            raise ValueError("No words available. Word list is empty.")

        # Generate random transformation rules (last one is always Transform_8)
        rules_num = random.randint(self.min_rules, self.max_rules)
        self.rules = []
        for i in range(rules_num - 1):
            k = random.choice(list(range(1, 8)))
            self.rules.append(f"Transform_{k}")
        self.rules.append("Transform_8")  # Always end with emoji conversion

        # Select random word and apply transformations
        self.original_word = random.choice(self.words)
        self.transformed_word = self._apply_transforms(self.original_word, self.rules)

        # Build question
        question = (
            f"Transformed word: {self.transformed_word}\n"
            f"Transforms applied: {', '.join(self.rules)}\n"
            "Please recover the original word from the above transformed word.\n"
        )

        observation = f"{self._get_instructions()}{question}"
        return observation, {"suffix": "Recover the original word."}

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
        correct_normalized = self.original_word.lower()

        # Check if answer is correct
        is_correct = user_answer == correct_normalized
        reward = 1.0 if is_correct else 0.0

        if is_correct:
            obs = f"Correct! The original word is '{self.original_word}'."
        else:
            obs = f"Incorrect. Your answer was '{user_answer}', but the correct word is '{self.original_word}'."

        return obs, reward, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random word (for testing, returns correct answer)
        """
        if self.original_word is not None:
            return f"\\boxed{{{self.original_word}}}"
        else:
            return f"\\boxed{{{random.choice(self.words)}}}"
