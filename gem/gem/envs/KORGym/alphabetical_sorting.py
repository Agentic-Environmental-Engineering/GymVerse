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

"""Alphabetical Sorting environment - Find word in 3x3 grid following a path."""

import os
import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class AlphabeticalSortingEnv(Env):
    """
    Alphabetical Sorting word puzzle environment.

    A 9-letter word is placed in a 3x3 grid following a continuous path.
    The agent must identify the word by following the path.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        grid_size: int = 3,
        **_,
    ):
        """
        Initialize Alphabetical Sorting environment.

        Args:
            grid_size: Size of the grid (default 3x3)
        """
        super().__init__()
        self.grid_size = grid_size
        self.word_length = grid_size * grid_size
        self.words = self._load_words()
        self.all_paths = self._generate_all_paths()
        self.board = None
        self.correct_word = None
        self.possible_answers = None

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
                            word = line.strip().lower()
                            # Filter by length
                            if len(word) == self.word_length:
                                words.append(word)
                    break
                except Exception:
                    continue

        if not words:
            # Fallback word list if file not found
            words = [
                "adventure", "beautiful", "challenge", "dangerous", "education",
                "fireworks", "geography", "happiness", "important", "knowledge",
            ]

        return words

    def _dfs(self, k: int, visited: List[List[bool]], i: int, j: int,
             path: List[List[int]], ans: List[List[List[int]]]):
        """DFS to generate all possible paths in the grid."""
        n = self.grid_size
        if i < 0 or i >= n or j < 0 or j >= n:
            return
        if visited[i][j]:
            return

        visited[i][j] = True
        path = path + [[i, j]]

        if k == n * n:
            ans.append(path[:])
            visited[i][j] = False
            return

        # Try all 4 directions
        self._dfs(k + 1, visited, i + 1, j, path, ans)
        self._dfs(k + 1, visited, i - 1, j, path, ans)
        self._dfs(k + 1, visited, i, j + 1, path, ans)
        self._dfs(k + 1, visited, i, j - 1, path, ans)

        visited[i][j] = False

    def _generate_all_paths(self) -> List[List[List[int]]]:
        """Generate all possible continuous paths in the grid."""
        n = self.grid_size
        all_paths = []

        # Try starting from each position
        for i in range(n):
            for j in range(n):
                ans = []
                visited = [[False] * n for _ in range(n)]
                self._dfs(1, visited, i, j, [], ans)
                all_paths.extend(ans)

        return all_paths

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
        Generate a new puzzle instance.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        if not self.words:
            raise ValueError("No words available. Word list is empty.")

        # Randomly select a word
        self.correct_word = random.choice(self.words)

        # Initialize empty grid
        n = self.grid_size
        self.board = [['' for _ in range(n)] for _ in range(n)]

        # Select a random path and fill the word
        path = random.choice(self.all_paths)
        for pos, char in zip(path, self.correct_word):
            i, j = pos
            self.board[i][j] = char

        # Build board display
        board_str = "\n".join(["|".join(row) for row in self.board])

        question = (
            f"Game rules: A word with a length of {self.word_length}, randomly select a starting point "
            f"in a {n}x{n} square, and fill in the letters in the order they appear in the word, "
            "selecting consecutive positions to place them in the grid. Please identify the word in the square.\n\n"
            f"board:\n{board_str}"
        )

        # Precompute possible answers for verification
        words_set = set(self.words)
        self.possible_answers = set()
        for p in self.all_paths:
            word = "".join([self.board[i][j] for i, j in p])
            if word in words_set:
                self.possible_answers.add(word.lower())

        observation = f"{self._get_instructions()}{question}"
        return observation, {"suffix": "Identify the word in the grid."}

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

        # Check if answer is in possible answers
        is_correct = user_answer in self.possible_answers
        reward = 1.0 if is_correct else 0.0

        if is_correct:
            obs = f"Correct! The word is '{user_answer}'."
        else:
            obs = (f"Incorrect. Your answer was '{user_answer}'. "
                  f"The correct answer could be one of: {', '.join(sorted(self.possible_answers)[:3])}...")

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
