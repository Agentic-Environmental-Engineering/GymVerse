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

"""WordPuzzle environment - Crossword puzzle game with multimodal observations."""

import ast
import base64
import io
import math
import os
import random
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class WordPuzzleEnv(Env):
    """
    Word Puzzle (Crossword) environment.

    Players are given clues and a partially filled crossword grid image.
    They must guess all the words that fit the grid based on the clues.

    This is a single-turn multimodal environment with partial credit scoring.
    """

    def __init__(
        self,
        min_words: int = 5,
        max_words: int = 15,
        min_difficulty: float = 0.5,
        max_difficulty: float = 0.9,
        grid_size: int = 20,
        word_clues_path: Optional[str] = None,
        **_,
    ):
        """
        Initialize WordPuzzle environment.

        Args:
            min_words: Minimum number of words in puzzle
            max_words: Maximum number of words in puzzle
            min_difficulty: Minimum difficulty (fraction of letters masked)
            max_difficulty: Maximum difficulty (fraction of letters masked)
            grid_size: Size of the grid for word placement
            word_clues_path: Path to word-clue CSV file
        """
        super().__init__()
        self.min_words = min_words
        self.max_words = max_words
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.grid_size = grid_size

        # Load word bank
        if word_clues_path is None:
            # Default path relative to this file (parent directory - korgym root)
            word_clues_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "high_quality_word_clues.csv"
            )
        self.word_bank = self._load_word_bank(word_clues_path)

        self.answer = None
        self.clues = None

    def _load_word_bank(self, path: str) -> Dict[str, str]:
        """Load word-clue pairs from CSV."""
        df = pd.read_csv(path)
        df['word'] = df['word'].str.strip().str.lower()
        df = df.drop_duplicates('word').set_index('word')
        word_dict = df['clue'].to_dict()
        # Filter to valid words (3-12 letters, alphabetic only)
        valid_words = {
            w: c for w, c in word_dict.items()
            if 3 <= len(w) <= 12 and w.isalpha()
        }
        return valid_words

    def _select_words(self, num: int, seed: int) -> Tuple[List[str], List[str]]:
        """Select words with bias toward mid-length words."""
        random.seed(seed)
        words = list(self.word_bank.keys())
        # Bias toward 5-8 letter words
        weights = [3 if 5 <= len(w) <= 8 else 1 for w in words]

        selected = []
        while len(selected) < num and words:
            chosen = random.choices(words, weights=weights, k=1)[0]
            if chosen not in selected:
                selected.append(chosen)
                idx = words.index(chosen)
                words.pop(idx)
                weights.pop(idx)

        descriptions = [self.word_bank[w] for w in selected]
        return selected, descriptions

    def _place_words(
        self,
        words: List[str],
        grid_size: int
    ) -> Tuple[List[List[Optional[str]]], List[Dict]]:
        """
        Place words in a crossword grid.

        Returns:
            Tuple of (grid, placed_info)
        """
        grid = [[None] * grid_size for _ in range(grid_size)]
        placed_info = []

        # Sort words by length (longest first)
        sorted_words = sorted(enumerate(words), key=lambda x: -len(x[1]))

        for original_idx, word in sorted_words:
            word = word.upper()
            placed = False
            max_attempts = 200 if original_idx == 0 else 500

            for _ in range(max_attempts):
                direction = random.choice(['across', 'down'])
                word_len = len(word)

                if direction == 'across':
                    max_col = grid_size - word_len
                    max_row = grid_size - 1
                else:
                    max_row = grid_size - word_len
                    max_col = grid_size - 1

                if max_row < 0 or max_col < 0:
                    continue

                start_row = random.randint(0, max_row)
                start_col = random.randint(0, max_col)

                # Check if placement is valid
                valid = True
                overlaps = 0
                temp_grid = [row[:] for row in grid]

                for i in range(word_len):
                    r = start_row + (i if direction == 'down' else 0)
                    c = start_col + (i if direction == 'across' else 0)

                    if temp_grid[r][c]:
                        if temp_grid[r][c] != word[i]:
                            valid = False
                            break
                        overlaps += 1
                    else:
                        temp_grid[r][c] = word[i]

                if valid and (overlaps > 0 or original_idx == 0):
                    # Place word in grid
                    for i in range(word_len):
                        r = start_row + (i if direction == 'down' else 0)
                        c = start_col + (i if direction == 'across' else 0)
                        grid[r][c] = temp_grid[r][c]

                    placed_info.append({
                        'number': original_idx + 1,
                        'row': start_row,
                        'col': start_col,
                        'direction': direction,
                        'word': word
                    })
                    placed = True
                    break

            if not placed:
                return grid, placed_info

        return grid, placed_info

    def _render_image(
        self,
        char_grid: List[List[Optional[str]]],
        placed_info: List[Dict],
        difficulty: float
    ) -> bytes:
        """
        Render crossword puzzle image with partial masking.

        Returns:
            PNG image as bytes
        """
        cell_size = 35

        # Find bounding box of puzzle
        min_row = min(info['row'] for info in placed_info)
        max_row = max(
            (info['row'] + len(info['word']) - 1)
            if info['direction'] == 'down' else info['row']
            for info in placed_info
        )
        min_col = min(info['col'] for info in placed_info)
        max_col = max(
            (info['col'] + len(info['word']) - 1)
            if info['direction'] == 'across' else info['col']
            for info in placed_info
        )

        padding = 25
        img = Image.new(
            'RGB',
            ((max_col - min_col + 1) * cell_size + 2 * padding,
             (max_row - min_row + 1) * cell_size + 2 * padding),
            color=(255, 255, 255)
        )
        draw = ImageDraw.Draw(img)

        # Determine masked positions
        masked_positions = set()
        for info in placed_info:
            word = info['word']
            word_len = len(word)
            k = max(1, math.ceil(difficulty * word_len))
            k = min(k, word_len)
            indices = random.sample(range(word_len), k)
            start_row, start_col = info['row'], info['col']
            direction = info['direction']

            for i in indices:
                if direction == 'across':
                    r = start_row
                    c = start_col + i
                else:
                    r = start_row + i
                    c = start_col

                masked_positions.add((r, c))

        # Try to load a font, fallback to default
        try:
            font_small = ImageFont.truetype("arial.ttf", 12)
            font_large = ImageFont.truetype("arial.ttf", 14)
        except:
            try:
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font_small = ImageFont.load_default()
                font_large = ImageFont.load_default()

        # Draw grid
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                x0 = padding + (c - min_col) * cell_size
                y0 = padding + (r - min_row) * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                if r >= len(char_grid) or c >= len(char_grid[0]):
                    continue

                is_active = char_grid[r][c] is not None

                if is_active:
                    # White cell with letter
                    draw.rectangle([x0, y0, x1, y1], fill='white', outline='#CCCCCC')

                    # Draw number if this is a word start
                    for info in placed_info:
                        if info['row'] == r and info['col'] == c:
                            draw.text(
                                (x0 + 2, y0 + 2),
                                str(info['number']),
                                fill='#FF4444',
                                font=font_small
                            )

                    # Draw letter or underline
                    if (r, c) not in masked_positions:
                        char = char_grid[r][c]
                        bbox = draw.textbbox((0, 0), char, font=font_large)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        tx = x0 + (cell_size - text_width) / 2
                        ty = y0 + (cell_size - text_height) / 2
                        draw.text((tx, ty), char, fill='black', font=font_large)
                    else:
                        # Draw underline for masked letter
                        underline_y = y0 + cell_size - 4
                        draw.line(
                            [x0 + 3, underline_y, x1 - 3, underline_y],
                            fill='#666666',
                            width=2
                        )
                else:
                    # Grey cell (not part of puzzle)
                    draw.rectangle([x0, y0, x1, y1], fill='#DDDDDD', outline='#CCCCCC')

        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board (as an image) and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: [\"happy\", \"person\"]'\n\n"
            "Alternatively, you can use \\boxed{[\"happy\", \"person\"]} format.\n\n"
            "You need to complete a crossword puzzle that consists of grey and white squares. "
            "The solver guesses the words corresponding to the horizontal and vertical directions "
            "based on a set of clues. During the decryption process, every white square must be "
            "filled with a letter, and the red number in the first white square of each entry "
            "corresponds to its clue number. You need to provide all the words in order in your "
            "answer as a list.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a new word puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle with retries
        max_attempts = 1000
        grid_size = self.grid_size

        for attempt in range(max_attempts):
            num_words = random.randint(self.min_words, self.max_words)
            difficulty = random.uniform(self.min_difficulty, self.max_difficulty)

            selected, descs = self._select_words(num_words, seed + attempt if seed else attempt)
            grid, placed = self._place_words(selected, grid_size)

            if len(placed) == num_words:
                # Success! All words placed
                placed_sorted = sorted(placed, key=lambda x: x['number'])
                self.answer = [p['word'].lower() for p in placed_sorted]
                self.clues = [descs[p['number'] - 1] for p in placed_sorted]

                # Generate image
                image_bytes = self._render_image(grid, placed, difficulty)

                # Create multimodal observation
                clues_text = "Clues:\n"
                for i, clue in enumerate(self.clues, 1):
                    clues_text += f"{i}. {clue}\n"

                observation = {
                    "text": f"{self._get_instructions()}{clues_text}",
                    "image": base64.b64encode(image_bytes).decode('utf-8'),
                    "image_format": "png"
                }

                return observation, {"suffix": f"Complete crossword puzzle ({num_words} words)."}

            # If failed, try with larger grid
            if attempt % 20 == 19:
                grid_size += 5
                if grid_size > 40:
                    grid_size = self.grid_size  # Reset

        # If all attempts failed, return a simple puzzle
        raise RuntimeError("Failed to generate valid crossword puzzle after many attempts")

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's solution.

        Args:
            action: Agent's response containing the list of words

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer using \boxed{} format first
        parsed_answer = extract_last_boxed_answer(action)

        # If not found, try "Answer:" format
        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(\[.+?\])', action, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: [\"word1\", \"word2\", ...]' or "
                "\\boxed{[\"word1\", \"word2\", ...]} format."
            )
            return obs, 0.0, True, False, {}

        # Parse the answer list
        try:
            answers = ast.literal_eval(parsed_answer)
        except Exception as e:
            obs = f"Failed to parse answer. Error: {e}"
            return obs, 0.0, True, False, {}

        if not isinstance(answers, list):
            obs = "Answer must be a list of words."
            return obs, 0.0, True, False, {}

        # Check each answer
        correct_count = 0
        total = len(self.answer)
        details = []

        if len(answers) != total:
            details.append(
                f"Warning: You provided {len(answers)} answers, "
                f"but {total} words are expected."
            )

        for i, correct_word in enumerate(self.answer):
            user_word = answers[i].strip() if i < len(answers) else ""
            is_correct = user_word.lower() == correct_word.strip().lower()

            if is_correct:
                correct_count += 1
                details.append(f"✓ Word {i+1}: '{user_word}' is correct!")
            else:
                details.append(
                    f"✗ Word {i+1}: You wrote '{user_word}', "
                    f"but correct answer is '{correct_word}'"
                )

        score = correct_count / total if total > 0 else 0
        obs = f"Score: {correct_count}/{total} = {score:.2f}\n" + "\n".join(details)

        return obs, score, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            The correct answer (for testing)
        """
        if self.answer is not None:
            return f"\\boxed{{{self.answer}}}"
        else:
            return "\\boxed{[\"example\"]}"
