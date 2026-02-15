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

"""JigsawPuzzle environment - Image puzzle matching game."""

import ast
import base64
import io
import os
import random
import string
from typing import Optional, Tuple, Dict, Any, List

from PIL import Image, ImageDraw, ImageFont

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class JigsawPuzzleEnv(Env):
    """
    Jigsaw Puzzle environment.

    Players are shown a grid with numbered positions and shuffled puzzle pieces
    with letter labels. They must match each position number to the correct
    piece letter to restore the original image.

    This is a single-turn multimodal environment with partial credit scoring.
    """

    def __init__(
        self,
        min_pieces: int = 4,
        max_pieces: int = 12,
        pictures_dir: Optional[str] = None,
        **_,
    ):
        """
        Initialize JigsawPuzzle environment.

        Args:
            min_pieces: Minimum number of puzzle pieces
            max_pieces: Maximum number of puzzle pieces
            pictures_dir: Directory containing source images
        """
        super().__init__()
        self.min_pieces = min_pieces
        self.max_pieces = max_pieces

        # Set pictures directory
        if pictures_dir is None:
            pictures_dir = os.path.join(
                os.path.dirname(__file__),
                "jigsaw_pictures"
            )
        self.pictures_dir = pictures_dir

        self.answer = None

    def _generate_puzzle(self, seed: int) -> Tuple[bytes, List[Tuple[int, str]]]:
        """
        Generate a jigsaw puzzle by slicing an image.

        Returns:
            Tuple of (image_bytes, correct_answer)
        """
        random.seed(seed)

        LETTER_HEIGHT = 60  # Height for letter labels
        SPACING = 10  # Spacing between option images
        LINE_WIDTH = 2  # Width of dividing lines

        # Select random input image
        png_files = [
            f for f in os.listdir(self.pictures_dir)
            if f.lower().endswith('.png')
        ]
        if not png_files:
            raise FileNotFoundError(f"No PNG files found in {self.pictures_dir}")

        img_name = random.choice(png_files)
        img_path = os.path.join(self.pictures_dir, img_name)

        # Open and preprocess image
        img = Image.open(img_path)
        width, height = img.size

        # Determine grid dimensions
        short = random.randint(2, 4)  # Short edge: 2-4 pieces
        # Long edge: at most 2*short, and total pieces <= 26
        long = random.randint(short, min(2 * short, 26 // short))

        if width >= height:
            w = long
            h = short
        else:
            w = short
            h = long

        # Ensure total pieces in range
        total_pieces = w * h
        if total_pieces < self.min_pieces or total_pieces > self.max_pieces:
            # Adjust to be within range
            if total_pieces < self.min_pieces:
                w = 3
                h = 2
            elif total_pieces > self.max_pieces:
                w = 3
                h = 3

        block_width = width // w
        block_height = height // h
        img = img.resize((block_width * w, block_height * h))  # Ensure divisible

        # Cut image into pieces
        pieces = []
        for y in range(h):
            for x in range(w):
                left = x * block_width
                upper = y * block_height
                pieces.append(
                    img.crop((left, upper, left + block_width, upper + block_height))
                )

        # Generate random mapping
        numbers = list(range(1, h * w + 1))
        letters = list(string.ascii_uppercase[:h * w])
        random.shuffle(letters)
        mapping = {n: l for n, l in zip(numbers, letters)}
        correct_answer = list(zip(numbers, letters))

        # Create blank board with numbers
        board = Image.new('RGB', img.size, 'white')
        draw = ImageDraw.Draw(board)

        # Draw dividing lines
        for x in range(1, w):
            draw.line(
                [(x * block_width, 0), (x * block_width, img.size[1])],
                fill='black',
                width=LINE_WIDTH
            )
        for y in range(1, h):
            draw.line(
                [(0, y * block_height), (img.size[0], y * block_height)],
                fill='black',
                width=LINE_WIDTH
            )

        # Add number labels
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 40)
            except:
                font = ImageFont.load_default()

        for idx in range(w * h):
            y_pos = idx // w
            x_pos = idx % w
            text = str(idx + 1)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                (x_pos * block_width + (block_width - text_width) / 2,
                 y_pos * block_height + (block_height - text_height) / 2),
                text,
                fill='black',
                font=font
            )

        # Create labeled pieces
        labeled_pieces = []
        for idx, piece in enumerate(pieces):
            new_img = Image.new('RGB', (block_width, block_height + LETTER_HEIGHT), 'white')
            new_img.paste(piece, (0, 0))

            draw = ImageDraw.Draw(new_img)
            letter = mapping[idx + 1]
            bbox = draw.textbbox((0, 0), letter, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text(
                ((block_width - text_width) // 2, block_height + 5),
                letter,
                fill='black',
                font=font
            )
            labeled_pieces.append(new_img)

        # Shuffle pieces
        random.shuffle(labeled_pieces)

        # Create options image
        option_width = w * block_width + (w - 1) * SPACING
        option_height = h * (block_height + LETTER_HEIGHT) + (h - 1) * SPACING
        fragments = Image.new('RGB', (option_width, option_height), 'white')

        for i, piece in enumerate(labeled_pieces):
            row = i // w
            col = i % w
            x = col * (block_width + SPACING)
            y = row * (block_height + LETTER_HEIGHT + SPACING)
            fragments.paste(piece, (x, y))

        # Combine board and fragments
        combined_width = max(board.width, fragments.width)
        combined_height = board.height + fragments.height + 20
        combined_image = Image.new('RGB', (combined_width, combined_height), 'white')
        combined_image.paste(board, (0, 0))
        combined_image.paste(fragments, (0, board.height + 20))

        # Convert to bytes
        img_bytes = io.BytesIO()
        combined_image.save(img_bytes, format='PNG')

        return img_bytes.getvalue(), correct_answer

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board (as an image) and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: [(1,'A'),(2,'D'),...]'\n\n"
            "Alternatively, you can use \\boxed{[(1,'A'),(2,'D'),...]} format.\n\n"
            "As shown in the picture, you need to solve the puzzle game. The top of the image "
            "is the puzzle board (represented by numbers) where pieces need to be filled in, "
            "and the bottom of the image shows the puzzle pieces (represented by letters). "
            "You need to place the puzzle pieces from the bottom into the blank spaces at the top "
            "to restore the original picture. The output should be in the format: "
            "[(position_number, piece_letter), ...]\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a new jigsaw puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        image_bytes, self.answer = self._generate_puzzle(seed if seed else random.randint(0, 1000000))

        # Create multimodal observation
        observation = {
            "text": self._get_instructions(),
            "image": base64.b64encode(image_bytes).decode('utf-8'),
            "image_format": "png"
        }

        return observation, {"suffix": f"Match {len(self.answer)} puzzle pieces to positions."}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's solution.

        Args:
            action: Agent's response containing the list of (number, letter) pairs

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
                "Please use 'Answer: [(1,'A'),(2,'B'),...]' or "
                "\\boxed{[(1,'A'),(2,'B'),...]} format."
            )
            return obs, 0.0, True, False, {}

        # Parse the answer
        try:
            user_answer = ast.literal_eval(parsed_answer)
        except Exception as e:
            obs = f"Failed to parse answer. Error: {e}"
            return obs, 0.0, True, False, {}

        if not isinstance(user_answer, list):
            obs = "Answer must be a list of (number, letter) tuples."
            return obs, 0.0, True, False, {}

        # Convert to sets for comparison
        correct_set = set(self.answer)
        user_set = set(user_answer)

        # Calculate score based on intersection
        intersection = correct_set & user_set
        score = len(intersection) / len(correct_set) if len(correct_set) > 0 else 0

        if score == 1.0:
            obs = f"Perfect! All {len(self.answer)} pieces matched correctly!"
        else:
            correct_count = len(intersection)
            obs = (
                f"Score: {correct_count}/{len(correct_set)} pieces matched correctly "
                f"({score:.2%}).\n"
            )
            # Show which are wrong
            wrong = correct_set - user_set
            if wrong:
                obs += f"Incorrect mappings for positions: {sorted([w[0] for w in wrong])}"

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
            return "\\boxed{[(1,'A')]}"
