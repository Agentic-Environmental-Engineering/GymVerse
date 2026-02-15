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

"""FillGame environment - Visual puzzle matching game."""

import base64
import io
import math
import os
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from PIL import Image, ImageDraw, ImageFont

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class FillGameEnv(Env):
    """
    FillGame environment.

    Players are shown an image with a missing rectangular piece and must
    choose the correct piece from multiple options to fill the gap.

    This is a single-turn multimodal environment with sparse rewards.
    """

    def __init__(
        self,
        min_options: int = 8,
        max_options: int = 12,
        target_size: Tuple[int, int] = (200, 200),
        option_size: Tuple[int, int] = (70, 70),
        **_,
    ):
        """
        Initialize FillGame environment.

        Args:
            min_options: Minimum number of answer options
            max_options: Maximum number of answer options
            target_size: Size of the main puzzle image
            option_size: Size of each option piece
        """
        super().__init__()
        self.min_options = min_options
        self.max_options = max_options
        self.target_size = target_size
        self.option_size = option_size
        self.correct_answer = None
        self.image_data = None

        # Get pictures directory path
        env_dir = Path(__file__).parent
        self.pictures_dir = env_dir / "fill_game_pictures"

    def _encode_image_to_base64(self, img: Image.Image) -> str:
        """
        Encode PIL Image to base64 string.

        Args:
            img: PIL Image

        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _generate_puzzle(self, seed: int) -> Dict[str, Any]:
        """
        Generate a fill game puzzle.

        Args:
            seed: Random seed

        Returns:
            Dictionary with puzzle data
        """
        random.seed(seed)

        # Select a random category and image
        categories = [d for d in self.pictures_dir.iterdir() if d.is_dir()]
        if not categories:
            raise FileNotFoundError(f"No picture categories found in {self.pictures_dir}")

        category = random.choice(categories)
        images = list(category.glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No PNG images found in {category}")

        # Load and resize main image
        main_img_path = random.choice(images)
        original_img = Image.open(main_img_path).convert("RGBA")
        original_img = original_img.resize(self.target_size, Image.LANCZOS)
        width, height = self.target_size

        # Define the cutout region (right side, middle)
        shape_w, shape_h = self.option_size
        offset_x = width - shape_w - 20
        offset_y = (height - shape_h) // 2

        # Create mask for the correct piece
        shape_mask = Image.new("L", self.target_size, 0)
        draw_mask = ImageDraw.Draw(shape_mask)
        draw_mask.rectangle(
            [offset_x, offset_y, offset_x + shape_w, offset_y + shape_h],
            fill=255
        )

        # Extract the correct piece
        correct_piece = Image.new("RGBA", self.target_size, (0, 0, 0, 0))
        correct_piece.paste(original_img, mask=shape_mask)
        correct_piece_cropped = correct_piece.crop(
            (offset_x, offset_y, offset_x + shape_w, offset_y + shape_h)
        )
        correct_piece_cropped = correct_piece_cropped.resize(self.option_size, Image.LANCZOS)

        # Create puzzle image with cutout
        puzzle_img = original_img.copy()
        inv_mask = Image.new("L", self.target_size, 255)
        inv_draw = ImageDraw.Draw(inv_mask)
        inv_draw.rectangle(
            [offset_x, offset_y, offset_x + shape_w, offset_y + shape_h],
            fill=0
        )

        puzzle_array = puzzle_img.load()
        inv_array = inv_mask.load()
        for y in range(height):
            for x in range(width):
                if inv_array[x, y] == 0:
                    puzzle_array[x, y] = (255, 255, 255, 0)  # Make transparent

        # Generate distractor options
        num_options = random.randint(self.min_options, self.max_options)
        distractor_pieces = []

        while len(distractor_pieces) < (num_options - 1):
            distract_img_path = random.choice(images)
            d_img = Image.open(distract_img_path).convert("RGBA")
            d_img = d_img.resize(self.target_size, Image.LANCZOS)

            # Random crop
            rand_x = random.randint(0, width - shape_w)
            rand_y = random.randint(0, height - shape_h)
            piece = d_img.crop((rand_x, rand_y, rand_x + shape_w, rand_y + shape_h))
            piece = piece.resize(self.option_size, Image.LANCZOS)

            # Random transformations
            if random.random() < 0.3:
                piece = piece.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.3:
                piece = piece.transpose(Image.ROTATE_90)

            distractor_pieces.append(piece)

        # Mix all options
        all_options = distractor_pieces + [correct_piece_cropped]
        random.shuffle(all_options)
        correct_idx = all_options.index(correct_piece_cropped)
        answer_options = [chr(ord('A') + i) for i in range(num_options)]
        correct_answer = answer_options[correct_idx]

        # Create final composite image
        title_text = "1. Please choose the shape that best fits the missing piece"
        try:
            font_title = ImageFont.truetype("arialbd.ttf", 40)
        except (IOError, OSError):
            try:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
            except (IOError, OSError):
                font_title = ImageFont.load_default()

        # Calculate title dimensions
        bbox = font_title.getbbox(title_text)
        title_w = bbox[2] - bbox[0]
        title_h = bbox[3] - bbox[1]

        # Layout calculations
        margin = 20
        columns = 4
        rows = math.ceil(num_options / columns)
        options_total_width = (self.option_size[0] + margin) * columns - margin
        options_total_height = (self.option_size[1] + margin) * rows - margin

        final_width = max(width, options_total_width, title_w) + margin * 2
        final_height = (margin + title_h + margin + height + margin +
                       options_total_height + margin)

        # Create final image
        final_image = Image.new("RGBA", (final_width, final_height), (255, 255, 255, 255))
        draw_final = ImageDraw.Draw(final_image)

        # Draw title
        title_x = (final_width - title_w) // 2
        title_y = margin
        draw_final.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font_title)

        # Paste puzzle image
        puzzle_x = (final_width - width) // 2
        puzzle_y = title_y + title_h + margin
        final_image.paste(puzzle_img, (puzzle_x, puzzle_y), puzzle_img)

        # Draw option pieces
        try:
            font_option = ImageFont.truetype("arialbd.ttf", 24)
        except (IOError, OSError):
            try:
                font_option = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except (IOError, OSError):
                font_option = ImageFont.load_default()

        options_start_x = (final_width - options_total_width) // 2
        options_start_y = puzzle_y + height + margin
        current_x = options_start_x
        current_y = options_start_y

        for i, piece_img in enumerate(all_options):
            final_image.paste(piece_img, (current_x, current_y), piece_img)
            draw_final.text(
                (current_x + 5, current_y + 5),
                answer_options[i],
                fill=(255, 0, 0),
                font=font_option
            )
            current_x += self.option_size[0] + margin
            if (i + 1) % columns == 0:
                current_x = options_start_x
                current_y += self.option_size[1] + margin

        return {
            "answer": correct_answer,
            "image": final_image,
            "num_options": num_options
        }

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board which is a picture and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., \"Answer: F\"\n\n"
            "Alternatively, you can use \\boxed{F} format.\n\n"
            "I will provide an image in which a rectangular piece is missing from a larger pattern. "
            "Below the pattern, there are several options for pieces that could fit into the blank "
            "space. Please choose the most suitable piece to fill the blank and output the letter "
            "corresponding to that option, e.g., 'Answer: G'.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a new FillGame puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle = self._generate_puzzle(seed if seed else random.randint(0, 1000000))
        self.correct_answer = puzzle["answer"]
        self.image_data = self._encode_image_to_base64(puzzle["image"])

        # Create multimodal observation
        observation = {
            "text": self._get_instructions(),
            "image": self.image_data,
            "image_format": "png"
        }

        return observation, {
            "suffix": f"Choose correct piece from {puzzle['num_options']} options."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's answer.

        Args:
            action: Agent's response containing the chosen letter

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer
        parsed_answer = extract_last_boxed_answer(action)

        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*([A-Z])', action, re.IGNORECASE)
            if match:
                parsed_answer = match.group(1).strip().upper()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: LETTER' or \\boxed{LETTER} format."
            )
            return obs, 0.0, True, False, {}

        # Normalize answer
        user_answer = parsed_answer.strip().upper()

        # Check answer
        if user_answer == self.correct_answer:
            obs = f"Correct! The answer is {self.correct_answer}."
            return obs, 1.0, True, False, {}
        else:
            obs = f"Incorrect. You chose {user_answer}, but the correct answer is {self.correct_answer}."
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            The correct answer
        """
        if self.correct_answer is not None:
            return f"\\boxed{{{self.correct_answer}}}"
        else:
            return "\\boxed{A}"
