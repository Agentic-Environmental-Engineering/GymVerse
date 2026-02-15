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

"""DateCount environment - Calculate dates with offsets."""

import random
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class DateCountEnv(Env):
    """
    Date calculation environment.

    The agent must calculate a target date given an offset from a base date.
    This is a single-turn environment where the agent provides one answer and the episode ends.

    Example:
        "The date 100 days ago is 2024/1/15, what is the date today?"
        Answer: \\boxed{2024/4/25}
    """

    def __init__(
        self,
        min_year: int = 500,
        max_year: int = 1525,
        max_offset: int = 100000,
        **_,
    ):
        """
        Initialize DateCount environment.

        Args:
            min_year: Minimum year for date generation
            max_year: Maximum year for date generation
            max_offset: Maximum absolute value for day offset
        """
        super().__init__()
        self.min_year = min_year
        self.max_year = max_year
        self.max_offset = max_offset
        self.current_problem = None
        self.correct_answer = None

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. You will be given a date calculation problem.\n"
            "Your task is:\n"
            "- First, calculate the correct date based on the given information.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), "
            "where YOUR_ANSWER is your final answer in YYYY/M/D format, e.g., 'Answer: 1992/5/18'\n\n"
            "Alternatively, you can use \\boxed{YYYY/M/D} format, e.g., \\boxed{1992/5/18}\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new date calculation problem.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate random date and offset
        year = random.randint(self.min_year, self.max_year)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # Use 28 to avoid month-specific logic
        offset = random.randint(-self.max_offset, self.max_offset)

        base_date = datetime(year, month, day)
        target_date = base_date + timedelta(days=offset)

        # Store correct answer
        self.correct_answer = target_date.strftime("%Y/%m/%d")

        # Create problem description
        direction = "ago" if offset > 0 else "later"
        abs_offset = abs(offset)
        self.current_problem = (
            f"The date {abs_offset} days {direction} is {base_date.year}/{base_date.month}/{base_date.day}, "
            f"what is the date today? (The output should be in the format: 'Answer: year/month/date')"
        )

        observation = f"{self._get_instructions()}\n{self.current_problem}"
        return observation, {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's answer.

        Args:
            action: Agent's response containing the date answer

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
                "Please use 'Answer: YYYY/M/D' or \\boxed{YYYY/M/D} format."
            )
            return obs, 0.0, True, False, {}

        # Normalize answer format (remove spaces, convert to consistent format)
        parsed_answer = parsed_answer.strip().replace(" ", "")
        correct_normalized = self.correct_answer.strip().replace(" ", "")

        # Check if answer is correct
        is_correct = parsed_answer == correct_normalized
        reward = 1.0 if is_correct else 0.0

        if is_correct:
            obs = f"Correct! The answer is {self.correct_answer}."
        else:
            obs = f"Incorrect. Your answer was '{parsed_answer}', but the correct answer is {self.correct_answer}."

        return obs, reward, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action (for testing purposes).

        Returns:
            A random date in the correct format
        """
        if self.correct_answer is not None:
            # For testing, return the correct answer
            return f"\\boxed{{{self.correct_answer}}}"
        else:
            # Return a random date
            year = random.randint(self.min_year, self.max_year)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            return f"\\boxed{{{year}/{month}/{day}}}"
