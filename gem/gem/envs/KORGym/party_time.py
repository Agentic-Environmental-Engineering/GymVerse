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

"""PartyTime environment - Count party participants with attributes."""

import random
import string
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class Participator:
    """Represents a party participant with various attributes."""

    def __init__(self):
        """Initialize participator with random attributes."""
        self.name = ''.join(random.choices(string.ascii_letters, k=random.randint(3, 8)))
        self.gender = [random.choice(['male', 'female'])]
        self.shirt_color = [random.choice(['red', 'orange', 'blue', 'green', 'yellow', 'purple', 'cyan'])]
        self.pants_color = [random.choice(['red', 'orange', 'blue', 'green', 'yellow', 'purple', 'cyan'])]
        self.hair_color = [random.choice(['red', 'orange', 'blue', 'green', 'yellow', 'purple', 'cyan'])]
        self.has_items = random.sample([
            "balloon", "snacks", "camera", "hat", "sunglasses", "lighter",
            "bottle", "phone", "book", "flowers",
            "candy", "guitar", "umbrella", "scarf",
            "perfume", "candle", "wallet", "pencil"
        ], random.randint(1, 6))


class PartyTimeEnv(Env):
    """
    PartyTime counting environment.

    Players are given descriptions of party participants with various attributes
    and items. They must count the total number of participants or specific items
    matching given criteria.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_participants: int = 70,
        max_participants: int = 200,
        **_,
    ):
        """
        Initialize PartyTime environment.

        Args:
            min_participants: Minimum number of participants
            max_participants: Maximum number of participants
        """
        super().__init__()
        self.min_participants = min_participants
        self.max_participants = max_participants
        self.question = None
        self.answer = None

    def _generate_puzzle(self, seed: int) -> Dict[str, Any]:
        """
        Generate a party counting puzzle.

        Args:
            seed: Random seed

        Returns:
            Dictionary with question and answer
        """
        random.seed(seed)

        # Generate participants
        nums = random.randint(self.min_participants, self.max_participants)
        participators = [Participator() for _ in range(nums)]

        # Choose query type
        query_objects = ['total number', 'items number']
        query_object = random.choice(query_objects)

        # Collect all attributes
        attributes = list(participators[0].__dict__.keys())
        attributes_features = {attribute: [] for attribute in attributes}

        # Select attributes for the question (exclude name)
        attributes.remove('name')
        selected_attributes = random.sample(
            attributes,
            random.randint(2, len(attributes))
        )

        # Store all participant attributes
        for participator in participators:
            attributes_features['name'].append(participator.name)
            attributes_features['gender'].extend(participator.gender)
            attributes_features['shirt_color'].extend(participator.shirt_color)
            attributes_features['pants_color'].extend(participator.pants_color)
            attributes_features['hair_color'].extend(participator.hair_color)
            attributes_features['has_items'].extend(participator.has_items)

        # Select attribute values for the question
        question_attribute = {}
        for attr in selected_attributes:
            unique_values = list(set(attributes_features[attr]))
            sample_size = random.randint(1, len(unique_values))
            question_attribute[attr] = random.sample(unique_values, sample_size)

        # Calculate answer
        answer = 0
        # For items number query, select a specific item to count
        sub_query_object = random.choice(list(set(attributes_features['has_items'])))

        for participator in participators:
            qualified = True
            for attr, values in question_attribute.items():
                # For list attributes, check if at least one value matches
                if not any(item in values for item in getattr(participator, attr)):
                    qualified = False
                    break

            if qualified:
                if query_object == 'total number':
                    answer += 1
                elif query_object == 'items number':
                    if sub_query_object in getattr(participator, 'has_items'):
                        answer += 1

        # Build question text
        question = "We invite some students to our party today. Their appearance and their belongings are as follows:\n"
        for i, participator in enumerate(participators):
            question += (
                f"Student({i + 1}): Name = {participator.name}, Gender = {participator.gender[0]}, "
                f"Shirt color = {participator.shirt_color[0]}, Pants color = {participator.pants_color[0]}, "
                f"Hair color = {participator.hair_color[0]}, Has items = {'/'.join(participator.has_items)};\n"
            )

        if query_object == 'total number':
            question += "Please help me calculate the total number of students that meet the following criteria, "
        else:
            question += (f"Please help me calculate the total number of {sub_query_object} "
                        f"of these students that meet the following criteria, ")

        question += (
            "and return the number in the following format: 'Answer: $YOUR_ANSWER' (without quotes), "
            "where YOUR_ANSWER is your final answer to the question, e.g., 'Answer: 16'.\nAll students that: "
        )

        # List all criteria
        i = 1
        for attr, values in question_attribute.items():
            question += f"{i}. {attr} belong to {'/'.join(str(v) for v in values)}; "
            i += 1

        question = question.rstrip('; ') + "."

        return {
            "question": question,
            "answer": answer
        }

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: 12'\n\n"
            "Alternatively, you can use \\boxed{12} format.\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new party counting puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle = self._generate_puzzle(seed if seed else random.randint(0, 1000000))
        self.question = puzzle["question"]
        self.answer = puzzle["answer"]

        # Build observation
        observation = f"{self._get_instructions()}{self.question}"

        return observation, {
            "suffix": f"Count party participants matching criteria."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's answer.

        Args:
            action: Agent's response containing the count

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer
        parsed_answer = extract_last_boxed_answer(action)

        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(\d+)', action, re.IGNORECASE)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: NUMBER' or \\boxed{NUMBER} format."
            )
            return obs, 0.0, True, False, {}

        # Parse number
        try:
            user_answer = int(parsed_answer)
        except ValueError:
            obs = f"Failed to parse answer as integer: '{parsed_answer}'"
            return obs, 0.0, True, False, {}

        # Check answer
        if user_answer == self.answer:
            obs = f"Correct! The answer is {self.answer}."
            return obs, 1.0, True, False, {}
        else:
            obs = f"Incorrect. You answered {user_answer}, but the correct answer is {self.answer}."
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            The correct answer
        """
        if self.answer is not None:
            return f"\\boxed{{{self.answer}}}"
        else:
            return "\\boxed{0}"
