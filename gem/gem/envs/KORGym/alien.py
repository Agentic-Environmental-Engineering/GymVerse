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

"""Alien environment - Counting alien creatures with attributes."""

import random
import string
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class Alien:
    """Represents an alien creature with various attributes."""

    def __init__(self):
        """Initialize alien with random attributes."""
        self.name = ''.join(random.choices(string.ascii_letters, k=random.randint(3, 8)))
        self.diet = random.choice([
            'herbivore', 'carnivore', 'Omnivore', 'Scavenger', 'Parasite', 'Insectivore'
        ])
        self.legs = random.randint(0, 10)
        self.horns = random.randint(0, 10)
        self.reproduction = random.choice([
            'mammal', 'oviparous', 'Viviparous', 'Asexual Reproduction', 'Spore Reproduction'
        ])
        self.color = random.choice([
            'red', 'orange', 'blue', 'green', 'yellow', 'purple', 'cyan'
        ])


class AlienEnv(Env):
    """
    Alien counting environment.

    Players are given descriptions of alien creatures with various attributes
    and quantities. They must count the total number, total horns, or total legs
    of creatures matching specific criteria.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_total: int = 70,
        max_total: int = 200,
        **_,
    ):
        """
        Initialize Alien environment.

        Args:
            min_total: Minimum total number of aliens
            max_total: Maximum total number of aliens
        """
        super().__init__()
        self.min_total = min_total
        self.max_total = max_total
        self.question = None
        self.answer = None

    def _generate_puzzle(self, seed: int) -> Dict[str, Any]:
        """
        Generate an alien counting puzzle.

        Args:
            seed: Random seed

        Returns:
            Dictionary with question and answer
        """
        random.seed(seed)

        # Generate total number of aliens
        nums = random.randint(self.min_total, self.max_total)

        # Generate alien species
        num_aliens = random.randint(1, nums)
        aliens = [Alien() for _ in range(num_aliens)]
        parts = [0] * num_aliens

        # Randomly distribute quantities to species
        for i in range(nums):
            index = random.randint(0, num_aliens - 1)
            parts[index] += 1

        # Collect all attributes
        query_objects = ['total number', 'horns', 'legs']
        attributes = list(aliens[0].__dict__.keys())
        attributes_features = {attribute: [] for attribute in attributes}

        attributes.remove('name')
        selected_attributes = random.sample(
            attributes,
            random.randint(2, len(attributes))
        )
        query_objects = [item for item in query_objects if item not in selected_attributes]
        query_object = random.choice(query_objects)

        # Store all alien attributes
        for alien in aliens:
            attributes_features['name'].append(alien.name)
            attributes_features['diet'].append(alien.diet)
            attributes_features['horns'].append(alien.horns)
            attributes_features['legs'].append(alien.legs)
            attributes_features['reproduction'].append(alien.reproduction)
            attributes_features['color'].append(alien.color)

        # Select attributes for the question
        question_attribute = {}
        for selected_attribute in selected_attributes:
            available = attributes_features[selected_attribute]
            sample_count = random.randint(1, len(available))
            question_attribute[selected_attribute] = list(
                set(random.sample(available, sample_count))
            )

        # Calculate answer
        answer = 0
        for i in range(len(aliens)):
            qualified = True
            for attribute in question_attribute.keys():
                if getattr(aliens[i], attribute) not in question_attribute[attribute]:
                    qualified = False
                    break

            if qualified:
                if query_object == 'total number':
                    answer += parts[i]
                elif query_object == 'horns':
                    answer += parts[i] * getattr(aliens[i], 'horns')
                elif query_object == 'legs':
                    answer += parts[i] * getattr(aliens[i], 'legs')

        # Build question text
        question = 'There are several alien beings on a distant planet. Their categories and corresponding features are as follows:\n'
        for i, alien in enumerate(aliens):
            question += (
                f"Alien({i + 1}): Name = {alien.name}, Diet = {alien.diet}, "
                f"Legs = {alien.legs}, Horns = {alien.horns}, "
                f"Reproduction = {alien.reproduction}, Color = {alien.color};\n"
            )

        question += 'Now, there are '
        for i in range(len(parts)):
            if parts[i] > 0:
                question += f"{parts[i]} {attributes_features['name'][i]}, "
        question = question.rstrip(", ")

        if query_object == 'total number':
            question += (' in this area. Please help me calculate the total number of '
                        'alien animals that meet the following criteria, ')
        else:
            question += (f" in this area, please help me calculate the total number of "
                        f"{query_object} of these alien animals that meet the following "
                        f"criteria, ")

        question += ("and return the number in the following format: "
                    "'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your "
                    "final answer to the question, e.g., 'Answer: 16'.\nAll alien animals that: ")

        i = 1
        for attribute in question_attribute.keys():
            values = '/'.join([
                str(item) if isinstance(item, int) else item
                for item in question_attribute[attribute]
            ])
            question += f"{i}. {attribute} are {values}; "
            i += 1

        question = question.rstrip('; ')
        question += '.'

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
        Generate a new alien counting puzzle.

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
            "suffix": f"Count aliens matching criteria."
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
