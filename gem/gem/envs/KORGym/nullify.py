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

"""Nullify environment - Number elimination puzzle game."""

import random
import math
from copy import deepcopy
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class UnitType(Enum):
    """Type of game unit."""
    NUMBER = 1  # Positive/negative number
    ORDINARY_OPERATOR = 2  # Multiply/divide
    OPERATOR = 3  # Other operators


class Unit:
    """Game unit representing a number or operator."""

    def __init__(self, unit_type: UnitType, symbol: str, value: Optional[float] = None):
        self.type = unit_type
        self.symbol = symbol
        self.value = value

    def __repr__(self):
        if self.type == UnitType.NUMBER:
            return f"{self.value}"
        elif self.type == UnitType.ORDINARY_OPERATOR:
            return f"{self.symbol}{self.value}"
        else:
            return f"{self.symbol}"


class NullifyEnv(Env):
    """
    Nullify environment.

    Players must combine units to eliminate all of them, with the goal of
    reaching a final result of 0. Units can be numbers (+/-), multiplication/
    division operators, or special operators (sqrt, square, reciprocal, floor, ceil).

    This is a multi-turn environment with sparse terminal reward.
    """

    def __init__(
        self,
        min_steps: int = 3,
        max_steps: int = 10,
        max_turns: int = 50,
        **_,
    ):
        """
        Initialize Nullify environment.

        Args:
            min_steps: Minimum generation steps
            max_steps: Maximum generation steps
            max_turns: Maximum number of turns
        """
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.max_turns = max_turns
        self.units = None
        self.turn = None

    def _generate_number_unit(self) -> Unit:
        """Generate a random number unit."""
        while True:
            value = round(random.randint(-10, 10), 4)
            if value != 0:
                sign = "+" if value >= 0 else "-"
                return Unit(UnitType.NUMBER, sign, value)

    def _generate_ordinary_operator_unit(self) -> Unit:
        """Generate a multiply/divide operator unit."""
        while True:
            value = round(random.randint(-10, 10), 4)
            if value != 0:
                sign = random.choice(["*", "/"])
                return Unit(UnitType.ORDINARY_OPERATOR, sign, value)

    def _generate_operator_unit(self) -> Unit:
        """Generate a special operator unit."""
        operator = random.choice(['floor', 'ceil', 'sqrt', 'square', 'reciprocal'])
        return Unit(UnitType.OPERATOR, operator)

    def _generate_puzzle(self, seed: int) -> List[Unit]:
        """Generate puzzle using reverse operations."""
        random.seed(seed)
        current_value = 0
        generation_steps = random.randint(self.min_steps, self.max_steps)
        units = []

        while generation_steps > 0:
            operation_type = random.choice(['add_or_minus', 'multiply', 'apply_operator'])

            if operation_type == "add_or_minus":
                number_unit = self._generate_number_unit()
                current_value -= number_unit.value
                units.append(number_unit)
                generation_steps -= 1

            elif operation_type == "multiply":
                if current_value == 0:
                    continue

                operator_unit = self._generate_ordinary_operator_unit()
                if operator_unit.symbol == "*":
                    current_value /= operator_unit.value
                elif operator_unit.symbol == "/":
                    current_value *= operator_unit.value

                units.append(operator_unit)
                generation_steps -= 1

            elif operation_type == "apply_operator":
                if current_value == 0:
                    continue

                operator_unit = self._generate_operator_unit()

                if operator_unit.symbol == "sqrt":
                    if current_value < 0:
                        continue
                    current_value = current_value ** 2
                elif operator_unit.symbol == "square":
                    current_value = math.sqrt(abs(current_value))
                elif operator_unit.symbol == "reciprocal":
                    current_value = 1 / current_value
                elif operator_unit.symbol in ["floor", "ceil"]:
                    int_part = math.floor(current_value)
                    frac_part = current_value - int_part
                    if frac_part != 0:
                        frac_symbol = "+" if frac_part > 0 else "-"
                        units.append(Unit(UnitType.NUMBER, frac_symbol, round(frac_part, 4)))
                    if operator_unit.symbol == "floor":
                        current_value = int_part + random.uniform(0, 1)
                    elif operator_unit.symbol == "ceil":
                        current_value = int_part - random.uniform(0, 1)

                units.append(operator_unit)
                generation_steps -= 1

        # Add final number to complete puzzle
        current_symbol = "+" if current_value > 0 else "-"
        units.append(Unit(UnitType.NUMBER, current_symbol, round(current_value, 4)))

        # Shuffle units
        random.shuffle(units)
        return units

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: 2 4'.\n\n"
            "Alternatively, you can use \\boxed{2 4} format.\n\n"
            "In the game, there are several independent units. Among them, units that consist of "
            "a sign (+ or -) followed by a number can combine with other units. For example, "
            "combining -10 with ×8 results in -80, and combining -7 with ÷3 yields -2.33... "
            "If the combination of two units results in 0 (i.e. when a positive and a negative "
            "number with equal absolute values are operated on), no new unit is produced.\n\n"
            "Game Objective: Eliminate all units, meaning that the final combined result is 0.\n\n"
            "Current Unit Types:\n"
            "+number: A positive number, which also represents the addition operation.\n"
            "-number: A negative number, which also represents the subtraction operation.\n"
            "*number: Multiplication operation.\n"
            "/number: Division operation.\n"
            "sqrt: Square root operation.\n"
            "square: Square operation.\n"
            "reciprocal: Reciprocal operation.\n"
            "floor: Floor (round down) operation.\n"
            "ceil: Ceiling (round up) operation.\n\n"
            "Please output the operation for the current turn by directly providing the two "
            "corresponding unit indices, separated by a space (e.g., 'Answer: 2 4').\n\n"
        )

    def _format_board(self) -> str:
        """Format the current game board."""
        lines = []
        for idx, unit in enumerate(self.units):
            lines.append(f"{idx} {unit}")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new Nullify puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate puzzle
        puzzle_seed = seed if seed is not None else random.randint(0, 1000000)
        self.units = self._generate_puzzle(puzzle_seed)
        self.turn = 0

        # Build observation
        board_str = self._format_board()
        observation = f"{self._get_instructions()}Board:\n{board_str}\n"

        return observation, {
            "suffix": f"Units: {len(self.units)}, Max turns: {self.max_turns}."
        }

    def _verify_action(self, i: int, j: int) -> int:
        """
        Verify and execute action.

        Returns:
            0: Valid operation, game continues
            1: Victory (all eliminated)
            -1: Invalid operation
            -2: Only one number left but not 0 (failure)
        """
        if i < 0 or j < 0 or i >= len(self.units) or j >= len(self.units) or i == j:
            return -1

        temp_units = deepcopy(self.units)
        unit1 = temp_units[i]
        unit2 = temp_units[j]

        # Two numbers combining
        if unit1.type == UnitType.NUMBER and unit2.type == UnitType.NUMBER:
            sum_val = unit1.value + unit2.value

            # Remove both units
            indices = sorted([i, j], reverse=True)
            for idx in indices:
                del temp_units[idx]

            # If sum is 0, just remove both
            if abs(sum_val) < 1e-9:
                self.units = temp_units
                return 1 if len(temp_units) == 0 else 0
            else:
                # Create new unit
                new_sign = '+' if sum_val >= 0 else '-'
                new_unit = Unit(UnitType.NUMBER, new_sign, sum_val)
                temp_units.append(new_unit)

                # Check if complete
                if len(temp_units) == 1:
                    if abs(new_unit.value) < 1e-3:
                        self.units = temp_units
                        return 1
                    else:
                        return -2
                else:
                    self.units = temp_units
                    return 0

        # Number and operator combining
        num_unit, op_unit = None, None
        if unit1.type == UnitType.NUMBER and unit2.type in (UnitType.ORDINARY_OPERATOR, UnitType.OPERATOR):
            num_unit, op_unit = unit1, unit2
        elif unit2.type == UnitType.NUMBER and unit1.type in (UnitType.ORDINARY_OPERATOR, UnitType.OPERATOR):
            num_unit, op_unit = unit2, unit1
        else:
            return -1  # Invalid combination

        try:
            if op_unit.type == UnitType.ORDINARY_OPERATOR:
                if op_unit.symbol == '*':
                    new_val = num_unit.value * op_unit.value
                elif op_unit.symbol == '/':
                    if abs(op_unit.value) < 1e-9:
                        return -1
                    new_val = num_unit.value / op_unit.value
                else:
                    return -1
            else:
                # Special operators
                if op_unit.symbol == 'floor':
                    new_val = math.floor(num_unit.value)
                elif op_unit.symbol == 'ceil':
                    new_val = math.ceil(num_unit.value)
                elif op_unit.symbol == 'sqrt':
                    if num_unit.value < 0:
                        return -1
                    new_val = math.sqrt(num_unit.value)
                elif op_unit.symbol == 'square':
                    new_val = num_unit.value ** 2
                elif op_unit.symbol == 'reciprocal':
                    if abs(num_unit.value) < 1e-9:
                        return -1
                    new_val = 1 / num_unit.value
                else:
                    return -1

            new_sign = '+' if new_val >= 0 else '-'
            new_unit = Unit(UnitType.NUMBER, new_sign, new_val)

        except (ValueError, ZeroDivisionError):
            return -1

        # Remove both units and add new one
        indices = sorted([i, j], reverse=True)
        for idx in indices:
            del temp_units[idx]
        temp_units.append(new_unit)

        # Check if complete
        if len(temp_units) == 1:
            if abs(new_unit.value) < 1e-4:
                self.units = temp_units
                return 1
            else:
                return -2
        else:
            self.units = temp_units
            return 0

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process player's action.

        Args:
            action: Two unit indices separated by space

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        parsed_action = extract_last_boxed_answer(action)

        if parsed_action is None:
            import re
            match = re.search(r'Answer:\s*(.+)', action, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_action = match.group(1).strip()

        if parsed_action is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: 2 4' or \\boxed{2 4} format."
            )
            return obs, 0.0, True, False, {}

        # Parse indices
        try:
            i, j = map(int, str(parsed_action).split())
        except Exception as e:
            obs = f"Failed to parse action: {parsed_action}. Error: {e}"
            return obs, 0.0, True, False, {}

        # Verify action
        result = self._verify_action(i, j)
        self.turn += 1

        if result == 1:
            obs = (
                f"Congratulations! You eliminated all units!\n"
                f"Turns used: {self.turn}/{self.max_turns}"
            )
            return obs, 1.0, True, False, {}
        elif result == -1:
            obs = "Invalid operation. Game over."
            return obs, 0.0, True, False, {}
        elif result == -2:
            obs = "Only one number left but not 0. Game over."
            return obs, 0.0, True, False, {}

        # Check turn limit
        if self.turn >= self.max_turns:
            obs = (
                f"Maximum turns reached ({self.max_turns}). Game over.\n"
                f"Remaining units:\n{self._format_board()}"
            )
            return obs, 0.0, True, False, {}

        # Continue game
        board_str = self._format_board()
        observation = f"{self._get_instructions()}Board:\n{board_str}\n"

        return observation, 0.0, False, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random valid action.

        Returns:
            Random action as string
        """
        # Try to find a valid action
        for _ in range(100):
            if len(self.units) < 2:
                break
            i = random.randint(0, len(self.units) - 1)
            j = random.randint(0, len(self.units) - 1)
            if i != j:
                # Check if valid
                temp_env = NullifyEnv()
                temp_env.units = deepcopy(self.units)
                result = temp_env._verify_action(i, j)
                if result >= 0:
                    return f"\\boxed{{{i} {j}}}"

        # Fallback
        if len(self.units) >= 2:
            return f"\\boxed{{0 1}}"
        return "\\boxed{0 0}"
