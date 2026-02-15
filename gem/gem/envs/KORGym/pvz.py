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

"""PVZ environment - Simplified Plants vs. Zombies game."""

import random
import math
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class PVZEnv(Env):
    """
    Plants vs. Zombies environment.

    A simplified tower defense game where plants must defend against waves
    of zombies on a 5×7 board. Plants cost sun and have various abilities,
    while zombies spawn from the right and move left.

    This is a multi-turn environment with dense reward (score-based).
    """

    ROWS = 5
    COLS = 7
    PLANT_COST = {
        'X': 50,    # Sunflower
        'W': 100,   # Peashooter
        'S': 325,   # Three-Line Shooter
        'J': 50,    # Wall-nut
        'H': 125,   # Torch Stump
        'F': 300    # Fire Chili
    }
    PLANT_HEALTH = {
        'X': 2,
        'W': 2,
        'S': 2,
        'J': 10,
        'H': 2,
        'F': float("inf")
    }
    ZOMBIE_HEALTH_AND_ATTACK = {
        'normal': {"health": 4, "attack": 1},
        'roadblock': {"health": 8, "attack": 1},
        'barrel': {"health": 12, "attack": 1},
        'high': {"health": 6, "attack": 3}
    }
    ZOMBIE_RENDER = {
        'normal': "N",
        'roadblock': "R",
        'barrel': "B",
        'high': "I"
    }

    def __init__(
        self,
        max_turns: int = 100,
        **_,
    ):
        """
        Initialize PVZ environment.

        Args:
            max_turns: Maximum number of turns
        """
        super().__init__()
        self.max_turns = max_turns
        self.plants = None
        self.zombies = None
        self.sun = None
        self.score = None
        self.turn = None
        self.game_over = None

    def _setup_game(self, seed: int):
        """Initialize game state."""
        random.seed(seed)
        self.plants = {}  # {(row, col): {"type": str, "health": int}}
        self.zombies = []  # [{'type': str, 'row': int, 'col': int, 'health': int, 'attack': int}]
        self.sun = 50
        self.score = 0
        self.turn = 0
        self.game_over = False

    def _cleanup(self):
        """Remove dead units."""
        self.zombies = [zombie for zombie in self.zombies if zombie["health"] > 0]
        self.plants = {pos: data for pos, data in self.plants.items() if data["health"] > 0}

    def _chilli_action(self):
        """Fire Chili clears all zombies in its row and self-destructs."""
        for pos in list(self.plants.keys()):
            plant_type = self.plants[pos]["type"]
            if plant_type == "F":
                row, _ = pos
                self.zombies = [zombie for zombie in self.zombies if zombie["row"] != row]
                del self.plants[pos]

    def _sun_flower_action(self):
        """Sunflowers generate extra sun."""
        for pos in list(self.plants.keys()):
            plant_type = self.plants[pos]["type"]
            if plant_type == "X":
                self.sun += 10

    def _calculate_damage(self, row: int, col: int) -> int:
        """Calculate damage with torch bonus."""
        base_damage = 1
        # Check if there's a torch between the plant and the edge
        for c in range(col + 1, self.COLS):
            if (row, c) in self.plants:
                if self.plants[(row, c)]["type"] == "H":
                    base_damage += 1
                    break
        return base_damage

    def _peas_action(self):
        """Pea shooters attack zombies."""
        peas = [(pos, data) for pos, data in self.plants.items() if data["type"] in ('W', 'S')]
        for pos, data in peas:
            row, col = pos
            if data["type"] == "W":
                lines = [row]
            elif data["type"] == "S":
                lines = [max(0, row - 1), row, min(self.ROWS - 1, row + 1)]
            else:
                lines = []

            for line in lines:
                zombies_in_line = [zombie for zombie in self.zombies if zombie["row"] == line]
                if not zombies_in_line:
                    continue
                # Target nearest zombie (smallest col)
                target_zombie = min(zombies_in_line, key=lambda x: x["col"])
                dmg = self._calculate_damage(line, col)
                target_zombie["health"] -= dmg

    def _plants_action(self):
        """All plants take their actions."""
        self._chilli_action()
        self._sun_flower_action()
        self._peas_action()

    def _zombies_action(self):
        """Zombies attack plants or move left."""
        new_zombies = []
        for zombie in self.zombies:
            row = zombie["row"]
            col = zombie["col"]
            attack = zombie["attack"]

            # If there's a plant at current position, attack it
            if (row, col) in self.plants:
                plant = self.plants[(row, col)]
                new_health = plant["health"] - attack
                if new_health <= 0:
                    del self.plants[(row, col)]
                else:
                    self.plants[(row, col)]["health"] = new_health
                new_zombies.append(zombie)
            else:
                # Move left
                new_col = col - 1
                if new_col < 0:
                    # Zombie reached left edge - game over
                    self.game_over = True
                    return
                zombie["col"] = new_col
                new_zombies.append(zombie)
        self.zombies = new_zombies

    def _select_zombie_type(self, round_num: int) -> str:
        """Select zombie type based on round number."""
        if round_num < 10:
            return "normal"
        elif round_num < 20:
            return random.choice(["normal", "roadblock"])
        else:
            return random.choice(["normal", "roadblock", "barrel", "high"])

    def _generate_new_zombies(self):
        """Generate new zombies every 5 turns."""
        if self.turn % 5 != 0:
            return

        base_count = 1 + math.floor(self.turn / 10)
        for _ in range(base_count):
            row = random.randint(0, self.ROWS - 1)
            zombie_type = self._select_zombie_type(self.turn)
            health = self.ZOMBIE_HEALTH_AND_ATTACK[zombie_type]["health"] + (self.turn // 10) * 4
            self.zombies.append({
                "type": zombie_type,
                "row": row,
                "col": self.COLS - 1,
                "health": health,
                "attack": self.ZOMBIE_HEALTH_AND_ATTACK[zombie_type]["attack"]
            })

    def _update_round(self):
        """Execute one round of game logic."""
        self._cleanup()
        self._plants_action()
        self._zombies_action()
        self._generate_new_zombies()
        self.sun += 25
        self.turn += 1
        self.score += 1

    def _process_action(self, actions: List[Tuple[str, int, int]]):
        """Process player's plant placements."""
        for plant_type, row, col in actions:
            # Check if position is occupied
            if (row, col) in self.plants:
                continue
            # Check if enough sun
            if self.sun >= self.PLANT_COST.get(plant_type, float("inf")):
                self.sun -= self.PLANT_COST[plant_type]
                self.plants[(row, col)] = {
                    "type": plant_type,
                    "health": self.PLANT_HEALTH[plant_type]
                }

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "In the following, you are required to participate in a simplified version of the "
            "Plants vs. Zombies game. The game is played on a 5×7 board, where zombies spawn from "
            "the far right side and move one step to the left each turn. The types of plants and "
            "zombies are as follows:\n\n"
            "Plants:\n"
            "- Sunflower (X): Costs 50 sun, has 2 HP, and generates an extra 10 sun each turn.\n"
            "- Peashooter (W): Costs 100 sun, has 2 HP, and deals 1 damage each turn to the first "
            "zombie in its current row.\n"
            "- Three-Line Shooter (S): Costs 325 sun, has 2 HP, and deals 1 damage each turn to the "
            "first zombie in its current row as well as the first zombie in each of the adjacent rows.\n"
            "- Wall-nut (J): Costs 50 sun and has 10 HP.\n"
            "- Torch Stump (H): Costs 125 sun, has 2 HP; it increases the damage of the plant to its "
            "left in the same row by +1.\n"
            "- Fire Chili (F): Costs 300 sun and eliminates all zombies in its row.\n\n"
            "Zombies:\n"
            "- Regular Zombie (N): Has 4 HP and deals 1 damage each turn to the plant that blocks its path.\n"
            "- Roadblock Zombie (R): Has 8 HP and deals 1 damage each turn to the plant that blocks its path.\n"
            "- Bucket Zombie (B): Has 12 HP and deals 1 damage each turn to the plant that blocks its path.\n"
            "- High-Attack Zombie (I): Has 6 HP and deals 3 damage each turn to the plant that blocks its path.\n\n"
            "Rules:\n"
            "- At least 25 sun is gained each turn.\n"
            "- A new zombie is spawned every 5 turns.\n"
            "- After every 10 turns, newly spawned zombies have their HP increased by 4, and the number "
            "of zombies spawned increases by 1.\n"
            "- Your score increases by 1 each turn.\n"
            "- The game lasts for a maximum of 100 turns.\n"
            "- Plants cannot be placed on the same grid cell, but zombies can coexist in the same cell.\n"
            "- Roadblock Zombies only spawn after turn 10, and Bucket Zombies and High-Attack Zombies only "
            "spawn after turn 20.\n\n"
            "Please input in the format 'PlantType Row Column'. If multiple plants need to be planted, "
            "separate them using a semicolon (`;`).\n"
            "Example: 'Answer: X 2 0;W 1 1'\n\n"
            "Alternatively, you can use \\boxed{X 2 0;W 1 1} format.\n\n"
        )

    def _format_board(self) -> str:
        """Format the game board as string."""
        header = f"Turn: {self.turn} | Sun: {self.sun} | Score: {self.score}\n\n"
        header += "Current Battlefield (X: Sunflower, W: Peashooter, S: Three-Line Shooter, "
        header += "J: Wall-nut, H: Torch Stump, F: Fire Chili, N: Zombie, R: Roadblock Zombie, "
        header += "B: Bucket Zombie, I: High-Attack Zombie)\n"

        grid = [['0'] * self.COLS for _ in range(self.ROWS)]

        # Place plants
        for (r, c), plant in self.plants.items():
            grid[r][c] = plant["type"]

        # Place zombies
        for zombie in self.zombies:
            r, c = zombie["row"], zombie["col"]
            z_char = self.ZOMBIE_RENDER.get(zombie["type"], "?")
            if grid[r][c] == '0':
                grid[r][c] = z_char
            else:
                grid[r][c] += z_char

        grid_str = "\n".join([f"Line{i}|" + "|".join(f"{cell:3}" for cell in row) for i, row in enumerate(grid)])
        return header + grid_str + "\n"

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new PVZ game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate game
        game_seed = seed if seed is not None else random.randint(0, 1000000)
        self._setup_game(game_seed)

        # Build observation
        board_str = self._format_board()
        observation = f"{self._get_instructions()}{board_str}\n"

        return observation, {
            "suffix": f"Max turns: {self.max_turns}."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process player's action.

        Args:
            action: Plant placements in format "Type Row Col;Type Row Col;..."

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
            # No action - just pass turn
            parsed_action = ""

        # Parse plant placements
        actions = []
        action_str = str(parsed_action).strip()
        if action_str:
            for cmd in action_str.split(';'):
                cmd = cmd.strip()
                if not cmd:
                    continue
                parts = cmd.split()
                if len(parts) != 3:
                    continue
                p_type = parts[0].upper()
                try:
                    row = int(parts[1])
                    col = int(parts[2])
                except ValueError:
                    continue
                if p_type not in self.PLANT_COST:
                    continue
                if not (0 <= row < self.ROWS) or not (0 <= col < self.COLS):
                    continue
                actions.append((p_type, row, col))

        # Process action and update round
        prev_score = self.score
        self._process_action(actions)
        self._update_round()

        # Calculate reward
        reward = float(self.score - prev_score)

        # Check game over
        if self.game_over:
            obs = (
                f"Game Over! A zombie reached your defenses.\n"
                f"Final score: {self.score}\n"
                f"Turns survived: {self.turn}\n"
                f"{self._format_board()}"
            )
            return obs, reward, True, False, {}

        # Check turn limit
        if self.turn >= self.max_turns:
            obs = (
                f"Congratulations! You survived {self.max_turns} turns!\n"
                f"Final score: {self.score}\n"
                f"{self._format_board()}"
            )
            return obs, reward, True, False, {}

        # Continue game
        board_str = self._format_board()
        observation = f"{self._get_instructions()}{board_str}\n"

        return observation, reward, False, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random valid action.

        Returns:
            Random action as string
        """
        # Try to place a random plant
        actions = []

        # Prioritize sunflowers early
        if self.turn < 20 and self.sun >= 50:
            # Try to place sunflower
            for _ in range(3):
                row = random.randint(0, self.ROWS - 1)
                col = random.randint(0, 2)
                if (row, col) not in self.plants and self.sun >= 50:
                    actions.append(f"X {row} {col}")
                    break

        # Place peashooters if we have sun
        if self.sun >= 100:
            for _ in range(2):
                row = random.randint(0, self.ROWS - 1)
                col = random.randint(1, self.COLS - 1)
                if (row, col) not in self.plants:
                    actions.append(f"W {row} {col}")
                    break

        if actions:
            return f"\\boxed{{" + ";".join(actions) + "}"
        return "\\boxed{}"
