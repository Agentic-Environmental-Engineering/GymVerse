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

"""MiniGrid navigation environment - Grid-based pathfinding with obstacles."""

import random
from typing import Optional, Tuple, Dict, Any
from enum import IntEnum

try:
    import gymnasium as gym
    from minigrid.wrappers import SymbolicObsWrapper
    import numpy as np
    MINIGRID_AVAILABLE = True
except ImportError:
    MINIGRID_AVAILABLE = False

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class ObjectType(IntEnum):
    """Object types in MiniGrid."""
    UNSEEN = 0
    EMPTY = 1
    WALL = 2
    FLOOR = 3
    DOOR = 4
    KEY = 5
    BALL = 6
    BOX = 7
    GOAL = 8
    LAVA = 9
    AGENT = 10
    UNKNOWN = 255


class Color(IntEnum):
    """Colors in MiniGrid."""
    RED = 0
    GREEN = 1
    BLUE = 2
    PURPLE = 3
    YELLOW = 4
    GREY = 5


class MiniGridEnv(Env):
    """
    MiniGrid navigation environment.

    Agent navigates a 2D grid to reach goals, collect keys, open doors, and
    avoid obstacles like lava. Various environment scenarios with different
    complexity levels.

    This is a multi-turn environment with sparse terminal rewards.
    """

    # Environment names by difficulty
    EASY_ENVS = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-GoToDoor-6x6-v0",
    ]

    MEDIUM_ENVS = [
        "MiniGrid-Empty-Random-6x6-v0",
        "MiniGrid-DoorKey-6x6-v0",
        "MiniGrid-GoToObject-6x6-N2-v0",
        "MiniGrid-LavaGapS5-v0",
        "MiniGrid-RedBlueDoors-6x6-v0",
        "MiniGrid-Unlock-v0",
    ]

    HARD_ENVS = [
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-FourRooms-v0",
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-BlockedUnlockPickup-v0",
    ]

    def __init__(
        self,
        difficulty: str = "medium",
        max_steps: int = 100,
        **_,
    ):
        """
        Initialize MiniGrid environment.

        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            max_steps: Maximum number of steps
        """
        super().__init__()

        if not MINIGRID_AVAILABLE:
            raise ImportError(
                "MiniGrid is not installed. Please install it with: "
                "pip install minigrid"
            )

        self.difficulty = difficulty
        self.max_steps = max_steps

        # Select environment pool based on difficulty
        if difficulty == "easy":
            self.env_pool = self.EASY_ENVS
        elif difficulty == "hard":
            self.env_pool = self.HARD_ENVS
        else:
            self.env_pool = self.MEDIUM_ENVS

        self.env = None
        self.env_name = None
        self.current_step = 0
        self.carrying = None
        self.agent_pos = None
        self.agent_dir = None

        # Action mappings
        self.action_meanings = {
            0: 'turn_left',
            1: 'turn_right',
            2: 'move_forward',
            3: 'pickup',
            4: 'drop',
            5: 'toggle',
            6: 'done'
        }
        self.reverse_action_map = {v: k for k, v in self.action_meanings.items()}

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game problem-solver. I'll give you a game board and rules.\\n"
            "Your task is:\\n"
            "- First, give your answer according to the game board and rules.\\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., 'Answer: move_forward'\\n\\n"
            "Alternatively, you can use \\\\boxed{move_forward} format.\\n\\n"
            "You are an AI agent navigating a grid environment. Your task is:\\n\\n"
            "1. Analyze environmental observations\\n"
            "2. Choose optimal actions to achieve the mission\\n\\n"
            "**Core Rules**:\\n"
            "- Strictly use ONLY the provided action list\\n"
            "- Prioritize shortest paths\\n"
            "- Avoid dangerous areas (lava)\\n"
            "- Manage inventory carefully (keys, etc)\\n"
            "- Closed doors require 'toggle' to open\\n"
            "- Locked doors need matching key (use pickup first)\\n\\n"
            "**Action Space** (REQUIRED RESPONSE FORMAT):\\n"
            "turn_left   : Rotate 90° counter-clockwise\\n"
            "turn_right  : Rotate 90° clockwise\\n"
            "move_forward: Advance if path clear\\n"
            "pickup      : Collect keys/objects\\n"
            "drop        : Drop carried object\\n"
            "toggle      : Open doors or interact (facing target required)\\n"
            "done        : ONLY when goal reached\\n\\n"
            "**Observation**:\\n"
            "You receive the entire grid as an observation, represented as a 3D array of shape (width, height, 3).\\n"
            "- Coordinates range from (0,0) at top-left to (width-1, height-1) at bottom-right.\\n"
            "- Each cell contains [object_type, color, state]:\\n"
            "  - object_type: 1=EMPTY, 2=WALL, 3=FLOOR, 4=DOOR, 5=KEY, 6=BALL, 7=BOX, 8=GOAL, 9=LAVA, 10=AGENT\\n"
            "  - color: 0=RED, 1=GREEN, 2=BLUE, 3=PURPLE, 4=YELLOW, 5=GREY\\n"
            "    - For AGENT (10), this is direction: 0=right, 1=down, 2=left, 3=up\\n"
            "  - state: For DOOR, 0=open, 1=closed, 2=locked; otherwise 0\\n"
            "- Your position is the cell with object_type=10.\\n"
            "- The mission is a string, e.g., 'get to the green goal square'.\\n\\n"
            "Respond with exactly one lowercase action word.\\n\\n"
        )

    def _get_observation_text(self, obs: Dict[str, Any]) -> str:
        """Generate readable observation description."""
        mission = obs['mission']

        dir_names = {
            0: "right (→)",
            1: "down (↓)",
            2: "left (←)",
            3: "up (↑)"
        }

        # Inventory status
        if self.carrying:
            try:
                item_color = Color(self.carrying.color).name.lower()
                item_type = ObjectType(self.carrying.type).name.lower()
                inventory_status = f"carrying a {item_color} {item_type}"
            except (ValueError, AttributeError):
                inventory_status = "carrying an object"
        else:
            inventory_status = "not carrying anything"

        # Grid size and agent info
        grid_width, grid_height = obs['image'].shape[0], obs['image'].shape[1]
        agent_x, agent_y = self.agent_pos
        agent_dir = self.agent_dir

        observation_text = (
            f"Mission: {mission}\\n"
            f"Grid size: {grid_width}x{grid_height}\\n"
            f"Agent at ({agent_x}, {agent_y}), facing {dir_names[agent_dir]}\\n"
            f"Status: {inventory_status} | {self.max_steps - self.current_step} steps remaining\\n"
            f"Observation:\\n{np.array2string(obs['image'], separator=', ')}\\n"
        )
        return observation_text

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new MiniGrid episode.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Select random environment
        self.env_name = random.choice(self.env_pool)

        # Create environment
        self.env = gym.make(
            self.env_name,
            render_mode="rgb_array",
            max_steps=self.max_steps
        )
        self.env = SymbolicObsWrapper(self.env)

        # Reset environment
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()

        self.current_step = 0
        self.carrying = obs.get('carrying')
        self.agent_dir = self.env.unwrapped.agent_dir
        self.agent_pos = self.env.unwrapped.agent_pos

        # Build observation
        obs_text = self._get_observation_text(obs)
        observation = f"{self._get_instructions()}{obs_text}"

        return observation, {
            "suffix": f"Step {self.current_step}/{self.max_steps}, Mission: {obs['mission']}"
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Agent's action choice

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        parsed_action = extract_last_boxed_answer(action)

        if parsed_action is None:
            import re
            match = re.search(r'Answer:\\s*(\\w+)', action, re.IGNORECASE)
            if match:
                parsed_action = match.group(1).strip().lower()

        if parsed_action is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: ACTION' or \\\\boxed{ACTION} format."
            )
            return obs, 0.0, True, False, {}

        action_str = parsed_action.strip().lower()

        # Convert action string to index
        if action_str not in self.reverse_action_map:
            obs = (
                f"Invalid action: {action_str}. Must be one of: "
                f"{', '.join(self.action_meanings.values())}"
            )
            return obs, 0.0, True, False, {}

        action_idx = self.reverse_action_map[action_str]

        # Execute action
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        self.current_step += 1

        # Update agent state
        self.agent_dir = self.env.unwrapped.agent_dir
        self.agent_pos = self.env.unwrapped.agent_pos
        self.carrying = obs.get('carrying')

        # Check if goal reached
        if "reached_goal" in info.get('reason', ''):
            terminated = True
            reward = 1.0

        # Normalize reward to 0 or 1
        if reward > 0:
            reward = 1.0
        elif reward < 0:
            reward = 0.0

        # Build observation
        obs_text = self._get_observation_text(obs)

        if terminated:
            observation = (
                f"{obs_text}\\n\\n"
                f"Mission completed! Goal reached in {self.current_step} steps.\\n"
            )
        elif truncated:
            observation = (
                f"{obs_text}\\n\\n"
                f"Mission failed! Maximum steps ({self.max_steps}) reached.\\n"
            )
        else:
            observation = obs_text

        return observation, reward, terminated, truncated, {
            "current_step": self.current_step,
            "agent_pos": self.agent_pos,
            "agent_dir": self.agent_dir,
            "mission": obs.get('mission', '')
        }

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            Random action
        """
        action = random.choice(list(self.action_meanings.values()))
        return f"\\\\boxed{{{action}}}"
