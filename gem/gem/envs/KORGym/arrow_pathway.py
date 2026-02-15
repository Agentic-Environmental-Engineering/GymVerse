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

"""ArrowPathway environment - Auto-moving waypoint puzzle game."""

import ast
import random
from typing import Optional, Tuple, Dict, Any, List

from gem.core import Env
from gem.utils.parsing import extract_last_boxed_answer


class ArrowPathwayEnv(Env):
    """
    Arrow Pathway environment.

    Players are given a maze with a protagonist 'P', numbered waypoints (1,2,3),
    walls 'X', and empty spaces 'E'. The player moves automatically in a given
    initial direction. Players must place direction-changing devices at specific
    positions to ensure waypoints are triggered in sequence.

    This is a single-turn environment with sparse rewards.
    """

    def __init__(
        self,
        min_size: int = 7,
        max_size: int = 15,
        num_waypoints: int = 3,
        **_,
    ):
        """
        Initialize ArrowPathway environment.

        Args:
            min_size: Minimum maze size (n×n)
            max_size: Maximum maze size (n×n)
            num_waypoints: Number of sequential waypoints
        """
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.num_waypoints = num_waypoints
        self.maze = None
        self.initial_direction = None
        self.device_actions = None
        self.n = None

    def _get_direction(self, a: Tuple[int, int], b: Tuple[int, int]) -> Optional[str]:
        """Get direction from point a to point b."""
        r0, c0 = a
        r1, c1 = b
        if r1 == r0 and c1 == c0 + 1:
            return "right"
        elif r1 == r0 and c1 == c0 - 1:
            return "left"
        elif r1 == r0 + 1 and c1 == c0:
            return "down"
        elif r1 == r0 - 1 and c1 == c0:
            return "up"
        return None

    def _manhattan_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        order: str
    ) -> List[Tuple[int, int]]:
        """
        Generate Manhattan path between two points.

        Args:
            start: Start position
            end: End position
            order: "horizontal_first" or "vertical_first"

        Returns:
            List of positions forming the path
        """
        path = [start]
        r0, c0 = start
        r1, c1 = end

        if order == "horizontal_first":
            # Horizontal movement
            if c1 > c0:
                for c in range(c0 + 1, c1 + 1):
                    path.append((r0, c))
            elif c1 < c0:
                for c in range(c0 - 1, c1 - 1, -1):
                    path.append((r0, c))
            # Vertical movement
            if r1 > r0:
                for r in range(r0 + 1, r1 + 1):
                    path.append((r, c1))
            elif r1 < r0:
                for r in range(r0 - 1, r1 - 1, -1):
                    path.append((r, c1))
        else:  # vertical_first
            if r1 > r0:
                for r in range(r0 + 1, r1 + 1):
                    path.append((r, c0))
            elif r1 < r0:
                for r in range(r0 - 1, r1 - 1, -1):
                    path.append((r, c0))
            if c1 > c0:
                for c in range(c0 + 1, c1 + 1):
                    path.append((r1, c))
            elif c1 < c0:
                for c in range(c0 - 1, c1 - 1, -1):
                    path.append((r1, c))

        return path

    def _count_turns(self, path: List[Tuple[int, int]], init_dir: str) -> int:
        """Count number of direction changes in a path."""
        cnt = 0
        d = init_dir
        for i in range(len(path) - 1):
            nd = self._get_direction(path[i], path[i + 1])
            if nd != d:
                cnt += 1
                d = nd
        return cnt

    def _generate_maze(self, seed: int) -> Dict[str, Any]:
        """
        Generate a solvable maze with waypoints and solution path.

        Args:
            seed: Random seed

        Returns:
            Dictionary with maze, initial_direction, device_actions, etc.
        """
        random.seed(seed)
        n = random.randint(self.min_size, self.max_size)
        directions = ["up", "down", "left", "right"]
        initial_direction = random.choice(directions)

        # Choose P position based on initial direction
        if initial_direction == "right":
            P = (random.randint(0, n - 1), 0)  # Left edge
        elif initial_direction == "left":
            P = (random.randint(0, n - 1), n - 1)  # Right edge
        elif initial_direction == "down":
            P = (0, random.randint(0, n - 1))  # Top edge
        else:  # up
            P = (n - 1, random.randint(0, n - 1))  # Bottom edge

        # Generate waypoints
        waypoints = []
        for _ in range(self.num_waypoints):
            pos = (random.randint(0, n - 1), random.randint(0, n - 1))
            while pos == P or pos in waypoints:
                pos = (random.randint(0, n - 1), random.randint(0, n - 1))
            waypoints.append(pos)

        # Plan Manhattan path through all waypoints
        full_path = []
        current = P
        current_dir = initial_direction

        for wp in waypoints:
            path_h = self._manhattan_path(current, wp, "horizontal_first")
            path_v = self._manhattan_path(current, wp, "vertical_first")

            cnt_h = self._count_turns(path_h, current_dir)
            cnt_v = self._count_turns(path_v, current_dir)

            if cnt_h <= cnt_v:
                chosen_path = path_h
                if len(chosen_path) >= 2:
                    current_dir = self._get_direction(chosen_path[-2], chosen_path[-1])
            else:
                chosen_path = path_v
                if len(chosen_path) >= 2:
                    current_dir = self._get_direction(chosen_path[-2], chosen_path[-1])

            # Append path avoiding duplication
            if full_path and full_path[-1] == chosen_path[0]:
                full_path.extend(chosen_path[1:])
            else:
                full_path.extend(chosen_path)
            current = wp

        # Generate minimal device list (only at turns)
        mapping = {"up": "U", "down": "D", "left": "L", "right": "R"}
        device_actions = []

        # Check first move
        if len(full_path) >= 2:
            first_move = self._get_direction(full_path[0], full_path[1])
            if first_move != initial_direction:
                device_actions.append([mapping[first_move], full_path[0][0], full_path[0][1]])
                current_dir = first_move
            else:
                current_dir = initial_direction

        # Find all turn points
        for i in range(1, len(full_path) - 1):
            d_prev = self._get_direction(full_path[i - 1], full_path[i])
            d_next = self._get_direction(full_path[i], full_path[i + 1])
            if d_next != d_prev:
                device_actions.append([mapping[d_next], full_path[i][0], full_path[i][1]])

        # Construct maze
        maze = [["X" for _ in range(n)] for _ in range(n)]

        # Mark path as empty
        for (r, c) in full_path:
            maze[r][c] = "E"

        # Mark player start
        maze[P[0]][P[1]] = "P"

        # Mark waypoints
        for i, wp in enumerate(waypoints, start=1):
            r, c = wp
            maze[r][c] = str(i)

        # Randomly fill non-path cells
        for i in range(n):
            for j in range(n):
                if (i, j) not in full_path:
                    maze[i][j] = "E" if random.random() < 0.4 else "X"

        return {
            "maze": maze,
            "initial_direction": initial_direction,
            "device_actions": device_actions,
            "n": n
        }

    def _verify_actions(
        self,
        actions: List[List],
        maze: List[List[str]],
        initial_direction: str
    ) -> bool:
        """
        Simulate player movement with placed devices.

        Args:
            actions: List of [direction, row, col]
            maze: Maze grid
            initial_direction: Initial movement direction

        Returns:
            True if all waypoints triggered in order
        """
        n = len(maze)
        device_index = 0
        mapping = {"U": "up", "D": "down", "L": "left", "R": "right"}

        # Find start position
        start = None
        for i in range(n):
            for j in range(n):
                if maze[i][j] == "P":
                    start = (i, j)
                    break
            if start:
                break

        if not start:
            return False

        pos = start
        direction = initial_direction
        expected = 1  # Next waypoint to trigger

        # Find max waypoint number
        max_waypoint = 0
        for i in range(n):
            for j in range(n):
                if maze[i][j].isdigit():
                    max_waypoint = max(max_waypoint, int(maze[i][j]))

        max_steps = 1000
        steps = 0

        while steps < max_steps:
            # Check if device at current position
            if device_index < len(actions):
                exp_act = actions[device_index]
                if pos == (exp_act[1], exp_act[2]):
                    direction = mapping[exp_act[0]]
                    device_index += 1

            # Check if waypoint reached
            cell_val = maze[pos[0]][pos[1]]
            if cell_val.isdigit() and int(cell_val) == expected:
                expected += 1
                if expected > max_waypoint and max_waypoint > 0:
                    return True

            # Move in current direction
            r, c = pos
            if direction == "up":
                nr, nc = r - 1, c
            elif direction == "down":
                nr, nc = r + 1, c
            elif direction == "left":
                nr, nc = r, c - 1
            elif direction == "right":
                nr, nc = r, c + 1
            else:
                return False

            # Check bounds and walls
            if not (0 <= nr < n and 0 <= nc < n) or maze[nr][nc] == "X":
                return False

            pos = (nr, nc)
            steps += 1

        return False

    def _get_instructions(self) -> str:
        """Return game instructions."""
        return (
            "You are a good game player. I'll give you a game board and rules.\n"
            "Your task is:\n"
            "- First, give your answer according to the game board and rules.\n"
            "- Second, output the answer in the required format. The last line of your response "
            "should be: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final "
            "answer, e.g., \"Answer: [['R',3,2], ['U',0,2], ...]\"\n\n"
            "Alternatively, you can use \\boxed{[['R',3,2], ['U',0,2], ...]} format.\n\n"
            "Given an n×n maze containing empty spaces ('E'), a protagonist ('P'), walls ('X'), "
            "and numbered waypoints ('digits') that must be visited in sequence. You are provided "
            "with an initial player movement direction ('up/down/left/right') and a series of "
            "player actions ('U/D/L/R') along with their respective counts. The player needs to "
            "produce an action sequence such that the protagonist changes direction automatically "
            "when reaching each waypoint, ensuring that waypoints are visited sequentially. "
            "The action sequence must trigger the waypoints strictly in order; if the second "
            "waypoint isn't triggered, subsequent waypoints will not be triggered even if visited. "
            "The coordinates in the top left corner are [0,0].\n\n"
            "Please output the sequence of actions and corresponding trigger positions in the "
            "following format: [['R',3,2], ['U',0,2], ...] where 'R' means change to right, "
            "'U' means change to up, etc., at position (row=3, col=2) and (row=0, col=2).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a new Arrow Pathway puzzle.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed)

        # Generate maze
        result = self._generate_maze(seed if seed else random.randint(0, 1000000))
        self.maze = result["maze"]
        self.initial_direction = result["initial_direction"]
        self.device_actions = result["device_actions"]
        self.n = result["n"]

        # Build observation
        board_str = "\n".join([" ".join(row) for row in self.maze])
        device_actions_str = [act[0] for act in self.device_actions]

        observation = (
            f"{self._get_instructions()}"
            f"Maze Board:\n{board_str}\n"
            f"Current Direction: {self.initial_direction}\n"
            f"Device Actions: {device_actions_str}"
        )

        return observation, {
            "suffix": f"Guide player through {self.num_waypoints} waypoints ({self.n}×{self.n} maze)."
        }

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Validate the agent's solution.

        Args:
            action: Agent's response containing device placement list

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Try to extract answer
        parsed_answer = extract_last_boxed_answer(action)

        if parsed_answer is None:
            import re
            match = re.search(r'Answer:\s*(\[\[.+?\]\])', action, re.IGNORECASE | re.DOTALL)
            if match:
                parsed_answer = match.group(1).strip()

        if parsed_answer is None:
            obs = (
                "Your response did not contain a valid answer. "
                "Please use 'Answer: [[...]]' or \\boxed{[[...]]} format."
            )
            return obs, 0.0, True, False, {}

        # Parse device actions
        try:
            actions = ast.literal_eval(parsed_answer)
        except Exception as e:
            obs = f"Failed to parse answer. Error: {e}"
            return obs, 0.0, True, False, {}

        if not isinstance(actions, list):
            obs = "Answer must be a list of device actions."
            return obs, 0.0, True, False, {}

        # Verify actions
        success = self._verify_actions(actions, self.maze, self.initial_direction)

        if success:
            obs = f"Correct! All {self.num_waypoints} waypoints triggered in order using {len(actions)} devices."
            return obs, 1.0, True, False, {}
        else:
            obs = f"Incorrect. Waypoints not triggered in correct order with your {len(actions)} device placements."
            return obs, 0.0, True, False, {}

    def sample_random_action(self) -> str:
        """
        Sample a random action.

        Returns:
            The computed solution
        """
        if self.device_actions is not None:
            return f"\\boxed{{{self.device_actions}}}"
        else:
            return "\\boxed{[['R',0,0]]}"
