import random
import re
from queue import Queue
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class MazeEnv(Env):
    """
    Maze shortest-path environment (single-turn Q&A).

    The agent is given an N×N maze with walls and open cells and must provide a shortest
    path from (0, 0) to (N-1, N-1) using moves L, R, U, D. The answer must be returned
    in \\boxed{...} format, where the content is a sequence of characters consisting only
    of L, R, U, D (without spaces).
    """

    action2delta = {
        "L": (0, -1),
        "R": (0, +1),
        "U": (-1, 0),
        "D": (+1, 0),
    }

    def __init__(
        self,
        N: int = 10,
        density: float = 0.3,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        unsuccessful_solution: float = -0.2,
        rewarding_strategy: str = "(gold/answer)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 3.0,
        **kwargs
    ):
        super().__init__()
        # Core parameters to control difficulty
        self.N = N
        self.density = density

        # Preserve original reward-related parameters (not used in GEM scoring)
        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "unsuccessful_solution": unsuccessful_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        # State for current episode
        self.maze: Optional[List[str]] = None
        self.reference_answer: Optional[str] = None
        self.current_problem: Optional[str] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are given a square grid representing a maze. Cells are walls (#) or open spaces (.).\n"
            "Find a shortest path from the top-left (0, 0) to the bottom-right (N-1, N-1).\n"
            "You may move only in four directions: L (left), R (right), U (up), D (down), and only through open cells.\n"
            "Output Format: Return your move sequence inside \\boxed{...}. Example: \\boxed{RRDDLLUU}\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new maze problem."""
        super().reset(seed)

        # Validate parameters
        assert isinstance(self.N, int) and self.N >= 2, "N should be an integer >= 2"
        assert 0.0 <= float(self.density) < 1.0, "density should be between 0.0 and 1.0"

        # Generate a random maze with at least one path from start to goal
        maze, reference_answer = self._generate_maze_with_shortest_path(self.N, float(self.density))

        self.maze = ["".join(row) for row in maze]
        self.reference_answer = reference_answer

        # Build problem statement
        maze_str = "\n".join(self.maze)
        self.current_problem = (
            f"You are given a {self.N}×{self.N} grid representing a maze. Each cell is either a wall (#) or an open space (.).\n"
            f"The maze is provided below:\n{maze_str}\n\n"
            f"Your task is to find a shortest path from (0, 0) to ({self.N - 1}, {self.N - 1}).\n"
            f"You may move only in the four cardinal directions: up (U), down (D), left (L), and right (R), and only through open spaces (.).\n"
            f"Return your answer as a sequence of moves placed inside \\boxed{{...}}.\n"
            f"For example: \\boxed{{RRDDLLUU}}\n"
        )

        obs = self._get_instructions() + self.current_problem
        return obs, {}

    def _generate_maze_with_shortest_path(self, N: int, density: float) -> Tuple[List[List[str]], str]:
        """
        Generate a random maze and compute one shortest path via BFS.
        Ensures the start and end are open and a path exists.
        """
        while True:
            # Create maze with given density
            maze = [["#" if random.random() < density else "." for _ in range(N)] for _ in range(N)]
            maze[0][0] = "."
            maze[N - 1][N - 1] = "."

            prev: List[List[Optional[Tuple[int, int]]]] = [[None] * N for _ in range(N)]
            prev[0][0] = (0, 0)

            q: Queue[Tuple[int, int]] = Queue()
            q.put((0, 0))

            # BFS to find any path and record predecessors
            while not q.empty():
                x, y = q.get()
                for dx, dy in self.action2delta.values():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N and maze[nx][ny] == "." and prev[nx][ny] is None:
                        prev[nx][ny] = (x, y)
                        q.put((nx, ny))

            # If goal is reachable, reconstruct one shortest path
            if prev[N - 1][N - 1] is not None:
                path_actions: List[str] = []
                x, y = N - 1, N - 1
                while (x, y) != (0, 0):
                    px, py = prev[x][y]  # type: ignore
                    # Determine which action leads from (px, py) to (x, y)
                    for action, (dx, dy) in self.action2delta.items():
                        if (x, y) == (px + dx, py + dy):
                            path_actions.append(action)
                            break
                    x, y = px, py  # type: ignore
                path_actions.reverse()
                reference_answer = "".join(path_actions)
                return maze, reference_answer

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Validate the user's proposed path.

        Rewards:
        - Correct (valid path to goal with shortest length): 1.0
        - Wrong answer (valid format but not shortest or invalid path): 0.0
        - Format error (cannot parse \\boxed{...} or contains invalid characters): -0.1
        """
        # Parse answer from boxed format
        answer = self._parse_answer(action)

        if answer is None:
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Ensure answer uses only allowed characters L, R, U, D without any other symbols
        cleaned = answer.strip().upper()
        # Remove all whitespace inside the box, if any
        cleaned = re.sub(r"\s+", "", cleaned)

        if any(ch not in self.action2delta for ch in cleaned):
            return TERMINAL_STATE, -0.1, True, False, {"error": "format_error"}

        # Validate moves on the maze
        assert self.maze is not None and self.reference_answer is not None, "Environment not initialized. Call reset() first."
        N = self.N
        x, y = 0, 0
        for ch in cleaned:
            dx, dy = self.action2delta[ch]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < N and 0 <= ny < N):
                info = {
                    "error": "invalid_move",
                    "reason": "out_of_bounds",
                    "user_length": len(cleaned),
                    "shortest_length": len(self.reference_answer),
                    "reference_answer": self.reference_answer,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            if self.maze[nx][ny] == "#":
                info = {
                    "error": "invalid_move",
                    "reason": "hit_wall",
                    "user_length": len(cleaned),
                    "shortest_length": len(self.reference_answer),
                    "reference_answer": self.reference_answer,
                }
                return TERMINAL_STATE, 0.0, True, False, info
            x, y = nx, ny

        goal_reached = (x, y) == (N - 1, N - 1)
        shortest_len = len(self.reference_answer)
        user_len = len(cleaned)

        if not goal_reached:
            info = {
                "correct": False,
                "goal_reached": False,
                "user_length": user_len,
                "shortest_length": shortest_len,
                "reference_answer": self.reference_answer,
                "error": "not_reach_goal",
            }
            return TERMINAL_STATE, 0.0, True, False, info

        # Correct only if user path length equals shortest path length
        is_correct = (user_len == shortest_len)
        reward = 1.0 if is_correct else 0.0

        info = {
            "correct": is_correct,
            "goal_reached": True,
            "user_length": user_len,
            "shortest_length": shortest_len,
            "reference_answer": self.reference_answer,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the last occurrence of \\boxed{...} content from the text."""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
        return None

    def sample_random_action(self) -> str:
        """Sample a plausible action. If a reference is known, return it to ensure correctness."""
        if self.reference_answer is not None:
            return f"\\boxed{{{self.reference_answer}}}"
        # Fallback: random short sequence of moves
        moves = "LRUD"
        length = random.randint(1, max(2, self.N))
        seq = "".join(random.choice(moves) for _ in range(length))
        return f"\\boxed{{{seq}}}"