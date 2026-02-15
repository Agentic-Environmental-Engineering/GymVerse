import random
import re
from typing import Any, Optional, SupportsFloat, Tuple, List

from gem.core import Env
from gem.utils.constants import TERMINAL_STATE


class JugPuzzleEnv(Env):
    """Jug puzzle environment - single-turn Q&A using boxed action list format."""

    def __init__(
        self,
        # Problem size parameters
        N: Optional[int] = None,
        steps: Optional[int] = None,
        N_min: int = 2,
        N_max: int = 10,
        steps_min: int = 2,
        steps_max: int = 10,
        # Generation parameters
        max_capacity_multiple: int = 10,
        operation_probabilities: Optional[List[float]] = None,
        # Rewards
        wrong_format: float = -0.1,
        invalid_solution: float = 0.0,
        wrong_solution: float = 0.0,
        correct_solution: float = 1.0,
        **kwargs
    ):
        """
        Initialize the JugPuzzleEnv environment.

        Parameters:
            N: Number of jugs. If None, a random value in [N_min, N_max] will be sampled at reset.
            steps: Number of action steps used to synthesize a target volume. If None, random value in [steps_min, steps_max] at reset.
            N_min: Minimum number of jugs when sampling N.
            N_max: Maximum number of jugs when sampling N.
            steps_min: Minimum number of steps when sampling steps.
            steps_max: Maximum number of steps when sampling steps.
            max_capacity_multiple: Upper bound factor for jug capacities (each capacity sampled in [2, N * max_capacity_multiple]).
            operation_probabilities: Probabilities for operations ["Fill", "Empty", "Pour"]; must have length 3 and sum > 0.
            wrong_format: Reward for format errors (no boxed content).
            invalid_solution: Reward for invalid actions (out-of-range jug indices or malformed action list).
            wrong_solution: Reward when the target volume is not achieved.
            correct_solution: Reward when the target volume is present in any jug after executing the action list.
        """
        super().__init__()
        # Parameter storage (some may be sampled at reset)
        self.N = N
        self.steps = steps
        self.N_min = N_min
        self.N_max = N_max
        self.steps_min = steps_min
        self.steps_max = steps_max

        self.max_capacity_multiple = max_capacity_multiple

        if operation_probabilities is None:
            operation_probabilities = [0.1, 0.1, 0.8]
        assert len(operation_probabilities) == 3, "operation_probabilities should have exactly 3 elements for Fill, Empty, and Pour operations"
        assert sum(operation_probabilities) > 0, "operation_probabilities should sum to a positive value"
        self.operation_probabilities = operation_probabilities

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "wrong_solution": wrong_solution,
            "correct_solution": correct_solution,
        }

        # Runtime state
        self.current_problem: Optional[str] = None
        self.reference_answer_actions: Optional[str] = None
        self.reference_target_volume: Optional[int] = None
        self.jug_capacities: Optional[List[int]] = None
        self.used_N: Optional[int] = None
        self.used_steps: Optional[int] = None

    def _get_instructions(self) -> str:
        """Return task instructions."""
        return (
            "You are solving a jug puzzle.\n"
            "You will be given N jugs (initially empty) with specified capacities.\n"
            "Your goal is to fill any jug with exactly the target volume of water by performing a sequence of actions.\n"
            "Allowed actions:\n"
            "- Fill i   — Fill jug i to its full capacity.\n"
            "- Empty i  — Empty all water from jug i.\n"
            "- Pour i j — Pour water from jug i to jug j until jug i is empty or jug j is full.\n\n"
            "Output Format:\n"
            "Provide your sequence of actions inside \\boxed{...}.\n"
            "Each action must be on its own line, exactly as shown above (without backticks or quotes).\n\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and generate a new jug puzzle problem."""
        super().reset(seed)

        # Sample or validate N and steps
        if self.N is None:
            self.used_N = random.randint(self.N_min, self.N_max)
        else:
            assert self.N >= 2, "N should be greater than or equal to 2"
            self.used_N = self.N

        if self.steps is None:
            self.used_steps = random.randint(self.steps_min, self.steps_max)
        else:
            assert self.steps >= 2, "steps should be greater than or equal to 2"
            self.used_steps = self.steps

        # Generate jug capacities
        capacities = [random.randint(2, self.used_N * self.max_capacity_multiple) for _ in range(self.used_N)]
        self.jug_capacities = capacities

        # Differences of capacities (for target volume selection filtering)
        differences = set(
            capacity_i - capacity_j
            for capacity_j in capacities
            for capacity_i in capacities
            if capacity_i != capacity_j
        )

        # Initial reference (fallback)
        initial_jug = random.randint(0, self.used_N - 1)
        reference_actions = f"Fill {initial_jug}\n"
        target_volume = capacities[initial_jug]

        # Synthesize a sequence of actions to discover a target volume
        volumes = [0] * self.used_N
        actions_text = ""
        existing_volumes = set()

        for _ in range(self.used_steps):
            while True:
                operation = random.choices(["Fill", "Empty", "Pour"], self.operation_probabilities)[0]
                if operation == "Fill":
                    jug = random.randint(0, self.used_N - 1)
                    if volumes[jug] < capacities[jug]:
                        actions_text += f"Fill {jug}\n"
                        volumes[jug] = capacities[jug]
                        break
                elif operation == "Empty":
                    jug = random.randint(0, self.used_N - 1)
                    if volumes[jug] > 0:
                        actions_text += f"Empty {jug}\n"
                        volumes[jug] = 0
                        break
                else:  # Pour
                    jug_i = random.randint(0, self.used_N - 1)
                    jug_j = random.randint(0, self.used_N - 1)
                    if jug_i != jug_j and volumes[jug_i] > 0 and volumes[jug_j] < capacities[jug_j]:
                        actions_text += f"Pour {jug_i} {jug_j}\n"
                        pour_amount = min(volumes[jug_i], capacities[jug_j] - volumes[jug_j])
                        volumes[jug_i] -= pour_amount
                        volumes[jug_j] += pour_amount
                        break

            candidate_target_volumes = set(v for v in volumes if v > 0) - existing_volumes - differences - set(capacities)
            if candidate_target_volumes:
                # Update the reference actions and target volume when a novel candidate appears
                reference_actions = actions_text
                target_volume = random.choice(list(candidate_target_volumes))
                existing_volumes |= candidate_target_volumes

        # Store reference solution and target
        self.reference_answer_actions = reference_actions.strip()
        self.reference_target_volume = target_volume

        # Build problem prompt
        jug_caps_block = "\n".join(f"Jug {i}'s capacity: {c} liters" for i, c in enumerate(capacities))
        self.current_problem = (
            f"You are given {self.used_N} jugs (initially empty) with the following capacities:\n"
            f"{jug_caps_block}\n\n"
            f"Please fill a jug (you pick the one) with exactly {self.reference_target_volume} liters of water. "
            f"You may perform the following actions:\n"
            f"- Fill i — Fill jug i to its full capacity.\n"
            f"- Empty i — Empty all water from jug i.\n"
            f"- Pour i j — Pour water from jug i to jug j until either jug i is empty or jug j is full.\n\n"
            f"Output Format: Each action should be written on its own line in the format shown above (without backticks or quotes). "
            f"Provide your sequence inside \\boxed{{...}}."
        )

        obs = self._get_instructions() + self.current_problem
        info = {
            "N": self.used_N,
            "steps": self.used_steps,
            "jug_capacities": self.jug_capacities,
            "target_volume": self.reference_target_volume,
            "reference_answer_actions": self.reference_answer_actions,
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step: parse and verify the submitted action list."""
        # Extract boxed content
        boxed_text = self._parse_answer(action)
        if boxed_text is None:
            return TERMINAL_STATE, self.rewards["wrong_format"], True, False, {"error": "format_error"}

        # Parse actions from boxed content
        parsed_actions = self._process_actions(boxed_text)
        if parsed_actions is None:
            return TERMINAL_STATE, self.rewards["invalid_solution"], True, False, {"error": "invalid_solution"}

        # Simulate actions
        volumes = [0] * (self.used_N if self.used_N is not None else 0)
        capacities = self.jug_capacities if self.jug_capacities is not None else []

        for act in parsed_actions:
            op = act[0]
            if op == "Fill":
                jug = act[1]
                if not (0 <= jug < len(capacities)):
                    return TERMINAL_STATE, self.rewards["invalid_solution"], True, False, {"error": "invalid_solution"}
                volumes[jug] = capacities[jug]
            elif op == "Empty":
                jug = act[1]
                if not (0 <= jug < len(capacities)):
                    return TERMINAL_STATE, self.rewards["invalid_solution"], True, False, {"error": "invalid_solution"}
                volumes[jug] = 0
            elif op == "Pour":
                jug_i, jug_j = act[1], act[2]
                if not (0 <= jug_i < len(capacities) and 0 <= jug_j < len(capacities) and jug_i != jug_j):
                    return TERMINAL_STATE, self.rewards["invalid_solution"], True, False, {"error": "invalid_solution"}
                pour_amount = min(volumes[jug_i], capacities[jug_j] - volumes[jug_j])
                volumes[jug_i] -= pour_amount
                volumes[jug_j] += pour_amount
            else:
                # Should be unreachable due to parser constraints
                return TERMINAL_STATE, self.rewards["invalid_solution"], True, False, {"error": "invalid_solution"}

        # Check if target volume achieved
        target = self.reference_target_volume
        is_correct = target in volumes
        reward = self.rewards["correct_solution"] if is_correct else self.rewards["wrong_solution"]

        info = {
            "correct": is_correct,
            "reference_answer_actions": self.reference_answer_actions,
            "target_volume": self.reference_target_volume,
            "jug_capacities": self.jug_capacities,
            "user_actions": parsed_actions,
        }
        return TERMINAL_STATE, reward, True, False, info

    def _parse_answer(self, text: str) -> Optional[str]:
        """Extract the content inside \\boxed{...}. Supports multi-line content."""
        pattern = r'\\boxed\{([\s\S]*?)\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
        return None

    def _process_actions(self, answer_text: Optional[str]) -> Optional[List[List[Any]]]:
        """Parse the multi-line action list into structured operations."""
        if answer_text is None:
            return None

        lines = answer_text.splitlines()
        actions: List[List[Any]] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                # Skip empty lines; these are allowed but ignored
                continue
            parts = line.split()
            if not parts:
                return None
            op = parts[0]
            if op in ("Fill", "Empty"):
                if len(parts) != 2:
                    return None
                try:
                    jug = int(parts[1])
                except ValueError:
                    return None
                actions.append([op, jug])
            elif op == "Pour":
                if len(parts) != 3:
                    return None
                try:
                    jug_i = int(parts[1])
                    jug_j = int(parts[2])
                except ValueError:
                    return None
                actions.append(["Pour", jug_i, jug_j])
            else:
                return None
        return actions

    def sample_random_action(self) -> str:
        """Sample a random action output. Defaults to the reference answer when available."""
        if self.reference_answer_actions:
            return f"\\boxed{{\n{self.reference_answer_actions}\n}}"
        # Fallback: produce a simple valid single action
        if self.jug_capacities:
            jug = random.randint(0, len(self.jug_capacities) - 1)
            return f"\\boxed{{\nFill {jug}\n}}"
        return "\\boxed{\n}"