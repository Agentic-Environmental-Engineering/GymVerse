import random
import re
from typing import Any, Dict, Optional, Tuple, List

from gem.core import Env
from gem.utils.constants import LanguageGameReward


class NetworkPlanningGameEnv(Env):
    """
    Network Planning Game (GEM environment) with Complexity Evolution Support

    Description:
        You manage a road network connecting towns. Each town has a level (development)
        and each road has a capacity (throughput). Your moves are global policies that
        reorganize the network deterministically. After your policy, a random event
        introduces a new town connected to the network. Your progress metric is the
        hub influence: for any town, influence = town_level + sum(capacities of incident roads).
        The episode ends when you reach a target hub influence threshold, when no valid
        actions remain (or budget is exhausted), or when the turn limit is reached.

    State:
        - nodes: dict[name -> level]
        - edges: dict[(name_a, name_b) sorted tuple -> capacity]
        - budget: int (planning points)
        - max_level: int (cap on town levels)
        - max_capacity: int (cap on road capacities)
        - target_influence: int (success threshold for max hub influence)
        - turn_count: int
        - last_message: str (feedback about last action)
        - rng: Python random module seeded per reset

    Actions (submit via \\boxed{...}):
        - "expand": Connect the highest-influence hub to up to two lowest-level towns (adding new roads
                    or increasing capacity if already connected). Consumes 1 budget.
        - "upgrade": Upgrade the top two hubs: increase each hub's level by 1 (up to max) and
                     increase capacities of all adjacent roads by 1 (up to max). Consumes 1 budget.
        - "consolidate": Merge the lowest-level town into the second-lowest by removing the lowest
                         town and transferring connections; the recipient's level increases by 1
                         (up to max). Consumes 1 budget.
        - "rebalance": Decrease capacity by 1 on the two weakest roads (removing them if they hit 0),
                       and increase capacity by 1 on the two strongest roads (up to max). Consumes 1 budget.

    Rewards:
        - Shaped: positive proportional to increase in max hub influence (delta / target_influence).
        - Success bonus: +1.0 when threshold reached.
        - Small negative for unsupported actions (well-formatted but unknown).
        - Small negative for no-op actions (protocol use that yields no structural change).
        - Format errors terminate with LanguageGameReward.format_error_reward.

    Complexity Evolution (RLVE-style):
        - complexity: 1-10 (controls environment difficulty center)
        - Parameters evolve based on complexity through linear interpolation
        - enable_param_randomization: If True, adds random variance around center values
          to prevent overfitting (each reset has slightly different parameters)
        - Call evolve(mean_episode_success) to potentially increase complexity
        - Training code can directly modify complexity to decrease difficulty (rollback)
    """

    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()

        # ========== 1. Core evolution parameters ==========
        self.complexity = complexity  # 1-10


        # ========== 2. Fixed parameters ==========
        self.max_turns = max_turns if max_turns is not None else 100
        self._supported_actions = ["expand", "upgrade", "consolidate", "rebalance"]

        # ========== 3. Define evolvable parameters' ranges ==========
        # Parameters that change with complexity
        self.complexity_params = {
            'start_towns': (3, 6),           # Initial town count: 3-6 (more = harder)
            'budget': (18, 10),              # Budget: 18-10 (less = harder)
            'target_influence': (10, 30),    # Target influence: 10-30 (higher = harder)
            'max_level': (4, 8),             # Max town level: 4-8 (higher = harder)
            'max_capacity': (3, 8),          # Max road capacity: 3-8 (higher = harder)
        }

        # ========== 4. RLVE-style: Parameter randomization ==========
        self.enable_param_randomization = enable_param_randomization

        # Define variance (absolute value) for each parameter
        # Following RLVE principles:
        #   - Small-range parameters (3-5 possible values): no randomization
        #   - Discrete parameters: ±1, with 10-20% relative variance
        #   - Continuous parameters: ±5-15% relative range
        self.param_variance = {
            'start_towns': 0,      # Small-range (4 values: 3,4,5,6) → Fixed at center
            'budget': 1,           # Discrete (9 values: 10-18) → ±1 (12.5% relative variance)
            'target_influence': 2, # Discrete (21 values: 10-30) → ±2 (10% relative variance)
            'max_level': 0,        # Small-range (5 values: 4-8) → Fixed at center
            'max_capacity': 0,     # Small-range (6 values: 3-8) → Fixed at center
        }

        # ========== 5. Placeholders for evolvable parameters (calculated in reset) ==========
        self.start_towns: int = 0
        self.budget: int = 0
        self.target_influence: int = 0
        self.max_level: int = 0
        self.max_capacity: int = 0

        # ========== 6. Game state ==========
        self.nodes: Dict[str, int] = {}
        self.edges: Dict[Tuple[str, str], int] = {}
        self.turn_count: int = 0
        self.last_message: str = ""
        self._next_id: int = 0
        self._prev_max_influence: int = 0

        # First reset
        self.reset()

    def _get_instructions(self) -> str:
        """
        Return game instructions for the LLM agent.
        """
        return (
            "Network Planning Game\n"
            "Goal: Achieve a hub influence at or above the target threshold. A town's influence is its level plus the\n"
            "sum of capacities of roads connected to it. Success occurs when the maximum town influence >= target.\n"
            "\n"
            "Resources and rules:\n"
            "- You have a planning budget. Each valid policy action consumes 1 budget.\n"
            "- After your deterministic policy, a random event introduces a new town connected to a random existing town.\n"
            "- Caps: towns cannot exceed max_level and roads cannot exceed max_capacity.\n"
            "- Episode ends on success, when no valid actions remain (or budget is exhausted), or when max turns are reached.\n"
            "\n"
            "Available actions (use exactly these keywords):\n"
            "- expand: Connect the top hub to up to two weakest towns (create/increase roads).\n"
            "- upgrade: Upgrade top two hubs (level + roads' capacities).\n"
            "- consolidate: Merge the weakest town into the second weakest (remove one town, transfer connections).\n"
            "- rebalance: Shift capacity from weakest roads to strongest roads.\n"
            "\n"
            "Format: Submit your action in \\boxed{...}.\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        """
        Return current state description and format instruction.
        """
        hubs = self._top_hubs(2)
        hub_desc = ", ".join([f"{name}(infl={self._influence(name)})" for name in hubs])
        nodes_desc = ", ".join([f"{name}:{lvl}" for name, lvl in sorted(self.nodes.items())])
        edges_desc = f"{len(self.edges)} roads"
        return (
            f"State:\n"
            f"- Towns: {nodes_desc}\n"
            f"- Roads: {edges_desc}\n"
            f"- Top hubs: {hub_desc if hub_desc else 'none'}\n"
            f"- Budget: {self.budget}\n"
            f"- Target influence: {self.target_influence}\n"
            f"- Turn: {self.turn_count}\n"
            f"Last: {self.last_message}\n"
            "Enter your action in \\boxed{action} using one of: expand, upgrade, consolidate, rebalance."
        )

    def _apply_complexity_params(self):
        """
        Apply complexity to calculate actual parameter values.

        RLVE-style improvement:
        - complexity controls the "center value" of parameters
        - If enable_param_randomization=True, adds random variance around center
        - This prevents overfitting to fixed parameter values

        complexity=1  → center at minimum values
        complexity=10 → center at maximum values
        """
        # Normalize complexity to [0, 1], clamped at 1.0 for levels > 10
        normalized = min(1.0, (self.complexity - 1) / 9.0)

        # Calculate actual values for each evolvable parameter
        for param_name, (min_val, max_val) in self.complexity_params.items():
            # 1. Calculate center value using linear interpolation
            center_value = min_val + (max_val - min_val) * normalized

            # 2. Optionally add random variance (RLVE-style)
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)

                if variance > 0:
                    # Add random offset in range [-variance, +variance]
                    offset = random.uniform(-variance, variance)
                    actual_value = center_value + offset

                    # Clamp to [min_val, max_val] range
                    # Note: budget is reversed (min_val=18, max_val=10)
                    if min_val > max_val:  # Reversed parameter (e.g., budget)
                        actual_value = max(max_val, min(min_val, actual_value))
                    else:  # Normal parameter
                        actual_value = max(min_val, min(max_val, actual_value))

                    # Convert to integer
                    actual_value = int(round(actual_value))
                else:
                    # No variance: use center value directly
                    actual_value = int(round(center_value))
            else:
                # Randomization disabled: use fixed center value
                actual_value = int(round(center_value))

            # 3. Set to instance attribute
            setattr(self, param_name, actual_value)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Reset environment and return initial observation.

        Key: Parameters are dynamically calculated based on current complexity.
        """
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        else:
            # ensure some randomness but reproducibility across run lifetimes if needed
            random.seed()

        # ========== Apply complexity-based parameters ==========
        self._apply_complexity_params()

        # Initialize nodes and edges
        self.nodes = {}
        self.edges = {}
        self.turn_count = 0
        self._next_id = 0
        self.last_message = "Episode start. Consider upgrading or expanding early."

        # Use dynamically calculated start_towns
        for _ in range(self.start_towns):
            self._add_town(level=random.randint(1, 2))

        # Create a simple chain of roads to ensure connectivity
        town_names = sorted(self.nodes.keys())
        for i in range(len(town_names) - 1):
            a, b = town_names[i], town_names[i + 1]
            self._set_edge(a, b, 1)

        self._prev_max_influence = self._max_influence()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with LLM's text action.
        """
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with a valid action keyword."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()}
            )

        # If budget exhausted before action, terminate failure
        if self.budget <= 0:
            obs = f"Failed! Budget exhausted at turn {self.turn_count}. No valid actions remain."
            return (
                obs,
                -0.5,
                True,
                False,
                {"suffix": self.get_task_suffix()}
            )

        prev_metric = self._max_influence()
        action_used_budget = False
        change_count = 0
        progress_delta = 0.0
        reward = 0.0
        msg_details = ""

        if parsed not in self._supported_actions:
            obs = (
                f"At turn {self.turn_count}, unsupported action: '{parsed}'. No change."
            )
            # small penalty but do not consume budget
            reward = -0.02
            self.last_message = "Unsupported action. Use: expand, upgrade, consolidate, rebalance."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        # Apply deterministic transform
        if parsed == "expand":
            change_count, msg_details = self._action_expand()
            action_used_budget = True
        elif parsed == "upgrade":
            change_count, msg_details = self._action_upgrade()
            action_used_budget = True
        elif parsed == "consolidate":
            change_count, msg_details = self._action_consolidate()
            action_used_budget = True
        elif parsed == "rebalance":
            change_count, msg_details = self._action_rebalance()
            action_used_budget = True

        if action_used_budget:
            self.budget = max(0, self.budget - 1)

        # If no structural change occurred
        if change_count == 0:
            obs = (
                f"At turn {self.turn_count}, no structural change (no-op) after '{parsed}'. "
                f"Reason: constraints or caps prevent modification."
            )
            reward = -0.005
            self.last_message = "No-op: try a different policy that can modify structure."
            # Still perform stochastic insertion to reflect time passage
            self._random_insertion_event()
            # After event, check end conditions
            current_metric = self._max_influence()
            if self._no_valid_actions_remain():
                obs = f"Failed! No valid actions remain after a no-op at turn {self.turn_count}."
                return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}
            if self.turn_count >= self.max_turns:
                obs = f"Reached max turns ({self.max_turns})."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            self._prev_max_influence = current_metric
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        # Stochastic insertion after deterministic transform
        event_msg = self._random_insertion_event()

        # Compute progress and rewards
        current_metric = self._max_influence()
        progress_delta = max(0.0, current_metric - prev_metric)
        shaped = progress_delta / float(self.target_influence)
        reward += shaped

        # Check success
        if current_metric >= self.target_influence:
            obs = (
                f"Success! Target influence reached (max influence={current_metric} >= {self.target_influence}). "
                f"Applied '{parsed}': {msg_details}. Random event: {event_msg}"
            )
            reward += 1.0
            self.last_message = "Success achieved."
            return obs, reward, True, False, {"suffix": self.get_task_suffix()}

        # Check failure due to no valid actions/budget
        if self._no_valid_actions_remain():
            obs = (
                f"Failed! No valid actions remain at turn {self.turn_count}. "
                f"Applied '{parsed}': {msg_details}. Random event: {event_msg}"
            )
            return obs, -0.5, True, False, {"suffix": self.get_task_suffix()}

        # Check max turns
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = (
            f"At turn {self.turn_count}, applied '{parsed}': {msg_details}. "
            f"Random event: {event_msg}. Progress delta: +{int(progress_delta)} (max influence now {current_metric})."
        )
        self.last_message = f"Applied {parsed}. Δ={int(progress_delta)}."
        self._prev_max_influence = current_metric
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[str]:
        """
        Parse action from LLM response text. Extract content from \\boxed{...}
        and normalize to lowercase stripped keyword.
        """
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        extracted = matches[-1].group(1).strip().lower()
        if not extracted:
            return None
        # Return exact keyword string; unknown actions will be handled upstream
        return extracted

    def sample_random_action(self) -> str:
        """Return example action in correct format."""
        return f"\\boxed{{{random.choice(self._supported_actions)}}}"

    # ---------- Domain helpers ----------

    def _edge_key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def _set_edge(self, a: str, b: str, cap: int) -> None:
        self.edges[self._edge_key(a, b)] = max(0, min(self.max_capacity, cap))

    def _get_edge(self, a: str, b: str) -> Optional[int]:
        return self.edges.get(self._edge_key(a, b), None)

    def _remove_edge(self, a: str, b: str) -> None:
        self.edges.pop(self._edge_key(a, b), None)

    def _add_town(self, level: int) -> str:
        name = f"Town-{self._next_id}"
        self._next_id += 1
        self.nodes[name] = max(1, min(self.max_level, level))
        return name

    def _degree(self, name: str) -> int:
        d = 0
        for (x, y) in self.edges.keys():
            if x == name or y == name:
                d += 1
        return d

    def _incident_edges(self, name: str) -> List[Tuple[str, str]]:
        return [k for k in self.edges.keys() if name in k]

    def _influence(self, name: str) -> int:
        inc_sum = sum(self.edges[k] for k in self._incident_edges(name))
        return self.nodes.get(name, 0) + inc_sum

    def _max_influence(self) -> int:
        if not self.nodes:
            return 0
        return max(self._influence(n) for n in self.nodes.keys())

    def _top_hubs(self, k: int = 2) -> List[str]:
        return sorted(self.nodes.keys(), key=lambda n: (-self._influence(n), n))[:k]

    def _weakest_towns(self, k: int = 2) -> List[str]:
        return sorted(self.nodes.keys(), key=lambda n: (self.nodes[n], self._degree(n), n))[:k]

    def _can_expand(self) -> bool:
        hubs = self._top_hubs(1)
        if not hubs:
            return False
        h = hubs[0]
        for t in sorted(self.nodes.keys()):
            if t == h:
                continue
            cap = self._get_edge(h, t)
            if cap is None or cap < self.max_capacity:
                return True
        return False

    def _can_upgrade(self) -> bool:
        hubs = self._top_hubs(2)
        if not hubs:
            return False
        for h in hubs:
            if self.nodes[h] < self.max_level:
                return True
            for k in self._incident_edges(h):
                if self.edges[k] < self.max_capacity:
                    return True
        return False

    def _can_consolidate(self) -> bool:
        # For gameplay balance, require at least 3 towns to consolidate.
        return len(self.nodes) >= 3

    def _can_rebalance(self) -> bool:
        if not self.edges:
            return False
        max_cap = max(self.edges.values()) if self.edges else 0
        min_cap_pos = min([c for c in self.edges.values() if c > 0], default=None)
        return (min_cap_pos is not None) and (max_cap < self.max_capacity)

    def _no_valid_actions_remain(self) -> bool:
        if self.budget <= 0:
            return True
        return not (self._can_expand() or self._can_upgrade() or self._can_consolidate() or self._can_rebalance())

    # ---------- Actions implementations ----------

    def _action_expand(self) -> Tuple[int, str]:
        """
        Connect top hub to up to two weakest towns (create or increase capacity).
        """
        hubs = self._top_hubs(1)
        if not hubs:
            return 0, "no hub available"
        hub = hubs[0]
        candidates = [t for t in self._weakest_towns(k=999) if t != hub]
        changes = 0
        applied_to: List[str] = []
        for t in candidates:
            if changes >= 2:
                break
            cap = self._get_edge(hub, t)
            if cap is None:
                self._set_edge(hub, t, 1)
                changes += 1
                applied_to.append(t)
            elif cap < self.max_capacity:
                self._set_edge(hub, t, cap + 1)
                changes += 1
                applied_to.append(t)
            # if already at max capacity, skip and continue
        if changes == 0:
            return 0, "constraints prevented any new connections or upgrades"
        return changes, f"strengthened hub {hub} toward {', '.join(applied_to)}"

    def _action_upgrade(self) -> Tuple[int, str]:
        """
        Upgrade top two hubs: level +1 and adjacent roads' capacity +1 (up to caps).
        """
        hubs = self._top_hubs(2)
        if not hubs:
            return 0, "no hubs to upgrade"
        changes = 0
        upgraded_desc = []
        for h in hubs:
            before = self.nodes[h]
            if self.nodes[h] < self.max_level:
                self.nodes[h] += 1
                changes += 1
            # upgrade adjacent roads
            incs = 0
            for k in self._incident_edges(h):
                if self.edges[k] < self.max_capacity:
                    self.edges[k] += 1
                    incs += 1
            upgraded_desc.append(f"{h}(level {before}->{self.nodes[h]}, roads +{incs})")
            changes += incs
        if changes == 0:
            return 0, "caps prevent upgrades"
        return changes, f"upgraded hubs: {', '.join(upgraded_desc)}"

    def _action_consolidate(self) -> Tuple[int, str]:
        """
        Merge the weakest town into the second weakest:
        - Remove the lowest-level town.
        - Recipient level +1 (up to cap).
        - Transfer edges: reattach to recipient (combined capacity up to cap).
        """
        if len(self.nodes) < 3:
            return 0, "not enough towns to consolidate"
        weak = self._weakest_towns(2)
        if len(weak) < 2:
            return 0, "not enough towns to consolidate"
        donor, recipient = weak[0], weak[1]
        changes = 0
        # Recipient gains level +1
        before_lvl = self.nodes[recipient]
        if self.nodes[recipient] < self.max_level:
            self.nodes[recipient] += 1
            changes += 1
        # Transfer donor edges
        donor_edges = self._incident_edges(donor)
        transferred_count = 0
        for (a, b) in donor_edges:
            neighbor = a if b == donor else b
            if neighbor == recipient:
                # existing edge between donor and recipient -> ignore when donor removed
                self._remove_edge(a, b)
                changes += 1
                continue
            existing = self._get_edge(recipient, neighbor)
            donor_cap = self.edges[(a, b)]
            # remove old donor edge
            self._remove_edge(a, b)
            changes += 1
            if existing is None:
                self._set_edge(recipient, neighbor, min(self.max_capacity, donor_cap))
            else:
                self._set_edge(recipient, neighbor, min(self.max_capacity, existing + donor_cap))
            transferred_count += 1
        # Remove donor town
        del self.nodes[donor]
        changes += 1
        return changes, f"merged {donor} into {recipient} (level {before_lvl}->{self.nodes[recipient]}, transferred {transferred_count} roads)"

    def _action_rebalance(self) -> Tuple[int, str]:
        """
        Reduce capacities on the two weakest roads by 1 (removing them at 0),
        and increase capacities on the two strongest roads by 1 (up to cap).
        """
        if not self.edges:
            return 0, "no roads to rebalance"
        # Sort edges by capacity
        sorted_edges = sorted(self.edges.items(), key=lambda kv: (kv[1], kv[0]))
        weakest = [sorted_edges[i] for i in range(min(2, len(sorted_edges)))]
        strongest = [sorted_edges[-1 - i] for i in range(min(2, len(sorted_edges)))]
        changes = 0
        reduced_desc = []
        for (k, cap) in weakest:
            if cap > 0:
                new_cap = cap - 1
                if new_cap == 0:
                    self._remove_edge(k[0], k[1])
                    reduced_desc.append(f"{k[0]}-{k[1]} removed")
                else:
                    self._set_edge(k[0], k[1], new_cap)
                    reduced_desc.append(f"{k[0]}-{k[1]} -> {new_cap}")
                changes += 1
        increased_desc = []
        for (k, cap) in strongest:
            if cap < self.max_capacity:
                self._set_edge(k[0], k[1], cap + 1)
                increased_desc.append(f"{k[0]}-{k[1]} -> {cap + 1}")
                changes += 1
        if changes == 0:
            return 0, "capacities already at bounds"
        return changes, f"reduced: {', '.join(reduced_desc)}; increased: {', '.join(increased_desc)}"

    def _random_insertion_event(self) -> str:
        """
        Stochastic event after deterministic transform:
        - Introduce a new town with base level (1 or 2 depending on progress).
        - Connect it to a random existing town with capacity 1 (or increase if exists).
        """
        base = 1
        if self._max_influence() >= int(0.8 * self.target_influence) and random.random() < 0.4:
            base = 2
        new_town = self._add_town(level=base)
        existing_names = list(self.nodes.keys())
        if existing_names:
            # choose a random existing town different from new_town
            candidates = [n for n in existing_names if n != new_town]
            if candidates:
                neighbor = random.choice(candidates)
                cap = self._get_edge(new_town, neighbor)
                if cap is None:
                    self._set_edge(new_town, neighbor, 1)
                    return f"new town {new_town}(lvl={base}) connected to {neighbor} with road 1"
                else:
                    self._set_edge(new_town, neighbor, min(self.max_capacity, cap + 1))
                    return f"new town {new_town}(lvl={base}) increased road to {neighbor} -> {cap + 1}"
        return f"new town {new_town}(lvl={base}) with no road (isolated)"



class NetworkPlanningGameEnvWithFeedback(NetworkPlanningGameEnv):
    """
    Wrapper that adds structured diagnostic feedback without changing game logic.

    Diagnostic Fields in info["diagnostic"]:
        - error_type: str (FormatError, ProtocolViolation, WrongDecision, UnsupportedAction, Timeout, OK, Failure, Success)
        - error_detail: dict (specific details)
        - hint: Optional[str] (actionable suggestion if feedback_level >= 2)
        - turn: int (current turn count)
        - state: dict (relevant snapshot: budget, target, max_influence, top_hub)

    Error mapping:
        - FormatError: invalid \\boxed{} format
        - UnsupportedAction: keyword not in supported list
        - ProtocolViolation: action produced no structural change (no-op due to caps/constraints)
        - WrongDecision: well-formed action that did not improve progress (delta <= 0), but changed structure
        - Timeout: reached max turns
        - Failure: no valid actions remain or budget exhausted
        - Success: target reached
        - OK: normal step with positive progress
    """

    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        # Parse action name from observation if present
        action_name = None
        m = re.search(r"applied '([a-z]+)'", text)
        if m:
            action_name = m.group(1)

        # Identify delta
        delta_match = re.search(r"progress delta: \+([0-9]+)", text)
        delta_val = int(delta_match.group(1)) if delta_match else None

        # Classify errors/outcomes
        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Use \\boxed{expand} or other valid keyword."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            bad = None
            bmatch = re.search(r"unsupported action: '([^']+)'", obs)
            bad = bmatch.group(1) if bmatch else None
            error_detail["action"] = bad
            error_detail["supported"] = ["expand", "upgrade", "consolidate", "rebalance"]
            hint = "Choose one of the supported actions. Consider 'upgrade' early to build hub influence."

        elif "no structural change (no-op)" in text or "no-op" in text:
            error_type = "ProtocolViolation"
            error_detail["reason"] = "constraints_or_caps"
            error_detail["action"] = action_name
            hint = "Your chosen policy couldn't change the network. Try a different policy (e.g., expand or consolidate)."

        elif "success!" in text:
            error_type = "Success"
            imatch = re.search(r"max influence=([0-9]+)", text)
            error_detail["max_influence"] = int(imatch.group(1)) if imatch else None
            error_detail["target"] = getattr(self, "target_influence", None)
            hint = "Well done! Strategy met the threshold."

        elif "failed!" in text:
            error_type = "Failure"
            if "budget exhausted" in text:
                error_detail["cause"] = "budget_exhausted"
                hint = "Budget ran out. Prioritize actions that increase hub influence efficiently (upgrade, expand)."
            else:
                error_detail["cause"] = "no_valid_actions_remain"
                hint = "No valid actions left. Earlier consolidations or upgrades might unlock progress."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = getattr(self, "max_turns", None)
            hint = "Time limit reached. Choose stronger actions earlier to meet the target sooner."

        else:
            # Normal case: inspect progress
            if delta_val is not None:
                if delta_val > 0:
                    error_type = "OK"
                    hint = "Progress made. Continue upgrading key hubs or expand toward weak towns."
                else:
                    # Changed structure but no progress
                    if action_name is not None:
                        error_type = "WrongDecision"
                        error_detail["action"] = action_name
                        hint = "No progress this turn. Try 'upgrade' or 'expand' to boost hub influence."
                    else:
                        error_type = "WrongDecision"
                        hint = "No progress. Use actions that improve hub level or adjacent road capacities."

        # Build diagnostic dict
        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "budget": getattr(self, "budget", None),
                "target": getattr(self, "target_influence", None),
                "max_influence": self._max_influence() if hasattr(self, "_max_influence") else None,
                "top_hub": self._top_hubs(1)[0] if hasattr(self, "_top_hubs") and self._top_hubs(1) else None,
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        # Initial diagnostic guidance
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start by 'upgrade' to boost hub influence, or 'expand' to connect weak towns.",
            "turn": 0,
            "state": {
                "budget": getattr(self, "budget", None),
                "target": getattr(self, "target_influence", None),
                "max_influence": self._max_influence() if hasattr(self, "_max_influence") else None,
                "top_hub": self._top_hubs(1)[0] if self._top_hubs(1) else None,
            }
        }
        return obs, info