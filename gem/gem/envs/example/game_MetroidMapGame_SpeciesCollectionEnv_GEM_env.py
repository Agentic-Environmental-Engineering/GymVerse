from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class MetroidMapGameEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 60,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 60

        self.complexity_params = {
            # Number of rooms in the map: larger maps have more states to consider → harder
            "num_rooms": (8, 30),
            # Number of distinct key types: more types increase combinatorial locking → harder
            "num_key_types": (2, 6),
            # Percentage (0-100) of rooms (except start) that require a key: higher ratio → harder
            "lock_ratio_percent": (20, 70),
            # Extra edges to add beyond a spanning tree: more connections → harder information structure
            "num_extra_edges": (0, 20),
            # Maximum loot value per room: higher values increase sums and range → slightly harder
            "loot_max": (9, 20),
            # REVERSED: starting number of keys in the start room: fewer starting keys → harder
            "start_keys_count": (2, 0),
            # Maximum number of different keys that can be in one room: higher allows deeper chains → harder
            "keys_per_room_max": (1, 3),
        }

        self.param_variance = {
            "num_rooms": 2,              # medium range → ±2
            "num_key_types": 1,          # small range (5 values) → ±1
            "lock_ratio_percent": 5,     # large range → ±5 percentage points
            "num_extra_edges": 3,        # medium range → ±3
            "loot_max": 2,               # medium range → ±2
            "start_keys_count": 0,       # tiny range (3 values) → 0
            "keys_per_room_max": 0,      # small range (3 values) → 0
        }

        # Placeholder attributes set by _apply_complexity_params
        self.num_rooms: int = 0
        self.num_key_types: int = 0
        self.lock_ratio_percent: int = 0
        self.num_extra_edges: int = 0
        self.loot_max: int = 0
        self.start_keys_count: int = 0
        self.keys_per_room_max: int = 0

        # Game state
        self.turn_count: int = 0
        self.start_room: int = 1
        self.key_types: Tuple[str, ...] = tuple()
        self.adj: Dict[int, set] = {}
        self.room_required_key: Dict[int, Optional[str]] = {}
        self.room_keys: Dict[int, set] = {}
        self.room_loot: Dict[int, int] = {}
        self.true_accessible_rooms: set = set()
        self.true_total_loot: int = 0

        self.marked_rooms: set = set()
        self.agent_keys: set = set()
        self.agent_accessible_rooms: set = set()

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            actual = center
            variance = self.param_variance.get(name, 0)
            if self.enable_param_randomization and variance > 0:
                actual = center + random.uniform(-variance, variance)
            # Clamp (supports reversed ranges)
            low, high = (max_val, min_val) if min_val > max_val else (min_val, max_val)
            actual = max(low, min(high, actual))
            setattr(self, name, int(round(actual)))

    def _build_connected_graph(self, n: int, extra_edges: int):
        self.adj = {i: set() for i in range(1, n + 1)}
        # Spanning tree
        for v in range(2, n + 1):
            u = random.randint(1, v - 1)
            self.adj[u].add(v)
            self.adj[v].add(u)
        # Add extra edges
        existing = set()
        for u in range(1, n + 1):
            for v in self.adj[u]:
                if u < v:
                    existing.add((u, v))
        attempts = 0
        added = 0
        max_attempts = n * 10
        while added < extra_edges and attempts < max_attempts:
            u = random.randint(1, n)
            v = random.randint(1, n)
            attempts += 1
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in existing:
                continue
            self.adj[u].add(v)
            self.adj[v].add(u)
            existing.add((a, b))
            added += 1

    def _compute_true_closure(self) -> Tuple[set, set]:
        accessible = set([self.start_room])
        keys = set(self.room_keys[self.start_room])  # start keys available
        changed = True
        while changed:
            changed = False
            for u in list(accessible):
                for v in self.adj[u]:
                    if v in accessible:
                        continue
                    req = self.room_required_key.get(v)
                    if req is None or req in keys:
                        accessible.add(v)
                        before_len = len(keys)
                        keys |= self.room_keys[v]
                        if len(keys) != before_len:
                            changed = True
                        changed = True
        return accessible, keys

    def _compute_agent_closure(self) -> set:
        accessible = set([self.start_room])
        keys = set(self.agent_keys)  # includes start keys by design
        changed = True
        while changed:
            changed = False
            for u in list(accessible):
                for v in self.adj[u]:
                    if v in accessible:
                        continue
                    req = self.room_required_key.get(v)
                    if req is None or req in keys:
                        accessible.add(v)
                        # Only keys from marked rooms are in agent_keys; do not add keys from unmarked rooms
                        changed = True
        return accessible

    def _get_instructions(self) -> str:
        return (
            "MetroidMapGame:\n"
            "You explore a map of rooms connected by passages. Some rooms require specific keys to enter. Keys are found in rooms.\n"
            "Goal: Compute the total loot value of all rooms that are reachable from the start by progressively collecting keys.\n"
            "You can query the map and mark rooms to record collected keys/loot in your notes (marking follows reachability from your current collected keys).\n"
            "Actions (use \\boxed{...}):\n"
            "- TASK: repeat task description\n"
            "- START: show the start room ID\n"
            "- NEIGHBORS i: list neighbors of room i and whether they are locked (by which key)\n"
            "- INSPECT i: reveal room i's lock requirement, loot value, and keys found there\n"
            "- MARK i: mark room i to add its keys and loot into your record (only if currently reachable)\n"
            "- COUNT: show how many rooms you've marked and the sum of their loot\n"
            "- SUBMIT x: submit your final answer (integer total)\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        marked_list = sorted(list(self.marked_rooms))
        keys_list = sorted(list(self.agent_keys))
        agent_accessible = sorted(list(self.agent_accessible_rooms))
        return (
            f"State: start={self.start_room}, marked={marked_list}, collected_keys={keys_list}, "
            f"agent_accessible_count={len(agent_accessible)}. "
            "Enter action as \\boxed{COMMAND [arg]}."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        # Key types
        self.key_types = tuple(f"Key{chr(65 + i)}" for i in range(self.num_key_types))

        # Build graph
        self._build_connected_graph(self.num_rooms, self.num_extra_edges)

        # Assign room lock requirements
        self.room_required_key = {}
        for i in range(1, self.num_rooms + 1):
            if i == self.start_room:
                self.room_required_key[i] = None
            else:
                if random.randint(1, 100) <= self.lock_ratio_percent:
                    self.room_required_key[i] = random.choice(self.key_types)
                else:
                    self.room_required_key[i] = None

        # Assign loot values
        self.room_loot = {i: random.randint(1, self.loot_max) for i in range(1, self.num_rooms + 1)}

        # Assign keys in rooms
        self.room_keys = {i: set() for i in range(1, self.num_rooms + 1)}
        # Start keys
        start_keys = set(random.sample(self.key_types, k=min(self.start_keys_count, len(self.key_types))))
        self.room_keys[self.start_room] = set(start_keys)

        # Distribute keys across rooms
        for i in range(1, self.num_rooms + 1):
            if i == self.start_room:
                continue
            max_k = self.keys_per_room_max
            count = random.randint(0, max_k)
            if count > 0:
                ks = set(random.sample(self.key_types, k=min(count, len(self.key_types))))
                self.room_keys[i] |= ks

        # Ensure each key type exists somewhere
        for kt in self.key_types:
            if not any(kt in s for s in self.room_keys.values()):
                target = random.randint(1, self.num_rooms)
                self.room_keys[target].add(kt)

        # Compute true closure and total loot; adjust minimal reachability if too small
        self.true_accessible_rooms, _ = self._compute_true_closure()
        min_required = max(2, int(0.3 * self.num_rooms))
        safety_loops = 0
        while len(self.true_accessible_rooms) < min_required and safety_loops < 5:
            # Boost start keys with one missing lock type that appears nearby
            missing_needed = set(self.room_required_key[i] for i in range(1, self.num_rooms + 1)
                                 if self.room_required_key[i] is not None) - self.room_keys[self.start_room]
            if missing_needed:
                add_key = random.choice(list(missing_needed))
                self.room_keys[self.start_room].add(add_key)
            self.true_accessible_rooms, _ = self._compute_true_closure()
            safety_loops += 1

        self.true_total_loot = sum(self.room_loot[i] for i in self.true_accessible_rooms)

        # Initialize agent state
        self.turn_count = 0
        self.marked_rooms = set()
        self.agent_keys = set(self.room_keys[self.start_room])  # start keys available immediately
        self.agent_accessible_rooms = self._compute_agent_closure()

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        cmd = parsed[0]
        info_suffix = {"suffix": self.get_task_suffix()}

        if cmd == "task":
            obs = "Task: Sum the loot values of all rooms reachable from the start by key progression. Queries help discover locks, keys, and neighbors."
            return obs, 0.0, False, False, info_suffix

        elif cmd == "start":
            obs = f"Start room is {self.start_room}."
            return obs, 0.0, False, False, info_suffix

        elif cmd == "neighbors":
            room_id = parsed[1]
            if room_id not in self.adj:
                obs = f"Room {room_id} does not exist."
                return obs, -0.1, False, False, info_suffix
            neighbors = sorted(list(self.adj[room_id]))
            details = []
            for nb in neighbors:
                req = self.room_required_key.get(nb)
                if req is None:
                    details.append(f"{nb}:unlocked")
                else:
                    details.append(f"{nb}:locked_by_{req}")
            obs = f"Neighbors of {room_id}: " + (", ".join(details) if details else "none")
            return obs, 0.0, False, False, info_suffix

        elif cmd == "inspect":
            room_id = parsed[1]
            if room_id not in self.adj:
                obs = f"Room {room_id} does not exist."
                return obs, -0.1, False, False, info_suffix
            req = self.room_required_key.get(room_id)
            req_txt = "none" if req is None else req
            loot = self.room_loot.get(room_id, 0)
            keys_here = sorted(list(self.room_keys.get(room_id, set())))
            obs = f"Room {room_id}: requires={req_txt}, loot={loot}, keys={keys_here}"
            return obs, 0.0, False, False, info_suffix

        elif cmd == "mark":
            room_id = parsed[1]
            if room_id not in self.adj:
                obs = f"Room {room_id} does not exist."
                return obs, -0.1, False, False, info_suffix
            # Recompute agent accessibility before marking
            self.agent_accessible_rooms = self._compute_agent_closure()
            if room_id in self.marked_rooms:
                obs = f"Room {room_id} already marked."
                return obs, 0.0, False, False, info_suffix
            if room_id not in self.agent_accessible_rooms:
                obs = f"Cannot mark: room {room_id} not reachable with collected keys."
                return obs, -0.1, False, False, info_suffix
            # Mark and collect
            self.marked_rooms.add(room_id)
            before_keys = set(self.agent_keys)
            self.agent_keys |= self.room_keys.get(room_id, set())
            self.agent_accessible_rooms = self._compute_agent_closure()
            delta_keys = sorted(list(self.agent_keys - before_keys))
            loot_gain = self.room_loot.get(room_id, 0)
            obs = f"Marked room {room_id}. Collected keys={delta_keys}, loot_gain={loot_gain}."
            return obs, 0.0, False, False, info_suffix

        elif cmd == "count":
            total_marked_loot = sum(self.room_loot[i] for i in self.marked_rooms)
            obs = f"Marked rooms: {len(self.marked_rooms)}; loot_sum={total_marked_loot}."
            return obs, 0.0, False, False, info_suffix

        elif cmd == "submit":
            submitted = parsed[1]
            # Terminal submission
            if submitted == self.true_total_loot:
                obs = "Success! Correct total loot."
                return obs, 1.0, True, False, info_suffix
            else:
                obs = "Failed! Incorrect total loot."
                return obs, -1.0, True, False, info_suffix

        elif cmd == "needs":
            room_id = parsed[1]
            if room_id not in self.adj:
                obs = f"Room {room_id} does not exist."
                return obs, -0.1, False, False, info_suffix
            req = self.room_required_key.get(room_id)
            req_txt = "none" if req is None else req
            obs = f"Room {room_id} requires {req_txt}."
            return obs, 0.0, False, False, info_suffix

        else:
            obs = f"Unsupported action: {cmd}."
            return obs, -0.5, True, False, info_suffix

        # Check timeout only if not terminated by actions above
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        content = m.group(1).strip()
        if not content:
            return None
        tokens = re.split(r'\s+', content)
        if len(tokens) == 0:
            return None
        cmd = tokens[0].lower()
        if cmd in ["task"]:
            return ("task",)
        if cmd in ["start"]:
            return ("start",)
        if cmd in ["neighbors"]:
            if len(tokens) < 2:
                return None
            try:
                room_id = int(tokens[1])
            except ValueError:
                return None
            return ("neighbors", room_id)
        if cmd in ["inspect"]:
            if len(tokens) < 2:
                return None
            try:
                room_id = int(tokens[1])
            except ValueError:
                return None
            return ("inspect", room_id)
        if cmd in ["mark"]:
            if len(tokens) < 2:
                return None
            try:
                room_id = int(tokens[1])
            except ValueError:
                return None
            return ("mark", room_id)
        if cmd in ["count"]:
            return ("count",)
        if cmd in ["submit"]:
            if len(tokens) < 2:
                return None
            try:
                val = int(tokens[1])
            except ValueError:
                return None
            return ("submit", val)
        if cmd in ["needs"]:
            if len(tokens) < 2:
                return None
            try:
                room_id = int(tokens[1])
            except ValueError:
                return None
            return ("needs", room_id)
        return (cmd,)  # Will be treated as unsupported in step()

    def sample_random_action(self) -> str:
        choices = [
            f"\\boxed{{TASK}}",
            f"\\boxed{{START}}",
            f"\\boxed{{NEIGHBORS {random.randint(1, max(1, self.num_rooms))}}}",
            f"\\boxed{{INSPECT {random.randint(1, max(1, self.num_rooms))}}}",
            f"\\boxed{{MARK {random.randint(1, max(1, self.num_rooms))}}}",
            f"\\boxed{{COUNT}}",
            f"\\boxed{{SUBMIT {self.true_total_loot}}}",
        ]
        return random.choice(choices)


class MetroidMapGameEnvWithFeedback(MetroidMapGameEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Wrap your command as \\boxed{COMMAND [arg]}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command"
            hint = "Use one of: TASK, START, NEIGHBORS i, INSPECT i, MARK i, COUNT, SUBMIT x."

        elif "cannot mark: room" in text and "not reachable" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "mark_unreachable"
            hint = "Inspect rooms and collect keys from reachable rooms first. Use NEIGHBORS to find reachable locks."

        elif "room" in text and "already marked" in text:
            error_type = "RedundantAction"
            error_detail["violation"] = "already_marked"
            hint = "Choose a different room. Use COUNT or NEIGHBORS to plan your next mark."

        elif "failed! incorrect total loot" in text:
            error_type = "WrongDecision"
            error_detail["expected_relation"] = "unknown"
            # Provide relative hint without revealing the exact value
            # We try to infer if the agent likely undercounted or overcounted by comparing marked sum with true total
            marked_sum = sum(self.room_loot[i] for i in self.marked_rooms)
            if marked_sum < self.true_total_loot:
                error_detail["expected_relation"] = "greater_than_submitted"
                hint = "You likely missed reachable rooms. Inspect neighbors and consider locks you can open with collected keys."
            else:
                error_detail["expected_relation"] = "less_than_submitted"
                hint = "You may have included rooms that are not reachable under key progression. Re-check lock requirements."

        elif "success! correct total loot" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well done."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Submit earlier or streamline queries. Use COUNT to track progress."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "num_rooms": self.num_rooms,
                "marked_count": len(self.marked_rooms),
                "agent_accessible_count": len(self.agent_accessible_rooms),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": f"Start by INSPECT {self.start_room} or NEIGHBORS {self.start_room} to learn locks and keys.",
            "turn": 0,
            "state": {
                "num_rooms": self.num_rooms,
                "marked_count": 0,
                "agent_accessible_count": len(self.agent_accessible_rooms),
            },
        }
        return obs, info