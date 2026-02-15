from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List, Set


class LabyrinthCourierEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 30,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, int(complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 30

        # Evolvable parameters
        self.complexity_params = {
            # number of rooms (nodes). Larger maze = more search and memory load = harder
            'num_rooms': (5, 20),
            # average branching per room (number of outgoing passages). Higher branching = more choices = harder
            'branching': (2, 4),
            # number of locked passages. More locks increase planning and inventory constraints = harder
            'num_locks': (0, 6),
            # number of keys available. REVERSED: fewer keys but same locks = harder; ensure feasibility in reset
            'num_keys': (2, 6),
            # observation richness: exits reveal neighbor names or not. REVERSED: less reveal = harder
            'visibility': (2, 1),  # 2=rich (names+lock info), 1=minimal (directions+lock info only)
            # trap density (rooms with trap that bounce you back). More traps = harder
            'num_traps': (0, 4),
        }

        # Variance settings
        self.param_variance = {
            'num_rooms': 1,
            'branching': 0,      # small integer range; keep stable for clarity
            'num_locks': 1,
            'num_keys': 1,
            'visibility': 0,     # categorical (1 or 2)
            'num_traps': 1,
        }

        # Placeholder attributes set during reset via _apply_complexity_params
        self.num_rooms: int = 0
        self.branching: int = 0
        self.num_locks: int = 0
        self.num_keys: int = 0
        self.visibility: int = 0
        self.num_traps: int = 0

        # Domain state
        self.turn_count: int = 0
        self.rooms: List[str] = []
        self.start_room: str = ""
        self.exit_room: str = ""
        self.current_room: str = ""
        # adjacency: room -> list of dicts with {'to': str, 'label': str, 'locked': bool, 'lock_id': Optional[str]}
        self.adj: Dict[str, List[Dict[str, Any]]] = {}
        # keys placed in rooms: room -> set of key ids present
        self.keys_in_rooms: Dict[str, Set[str]] = {}
        self.inventory: Set[str] = set()
        # traps: rooms with trap bounce behavior
        self.traps: Set[str] = set()
        # direction labels used for exits for immersion (not grid, just labels)
        self.dir_vocab = ["north", "east", "south", "west", "up", "down", "left", "right", "forward", "back"]

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            if self.enable_param_randomization:
                variance = self.param_variance.get(param_name, 0)
                if variance > 0:
                    actual_value = center_value + random.uniform(-variance, variance)
                    lo, hi = (max_val, min_val) if min_val > max_val else (min_val, max_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "You are navigating a labyrinth of named rooms connected by labeled passages.\n"
            "Goal: reach the EXIT room by moving through passages and unlocking any locked passages using keys you find.\n"
            "You perceive only local room information (exits, any keys present, and whether exits are locked).\n"
            "Rules:\n"
            "- Use actions in \\boxed{...} format.\n"
            "- Supported actions:\n"
            "  move dir=<label>          -> traverse the passage with the given label from the current room\n"
            "  take key=<key_id>         -> pick up a key present in the current room\n"
            "  inspect                   -> reprint the current room description (no movement)\n"
            "- Locked passages show lock ids (e.g., lock=k2). You can move through if you have the matching key id in inventory.\n"
            "- Some rooms contain traps that bounce you back to the previous room.\n"
            "Formatting:\n"
            "- All actions must be inside \\boxed{...}. Unknown actions or missing parameters are invalid.\n"
            "Example action:\n"
            f"{example}\n"
        )

    def get_task_suffix(self) -> str:
        return self._render_room_description() + "\nEnter your action in \\boxed{...} format."

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        # Ensure feasibility: adjust locks/keys relative to rooms
        max_possible_locks = max(0, self.num_rooms - 2)  # can't lock start without key and must allow connectivity
        self.num_locks = min(self.num_locks, max_possible_locks)
        # Ensure at least as many keys as locks (for solvability)
        self.num_keys = max(self.num_locks, self.num_keys)
        # Cap keys to avoid clutter
        self.num_keys = min(self.num_keys, max(1, self.num_rooms // 2))

        # Initialize rooms
        self.rooms = [f"R{i}" for i in range(self.num_rooms)]
        self.start_room = self.rooms[0]
        self.exit_room = self.rooms[-1]
        self.current_room = self.start_room
        self.turn_count = 0
        self.inventory = set()
        self.adj = {r: [] for r in self.rooms}
        self.keys_in_rooms = {r: set() for r in self.rooms}
        self.traps = set()

        # Build a connected backbone (spanning tree) to ensure connectivity
        unvisited = self.rooms[1:]
        parent = self.start_room
        while unvisited:
            child = unvisited.pop(0)
            label = self._unique_label_for(parent)
            self.adj[parent].append({'to': child, 'label': label, 'locked': False, 'lock_id': None})
            parent = child

        # Add extra edges to increase branching
        extra_edges_target = max(0, self.branching * self.num_rooms // 2 - (self.num_rooms - 1))
        added = 0
        attempts = 0
        while added < extra_edges_target and attempts < extra_edges_target * 5:
            attempts += 1
            a, b = random.sample(self.rooms, 2)
            # Avoid duplicate labels from same room
            existing_labels = {e['label'] for e in self.adj[a]}
            if len(existing_labels) >= len(self.dir_vocab):
                continue
            if any(e['to'] == b for e in self.adj[a]):
                continue
            label = self._unique_label_for(a)
            self.adj[a].append({'to': b, 'label': label, 'locked': False, 'lock_id': None})
            added += 1

        # Create locks; ensure each locked edge is solvable by placing key upstream
        lock_ids = [f"k{i+1}" for i in range(self.num_keys)]
        locked_edges = []
        all_edges = [(u, e_idx) for u in self.rooms for e_idx, _ in enumerate(self.adj[u])]
        random.shuffle(all_edges)
        for i in range(self.num_locks):
            if not all_edges:
                break
            u, e_idx = all_edges.pop()
            edge = self.adj[u][e_idx]
            # Do not lock the immediate start edge to avoid mandatory deadlock unless key in start
            lock_id = lock_ids[i % len(lock_ids)]
            edge['locked'] = True
            edge['lock_id'] = lock_id
            locked_edges.append((u, e_idx, lock_id))

        # Place keys ensuring reachability before the corresponding lock
        # We approximate "upstream" by placing keys in rooms along the backbone from start to the source room
        backbone_order = {self.rooms[i]: i for i in range(self.num_rooms)}
        for (u, e_idx, lock_id) in locked_edges:
            # candidate rooms from start up to u (inclusive), avoid exit
            idx_u = backbone_order.get(u, 0)
            candidate_rooms = [self.rooms[i] for i in range(0, max(1, idx_u + 1))]
            # ensure not placing in a room that itself is unreachable due to locks; heuristic: allow any candidate
            place_room = random.choice(candidate_rooms)
            # Avoid exit for placing keys
            if place_room == self.exit_room and self.num_rooms > 2:
                place_room = self.rooms[1]
            self.keys_in_rooms[place_room].add(lock_id)

        # Place remaining extra keys randomly (duplicates allowed)
        used_keys = {lock_id for _, _, lock_id in locked_edges}
        remaining_keys = [kid for kid in lock_ids if kid not in used_keys]
        for kid in remaining_keys:
            room = random.choice(self.rooms[:-1]) if self.num_rooms > 1 else self.start_room
            self.keys_in_rooms[room].add(kid)

        # Place traps in random non-start, non-exit rooms
        candidates_for_traps = [r for r in self.rooms if r not in {self.start_room, self.exit_room}]
        random.shuffle(candidates_for_traps)
        self.traps = set(candidates_for_traps[:min(self.num_traps, len(candidates_for_traps))])
        self._previous_room: Optional[str] = None

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{...} with a supported action."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get('action', '')
        reward = 0.0
        msg = ""

        if act == 'inspect':
            msg = "You look around."
            obs = self._render_room_description(prefix=msg)
        elif act == 'take':
            key_id = parsed.get('key')
            if not key_id:
                obs = "UNSUPPORTED ACTION: 'take' requires key=<key_id>."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if key_id in self.keys_in_rooms.get(self.current_room, set()):
                self.keys_in_rooms[self.current_room].remove(key_id)
                self.inventory.add(key_id)
                obs = f"Taken {key_id}. " + self._render_room_description()
            else:
                obs = f"FAILED: key {key_id} not present here. " + self._render_room_description()
        elif act == 'move':
            dir_label = parsed.get('dir')
            if not dir_label:
                obs = "UNSUPPORTED ACTION: 'move' requires dir=<label>."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            exits = self.adj.get(self.current_room, [])
            matching = None
            for e in exits:
                if e['label'] == dir_label:
                    matching = e
                    break
            if matching is None:
                obs = f"FAILED: no exit labeled '{dir_label}' from here. " + self._render_room_description()
            else:
                if matching['locked']:
                    lock_id = matching['lock_id']
                    if lock_id in self.inventory:
                        # consume the key to unlock (design choice)
                        self.inventory.remove(lock_id)
                        matching['locked'] = False
                        matching['lock_id'] = None
                        msg = f"Unlocked using {lock_id}. "
                    else:
                        obs = f"FAILED: exit '{dir_label}' is locked (lock={lock_id}) and you lack the key."
                        return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}
                # Move
                self._previous_room = self.current_room
                self.current_room = matching['to']
                # Trap check
                if self.current_room in self.traps:
                    bounced_from = self.current_room
                    self.current_room = self._previous_room if self._previous_room else self.current_room
                    obs = f"TRAP: Room {bounced_from} bounced you back to {self.current_room}. " + self._render_room_description()
                else:
                    obs = msg + self._render_room_description()
        else:
            obs = f"UNSUPPORTED ACTION: '{act}'. Allowed: move, take, inspect."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.current_room == self.exit_room:
            success_obs = "Success! You have reached the EXIT."
            return success_obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            truncated = True
            terminated = True
            obs = f"TIMEOUT: Reached max turns ({self.max_turns})."
            return obs, 0.0, terminated, truncated, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        # Find all boxed patterns and take the last one
        matches = re.findall(r"\\boxed\{([^}]+)\}", action.strip(), flags=re.DOTALL)
        if not matches:
            return None
        inner = matches[-1].strip()  # Take the last match
        if not inner:
            return None
        parts = inner.split()
        tokens: Dict[str, Any] = {}
        tokens['action'] = parts[0]
        for part in parts[1:]:
            if '=' in part:
                k, v = part.split('=', 1)
                tokens[k.strip()] = v.strip()
        return tokens

    def sample_random_action(self) -> str:
        # Prefer a plausible action given starting state when available
        if hasattr(self, 'adj') and self.adj and self.current_room in self.adj and self.adj[self.current_room]:
            choice = random.choice(self.adj[self.current_room])
            return rf"\boxed{{move dir={choice['label']}}}"
        return r"\boxed{inspect}"

    def _unique_label_for(self, room: str) -> str:
        used = {e['label'] for e in self.adj.get(room, [])}
        candidates = [d for d in self.dir_vocab if d not in used]
        if not candidates:
            # fallback: synthesize new label
            i = 1
            while f"path{i}" in used:
                i += 1
            return f"path{i}"
        return random.choice(candidates)

    def _render_room_description(self, prefix: str = "") -> str:
        exits = self.adj.get(self.current_room, [])
        keys_here = sorted(list(self.keys_in_rooms.get(self.current_room, set())))
        trap_note = "Yes" if self.current_room in self.traps else "No"
        if self.visibility == 2:
            # rich: show exit labels, lock status, and destination names
            exits_desc = []
            for e in exits:
                lock_str = f" lock={e['lock_id']}" if e['locked'] and e['lock_id'] else ""
                exits_desc.append(f"{e['label']} -> {e['to']}{lock_str}")
            exits_text = "; ".join(exits_desc) if exits_desc else "None"
        else:
            # minimal: show exit labels and whether locked, but not destination names
            exits_desc = []
            for e in exits:
                lock_str = f" lock={e['lock_id']}" if e['locked'] and e['lock_id'] else ""
                exits_desc.append(f"{e['label']}{lock_str}")
            exits_text = "; ".join(exits_desc) if exits_desc else "None"
        keys_text = ", ".join(keys_here) if keys_here else "None"
        inv_text = ", ".join(sorted(list(self.inventory))) if self.inventory else "Empty"
        return (
            f"{prefix}Room: {self.current_room} | Exit: {self.exit_room}\n"
            f"Exits: {exits_text}\n"
            f"Keys here: {keys_text} | Inventory: {inv_text} | Trap: {trap_note}"
        )


class LabyrinthCourierEnvWithFeedback(LabyrinthCourierEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_or_syntax"
            hint = "Wrap your command like \\boxed{move dir=north} or \\boxed{inspect}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            if "requires" in text:
                error_detail["issue"] = "missing_parameter"
                hint = "Provide the required parameter, e.g., \\boxed{move dir=<label>} or \\boxed{take key=k1}."
            else:
                error_detail["issue"] = "unknown_command"
                hint = "Use one of: move, take, inspect."
        elif text.startswith("failed:"):
            error_type = "WrongDecision"
            if "not present" in text:
                error_detail["issue"] = "take_key_missing"
                hint = "Use inspect to list keys here. Only take keys that are listed."
            elif "no exit labeled" in text:
                error_detail["issue"] = "move_label_invalid"
                hint = "Use a label listed under Exits."
            elif "is locked" in text:
                error_detail["issue"] = "locked_without_key"
                hint = "Find and take the key matching the lock id shown."
        elif text.startswith("trap:"):
            error_type = "OK"
            error_detail["event"] = "trap_bounce"
            hint = "Try a different exit from your previous room or seek a key first."
        elif "timeout" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan a shorter route: inspect, collect only necessary keys, then move efficiently."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        else:
            error_type = "OK"
            error_detail["outcome"] = "progress"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "room": getattr(self, "current_room", None),
                "inventory": sorted(list(getattr(self, "inventory", set()))),
                "exit": getattr(self, "exit_room", None),
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
            "hint": "Start with \\boxed{inspect} to see exits and any keys.",
            "turn": 0,
            "state": {
                "room": getattr(self, "current_room", None),
                "inventory": [],
                "exit": getattr(self, "exit_room", None),
            },
        }
        return obs, info