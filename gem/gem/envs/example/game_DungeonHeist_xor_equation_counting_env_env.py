from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class DungeonHeistEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        # Evolvable parameters
        self.complexity_params = {
            # Corridor length: more rooms increases path planning difficulty
            'path_length': (8, 25),
            # Number of locked doors to confront on the path: more doors require more keys/decisions
            'num_doors': (0, 4),
            # Number of trap rooms along the corridor: more traps increase risk management difficulty
            'num_traps': (0, 10),
            # REVERSED: how many rooms ahead/behind are passively visible; less visibility is harder
            'vision_ahead': (2, 0),
            # REVERSED: starting health; less health is harder
            'starting_health': (6, 2),
            # REVERSED: disarm kits available; fewer kits is harder
            'disarm_kits': (3, 0),
            # Number of different door/key colors present; more colors increase matching complexity
            'door_colors': (1, 3),
            # REVERSED: extra keys beyond the required minimum; fewer extras is harder
            'extra_keys': (2, 0),
        }

        # Variance settings
        self.param_variance = {
            'path_length': 2,       # ~10% of range
            'num_doors': 1,
            'num_traps': 2,         # ~20% relative for larger range
            'vision_ahead': 0,      # small range → fix to center per level
            'starting_health': 0,   # small range → keep deterministic with level
            'disarm_kits': 1,       # small integer range → slight randomness
            'door_colors': 0,       # small range → fixed per level
            'extra_keys': 1,        # small integer range
        }

        # Placeholder attributes (set in _apply_complexity_params)
        self.path_length: int = 0
        self.num_doors: int = 0
        self.num_traps: int = 0
        self.vision_ahead: int = 0
        self.starting_health: int = 0
        self.disarm_kits: int = 0
        self.door_colors: int = 0
        self.extra_keys: int = 0

        # State
        self.turn_count: int = 0
        self.tiles: List[Dict[str, Any]] = []
        self.unlocked_doors: set = set()
        self.pos: int = 0
        self.health: int = 0
        self.kits: int = 0
        self.inventory: Dict[str, int] = {}
        self.look_revealed: set = set()
        self.color_pool: List[str] = ["Red", "Blue", "Green"]
        self.last_parsed_action: Optional[Tuple[str, Optional[str]]] = None

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_val = min_val + (max_val - min_val) * normalized
            variance = self.param_variance.get(param_name, 0)
            actual_value = center_val
            if self.enable_param_randomization and variance > 0:
                actual_value = center_val + random.uniform(-variance, variance)
            lo = min(min_val, max_val)
            hi = max(min_val, max_val)
            actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        return (
            "Dungeon Heist: Reach the vault at the end of the corridor without losing all health.\n"
            "You start at position 0 and must reach the last room to claim the treasure.\n"
            "Some rooms contain traps that harm you when you enter. Some passages are blocked by locked doors.\n"
            "Keys are color-coded and each key opens one door of the matching color. Doors must be unlocked before you can move through them.\n"
            "You always passively see a limited number of rooms ahead/behind. Use LOOK to extend your knowledge further temporarily.\n"
            "You may DISARM a trap in the next room if you have a disarm kit. Using a key consumes it.\n"
            "\n"
            "Available actions (use exactly one per turn):\n"
            "- MOVE FORWARD or MOVE BACK\n"
            "- LOOK\n"
            "- PICK [COLOR] (if a key is in your current room)\n"
            "- USE [COLOR] (unlocks the next door if color matches and you have the key)\n"
            "- DISARM (disarm the trap in the next room if present and you have a kit)\n"
            "- WAIT\n"
            "\n"
            "Format your action as \\boxed{...} with uppercase or lowercase accepted. Example:\n"
            f"{example}\n"
        )

    def get_task_suffix(self) -> str:
        visible = self._compute_visible_indices()
        desc_lines = []
        desc_lines.append(f"Position: {self.pos}/{self.path_length - 1}")
        desc_lines.append(f"Health: {self.health}")
        desc_lines.append(f"Disarm kits: {self.kits}")
        inv_str = ", ".join([f"{c}:{n}" for c, n in self.inventory.items() if n > 0]) or "none"
        desc_lines.append(f"Keys: {inv_str}")
        ahead = []
        behind = []
        for i in sorted(visible):
            if i == self.pos:
                continue
            if i > self.pos and i <= self.pos + max(0, self.vision_ahead + 2):
                ahead.append(f"{i}:{self._tile_brief(i)}")
            if i < self.pos and i >= max(0, self.pos - max(0, self.vision_ahead + 2)):
                behind.append(f"{i}:{self._tile_brief(i)}")
        if ahead:
            desc_lines.append("Ahead known: " + ", ".join(ahead))
        if behind:
            desc_lines.append("Behind known: " + ", ".join(behind))
        dist = (self.path_length - 1) - self.pos
        desc_lines.append(f"Distance to goal: {dist}")
        desc_lines.append("Enter your action in \\boxed{...} format. One action per turn.")
        return "\n".join(desc_lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()

        L = max(5, int(self.path_length))
        self.path_length = L

        colors = self.color_pool[: max(1, self.door_colors)]
        self.turn_count = 0
        self.pos = 0
        self.health = int(self.starting_health)
        self.kits = int(self.disarm_kits)
        self.inventory = {c: 0 for c in colors}
        self.unlocked_doors = set()
        self.look_revealed = set()
        self.tiles = []
        for _ in range(L):
            self.tiles.append({
                "door_color": None,
                "key_colors": [],
                "trap": False,
                "trap_cleared": False,
            })

        desired_doors = min(self.num_doors, max(0, L // 4))
        door_positions = set()
        possible_positions = list(range(2, L - 1))  # avoid placing at start, allow door just before goal
        random.shuffle(possible_positions)
        for p in possible_positions:
            if len(door_positions) >= desired_doors:
                break
            if all(abs(p - dp) >= 2 for dp in door_positions):
                door_positions.add(p)
        door_positions = sorted(door_positions)

        for i, dp in enumerate(door_positions):
            col = colors[i % len(colors)]
            self.tiles[dp]["door_color"] = col

        used_positions = set(door_positions)
        required_keys = len(door_positions)
        key_positions = []
        for i, dp in enumerate(door_positions):
            placed = False
            candidates = [k for k in range(1, dp) if k not in used_positions and self.tiles[k]["door_color"] is None]
            random.shuffle(candidates)
            for kp in candidates:
                self.tiles[kp]["key_colors"].append(colors[i % len(colors)])
                used_positions.add(kp)
                key_positions.append(kp)
                placed = True
                break
            if not placed:
                # fallback: ensure feasibility by placing at the earliest available before dp
                for kp in range(1, dp):
                    if kp not in used_positions and self.tiles[kp]["door_color"] is None:
                        self.tiles[kp]["key_colors"].append(colors[i % len(colors)])
                        used_positions.add(kp)
                        key_positions.append(kp)
                        placed = True
                        break
                if not placed:
                    # if still not placed (very rare in small L), drop the door to maintain solvability
                    self.tiles[dp]["door_color"] = None

        # Extra keys
        extras = max(0, int(self.extra_keys))
        extras_placed = 0
        available_for_extras = [i for i in range(1, L - 1) if i not in used_positions]
        random.shuffle(available_for_extras)
        for kp in available_for_extras:
            if extras_placed >= extras:
                break
            col = random.choice(colors)
            self.tiles[kp]["key_colors"].append(col)
            used_positions.add(kp)
            extras_placed += 1

        # Traps: cap to ensure solvable given health and kits: traps_on_path <= health - 1 + kits
        t_cap = max(0, self.health - 1 + self.kits)
        desired_traps = min(int(self.num_traps), t_cap)
        trap_positions = set()
        trap_candidates = [i for i in range(1, L - 1) if i not in door_positions]
        random.shuffle(trap_candidates)
        for tp in trap_candidates:
            if len(trap_positions) >= desired_traps:
                break
            trap_positions.add(tp)
        for tp in trap_positions:
            self.tiles[tp]["trap"] = True
            self.tiles[tp]["trap_cleared"] = False

        obs = self._get_instructions()
        return obs, {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False
        base_reward = -0.01
        extra_penalty = 0.0
        messages = []

        parsed = self._parse_action(action)
        self.last_parsed_action = parsed

        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} exactly."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        verb, arg = parsed

        if verb == "MOVE":
            direction = arg
            if direction == "FORWARD":
                next_pos = self.pos + 1
                if next_pos >= self.path_length:
                    messages.append("Cannot move beyond the corridor.")
                    extra_penalty -= 0.05
                else:
                    if self._is_locked(next_pos):
                        messages.append("Blocked by locked door ahead.")
                        extra_penalty -= 0.05
                    else:
                        self.pos = next_pos
                        if self._has_active_trap(self.pos):
                            self.tiles[self.pos]["trap_cleared"] = True
                            self.health -= 1
                            messages.append("You triggered a trap! Lost 1 health.")
                            extra_penalty -= 0.10
                        if self.pos == self.path_length - 1:
                            terminated = True
                            messages.append("Success! You reach the vault and claim the treasure.")
            elif direction == "BACK":
                prev_pos = self.pos - 1
                if prev_pos < 0:
                    messages.append("Cannot move beyond the corridor.")
                    extra_penalty -= 0.05
                else:
                    # Doors you already passed must be unlocked or behind you; moving back is always allowed
                    self.pos = prev_pos
                    if self._has_active_trap(self.pos):
                        self.tiles[self.pos]["trap_cleared"] = True
                        self.health -= 1
                        messages.append("You triggered a trap! Lost 1 health.")
                        extra_penalty -= 0.10
            else:
                messages.append("Unsupported action. Use MOVE FORWARD or MOVE BACK.")
                extra_penalty -= 0.05

        elif verb == "LOOK":
            newly = self._reveal_with_look(extra=2)
            if newly:
                messages.append("You scout ahead and behind: " + ", ".join(newly))
            else:
                messages.append("You gain no new information.")

        elif verb == "PICK":
            key_here = list(self.tiles[self.pos]["key_colors"])
            if not key_here:
                messages.append("No such key here.")
                extra_penalty -= 0.05
            else:
                if arg is None:
                    if len(key_here) == 1:
                        col = key_here[0]
                        self.inventory[col] = self.inventory.get(col, 0) + 1
                        self.tiles[self.pos]["key_colors"].remove(col)
                        messages.append(f"Picked up {col} key.")
                    else:
                        messages.append("Multiple keys here. Specify which key color to pick.")
                        extra_penalty -= 0.05
                else:
                    col_try = arg.title()
                    if col_try in key_here:
                        self.inventory[col_try] = self.inventory.get(col_try, 0) + 1
                        self.tiles[self.pos]["key_colors"].remove(col_try)
                        messages.append(f"Picked up {col_try} key.")
                    else:
                        messages.append("No such key here.")
                        extra_penalty -= 0.05

        elif verb == "USE":
            next_pos = self.pos + 1
            if next_pos >= self.path_length:
                messages.append("No door ahead.")
                extra_penalty -= 0.05
            else:
                if self.tiles[next_pos]["door_color"] is None:
                    messages.append("No door ahead.")
                    extra_penalty -= 0.05
                else:
                    needed = self.tiles[next_pos]["door_color"]
                    if arg is None:
                        messages.append("Specify which key color to use.")
                        extra_penalty -= 0.05
                    else:
                        col_try = arg.title()
                        if col_try != needed:
                            if self.inventory.get(col_try, 0) > 0:
                                messages.append("Wrong key color for the door ahead.")
                            else:
                                messages.append("No key of that color.")
                            extra_penalty -= 0.05
                        else:
                            if self.inventory.get(col_try, 0) <= 0:
                                messages.append("No key of that color.")
                                extra_penalty -= 0.05
                            else:
                                self.inventory[col_try] -= 1
                                self.unlocked_doors.add(next_pos)
                                messages.append(f"Unlocked the {col_try} door ahead.")

        elif verb == "DISARM":
            next_pos = self.pos + 1
            if next_pos >= self.path_length:
                messages.append("No trap to disarm ahead.")
                extra_penalty -= 0.05
            else:
                if not self._has_active_trap(next_pos):
                    messages.append("No trap to disarm ahead.")
                    extra_penalty -= 0.05
                else:
                    if self.kits <= 0:
                        messages.append("No disarm kits left.")
                        extra_penalty -= 0.05
                    else:
                        self.kits -= 1
                        self.tiles[next_pos]["trap_cleared"] = True
                        messages.append("Trap ahead disarmed.")

        elif verb == "WAIT":
            messages.append("You wait and observe your surroundings.")

        else:
            messages.append("Unsupported action.")
            extra_penalty -= 0.05

        if self.health <= 0 and not terminated:
            terminated = True
            messages.append("Failed! You ran out of health.")

        if not terminated and self.turn_count >= self.max_turns:
            terminated = True
            truncated = True
            messages.append(f"Reached max turns ({self.max_turns}).")

        obs = f"At turn {self.turn_count}: " + " ".join(messages) if messages else f"At turn {self.turn_count}: ..."
        reward = 0.0
        if "Success!" in obs:
            reward = 1.0
        elif "Failed!" in obs:
            reward = -1.0
        elif "invalid action format" in obs:
            reward = LanguageGameReward.format_error_reward
        else:
            reward = base_reward + extra_penalty

        return obs, reward, terminated, truncated, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Tuple[str, Optional[str]]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        tokens = re.split(r'\s+', content.strip())
        if not tokens:
            return None
        tok = [t.upper() for t in tokens]
        if tok[0] in ("MOVE", "GO", "STEP"):
            if len(tok) >= 2:
                if tok[1] in ("FORWARD", "F", "AHEAD"):
                    return ("MOVE", "FORWARD")
                if tok[1] in ("BACK", "B", "BACKWARD"):
                    return ("MOVE", "BACK")
            return ("MOVE", None)
        if tok[0] == "LOOK":
            return ("LOOK", None)
        if tok[0] == "WAIT":
            return ("WAIT", None)
        if tok[0] == "DISARM":
            return ("DISARM", None)
        if tok[0] == "PICK":
            if len(tok) >= 2:
                return ("PICK", tok[1])
            return ("PICK", None)
        if tok[0] == "USE":
            if len(tok) >= 2:
                return ("USE", tok[1])
            return ("USE", None)
        return ("UNSUPPORTED", None)

    def sample_random_action(self) -> str:
        choices = [
            "\\boxed{MOVE FORWARD}",
            "\\boxed{MOVE BACK}",
            "\\boxed{LOOK}",
            "\\boxed{DISARM}",
            "\\boxed{WAIT}",
            "\\boxed{PICK RED}",
            "\\boxed{USE BLUE}",
        ]
        return random.choice(choices)

    def _compute_visible_indices(self) -> set:
        vis = set()
        r = max(0, self.vision_ahead)
        for d in range(-r, r + 1):
            idx = self.pos + d
            if 0 <= idx < self.path_length:
                vis.add(idx)
        vis |= self.look_revealed
        return vis

    def _reveal_with_look(self, extra: int = 2) -> List[str]:
        newly = []
        ahead_end = min(self.path_length - 1, self.pos + self.vision_ahead + extra)
        behind_start = max(0, self.pos - self.vision_ahead - extra)
        for i in range(self.pos + self.vision_ahead + 1, ahead_end + 1):
            if i not in self.look_revealed:
                self.look_revealed.add(i)
                newly.append(f"{i}:{self._tile_brief(i)}")
        for i in range(behind_start, self.pos - self.vision_ahead):
            if i not in self.look_revealed:
                self.look_revealed.add(i)
                newly.append(f"{i}:{self._tile_brief(i)}")
        return newly

    def _tile_brief(self, i: int) -> str:
        t = self.tiles[i]
        parts = []
        if t["door_color"]:
            parts.append(f"door({t['door_color']})")
        if t["key_colors"]:
            ks = "/".join(t["key_colors"])
            parts.append(f"key({ks})")
        if t["trap"] and not t["trap_cleared"]:
            parts.append("trap")
        if not parts:
            return "empty"
        return "+".join(parts)

    def _is_locked(self, idx: int) -> bool:
        if idx < 0 or idx >= self.path_length:
            return True
        t = self.tiles[idx]
        if t["door_color"] is None:
            return False
        return idx not in self.unlocked_doors

    def _has_active_trap(self, idx: int) -> bool:
        if idx < 0 or idx >= self.path_length:
            return False
        t = self.tiles[idx]
        return bool(t["trap"] and not t["trap_cleared"])


class DungeonHeistEnvWithFeedback(DungeonHeistEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        text = obs.lower()

        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_boxed_format"
            hint = "Use \\boxed{...} around exactly one action, e.g., \\boxed{MOVE FORWARD}."
        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["turn_limit"] = self.max_turns
            hint = "Plan shorter routes: use LOOK sparingly and avoid unnecessary WAIT or backtracking."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["got"] = action
            hint = "Allowed verbs: MOVE FORWARD/BACK, LOOK, PICK [COLOR], USE [COLOR], DISARM, WAIT."
        elif ("blocked by locked door" in text or
              "no such key here" in text or
              "multiple keys here. specify which key color to pick" in text or
              "no trap to disarm ahead" in text or
              "no disarm kits left" in text or
              "cannot move beyond the corridor" in text or
              "no door ahead" in text or
              "specify which key color to use" in text):
            error_type = "ProtocolViolation"
            if "blocked by locked door" in text:
                error_detail["violation"] = "tried_to_move_into_locked_door"
                hint = "Use LOOK to identify the door color, then USE that color if you have the key."
            elif "no such key here" in text:
                error_detail["violation"] = "tried_to_pick_missing_key"
                hint = "Check the room description; only pick a key if it’s listed at your position."
            elif "multiple keys here" in text:
                error_detail["violation"] = "ambiguous_pick"
                hint = "Specify the key color explicitly, e.g., \\boxed{PICK RED}."
            elif "no trap to disarm ahead" in text:
                error_detail["violation"] = "disarm_without_trap"
                hint = "Use LOOK to confirm a trap exists in the next room before using DISARM."
            elif "no disarm kits left" in text:
                error_detail["violation"] = "no_kits"
                hint = "Avoid stepping on traps, or plan earlier disarms. LOOK to route around when possible."
            elif "cannot move beyond the corridor" in text:
                error_detail["violation"] = "move_out_of_bounds"
                hint = "You are at an end. Try the other direction or use another action."
            elif "no door ahead" in text:
                error_detail["violation"] = "use_without_door"
                hint = "Use LOOK to confirm a locked door is immediately ahead before using a key."
            elif "specify which key color to use" in text:
                error_detail["violation"] = "missing_color_for_use"
                hint = "Include a color, e.g., \\boxed{USE BLUE}."
        elif "wrong key color for the door ahead" in text or "no key of that color" in text:
            error_type = "WrongDecision"
            next_idx = self.pos + 1
            expected = None
            if 0 <= next_idx < self.path_length:
                expected = self.tiles[next_idx]["door_color"]
            error_detail["expected_color"] = expected
            if self.last_parsed_action and self.last_parsed_action[0] == "USE":
                error_detail["got"] = (self.last_parsed_action[1].title() if self.last_parsed_action[1] else None)
            else:
                error_detail["got"] = None
            if expected:
                hint = f"Use \\boxed{{USE {expected.upper()}}} if you have that key; otherwise go find it first."
            else:
                hint = "Confirm the door color ahead with LOOK, then use the matching key."
        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None
        elif "failed! you ran out of health" in text:
            error_type = "WrongDecision"
            error_detail["outcome"] = "death"
            hint = "Avoid stepping on traps: DISARM before moving forward or adjust route/backtrack after LOOK."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            state_info = {
                "position": getattr(self, "pos", None),
                "health": getattr(self, "health", None),
                "kits": getattr(self, "kits", None),
                "distance_to_goal": (self.path_length - 1 - self.pos) if hasattr(self, "path_length") else None,
            }
            # next tile snapshot
            nxt = self.pos + 1
            if 0 <= nxt < getattr(self, "path_length", 0):
                t = self.tiles[nxt]
                state_info["next_tile"] = {
                    "door_color": t["door_color"],
                    "trap": bool(t["trap"] and not t["trap_cleared"]),
                    "keys_here": list(t["key_colors"]),
                }
            diagnostic["state"] = state_info
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": "Start with \\boxed{LOOK} to reveal more rooms, then move forward cautiously.",
            "turn": 0,
            "state": {
                "position": getattr(self, "pos", None),
                "health": getattr(self, "health", None),
                "kits": getattr(self, "kits", None),
                "distance_to_goal": (self.path_length - 1 - self.pos),
            },
        }
        return obs, info