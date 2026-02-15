from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class ChronoWeaverEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 100,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 100

        self.complexity_params = {
            # Number of timeline slots to schedule actions; larger = more combinatorial choices = harder
            "num_slots": (3, 9),
            # Number of tiles in hand; larger = more branching and search depth = harder
            "hand_size": (4, 12),
            # Ordered goal length; longer sequences are harder to satisfy exactly
            "goal_length": (2, 5),
            # REVERSED: less gold makes planning tighter and harder
            "starting_gold": (6, 2),
            # REVERSED: less ore makes planning tighter and harder
            "starting_ore": (5, 1),
            # REVERSED: less intel makes planning tighter and harder
            "starting_intel": (4, 1),
            # Per-tile prerequisites count; more resource types per tile increases constraint complexity
            "prereq_level": (1, 2),
            # Cost scale; higher costs consume resources faster and increase difficulty
            "cost_scale": (1, 3),
            # Variety of distinct event types in the deck; more types increase state space and misplacement risk
            "deck_variety": (4, 8),
        }

        self.param_variance = {
            "num_slots": 1,
            "hand_size": 1,
            "goal_length": 1,
            "starting_gold": 1,
            "starting_ore": 1,
            "starting_intel": 1,
            "prereq_level": 0,
            "cost_scale": 0,
            "deck_variety": 1,
        }

        self.num_slots: int = 0
        self.hand_size: int = 0
        self.goal_length: int = 0
        self.starting_gold: int = 0
        self.starting_ore: int = 0
        self.starting_intel: int = 0
        self.prereq_level: int = 0
        self.cost_scale: int = 0
        self.deck_variety: int = 0

        self.available_event_types = [
            "SCOUT",
            "FORGE",
            "TRADE",
            "SNEAK",
            "BATTLE",
            "DEFEND",
            "RESEARCH",
            "BUILD",
            "NAVIGATE",
            "HUNT",
            "CRAFT",
            "EMBASSY",
        ]

        self.turn_count: int = 0
        self.slots: Dict[int, Optional[str]] = {}
        self.hand: Dict[str, Dict[str, Any]] = {}
        self.event_by_tile: Dict[str, str] = {}
        self.cost_by_tile: Dict[str, Dict[str, int]] = {}
        self.goal_sequence: list = []
        self.next_expected_index: int = 0

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
            # Clamp handling reversed ranges too
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        functions_desc = (
            "- place(tile=\"TILE_NAME\", slot=INT): Put a tile from your hand into an empty timeline slot.\n"
            "- swap(a=INT, b=INT): Swap the tiles currently in two occupied slots.\n"
            "- erase(slot=INT): Remove the tile from a slot and return it to your hand.\n"
            "- run(): Execute the timeline from slot 1 to slot N, consuming resources and triggering events; ends the episode."
        )
        return (
            "ChronoWeaver: Arrange action tiles into a timeline to trigger the target events in exact order.\n"
            "Goal: Execute tiles so that the event sequence matches the target list from first to last.\n"
            "Resources: gold, ore, intel. Each tile has a cost in these resources; if insufficient at execution, the tile is skipped.\n"
            "Rules:\n"
            "- You may place tiles only from your hand into empty slots.\n"
            "- You may swap only tiles in occupied slots; erase returns a tile to your hand.\n"
            "- The run() function executes the timeline sequentially and verifies the goal; success terminates with reward 1.0.\n"
            "Available functions:\n"
            f"{functions_desc}\n"
            "Format your action exactly as:\n"
            "<action>[place(tile=\"SCOUT-1\", slot=1)]</action>\n"
            "or <action>[swap(a=1, b=2)]</action>\n"
            "or <action>[erase(slot=3)]</action>\n"
            "or <action>[run()]</action>\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        hand_lines = []
        for tname, meta in sorted(self.hand.items()):
            cost = meta["cost"]
            cost_str = ", ".join(f"{k}:{v}" for k, v in cost.items()) if cost else "free"
            hand_lines.append(f"{tname} -> event:{meta['event']} cost({cost_str})")
        slots_lines = []
        for s in range(1, self.num_slots + 1):
            tile = self.slots.get(s)
            slots_lines.append(f"slot {s}: {tile if tile else 'EMPTY'}")
        next_target = (
            self.goal_sequence[self.next_expected_index]
            if self.next_expected_index < len(self.goal_sequence)
            else "DONE"
        )
        return (
            "State:\n"
            f"- Resources: gold={self.starting_gold}, ore={self.starting_ore}, intel={self.starting_intel}\n"
            f"- Timeline slots ({self.num_slots}): " + " | ".join(slots_lines) + "\n"
            f"- Hand ({len(self.hand)} tiles):\n" + ("\n".join(hand_lines) if hand_lines else "EMPTY") + "\n"
            f"- Target sequence ({len(self.goal_sequence)}): " + " -> ".join(self.goal_sequence) + "\n"
            f"- Next expected: {next_target}\n"
            'Enter your action: <action>[place(tile="TILE_NAME", slot=INT)]</action>, '
            '<action>[swap(a=INT, b=INT)]</action>, <action>[erase(slot=INT)]</action>, or <action>[run()]</action>'
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        self.turn_count = 0
        self.slots = {i: None for i in range(1, self.num_slots + 1)}
        self.hand = {}
        self.event_by_tile = {}
        self.cost_by_tile = {}
        self.goal_sequence = []
        self.next_expected_index = 0

        event_types = random.sample(self.available_event_types, self.deck_variety)
        # Assign base costs per event type
        base_costs: Dict[str, Dict[str, int]] = {}
        for evt in event_types:
            resources = ["gold", "ore", "intel"]
            chosen = random.sample(resources, self.prereq_level)
            cost = {r: random.randint(1, self.cost_scale) for r in chosen}
            base_costs[evt] = cost

        # Build hand with unique tile names and properties
        event_pool = [random.choice(event_types) for _ in range(self.hand_size)]
        counts: Dict[str, int] = {evt: 0 for evt in event_types}
        for evt in event_pool:
            counts[evt] += 1
            tname = f"{evt}-{counts[evt]}"
            self.hand[tname] = {"event": evt, "cost": dict(base_costs[evt])}
            self.event_by_tile[tname] = evt
            self.cost_by_tile[tname] = dict(base_costs[evt])

        # Prepare selection helper
        evt_to_tiles: Dict[str, list] = {evt: [] for evt in event_types}
        for tname, meta in self.hand.items():
            evt_to_tiles[meta["event"]].append(tname)

        # Choose a solvable goal sequence under current budgets by greedy fit; force-free tiles if needed
        budgets = {"gold": self.starting_gold, "ore": self.starting_ore, "intel": self.starting_intel}
        used = {"gold": 0, "ore": 0, "intel": 0}
        sorted_evts = sorted(event_types, key=lambda e: sum(base_costs[e].values()) if base_costs[e] else 0)
        zeroed_tiles = set()

        i = 0
        while len(self.goal_sequence) < self.goal_length and i < len(sorted_evts) * 3:  # allow multiple passes
            i += 1
            random.shuffle(sorted_evts)
            for evt in sorted_evts:
                if len(self.goal_sequence) >= self.goal_length:
                    break
                # Skip if no tiles of this event
                if not evt_to_tiles.get(evt):
                    continue
                cost = base_costs[evt]
                fits = True
                for r, c in cost.items():
                    if used[r] + c > budgets[r]:
                        fits = False
                        break
                if fits:
                    self.goal_sequence.append(evt)
                    for r, c in cost.items():
                        used[r] += c
                else:
                    # Try to force one specific tile of this event to be free to ensure solvability
                    # Only if we still need more events
                    # Choose a tile not yet zeroed
                    candidates = [t for t in evt_to_tiles[evt] if t not in zeroed_tiles]
                    if candidates and len(self.goal_sequence) < self.goal_length:
                        chosen_tile = random.choice(candidates)
                        self.hand[chosen_tile]["cost"] = {}
                        self.cost_by_tile[chosen_tile] = {}
                        zeroed_tiles.add(chosen_tile)
                        self.goal_sequence.append(evt)
                        # No resource consumption added
        # If still not enough, pad with any event that has at least one tile, making that tile free
        while len(self.goal_sequence) < self.goal_length:
            evt = random.choice(event_types)
            candidates = [t for t in evt_to_tiles.get(evt, []) if t not in zeroed_tiles]
            if not candidates:
                # create a virtual free tile by zeroing an existing one anyway
                all_tiles = list(self.hand.keys())
                chosen_tile = random.choice(all_tiles) if all_tiles else None
            else:
                chosen_tile = random.choice(candidates)
            if chosen_tile:
                self.hand[chosen_tile]["cost"] = {}
                self.cost_by_tile[chosen_tile] = {}
                zeroed_tiles.add(chosen_tile)
                self.goal_sequence.append(self.event_by_tile[chosen_tile])
            else:
                break  # no tiles at all; rare edge case

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = (
                f"At turn {self.turn_count}, invalid action format. "
                "Use <action>[place(tile=\"NAME\", slot=INT)]</action>, <action>[swap(a=INT, b=INT)]</action>, "
                "<action>[erase(slot=INT)]</action>, or <action>[run()]</action>."
            )
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed["name"]
        params = parsed["parameters"]

        if name not in {"place", "swap", "erase", "run"}:
            obs = f"Unsupported action '{name}'. Allowed: place, swap, erase, run."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        reward = 0.0
        if name == "place":
            tile = params.get("tile")
            slot = params.get("slot")
            if not isinstance(tile, str) or not isinstance(slot, int):
                obs = "Protocol violation: place requires tile(str) and slot(int)."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if slot < 1 or slot > self.num_slots:
                obs = f"Protocol violation: slot {slot} out of range 1..{self.num_slots}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.slots[slot] is not None:
                obs = f"Protocol violation: slot {slot} is not empty."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if tile not in self.hand:
                obs = f"Protocol violation: tile '{tile}' not in hand."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.slots[slot] = tile
            # remove from hand
            meta = self.hand.pop(tile)
            # reflect state
            obs = f"Placed {tile} (event {meta['event']}) at slot {slot}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if name == "swap":
            a = params.get("a")
            b = params.get("b")
            if not isinstance(a, int) or not isinstance(b, int):
                obs = "Protocol violation: swap requires a(int), b(int)."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if a < 1 or a > self.num_slots or b < 1 or b > self.num_slots:
                obs = f"Protocol violation: swap indices out of range 1..{self.num_slots}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.slots[a] is None or self.slots[b] is None:
                obs = "Protocol violation: both slots must be occupied to swap."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.slots[a], self.slots[b] = self.slots[b], self.slots[a]
            obs = f"Swapped slots {a} and {b}."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if name == "erase":
            slot = params.get("slot")
            if not isinstance(slot, int):
                obs = "Protocol violation: erase requires slot(int)."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if slot < 1 or slot > self.num_slots:
                obs = f"Protocol violation: slot {slot} out of range 1..{self.num_slots}."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            tile = self.slots.get(slot)
            if tile is None:
                obs = "Protocol violation: slot is already empty."
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.slots[slot] = None
            # return tile to hand
            self.hand[tile] = {"event": self.event_by_tile[tile], "cost": dict(self.cost_by_tile[tile])}
            obs = f"Erased slot {slot}; returned {tile} to hand."
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        if name == "run":
            gold = self.starting_gold
            ore = self.starting_ore
            intel = self.starting_intel
            pointer = 0
            log = []
            for s in range(1, self.num_slots + 1):
                tile = self.slots.get(s)
                if not tile:
                    continue
                cost = self.cost_by_tile[tile]
                can_pay = True
                if cost:
                    if cost.get("gold", 0) > gold:
                        can_pay = False
                    if cost.get("ore", 0) > ore:
                        can_pay = False
                    if cost.get("intel", 0) > intel:
                        can_pay = False
                if not can_pay:
                    log.append(f"slot {s}: {tile} skipped (insufficient resources)")
                    continue
                # Pay
                gold -= cost.get("gold", 0)
                ore -= cost.get("ore", 0)
                intel -= cost.get("intel", 0)
                evt = self.event_by_tile[tile]
                log.append(f"slot {s}: {tile} executed -> {evt}")
                if pointer < len(self.goal_sequence) and evt == self.goal_sequence[pointer]:
                    pointer += 1
            if pointer >= len(self.goal_sequence):
                obs = "Success! Goal sequence completed. " + \
                      f"Triggered events: [{' | '.join(e for e in [entry.split('->')[-1].strip() for entry in log if 'executed' in entry])}] " + \
                      f"Resources left: gold={gold}, ore={ore}, intel={intel}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                next_needed = self.goal_sequence[pointer] if pointer < len(self.goal_sequence) else "DONE"
                obs = (
                    "Failed! Goal not satisfied after execution. "
                    f"Progress {pointer}/{len(self.goal_sequence)}; next expected was {next_needed}. "
                    f"Resources left: gold={gold}, ore={ore}, intel={intel}. "
                    "Log: " + " | ".join(log)
                )
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"At turn {self.turn_count}, no state change."
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        from gem.utils.parsing import extract_action_parameters
        content = extract_action_parameters(action)
        if not content:
            return None
        content = content.strip()
        if not (content.startswith("[") and content.endswith("]")):
            return None
        func_call_str = content[1:-1].strip()
        func_pattern = re.compile(r"^(\w+)\((.*)\)$", re.DOTALL)
        func_match = func_pattern.match(func_call_str)
        if not func_match:
            return None
        func_name = func_match.group(1)
        params_str = func_match.group(2).strip()
        parameters: Dict[str, Any] = {}
        if params_str:
            param_pairs = re.findall(r"(\w+)\s*=\s*([^,]+?)(?:,|$)", params_str)
            for key, value in param_pairs:
                v = value.strip()
                try:
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        parameters[key] = v[1:-1]
                    elif v.lower() in {"true", "false"}:
                        parameters[key] = v.lower() == "true"
                    elif re.match(r"^-?\d+$", v):
                        parameters[key] = int(v)
                    elif re.match(r"^-?\d+\.\d+$", v):
                        parameters[key] = float(v)
                    else:
                        parameters[key] = v
                except Exception:
                    parameters[key] = v
        return {"name": func_name, "parameters": parameters}

    def sample_random_action(self) -> str:
        # Prefer a meaningful example based on current state
        empty_slots = [s for s in range(1, self.num_slots + 1) if self.slots.get(s) is None]
        if self.hand and empty_slots:
            tile = random.choice(list(self.hand.keys()))
            slot = random.choice(empty_slots)
            return f'<action>[place(tile="{tile}", slot={slot})]</action>'
        elif sum(1 for s in self.slots.values() if s) >= 2:
            filled = [i for i in range(1, self.num_slots + 1) if self.slots.get(i)]
            a, b = random.sample(filled, 2)
            return f'<action>[swap(a={a}, b={b})]</action>'
        else:
            return "<action>[run()]</action>"


class ChronoWeaverEnvWithFeedback(ChronoWeaverEnv):
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
            error_detail["issue"] = "bad_action_tags_or_syntax"
            hint = 'Use <action>[place(tile="NAME", slot=INT)]</action>, <action>[swap(a=INT, b=INT)]</action>, <action>[erase(slot=INT)]</action>, or <action>[run()]</action>.'

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["supported"] = ["place", "swap", "erase", "run"]
            hint = "Choose one of the supported functions and match parameter types."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "out of range" in text:
                error_detail["violation"] = "slot_index_out_of_range"
                hint = "Select slot within the displayed range. Check State section for slot count."
            elif "not empty" in text:
                error_detail["violation"] = "placing_into_occupied_slot"
                hint = "Erase or swap to free the slot, or choose an empty slot."
            elif "not in hand" in text:
                error_detail["violation"] = "tile_not_available"
                hint = "Use a tile listed under Hand. Copy tile name exactly."
            elif "both slots must be occupied" in text:
                error_detail["violation"] = "swap_requires_two_occupied_slots"
                hint = "Place tiles into both slots first, then swap."
            else:
                error_detail["violation"] = "invalid_parameters_or_state"
                hint = "Verify parameter types and current state; follow function signatures."

        elif "reached max turns" in text or "max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan faster: place the needed tiles for the first target events, then call run()."

        elif "failed!" in text:
            error_type = "WrongDecision"
            # Extract next expected event if present
            m = re.search(r"next expected was (\w+)", text)
            expected = m.group(1).upper() if m else None
            error_detail["expected_next_event"] = expected
            error_detail["progress"] = getattr(self, "next_expected_index", None)
            hint = "Place tiles whose event matches the target order early in the timeline, and ensure their costs fit your resources."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "next_target": (
                    self.goal_sequence[self.next_expected_index]
                    if self.next_expected_index < len(self.goal_sequence)
                    else "DONE"
                ),
                "resources": {
                    "gold": self.starting_gold,
                    "ore": self.starting_ore,
                    "intel": self.starting_intel,
                },
                "slots": {i: self.slots.get(i) for i in range(1, self.num_slots + 1)},
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
            "hint": "Start by placing a tile matching the first target event into an early slot, then arrange the rest before calling run().",
            "turn": 0,
            "state": {
                "next_target": self.goal_sequence[0] if self.goal_sequence else "DONE",
                "resources": {
                    "gold": self.starting_gold,
                    "ore": self.starting_ore,
                    "intel": self.starting_intel,
                },
            },
        }
        return obs, info