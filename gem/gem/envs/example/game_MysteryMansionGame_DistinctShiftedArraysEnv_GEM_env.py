from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class MysteryMansionGameEnv(Env):
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
            # Number of suspects: more options = harder
            "num_suspects": (4, 12),
            # Number of weapons: more options = harder
            "num_weapons": (3, 10),
            # Number of rooms: more options = harder
            "num_rooms": (4, 12),
            # REVERSED: fewer full/attribute reveals allowed = harder
            "max_reveals": (4, 1),
            # REVERSED: fewer comparisons allowed = harder
            "max_comparisons": (6, 2),
            # REVERSED: fewer property tests allowed = harder
            "max_tests": (5, 2),
        }

        self.param_variance = {
            "num_suspects": 1,
            "num_weapons": 1,
            "num_rooms": 1,
            "max_reveals": 1,
            "max_comparisons": 1,
            "max_tests": 1,
        }

        self.num_suspects: int = 0
        self.num_weapons: int = 0
        self.num_rooms: int = 0
        self.max_reveals: int = 0
        self.max_comparisons: int = 0
        self.max_tests: int = 0

        self.turn_count: int = 0
        self.suspects: Dict[str, Dict[str, Any]] = {}
        self.weapons: Dict[str, Dict[str, Any]] = {}
        self.rooms: Dict[str, Dict[str, Any]] = {}

        self.truth: Dict[str, str] = {}
        self.reveals_used: int = 0
        self.comparisons_used: int = 0
        self.tests_used: int = 0
        self.revealed_info: Dict[str, Any] = {}
        self.last_result: str = ""

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
            # Clamp across reversed ranges too
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Mystery Mansion: Deduce the culprit, weapon, and room.\n"
            "Goal: Submit the exact triple (culprit, weapon, room).\n"
            "You can:\n"
            "- LIST SUSPECTS | LIST WEAPONS | LIST ROOMS\n"
            "- REVEAL <KEY> where KEY ∈ {SUSPECT, WEAPON, ROOM, SUSPECT_INITIALS, WEAPON_TYPE, ROOM_FLOOR}\n"
            f"  (Reveal budget: {self.max_reveals})\n"
            "- TEST SUSPECT=<Name> | TEST WEAPON=<Name> | TEST ROOM=<Name> (uses test budget)\n"
            f"  (Test budget: {self.max_tests})\n"
            "- SCORE suspect=<Name>; weapon=<Name>; room=<Name> → returns match signature (suspect/weapon/room)\n"
            "- COMPARE <triple1> | <triple2> where triple is 'suspect=...; weapon=...; room=...' (uses comparison budget)\n"
            f"  (Comparison budget: {self.max_comparisons})\n"
            "- SUBMIT suspect=<Name>; weapon=<Name>; room=<Name> (terminal)\n"
            "Format all actions as \\boxed{...}\n"
            f"Example: {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Turn {self.turn_count}\n"
            f"Reveals left: {max(0, self.max_reveals - self.reveals_used)} | "
            f"Comparisons left: {max(0, self.max_comparisons - self.comparisons_used)} | "
            f"Tests left: {max(0, self.max_tests - self.tests_used)}\n"
            f"Known clues: {self.revealed_info if self.revealed_info else 'None'}\n"
            "Enter your action using \\boxed{...}.\n"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        suspect_pool = [
            "Green", "Mustard", "Scarlet", "Peacock", "Plum", "White",
            "Azure", "Crimson", "Olive", "Amber", "Violet", "Onyx", "Cyan", "Sable"
        ]
        weapon_pool = [
            "Candlestick", "Knife", "Lead Pipe", "Rope", "Wrench",
            "Poison", "Dagger", "Revolver", "Trophy", "Fire Poker"
        ]
        room_pool = [
            "Kitchen", "Study", "Library", "Hall", "Lounge", "Conservatory",
            "Dining Room", "Ballroom", "Cellar", "Billiard Room", "Courtyard", "Garden"
        ]

        suspects = random.sample(suspect_pool, self.num_suspects)
        weapons = random.sample(weapon_pool, self.num_weapons)
        rooms = random.sample(room_pool, self.num_rooms)

        self.suspects = {}
        for name in suspects:
            initials = (name[:2]).upper()
            self.suspects[name] = {
                "initials": initials,
                "alibi": random.choice(["strong", "weak"]),
                "motive": random.randint(1, 10),
            }

        self.weapons = {}
        weapon_type_map = {
            "Candlestick": "blunt",
            "Knife": "sharp",
            "Lead Pipe": "blunt",
            "Rope": "strangle",
            "Wrench": "blunt",
            "Poison": "poison",
            "Dagger": "sharp",
            "Revolver": "projectile",
            "Trophy": "blunt",
            "Fire Poker": "blunt",
        }
        for w in weapons:
            self.weapons[w] = {
                "type": weapon_type_map.get(w, "unknown"),
                "weight": random.choice(["light", "medium", "heavy"]),
            }

        self.rooms = {}
        for r in rooms:
            self.rooms[r] = {
                "floor": random.choice([1, 2, 3]),
                "indoor": True if r != "Garden" and r != "Courtyard" else False,
            }

        self.truth = {
            "suspect": random.choice(suspects),
            "weapon": random.choice(weapons),
            "room": random.choice(rooms),
        }

        self.turn_count = 0
        self.reveals_used = 0
        self.comparisons_used = 0
        self.tests_used = 0
        self.revealed_info = {}
        self.last_result = ""

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        a_type = parsed.get("type")
        reward = 0.0

        if a_type == "list":
            category = parsed.get("category")
            if category == "SUSPECTS":
                obs = f"Suspects: {sorted(list(self.suspects.keys()))}"
            elif category == "WEAPONS":
                obs = f"Weapons: {sorted(list(self.weapons.keys()))}"
            elif category == "ROOMS":
                obs = f"Rooms: {sorted(list(self.rooms.keys()))}"
            else:
                obs = f"Unsupported action: LIST {category}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.last_result = obs
            # Continue
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "reveal":
            key = parsed.get("key")
            if self.reveals_used >= self.max_reveals:
                obs = "Protocol violation: Reveal budget exceeded."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            self.reveals_used += 1
            if key == "SUSPECT":
                val = self.truth["suspect"]
                self.revealed_info["suspect"] = val
                obs = f"Reveal: culprit is {val}."
            elif key == "WEAPON":
                val = self.truth["weapon"]
                self.revealed_info["weapon"] = val
                obs = f"Reveal: weapon is {val}."
            elif key == "ROOM":
                val = self.truth["room"]
                self.revealed_info["room"] = val
                obs = f"Reveal: room is {val}."
            elif key == "SUSPECT_INITIALS":
                val = self.suspects[self.truth["suspect"]]["initials"]
                self.revealed_info["suspect_initials"] = val
                obs = f"Reveal: suspect initials are {val}."
            elif key == "WEAPON_TYPE":
                val = self.weapons[self.truth["weapon"]]["type"]
                self.revealed_info["weapon_type"] = val
                obs = f"Reveal: weapon type is {val}."
            elif key == "ROOM_FLOOR":
                val = self.rooms[self.truth["room"]]["floor"]
                self.revealed_info["room_floor"] = val
                obs = f"Reveal: room floor is {val}."
            else:
                obs = f"Unsupported action: REVEAL {key}."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            self.last_result = obs
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "test":
            target = parsed.get("target")
            value = parsed.get("value")
            if self.tests_used >= self.max_tests:
                obs = "Protocol violation: Test budget exceeded."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            # Validate value is in known sets
            valid = False
            if target == "SUSPECT":
                valid = value in self.suspects
            elif target == "WEAPON":
                valid = value in self.weapons
            elif target == "ROOM":
                valid = value in self.rooms
            if not valid:
                obs = f"Protocol violation: Unknown {target.lower()} '{value}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            self.tests_used += 1
            truth_val = self.truth[target.lower()]
            result = (truth_val == value)
            obs = f"Test: {target}={value} → {'TRUE' if result else 'FALSE'}."
            self.last_result = obs
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "score":
            tri = parsed.get("triple", {})
            s = tri.get("suspect", "")
            w = tri.get("weapon", "")
            r = tri.get("room", "")
            # Validate membership
            if s not in self.suspects or w not in self.weapons or r not in self.rooms:
                obs = "Protocol violation: SCORE triple contains unknown names."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            sm = 1 if s == self.truth["suspect"] else 0
            wm = 1 if w == self.truth["weapon"] else 0
            rm = 1 if r == self.truth["room"] else 0
            obs = f"SCORE: suspect_match={sm}, weapon_match={wm}, room_match={rm}."
            self.last_result = obs
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "compare":
            if self.comparisons_used >= self.max_comparisons:
                obs = "Protocol violation: Comparison budget exceeded."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            tri1 = parsed.get("triple1", {})
            tri2 = parsed.get("triple2", {})
            s1, w1, r1 = tri1.get("suspect", ""), tri1.get("weapon", ""), tri1.get("room", "")
            s2, w2, r2 = tri2.get("suspect", ""), tri2.get("weapon", ""), tri2.get("room", "")

            # Validate
            for name, coll in [(s1, self.suspects), (w1, self.weapons), (r1, self.rooms),
                               (s2, self.suspects), (w2, self.weapons), (r2, self.rooms)]:
                if name not in coll:
                    obs = "Protocol violation: COMPARE triple contains unknown names."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            score1 = (1 if s1 == self.truth["suspect"] else 0) + \
                     (1 if w1 == self.truth["weapon"] else 0) + \
                     (1 if r1 == self.truth["room"] else 0)
            score2 = (1 if s2 == self.truth["suspect"] else 0) + \
                     (1 if w2 == self.truth["weapon"] else 0) + \
                     (1 if r2 == self.truth["room"] else 0)
            self.comparisons_used += 1
            if score1 > score2:
                obs = f"COMPARE: first triple is closer ({score1} vs {score2})."
            elif score2 > score1:
                obs = f"COMPARE: second triple is closer ({score2} vs {score1})."
            else:
                obs = f"COMPARE: both triples equally close ({score1} vs {score2})."
            self.last_result = obs
            if self.turn_count >= self.max_turns:
                return f"Reached max turns ({self.max_turns})", 0.0, True, True, {"suffix": self.get_task_suffix()}
            return obs, reward, False, False, {"suffix": self.get_task_suffix()}

        elif a_type == "submit":
            tri = parsed.get("triple", {})
            s = tri.get("suspect", "")
            w = tri.get("weapon", "")
            r = tri.get("room", "")
            # Validate candidate names first
            if s not in self.suspects or w not in self.weapons or r not in self.rooms:
                obs = "Failed! Submission contains unknown names."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            correct = (s == self.truth["suspect"] and w == self.truth["weapon"] and r == self.truth["room"])
            if correct:
                obs = f"Success! Correct solution: {s} with {w} in the {r}."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Failed! Incorrect solution."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action: {a_type}."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action:
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        m = pattern.search(action)
        if not m:
            return None
        inner = m.group(1).strip()

        def parse_triple(text: str) -> Optional[Dict[str, str]]:
            s_pat = re.compile(r'suspect\s*=\s*(.*?);', re.IGNORECASE)
            w_pat = re.compile(r'weapon\s*=\s*(.*?);', re.IGNORECASE)
            r_pat = re.compile(r'room\s*=\s*(.*)', re.IGNORECASE)
            ms = s_pat.search(text)
            mw = w_pat.search(text)
            mr = r_pat.search(text)
            if not (ms and mw and mr):
                return None
            s = ms.group(1).strip()
            w = mw.group(1).strip()
            r = mr.group(1).strip()
            return {"suspect": s, "weapon": w, "room": r}

        u = inner.strip()
        if re.match(r'^LIST\s+\w+', u, re.IGNORECASE):
            cat = re.sub(r'^LIST\s+', '', u, flags=re.IGNORECASE).strip().upper()
            return {"type": "list", "category": cat}

        if re.match(r'^REVEAL\s+[A-Z_]+$', u, re.IGNORECASE):
            key = re.sub(r'^REVEAL\s+', '', u, flags=re.IGNORECASE).strip().upper()
            return {"type": "reveal", "key": key}

        mtest = re.match(r'^TEST\s+(SUSPECT|WEAPON|ROOM)\s*=\s*(.+)$', u, re.IGNORECASE)
        if mtest:
            target = mtest.group(1).upper()
            value = mtest.group(2).strip()
            return {"type": "test", "target": target, "value": value}

        if re.match(r'^SCORE\s+', u, re.IGNORECASE):
            triple_text = re.sub(r'^SCORE\s+', '', u, flags=re.IGNORECASE).strip()
            tri = parse_triple(triple_text)
            if tri:
                return {"type": "score", "triple": tri}
            else:
                return {"type": "unsupported"}

        if re.match(r'^COMPARE\s+', u, re.IGNORECASE):
            cmp_text = re.sub(r'^COMPARE\s+', '', u, flags=re.IGNORECASE).strip()
            parts = re.split(r'\|\s*', cmp_text)
            if len(parts) != 2:
                return {"type": "unsupported"}
            tri1 = parse_triple(parts[0].strip())
            tri2 = parse_triple(parts[1].strip())
            if tri1 and tri2:
                return {"type": "compare", "triple1": tri1, "triple2": tri2}
            else:
                return {"type": "unsupported"}

        if re.match(r'^SUBMIT\s+', u, re.IGNORECASE):
            triple_text = re.sub(r'^SUBMIT\s+', '', u, flags=re.IGNORECASE).strip()
            tri = parse_triple(triple_text)
            if tri:
                return {"type": "submit", "triple": tri}
            else:
                return {"type": "unsupported"}

        return {"type": "unsupported"}

    def sample_random_action(self) -> str:
        if self.suspects and self.weapons and self.rooms:
            s = random.choice(list(self.suspects.keys()))
            w = random.choice(list(self.weapons.keys()))
            r = random.choice(list(self.rooms.keys()))
            return f"\\boxed{{SCORE suspect={s}; weapon={w}; room={r}}}"
        return "\\boxed{LIST SUSPECTS}"


class MysteryMansionGameEnvWithFeedback(MysteryMansionGameEnv):
        def __init__(self, feedback_level: int = 2, **kwargs):
            self.feedback_level = feedback_level
            super().__init__(**kwargs)

        def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
            obs, reward, terminated, truncated, info = super().step(action)
            text = obs.lower()
            error_type = "OK"
            error_detail = {}
            hint = None

            if "invalid action format" in text or "use \\boxed" in text:
                error_type = "FormatError"
                error_detail["issue"] = "missing_boxed_format"
                hint = "Wrap your command in \\boxed{...} and follow the syntax shown in instructions."
            elif "unsupported action" in text:
                error_type = "UnsupportedAction"
                error_detail["issue"] = "unknown_or_malformed_command"
                hint = "Use one of: LIST, REVEAL, TEST, SCORE, COMPARE, SUBMIT."
            elif "protocol violation" in text and "reveal budget exceeded" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "reveal_budget_exceeded"
                hint = "Stop using REVEAL; switch to SCORE or TEST to gather information."
            elif "protocol violation" in text and "comparison budget exceeded" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "comparison_budget_exceeded"
                hint = "Avoid COMPARE; use SCORE repeatedly to evaluate candidates."
            elif "protocol violation" in text and "test budget exceeded" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "test_budget_exceeded"
                hint = "Use SCORE instead of TEST to check candidate components."
            elif "protocol violation" in text and "unknown" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "unknown_entity"
                hint = "List options first (e.g., LIST SUSPECTS) and use exact names."
            elif "failed! incorrect solution" in text or "failed! submission contains unknown names" in text:
                error_type = "WrongDecision"
                error_detail["expected"] = {
                    "suspect": self.truth.get("suspect"),
                    "weapon": self.truth.get("weapon"),
                    "room": self.truth.get("room"),
                }
                hint = "Use SCORE to see which components match; combine with REVEAL or TEST before submitting."
            elif "reached max turns" in text:
                error_type = "Timeout"
                error_detail["limit"] = self.max_turns
                hint = "Act earlier: LIST options, REVEAL a key if budget allows, then use SCORE to narrow and SUBMIT."

            diagnostic = {"error_type": error_type}
            if self.feedback_level >= 1:
                diagnostic["error_detail"] = error_detail
                diagnostic["turn"] = getattr(self, "turn_count", None)
                diagnostic["budgets"] = {
                    "reveals_left": max(0, self.max_reveals - self.reveals_used),
                    "comparisons_left": max(0, self.max_comparisons - self.comparisons_used),
                    "tests_left": max(0, self.max_tests - self.tests_used),
                }
                diagnostic["last_result"] = getattr(self, "last_result", "")
            if self.feedback_level >= 2:
                diagnostic["hint"] = hint

            info["diagnostic"] = diagnostic
            return obs, reward, terminated, truncated, info

        def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
            obs, info = super().reset(seed)
            info["diagnostic"] = {
                "error_type": "OK",
                "error_detail": {"outcome": "episode_start"},
                "hint": "Start with LIST SUSPECTS/WEAPONS/ROOMS, then use SCORE on plausible triples.",
                "turn": 0,
                "budgets": {
                    "reveals_left": self.max_reveals,
                    "comparisons_left": self.max_comparisons,
                    "tests_left": self.max_tests,
                },
            }
            return obs, info