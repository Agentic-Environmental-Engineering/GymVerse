from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class DetectiveDeductionGameEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 40,
        **_,
    ):
        super().__init__()
        self.complexity = int(max(1, min(10, complexity)))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 40

        self.complexity_params = {
            # Number of suspects: more candidates increase branching -> harder
            "suspect_count": (3, 8),
            # Number of weapons: more candidates increase branching -> harder
            "weapon_count": (3, 8),
            # Number of rooms: more candidates increase branching -> harder
            "room_count": (3, 10),
            # REVERSED: initial eliminations revealed at start: fewer hints -> harder
            "initial_eliminations": (4, 0),
            # REVERSED: suffix visibility (how many names are listed in suffix per category): less shown -> harder
            "suffix_visibility": (6, 2),
        }

        self.param_variance = {
            "suspect_count": 1,
            "weapon_count": 1,
            "room_count": 1,
            "initial_eliminations": 0,
            "suffix_visibility": 0,
        }

        self.turn_count: int = 0

        self._all_suspect_names = [
            "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"
        ]
        self._all_weapon_names = [
            "Dagger", "Revolver", "Rope", "Wrench", "Poison", "Candlestick", "Lead Pipe", "Axe", "Pistol", "Knife"
        ]
        self._all_room_names = [
            "Kitchen", "Study", "Ballroom", "Conservatory", "Library", "Lounge", "Hall", "Dining Room", "Billiard Room"
            , "Cellar"
        ]

        self.suspects = []
        self.weapons = []
        self.rooms = []
        self.true_suspect = None
        self.true_weapon = None
        self.true_room = None
        self.candidates = {"suspect": set(), "weapon": set(), "room": set()}
        self.pins = {"suspect": None, "weapon": None, "room": None}

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for name, (min_val, max_val) in self.complexity_params.items():
            center = min_val + (max_val - min_val) * normalized
            if self.enable_param_randomization:
                var = self.param_variance.get(name, 0)
                if var > 0:
                    val = center + random.uniform(-var, var)
                else:
                    val = center
            else:
                val = center
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            val = max(low, min(high, val))
            setattr(self, name, int(round(val)))

    def _get_instructions(self) -> str:
        ex = self.sample_random_action()
        return (
            "You are playing the Detective Deduction Game.\n"
            "Goal: Determine the hidden triple (Suspect, Weapon, Room) and accuse correctly.\n"
            "You may:\n"
            "- test S; W; R: propose a hypothesis triple to get a deterministic disproval of the first incorrect component.\n"
            "- pin suspect NAME | pin weapon NAME | pin room NAME: set a working hypothesis piece.\n"
            "- query candidates: see counts and a partial list of remaining candidates.\n"
            "- query notebook: see your current pinned hypothesis.\n"
            "- accuse S; W; R: make a final accusation (ends episode, success if exactly correct).\n"
            "- submit: accuse using all three pinned items (must have all three pinned).\n"
            "Notes:\n"
            "- Names are case-insensitive but must match an existing item from the instance.\n"
            "- A test that returns a disproved item eliminates it from candidates.\n"
            "- Wrong or unsupported actions end the episode with a penalty.\n"
            "Format your action as \\boxed{...} only, with one command per step.\n"
            f"Example: {ex}\n"
        )

    def get_task_suffix(self) -> str:
        def fmt_group(name: str, items: set) -> str:
            lst = sorted(items)
            if len(lst) <= self.suffix_visibility:
                shown = ", ".join(lst)
                return f"{name} ({len(lst)}): {shown}"
            else:
                shown = ", ".join(lst[:self.suffix_visibility]) + ", ..."
                return f"{name} ({len(lst)}): {shown}"

        cand_text = "\n".join([
            fmt_group("Suspects", self.candidates["suspect"]),
            fmt_group("Weapons", self.candidates["weapon"]),
            fmt_group("Rooms", self.candidates["room"]),
        ])
        pin_txt = f"Pinned: suspect={self.pins['suspect']}, weapon={self.pins['weapon']}, room={self.pins['room']}"
        return (
            f"Turn {self.turn_count}/{self.max_turns}\n"
            f"Remaining candidates:\n{cand_text}\n"
            f"{pin_txt}\n"
            "Enter your next action in \\boxed{...} format."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0

        self.suspects = random.sample(self._all_suspect_names, self.suspect_count)
        self.weapons = random.sample(self._all_weapon_names, self.weapon_count)
        self.rooms = random.sample(self._all_room_names, self.room_count)

        self.true_suspect = random.choice(self.suspects)
        self.true_weapon = random.choice(self.weapons)
        self.true_room = random.choice(self.rooms)

        self.candidates = {
            "suspect": set(self.suspects),
            "weapon": set(self.weapons),
            "room": set(self.rooms),
        }
        self.pins = {"suspect": None, "weapon": None, "room": None}

        elim_budget = int(self.initial_eliminations)
        cat_cycle = ["suspect", "weapon", "room"]
        ci = 0
        while elim_budget > 0:
            cat = cat_cycle[ci % 3]
            ci += 1
            true_item = {
                "suspect": self.true_suspect,
                "weapon": self.true_weapon,
                "room": self.true_room,
            }[cat]
            pool = [x for x in self.candidates[cat] if x != true_item]
            if len(pool) > 0:
                pick = random.choice(pool)
                self.candidates[cat].discard(pick)
                elim_budget -= 1
            if (len(self.candidates["suspect"]) == 1 and
                len(self.candidates["weapon"]) == 1 and
                len(self.candidates["room"]) == 1):
                break

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{...}} with a supported command."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        if parsed.get("type") == "unsupported":
            obs = f"Unsupported action '{parsed.get('raw', '')}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        terminated = False
        truncated = False
        reward = 0.0
        obs = ""

        def norm_name(name: str) -> str:
            return name.strip()

        if parsed["type"] == "test":
            sus = norm_name(parsed["suspect"])
            wea = norm_name(parsed["weapon"])
            roo = norm_name(parsed["room"])
            if sus.title() not in self.candidates["suspect"] and sus.title() not in self.suspects:
                obs = f"Protocol violation: unknown suspect '{sus}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if wea.title() not in self.candidates["weapon"] and wea.title() not in self.weapons:
                obs = f"Protocol violation: unknown weapon '{wea}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if roo.title() not in self.candidates["room"] and roo.title() not in self.rooms:
                obs = f"Protocol violation: unknown room '{roo}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

            sus_t = sus.title()
            wea_t = wea.title()
            roo_t = roo.title()
            if sus_t != self.true_suspect:
                self.candidates["suspect"].discard(sus_t)
                obs = f"Test result: disproved suspect {sus_t}."
            elif wea_t != self.true_weapon:
                self.candidates["weapon"].discard(wea_t)
                obs = f"Test result: disproved weapon {wea_t}."
            elif roo_t != self.true_room:
                self.candidates["room"].discard(roo_t)
                obs = f"Test result: disproved room {roo_t}."
            else:
                obs = "Test result: no disproof. This triple may be correct."
            # continue, no termination

        elif parsed["type"] == "pin":
            cat = parsed["category"]
            name = norm_name(parsed["name"]).title()
            if cat not in ("suspect", "weapon", "room"):
                obs = f"Unsupported action 'pin {cat}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if name not in self.candidates[cat]:
                if (name in (self.suspects if cat == "suspect" else self.weapons if cat == "weapon" else self.rooms)):
                    obs = f"Protocol violation: cannot pin eliminated {cat} '{name}'."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
                else:
                    obs = f"Protocol violation: unknown {cat} '{name}'."
                    return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            self.pins[cat] = name
            obs = f"Pinned {cat} {name}."

        elif parsed["type"] == "query":
            what = parsed["what"]
            if what == "candidates":
                obs = "Candidates snapshot provided."
            elif what == "notebook":
                ps = self.pins["suspect"]
                pw = self.pins["weapon"]
                pr = self.pins["room"]
                obs = f"Notebook: suspect={ps}, weapon={pw}, room={pr}."
            elif what in ("suspects", "weapons", "rooms"):
                mapping = {"suspects": "suspect", "weapons": "weapon", "rooms": "room"}
                cat = mapping[what]
                lst = sorted(self.candidates[cat])
                shown = ", ".join(lst[:min(len(lst), max(1, self.suffix_visibility * 2))])
                tail = "" if len(lst) <= self.suffix_visibility * 2 else ", ..."
                obs = f"Remaining {what}: {shown}{tail} (total {len(lst)})."
            else:
                obs = f"Unsupported action 'query {what}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif parsed["type"] == "accuse":
            sus = norm_name(parsed["suspect"]).title()
            wea = norm_name(parsed["weapon"]).title()
            roo = norm_name(parsed["room"]).title()
            if sus not in self.suspects:
                obs = f"Protocol violation: unknown suspect '{sus}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if wea not in self.weapons:
                obs = f"Protocol violation: unknown weapon '{wea}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if roo not in self.rooms:
                obs = f"Protocol violation: unknown room '{roo}'."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if (sus == self.true_suspect) and (wea == self.true_weapon) and (roo == self.true_room):
                obs = "Success! Correct accusation."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Incorrect accusation. The case remains unsolved."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        elif parsed["type"] == "submit":
            ps = self.pins["suspect"]
            pw = self.pins["weapon"]
            pr = self.pins["room"]
            if ps is None or pw is None or pr is None:
                obs = "Protocol violation: cannot submit; working hypothesis incomplete."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}
            if (ps == self.true_suspect) and (pw == self.true_weapon) and (pr == self.true_room):
                obs = "Success! Correct accusation via submit."
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = "Incorrect accusation via submit."
                return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            obs = f"Unsupported action '{parsed.get('type', '')}'."
            return obs, -1.0, True, False, {"suffix": self.get_task_suffix()}

        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns})."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        m = list(re.finditer(r'\\boxed\{(.+?)\}', action, flags=re.IGNORECASE | re.DOTALL))
        if not m:
            return None
        content = m[-1].group(1).strip()
        low = content.lower().strip()

        def split_triple(s: str):
            parts = re.split(r'[;|,]', s)
            parts = [p.strip() for p in parts if p.strip() != ""]
            if len(parts) != 3:
                return None
            return parts

        if low.startswith("test "):
            rest = content[5:].strip()
            triple = split_triple(rest)
            if not triple:
                return {"type": "unsupported", "raw": content}
            return {"type": "test", "suspect": triple[0], "weapon": triple[1], "room": triple[2]}

        if low.startswith("accuse "):
            rest = content[7:].strip()
            triple = split_triple(rest)
            if not triple:
                return {"type": "unsupported", "raw": content}
            return {"type": "accuse", "suspect": triple[0], "weapon": triple[1], "room": triple[2]}

        if low == "submit":
            return {"type": "submit"}

        if low.startswith("pin "):
            rest = content[4:].strip()
            if rest.lower().startswith("suspect "):
                name = rest[8:].strip()
                if not name:
                    return {"type": "unsupported", "raw": content}
                return {"type": "pin", "category": "suspect", "name": name}
            if rest.lower().startswith("weapon "):
                name = rest[7:].strip()
                if not name:
                    return {"type": "unsupported", "raw": content}
                return {"type": "pin", "category": "weapon", "name": name}
            if rest.lower().startswith("room "):
                name = rest[5:].strip()
                if not name:
                    return {"type": "unsupported", "raw": content}
                return {"type": "pin", "category": "room", "name": name}
            return {"type": "unsupported", "raw": content}

        if low.startswith("query "):
            what = content[6:].strip().lower()
            if what in ("candidates", "notebook", "suspects", "weapons", "rooms"):
                return {"type": "query", "what": what}
            return {"type": "unsupported", "raw": content}

        return {"type": "unsupported", "raw": content}

    def sample_random_action(self) -> str:
        if self.suspects and self.weapons and self.rooms:
            s = random.choice(self.suspects)
            w = random.choice(self.weapons)
            r = random.choice(self.rooms)
            return f"\\boxed{{test {s}; {w}; {r}}}"
        return "\\boxed{query candidates}"


class DetectiveDeductionGameEnvWithFeedback(DetectiveDeductionGameEnv):
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
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Wrap exactly one command in \\boxed{...}, e.g., \\boxed{test Alice; Dagger; Kitchen}."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_command_or_args"
            hint = "Use one of: test S; W; R | pin suspect/weapon/room NAME | query candidates/notebook | accuse S; W; R | submit."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            if "unknown suspect" in text:
                error_detail["violation"] = "unknown_suspect"
                hint = "Use a suspect from the candidate list. Try 'query candidates' to view remaining names."
            elif "unknown weapon" in text:
                error_detail["violation"] = "unknown_weapon"
                hint = "Use a weapon from the candidate list. Try 'query candidates'."
            elif "unknown room" in text:
                error_detail["violation"] = "unknown_room"
                hint = "Use a room from the candidate list. Try 'query candidates'."
            elif "cannot submit" in text:
                error_detail["violation"] = "submit_without_all_pins"
                hint = "Either accuse with all three names (accuse S; W; R) or pin all three then 'submit'."
            elif "cannot pin eliminated" in text:
                error_detail["violation"] = "pin_eliminated_item"
                hint = "Pin only remaining candidates. Use 'query candidates' to see what remains."
            else:
                error_detail["violation"] = "generic_protocol_violation"
                hint = "Follow the command formats and ensure names exist among candidates."

        elif "incorrect accusation" in text:
            error_type = "WrongDecision"
            error_detail["decision"] = "wrong_final_accuse"
            hint = "Before accusing, use 'test S; W; R' to eliminate items. Accuse only when confident."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = getattr(self, "max_turns", None)
            hint = "Make focused tests to narrow candidates quickly. Use 'query candidates' to track progress."

        elif "success" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "suspects_remaining": len(self.candidates["suspect"]),
                "weapons_remaining": len(self.candidates["weapon"]),
                "rooms_remaining": len(self.candidates["room"]),
                "pins": dict(self.pins),
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
            "hint": "Start by using 'query candidates' to see remaining names, then 'test S; W; R' to eliminate.",
            "turn": 0,
            "state": {
                "suspects_remaining": len(self.candidates["suspect"]),
                "weapons_remaining": len(self.candidates["weapon"]),
                "rooms_remaining": len(self.candidates["room"]),
                "pins": dict(self.pins),
            },
        }
        return obs, info