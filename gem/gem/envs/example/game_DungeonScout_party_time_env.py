from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional


class DungeonScoutEnv(Env):
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

        # Evolvable parameters
        self.complexity_params = {
            "num_rooms": (3, 12),  # More rooms increase search space and cognitive load
            "min_creatures_per_room": (1, 3),  # Higher minimum increases base population size
            "max_creatures_per_room": (3, 8),  # Higher maximum increases variance and complexity
            "predicate_terms": (1, 4),  # More conjunction terms increase reasoning difficulty
            "reveal_allowance": (20, 12),  # REVERSED: fewer reveals harder; will be clamped to >= num_rooms
            "species_variety": (3, 6),  # More species types increase combinatorial complexity
            "element_variety": (3, 6),  # More element types increase combinatorial complexity
            "level_range_max": (6, 10),  # Wider level range introduces more numeric variability
            "decoy_attributes": (0, 3),  # More decoy details increase distraction without helping
            "include_negations": (0, 1),  # Use negated conditions (0=no, 1=yes) increases reasoning difficulty
        }

        # Parameter variance
        self.param_variance = {
            "num_rooms": 1,
            "min_creatures_per_room": 1,
            "max_creatures_per_room": 1,
            "predicate_terms": 1,
            "reveal_allowance": 2,
            "species_variety": 1,
            "element_variety": 1,
            "level_range_max": 1,
            "decoy_attributes": 0,
            "include_negations": 0,
        }

        # Placeholder attributes
        self.num_rooms: int = 0
        self.min_creatures_per_room: int = 0
        self.max_creatures_per_room: int = 0
        self.predicate_terms: int = 0
        self.reveal_allowance: int = 0
        self.species_variety: int = 0
        self.element_variety: int = 0
        self.level_range_max: int = 0
        self.decoy_attributes: int = 0
        self.include_negations: int = 0

        # State
        self.turn_count: int = 0
        self.rooms: Dict[str, Any] = {}
        self.revealed_rooms: Dict[str, Any] = {}
        self.unrevealed_rooms: Dict[str, Any] = {}
        self.reveals_used: int = 0
        self.ground_truth_count: int = 0
        self.predicate_description: str = ""
        self.predicate_terms_list: Any = []
        self._predicate = None

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
                    if min_val > max_val:
                        # reversed range
                        actual_value = max(max_val, min(min_val, actual_value))
                    else:
                        actual_value = max(min_val, min(max_val, actual_value))
            else:
                actual_value = center_value
            setattr(self, param_name, int(round(actual_value)))

    def _get_instructions(self) -> str:
        return (
            "Dungeon Scout: Explore rooms to count creatures matching the predicate.\n"
            "Goal: Compute the exact number of creatures across all rooms that satisfy the given predicate.\n"
            "Actions:\n"
            "- Reveal a room: use \\boxed{reveal X} where X is a room letter (e.g., A, B, C).\n"
            "- Submit your final count: use \\boxed{answer N} where N is a nonnegative integer.\n"
            f"- Reveals available: {self.reveal_allowance}. Each reveal consumes 1.\n"
            "Protocol:\n"
            "- You may reveal rooms in any order until reveal limit or until you answer.\n"
            "- The episode ends when you submit an answer, exceed the reveal limit, use invalid format, or hit timeout.\n"
            "Example actions:\n"
            f"- {self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        lines = []
        lines.append(f"Predicate: {self.predicate_description}")
        lines.append("Rooms:")
        for name in sorted(self.rooms.keys()):
            if name in self.revealed_rooms:
                creatures = self.revealed_rooms[name]["creatures"]
                lines.append(f"- Room {name} [REVEALED]: {len(creatures)} creatures")
                for c in creatures:
                    shield_text = "shield" if c["shield"] else "no-shield"
                    lines.append(
                        f"  • {c['species']} | {c['element']} | lvl {c['level']} | {c['weapon']} | {shield_text}"
                    )
            else:
                lines.append(f"- Room {name} [UNREVEALED]")
        if self.decoy_attributes > 0:
            lines.append(self._decoy_text())
        lines.append(f"Reveals used: {self.reveals_used}/{self.reveal_allowance}")
        lines.append("Enter your action in \\boxed{...} format: reveal X or answer N")
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()

        # Safeguard solvability: ensure reveal allowance >= num_rooms
        self.reveal_allowance = max(self.reveal_allowance, self.num_rooms)

        self.turn_count = 0
        self.reveals_used = 0
        self.rooms = {}
        self.revealed_rooms = {}
        self.unrevealed_rooms = {}
        self.ground_truth_count = 0
        self._predicate = None
        self.predicate_terms_list = []

        # Domain attribute pools
        base_species = ["goblin", "skeleton", "orc", "mage", "troll", "bandit"]
        base_elements = ["fire", "water", "earth", "air", "shadow", "light"]
        weapons = ["sword", "bow", "staff", "axe", "dagger", "mace"]

        species_pool = random.sample(base_species, self.species_variety)
        element_pool = random.sample(base_elements, self.element_variety)

        # Create rooms and creatures
        room_names = [chr(ord("A") + i) for i in range(self.num_rooms)]
        for rn in room_names:
            n_creatures = random.randint(self.min_creatures_per_room, self.max_creatures_per_room)
            creatures = []
            for _ in range(n_creatures):
                c = {
                    "species": random.choice(species_pool),
                    "element": random.choice(element_pool),
                    "level": random.randint(1, self.level_range_max),
                    "shield": random.choice([True, False]),
                    "weapon": random.choice(weapons),
                }
                creatures.append(c)
            self.rooms[rn] = {"name": rn, "creatures": creatures}
            self.unrevealed_rooms[rn] = self.rooms[rn]

        # Build predicate with conjunction terms
        available_terms = ["species", "element", "level", "shield", "weapon"]
        random.shuffle(available_terms)
        selected_terms = available_terms[: self.predicate_terms]
        terms_desc = []
        predicates = []

        negation_enabled = bool(self.include_negations)

        for term in selected_terms:
            if term == "species":
                subset_size = 1 if len(species_pool) <= 3 else 2
                subset = sorted(random.sample(species_pool, subset_size))
                use_neg = negation_enabled and random.choice([True, False])
                if use_neg:
                    terms_desc.append(f"species NOT IN {{{', '.join(subset)}}}")
                    predicates.append(lambda c, subset=subset: c["species"] not in subset)
                else:
                    terms_desc.append(f"species IN {{{', '.join(subset)}}}")
                    predicates.append(lambda c, subset=subset: c["species"] in subset)

            elif term == "element":
                choice = random.choice(element_pool)
                use_neg = negation_enabled and random.choice([True, False])
                if use_neg:
                    terms_desc.append(f"element != {choice}")
                    predicates.append(lambda c, el=choice: c["element"] != el)
                else:
                    terms_desc.append(f"element == {choice}")
                    predicates.append(lambda c, el=choice: c["element"] == el)

            elif term == "level":
                threshold = random.randint(2, max(3, self.level_range_max - 1))
                direction = random.choice([">=", "<="])
                if direction == ">=":
                    terms_desc.append(f"level >= {threshold}")
                    predicates.append(lambda c, t=threshold: c["level"] >= t)
                else:
                    terms_desc.append(f"level <= {threshold}")
                    predicates.append(lambda c, t=threshold: c["level"] <= t)

            elif term == "shield":
                val = random.choice([True, False])
                terms_desc.append(f"has_shield == {str(val).lower()}")
                predicates.append(lambda c, v=val: c["shield"] == v)

            elif term == "weapon":
                subset_size = 1 if len(weapons) <= 4 else 2
                subset = sorted(random.sample(weapons, subset_size))
                use_neg = negation_enabled and random.choice([True, False])
                if use_neg:
                    terms_desc.append(f"weapon NOT IN {{{', '.join(subset)}}}")
                    predicates.append(lambda c, subset=subset: c["weapon"] not in subset)
                else:
                    terms_desc.append(f"weapon IN {{{', '.join(subset)}}}")
                    predicates.append(lambda c, subset=subset: c["weapon"] in subset)

        self.predicate_description = " AND ".join(terms_desc)

        def predicate(c):
            for p in predicates:
                if not p(c):
                    return False
            return True

        self._predicate = predicate

        # Compute ground truth count
        total = 0
        for rn in room_names:
            for c in self.rooms[rn]["creatures"]:
                if self._predicate(c):
                    total += 1
        self.ground_truth_count = total

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)

        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{reveal X}} or \\boxed{{answer N}}."
            return (
                obs,
                LanguageGameReward.format_error_reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )

        if parsed["type"] == "answer":
            n = parsed["value"]
            correct = n == self.ground_truth_count
            if correct:
                obs = (
                    f"Success! Correct count {self.ground_truth_count}. "
                    "Episode ends."
                )
                return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                obs = (
                    f"Incorrect. You answered {n}, but the correct count is {self.ground_truth_count}. "
                    "Episode ends."
                )
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        elif parsed["type"] == "reveal":
            room = parsed["room"]
            if self.reveals_used >= self.reveal_allowance:
                obs = (
                    f"Reveal limit reached ({self.reveals_used}/{self.reveal_allowance}). "
                    "No further reveals allowed. Episode ends."
                )
                return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

            if room not in self.rooms:
                obs = (
                    f"Unsupported or unknown room '{room}'. "
                    "Valid rooms: " + ", ".join(sorted(self.rooms.keys()))
                )
                # Not a format error; continue episode
                if self.turn_count >= self.max_turns:
                    obs = f"{obs}\nReached max turns ({self.max_turns})."
                    return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

            if room in self.revealed_rooms:
                obs = f"Room {room} is already revealed. Choose another room."
                if self.turn_count >= self.max_turns:
                    obs = f"{obs}\nReached max turns ({self.max_turns})."
                    return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
                return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

            # Reveal room
            self.reveals_used += 1
            self.revealed_rooms[room] = self.rooms[room]
            if room in self.unrevealed_rooms:
                del self.unrevealed_rooms[room]
            creatures = self.revealed_rooms[room]["creatures"]
            lines = [
                f"Revealed Room {room}: {len(creatures)} creatures.",
                f"Reveals used: {self.reveals_used}/{self.reveal_allowance}",
            ]
            for c in creatures:
                shield_text = "shield" if c["shield"] else "no-shield"
                lines.append(
                    f"- {c['species']} | {c['element']} | lvl {c['level']} | {c['weapon']} | {shield_text}"
                )
            obs = "\n".join(lines)

            if self.turn_count >= self.max_turns:
                obs = f"{obs}\nReached max turns ({self.max_turns})."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        else:
            obs = "Unsupported action. Use \\boxed{reveal X} or \\boxed{answer N}."
            # Terminate on unsupported command to enforce protocol strictness
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Any]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r"\\boxed\{(.+?)\}", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        content_l = content.lower()

        # answer N
        m_ans = re.match(r"answer\s+(-?\d+)", content_l)
        if m_ans:
            try:
                n = int(m_ans.group(1))
                if n < 0:
                    n = 0
                return {"type": "answer", "value": n}
            except Exception:
                return None

        # reveal X (room letter)
        m_rev = re.match(r"reveal\s+([a-z])", content_l)
        if m_rev:
            room_letter = m_rev.group(1).upper()
            return {"type": "reveal", "room": room_letter}

        # Otherwise unsupported
        return {"type": "unsupported"}

    def sample_random_action(self) -> str:
        if random.random() < 0.5 and self.rooms:
            room = random.choice(list(self.rooms.keys()))
            return f"\\boxed{{reveal {room}}}"
        else:
            guess = random.randint(0, max(0, self.max_turns // 2))
            return f"\\boxed{{answer {guess}}}"

    def _decoy_text(self) -> str:
        decoys = [
            "Note: Some rooms contain cracked pillars.",
            "Tip: Wind drafts may affect torchlight but not creatures.",
            "Lore: Ancient runes are etched on the door frames.",
        ]
        return "Decoy: " + " ".join(random.sample(decoys, k=min(self.decoy_attributes, len(decoys))))


class DungeonScoutEnvWithFeedback(DungeonScoutEnv):
    def __init__(self, feedback_level: int = 2, **kwargs):
        self.feedback_level = feedback_level
        super().__init__(**kwargs)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        text = obs.lower()
        error_type = "OK"
        error_detail: Dict[str, Any] = {}
        hint: Optional[str] = None

        if "invalid action format" in text or "use \\boxed" in text and "invalid" in text:
            error_type = "FormatError"
            error_detail["issue"] = "missing_or_bad_boxed_format"
            hint = "Use \\boxed{reveal X} to reveal a room or \\boxed{answer N} to submit the count."

        elif "unsupported action" in text or "unsupported or unknown room" in text:
            error_type = "UnsupportedAction"
            if "unknown room" in text:
                error_detail["issue"] = "unknown_room_letter"
                hint = "Check the listed room letters and use one of them, e.g., \\boxed{reveal A}."
            else:
                error_detail["issue"] = "unsupported_command"
                hint = "Only 'reveal X' and 'answer N' are supported."

        elif "reveal limit reached" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "exceeded_reveal_allowance"
            hint = "You cannot reveal more rooms. Submit your final count using \\boxed{answer N}."

        elif "already revealed" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "redundant_reveal"
            hint = "Choose a different unrevealed room letter to reveal."

        elif "incorrect. you answered" in text and "correct count is" in text:
            error_type = "WrongDecision"
            # Extract numbers if possible
            got_match = re.search(r"you answered\s+(-?\d+)", text)
            exp_match = re.search(r"correct count is\s+(-?\d+)", text)
            if got_match:
                error_detail["got"] = int(got_match.group(1))
            if exp_match:
                error_detail["expected"] = int(exp_match.group(1))
            hint = "Revisit the predicate conditions and tally creatures across all rooms. Ensure all terms are applied."

        elif "success! correct count" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = "Well done."

        elif "reached max turns" in text:
            error_type = "Timeout"
            error_detail["limit"] = self.max_turns
            hint = "Plan reveals and answer earlier to avoid timeout."

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "reveals_used": getattr(self, "reveals_used", None),
                "reveal_allowance": getattr(self, "reveal_allowance", None),
                "num_rooms": getattr(self, "num_rooms", None),
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
            "hint": "Start by revealing a room: \\boxed{reveal A}. Read the predicate carefully.",
            "turn": 0,
            "state": {
                "reveals_used": 0,
                "reveal_allowance": getattr(self, "reveal_allowance", None),
                "num_rooms": getattr(self, "num_rooms", None),
            },
        }
        return obs, info