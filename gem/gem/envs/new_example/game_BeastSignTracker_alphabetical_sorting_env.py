from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional

class BeastSignTrackerEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 16,
        **_,
    ):
        super().__init__()
        self.complexity = complexity
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 16

        # Evolvable parameters
        self.complexity_params = {
            # Number of suspects (more suspects = harder)
            'num_suspects': (3, 8),
            # Number of trail sites (more sites to inspect = larger search space = harder)
            'num_sites': (2, 6),
            # REVERSED: Probe budget (fewer probes = harder)
            'probe_limit': (5, 2),
            # REVERSED: Number of key clue categories that are revealed (fewer discriminative clues = harder)
            'key_clues_count': (2, 1),
            # Number of valid beasts consistent with clues (more valid targets = slightly harder ambiguity)
            'valid_targets_count': (1, 2),
        }

        # Parameter variance (small ranges → 0; keep stable to ensure curriculum integrity)
        self.param_variance = {
            'num_suspects': 0,
            'num_sites': 0,
            'probe_limit': 0,
            'key_clues_count': 0,
            'valid_targets_count': 0,
        }

        # Placeholders for evolvable parameters
        self.num_suspects: int = 0
        self.num_sites: int = 0
        self.probe_limit: int = 0
        self.key_clues_count: int = 0
        self.valid_targets_count: int = 0

        # Static domain data
        self.beast_db = {
            "Glimmerfox":   {"stride": "short",  "claws": "shallow", "time": "dusk",  "terrain": "forest",  "toes": 4, "residue": "hair"},
            "Riverdrake":   {"stride": "long",   "claws": "deep",    "time": "day",   "terrain": "river",   "toes": 3, "residue": "scales"},
            "Shadecat":     {"stride": "medium", "claws": "shallow", "time": "night", "terrain": "forest",  "toes": 5, "residue": "hair"},
            "Briarboar":    {"stride": "short",  "claws": "deep",    "time": "day",   "terrain": "swamp",   "toes": 4, "residue": "bones"},
            "Sunstalker":   {"stride": "long",   "claws": "deep",    "time": "day",   "terrain": "mountain","toes": 4, "residue": "bones"},
            "Silkstrider":  {"stride": "long",   "claws": "shallow", "time": "dusk",  "terrain": "forest",  "toes": 3, "residue": "hair"},
            "Gravelmaw":    {"stride": "medium", "claws": "deep",    "time": "night", "terrain": "mountain","toes": 5, "residue": "bones"},
            "Thornback":    {"stride": "short",  "claws": "deep",    "time": "dusk",  "terrain": "forest",  "toes": 3, "residue": "scales"},
            "Brinewyrm":    {"stride": "long",   "claws": "deep",    "time": "night", "terrain": "river",   "toes": 4, "residue": "scales"},
            "Moondrake":    {"stride": "medium", "claws": "shallow", "time": "night", "terrain": "swamp",   "toes": 3, "residue": "scales"},
        }

        self.site_pool = [
            "Mill Path", "Pine Grove", "River Bend", "Old Shrine", "Cliff Ledge",
            "Bog Crossing", "Meadow Verge", "Stone Bridge", "Cave Mouth", "Reedbank"
        ]

        # Dynamic state
        self.turn_count: int = 0
        self.remaining_probes: int = 0
        self.suspects: Dict[str, Dict[str, Any]] = {}
        self.site_names: list = []
        self.site_clues: Dict[str, Dict[str, Any]] = {}
        self.revealed_sites: set = set()
        self.current_hypothesis: Optional[str] = None
        self.valid_beasts: set = set()
        self.key_categories: list = []
        self.stage: str = "explore"

        self.reset()

    def _apply_complexity_params(self):
        normalized = min(1.0, (self.complexity - 1) / 9.0)
        for param_name, (min_val, max_val) in self.complexity_params.items():
            center_value = min_val + (max_val - min_val) * normalized
            actual_value = center_value
            variance = self.param_variance.get(param_name, 0)
            if self.enable_param_randomization and variance > 0:
                actual_value = center_value + random.uniform(-variance, variance)
            # Clamp to range; supports reversed params
            low, high = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
            actual_value = max(low, min(high, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _construct_instance(self):
        # Choose suspects
        all_names = list(self.beast_db.keys())
        random.shuffle(all_names)
        self.suspects = {name: self.beast_db[name] for name in all_names[:self.num_suspects]}

        # Choose pivot beast and categories to define shared attributes
        pivot_name = random.choice(list(self.suspects.keys()))
        pivot_attrs = self.suspects[pivot_name]
        categories = ["stride", "claws", "time", "terrain", "toes", "residue"]
        random.shuffle(categories)

        # Find categories that yield enough matches when fixed to pivot's values
        selected_cats = []
        matched_set = set(self.suspects.keys())
        # Try to select up to key_clues_count categories that still yield >= valid_targets_count matches
        for cat in categories:
            if len(selected_cats) >= self.key_clues_count:
                break
            target_val = pivot_attrs[cat]
            candidates = {n for n in matched_set if self.suspects[n][cat] == target_val}
            # Only add this category if enough candidates remain
            if len(candidates) >= self.valid_targets_count:
                selected_cats.append(cat)
                matched_set = candidates

        # If we could not select enough discriminative categories, relax to at least one category
        if len(selected_cats) == 0:
            # pick a category that maximizes matches
            best_cat = None
            best_candidates = set(self.suspects.keys())
            for cat in categories:
                target_val = pivot_attrs[cat]
                candidates = {n for n in self.suspects.keys() if self.suspects[n][cat] == target_val}
                if len(candidates) >= self.valid_targets_count and len(candidates) <= len(best_candidates):
                    best_cat = cat
                    best_candidates = candidates
            if best_cat is None:
                # Fallback: choose any category; accept that matched_set may be large
                best_cat = random.choice(categories)
                best_candidates = {n for n in self.suspects.keys() if self.suspects[n][best_cat] == pivot_attrs[best_cat]}
                if len(best_candidates) == 0:
                    best_candidates = set(self.suspects.keys())
            selected_cats = [best_cat]
            matched_set = best_candidates

        # If intersection still too small, expand by dropping categories (already minimal); ensure at least valid_targets_count by adding random suspects if needed
        if len(matched_set) < self.valid_targets_count:
            pool = [n for n in self.suspects.keys() if n not in matched_set]
            while len(matched_set) < self.valid_targets_count and pool:
                matched_set.add(pool.pop())

        # Now choose the valid beasts
        self.valid_beasts = set(random.sample(list(matched_set), self.valid_targets_count))
        self.key_categories = selected_cats

        # Build sites and clues
        self.site_names = random.sample(self.site_pool, self.num_sites)
        self.site_clues = {}
        # Assign key clues to some sites
        cats_to_assign = self.key_categories[:]
        random.shuffle(cats_to_assign)
        assigned_sites = set()
        for cat in cats_to_assign:
            if len(assigned_sites) >= self.num_sites:
                break
            site = random.choice([s for s in self.site_names if s not in assigned_sites])
            assigned_sites.add(site)
            # Consistent value derived from pivot (shared by valid beasts)
            self.site_clues[site] = {cat: pivot_attrs[cat]}

        # Assign neutral clues to remaining sites; keep consistency with at least one valid beast
        remaining_sites = [s for s in self.site_names if s not in assigned_sites]
        neutral_cats = [c for c in categories if c not in self.key_categories]
        for site in remaining_sites:
            # pick one neutral category
            cat = random.choice(neutral_cats) if neutral_cats else random.choice(categories)
            # choose a value that at least one valid beast has
            possible_vals = set(self.suspects[b][cat] for b in self.valid_beasts)
            val = random.choice(list(possible_vals))
            self.site_clues[site] = {cat: val}

        self.revealed_sites = set()
        self.remaining_probes = self.probe_limit
        self.current_hypothesis = None
        self.stage = "explore"

    def _get_instructions(self) -> str:
        return (
            "BeastSignTracker: Track a mysterious creature by inspecting signs at trail sites.\n"
            "Goal: Accuse any beast that is consistent with all revealed clues.\n"
            "Functions:\n"
            "- list_options(): Show suspects and available sites.\n"
            "- inspect(site=\"SITE_NAME\"): Reveal the clue at a site. Consumes 1 probe. Re-inspecting yields no new info.\n"
            "- hypothesis(beast=\"BEAST_NAME\"): Record a provisional guess (does not end the episode).\n"
            "- accuse(beast=\"BEAST_NAME\"): Make a final accusation and end the episode. You must inspect at least one site before accusing.\n"
            "Rules:\n"
            "- Probe budget limits how many sites you can inspect.\n"
            "- Clues are designed to be consistent with the hidden valid beasts; some clues are more discriminative.\n"
            "- Success reward is granted only if your accused beast is valid.\n"
            "Format:\n"
            "Use <action>[func_name(param=value)]</action> or <action>[func_name()]</action>\n"
            f"Example:\n{self.sample_random_action()}\n"
        )

    def get_task_suffix(self) -> str:
        suspects = ", ".join(sorted(self.suspects.keys()))
        available_sites = ", ".join(s for s in self.site_names if s not in self.revealed_sites)
        revealed_sites = ", ".join(sorted(self.revealed_sites)) if self.revealed_sites else "none"
        return (
            f"Suspects: {suspects}\n"
            f"Trail sites: {', '.join(self.site_names)}\n"
            f"Revealed sites: {revealed_sites}\n"
            f"Remaining probes: {self.remaining_probes}\n"
            "Enter your action: <action>[list_options()]</action>, <action>[inspect(site=\"SITE_NAME\")]</action>, "
            "<action>[hypothesis(beast=\"BEAST_NAME\")]</action>, or <action>[accuse(beast=\"BEAST_NAME\")]</action>"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)
        self._apply_complexity_params()
        self.turn_count = 0
        self._construct_instance()
        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        terminated = False
        truncated = False

        parsed = self._parse_action(action)
        if not parsed:
            obs = "Invalid action format. Use <action>[func_name(...)]</action>."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        name = parsed["name"]
        params = parsed["parameters"]

        reward = 0.0
        text = ""

        if name == "list_options":
            suspects_line = ", ".join(sorted(self.suspects.keys()))
            sites_line = ", ".join(s for s in self.site_names if s not in self.revealed_sites)
            text = (
                f"Options listed. Suspects: {suspects_line}. "
                f"Available sites: {sites_line if sites_line else 'none'}. "
                f"Remaining probes: {self.remaining_probes}."
            )

        elif name == "inspect":
            site = params.get("site", None)
            if not isinstance(site, str):
                text = "Unsupported action: inspect requires parameter site=\"SITE_NAME\"."
                return text, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if site not in self.site_names:
                text = f"Unsupported action: site '{site}' does not exist."
                return text, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if self.remaining_probes <= 0:
                text = "No probes left. You cannot inspect further; consider accusing."
            elif site in self.revealed_sites:
                text = f"Site '{site}' already inspected. No new information."
            else:
                self.remaining_probes -= 1
                self.revealed_sites.add(site)
                clue_dict = self.site_clues[site]
                cat = list(clue_dict.keys())[0]
                val = clue_dict[cat]
                text = f"Inspection at '{site}' reveals {cat}='{val}'. Probes left: {self.remaining_probes}."

        elif name == "hypothesis":
            beast = params.get("beast", None)
            if not isinstance(beast, str) or beast not in self.suspects:
                text = "Unsupported action: hypothesis requires a known beast name."
                return text, 0.0, True, False, {"suffix": self.get_task_suffix()}
            self.current_hypothesis = beast
            text = f"Hypothesis recorded: {beast}. This does not end the episode."

        elif name == "accuse":
            beast = params.get("beast", None)
            if not isinstance(beast, str) or beast not in self.suspects:
                text = "Unsupported action: accuse requires a known beast name."
                return text, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if len(self.revealed_sites) == 0:
                text = "Protocol violation: must inspect at least one site before accusing."
                return text, 0.0, True, False, {"suffix": self.get_task_suffix()}
            if beast in self.valid_beasts:
                text = f"Success! You correctly identified the trail-maker as '{beast}'."
                return text, 1.0, True, False, {"suffix": self.get_task_suffix()}
            else:
                text = f"Incorrect accusation. '{beast}' is inconsistent with the revealed clues."
                return text, 0.0, True, False, {"suffix": self.get_task_suffix()}

        else:
            text = f"Unsupported action '{name}'."
            return text, 0.0, True, False, {"suffix": self.get_task_suffix()}

        # Timeout check
        if self.turn_count >= self.max_turns:
            obs = f"Reached max turns ({self.max_turns}). Episode timed out."
            return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}

        obs = f"Turn {self.turn_count}: {text}"
        return obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action:
            return None
        from gem.utils.parsing import extract_action_parameters
        content = extract_action_parameters(action)
        if not content:
            return None
        content = content.strip()
        if not (content.startswith('[') and content.endswith(']')):
            return None
        func_call_str = content[1:-1].strip()
        func_pattern = re.compile(r'^(\w+)\((.*)\)$', re.DOTALL)
        func_match = func_pattern.match(func_call_str)
        if not func_match:
            return None
        func_name = func_match.group(1)
        params_str = func_match.group(2).strip()
        parameters: Dict[str, Any] = {}
        if params_str:
            pairs = re.findall(r'(\w+)\s*=\s*([^,]+?)(?:,|$)', params_str)
            for key, value in pairs:
                v = value.strip()
                try:
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        parameters[key] = v[1:-1]
                    elif '.' in v:
                        parameters[key] = float(v)
                    elif v.isdigit() or (v.startswith('-') and v[1:].isdigit()):
                        parameters[key] = int(v)
                    elif v.lower() == 'true':
                        parameters[key] = True
                    elif v.lower() == 'false':
                        parameters[key] = False
                    else:
                        parameters[key] = v
                except Exception:
                    parameters[key] = v
        return {"name": func_name, "parameters": parameters}

    def sample_random_action(self) -> str:
        if random.random() < 0.4 and self.site_names:
            site = random.choice(self.site_names)
            return f'<action>[inspect(site="{site}")]</action>'
        elif random.random() < 0.7 and self.suspects:
            beast = random.choice(list(self.suspects.keys()))
            return f'<action>[hypothesis(beast="{beast}")]</action>'
        else:
            return '<action>[list_options()]</action>'


class BeastSignTrackerEnvWithFeedback(BeastSignTrackerEnv):
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
            error_detail["issue"] = "missing_or_malformed_action_tags"
            hint = "Wrap a single function call in <action>[func_name(params)]</action>."

        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            error_detail["issue"] = "unknown_function_or_bad_parameters"
            hint = "Use one of: list_options(), inspect(site=\"SITE_NAME\"), hypothesis(beast=\"BEAST_NAME\"), accuse(beast=\"BEAST_NAME\")."

        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["violation"] = "accuse_before_inspection"
            hint = "Inspect at least one site with inspect(site=\"...\") to reveal a clue before accusing."

        elif "incorrect accusation" in text:
            error_type = "WrongDecision"
            # Provide non-spoiling detail
            error_detail["revealed_sites"] = list(getattr(self, "revealed_sites", []))
            error_detail["remaining_probes"] = getattr(self, "remaining_probes", None)
            hint = "Revisit the revealed clues; check if the accused beast matches all revealed attribute values."

        elif "reached max turns" in text or "timed out" in text:
            error_type = "Timeout"
            error_detail["max_turns"] = getattr(self, "max_turns", None)
            hint = "Plan to inspect early and accuse once enough evidence aligns."

        elif "already inspected" in text:
            error_type = "OK"
            error_detail["note"] = "redundant_inspection"
            hint = "Choose a new site or proceed to accuse if you have sufficient evidence."

        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"
            hint = None

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "remaining_probes": getattr(self, "remaining_probes", None),
                "available_sites": [s for s in getattr(self, "site_names", []) if s not in getattr(self, "revealed_sites", set())],
                "suspects_count": len(getattr(self, "suspects", {})),
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
            "hint": "Start by calling list_options() and then inspect a site to reveal your first clue.",
            "turn": 0,
            "state": {
                "remaining_probes": getattr(self, "remaining_probes", None),
                "available_sites": [s for s in getattr(self, "site_names", [])],
                "suspects_count": len(getattr(self, "suspects", {})),
            },
        }
        return obs, info