from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class GlyphConclaveEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 20,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 20

        # Evolvable parameters (domain-specific)
        self.complexity_params = {
            # number of tiles (larger = more entities to scan, harder)
            "num_tiles": (8, 30),
            # number of attributes per tile from this pool (more attrs = more combinations to reason about)
            "num_attribute_types": (3, 5),
            # number of clues in the conjunction (more constraints = harder logic)
            "num_clues": (1, 5),
            # fraction of tiles that are distractors with near-miss attributes (higher = harder discrimination)
            "distractor_density_pct": (5, 35),
            # selector complexity level (controls advanced clue types included)
            "clue_variety_level": (1, 4),  # introduces uniqueness/adjacency and nested attribute-count clue
            # REVERSED: number of hints available to request (fewer hints = harder)
            "hint_budget": (2, 0),
        }

        # Variance settings
        self.param_variance = {
            "num_tiles": 3,                 # ~10-15% variance
            "num_attribute_types": 0,       # small discrete range; keep stable per level
            "num_clues": 1,                 # integer ±1 variation
            "distractor_density_pct": 3,    # small percent wobble
            "clue_variety_level": 0,        # categorical complexity; fixed per level
            "hint_budget": 0,               # keep deterministic for curriculum predictability
        }

        # Placeholder attributes
        self.num_tiles: int = 0
        self.num_attribute_types: int = 0
        self.num_clues: int = 0
        self.distractor_density_pct: int = 0
        self.clue_variety_level: int = 0
        self.hint_budget: int = 0

        # Domain state
        self.turn_count: int = 0
        self.tiles: List[Dict[str, str]] = []
        self.attributes_pool: Dict[str, List[str]] = {}
        self.clues: List[Dict[str, Any]] = []
        self.target_count: int = 0
        self.question_mode: str = "count_tiles"  # or "count_attribute"
        self.attribute_to_count: Optional[Dict[str, str]] = None
        self.solved: bool = False
        self.hints_used: int = 0

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
                    lo, hi = (min_val, max_val) if min_val <= max_val else (max_val, min_val)
                    actual_value = max(lo, min(hi, actual_value))
            setattr(self, param_name, int(round(actual_value)))

    def _build_attribute_pool(self):
        full_pool = {
            "shape": ["circle", "triangle", "square", "hexagon", "star"],
            "color": ["red", "blue", "green", "yellow", "purple"],
            "fill": ["solid", "striped", "dotted", "hollow", "gradient"],
            "border": ["thin", "thick", "double", "dashed", "glow"],
            "symbol": ["alpha", "beta", "gamma", "delta", "omega"],
        }
        all_keys = list(full_pool.keys())
        random.shuffle(all_keys)
        chosen_keys = all_keys[: self.num_attribute_types]
        self.attributes_pool = {k: full_pool[k] for k in chosen_keys}

    def _sample_tiles(self):
        self.tiles = []
        keys = list(self.attributes_pool.keys())
        for _ in range(self.num_tiles):
            tile = {}
            for k in keys:
                tile[k] = random.choice(self.attributes_pool[k])
            self.tiles.append(tile)

    def _generate_base_clause(self):
        # simple attribute equalities
        keys = list(self.attributes_pool.keys())
        chosen_key = random.choice(keys)
        chosen_val = random.choice(self.attributes_pool[chosen_key])
        return {"type": "attr_eq", "key": chosen_key, "value": chosen_val}

    def _generate_unique_clause(self):
        # uniqueness: tiles where an attribute value is unique among all tiles
        key = random.choice(list(self.attributes_pool.keys()))
        return {"type": "unique_attr", "key": key}

    def _generate_adjacency_clause(self):
        # adjacency-like: tiles whose attribute matches either of two pivot values (proxy adjacency in attribute space)
        key = random.choice(list(self.attributes_pool.keys()))
        vals = random.sample(self.attributes_pool[key], 2)
        return {"type": "attr_in_set", "key": key, "values": vals}

    def _generate_nested_count_clause(self):
        # nested attribute count inside a tile: count how many of its attributes are from a given subset
        # For puzzle flavor, define a subset of allowed values across 2 attributes; tile must match at least t of them
        keys = list(self.attributes_pool.keys())
        if len(keys) < 2:
            return self._generate_base_clause()
        subkeys = random.sample(keys, 2)
        subset = {}
        for k in subkeys:
            subset[k] = random.sample(self.attributes_pool[k], 2)
        threshold = random.choice([1, 2])  # need to match at least threshold among these sub-choices
        return {"type": "nested_attr_hits", "subset": subset, "threshold": threshold}

    def _make_clues(self):
        # Always include at least one base clause to anchor solvability
        clues = [self._generate_base_clause()]
        advanced_catalog = []
        if self.clue_variety_level >= 2:
            advanced_catalog.append(self._generate_unique_clause)
        if self.clue_variety_level >= 3:
            advanced_catalog.append(self._generate_adjacency_clause)
        if self.clue_variety_level >= 4:
            advanced_catalog.append(self._generate_nested_count_clause)

        while len(clues) < self.num_clues:
            if advanced_catalog and random.random() < 0.6:
                gen = random.choice(advanced_catalog)
                clause = gen()
            else:
                clause = self._generate_base_clause()
            clues.append(clause)

        # Simplify: avoid contradictory equalities on same key
        eq_constraints = {}
        final = []
        for c in clues:
            if c["type"] == "attr_eq":
                k = c["key"]
                if k in eq_constraints and eq_constraints[k] != c["value"]:
                    continue
                eq_constraints[k] = c["value"]
            final.append(c)
        self.clues = final

    def _apply_clues_to_tile(self, tile: Dict[str, str]) -> bool:
        for c in self.clues:
            t = c["type"]
            if t == "attr_eq":
                if tile.get(c["key"]) != c["value"]:
                    return False
            elif t == "unique_attr":
                key = c["key"]
                val = tile.get(key)
                count = sum(1 for x in self.tiles if x.get(key) == val)
                if count != 1:
                    return False
            elif t == "attr_in_set":
                if tile.get(c["key"]) not in c["values"]:
                    return False
            elif t == "nested_attr_hits":
                hits = 0
                for k, vals in c["subset"].items():
                    if tile.get(k) in vals:
                        hits += 1
                if hits < c["threshold"]:
                    return False
        return True

    def _inject_distractors(self):
        # Slightly mutate some tiles to near-miss variants to increase difficulty, preserving solvability
        if self.distractor_density_pct <= 0:
            return
        num_distractors = max(0, int(self.num_tiles * self.distractor_density_pct / 100.0))
        indices = list(range(self.num_tiles))
        random.shuffle(indices)
        chosen = indices[:num_distractors]
        for idx in chosen:
            tile = self.tiles[idx]
            # pick an attribute used in clues or a random one
            keys = list(self.attributes_pool.keys())
            prefer = []
            for c in self.clues:
                if c["type"] in ("attr_eq", "attr_in_set"):
                    prefer.append(c["key"])
                elif c["type"] == "unique_attr":
                    prefer.append(c["key"])
                elif c["type"] == "nested_attr_hits":
                    prefer.extend(list(c["subset"].keys()))
            key_set = prefer if prefer else keys
            key = random.choice(key_set)
            # mutate to a different value
            candidates = [v for v in self.attributes_pool[key] if v != tile[key]]
            if candidates:
                tile[key] = random.choice(candidates)

    def _choose_question_mode(self):
        # Two modes: count tiles meeting clues OR among those tiles count occurrences of a specific attribute value
        if self.num_attribute_types >= 4 and self.num_tiles >= 12:
            self.question_mode = random.choice(["count_tiles", "count_attribute"])
        else:
            self.question_mode = "count_tiles"

    def _compute_target(self):
        selected = [tile for tile in self.tiles if self._apply_clues_to_tile(tile)]
        if self.question_mode == "count_tiles":
            self.target_count = len(selected)
            self.attribute_to_count = None
        else:
            # choose an attribute-value pair; count its occurrences within selected tiles
            key = random.choice(list(self.attributes_pool.keys()))
            val = random.choice(self.attributes_pool[key])
            self.attribute_to_count = {"key": key, "value": val}
            self.target_count = sum(1 for t in selected if t.get(key) == val)

    def _instance_feasible(self) -> bool:
        # Ensure nontrivial and solvable target; avoid degenerate 0 with impossible clues too often
        # Allow zero counts sometimes but keep a minimum chance of >0
        if self.target_count == 0:
            # 50% of instances with zero are accepted to maintain variety
            return random.random() < 0.5
        return True

    def _render_tiles_text(self) -> str:
        lines = []
        for i, t in enumerate(self.tiles, start=1):
            parts = [f"{k}={t[k]}" for k in sorted(t.keys())]
            lines.append(f"Tile {i}: " + ", ".join(parts))
        return "\n".join(lines)

    def _clue_to_text(self, c: Dict[str, Any]) -> str:
        t = c["type"]
        if t == "attr_eq":
            return f"Tiles with {c['key']} = {c['value']}"
        if t == "unique_attr":
            return f"Tiles whose {c['key']} value is unique among all tiles"
        if t == "attr_in_set":
            vs = ", ".join(c["values"])
            return f"Tiles with {c['key']} in {{{vs}}}"
        if t == "nested_attr_hits":
            subs = []
            for k, vals in c["subset"].items():
                subs.append(f"{k} in {{{', '.join(vals)}}}")
            return f"Tiles matching at least {c['threshold']} of: " + "; ".join(subs)
        return "Unknown clue"

    def _get_instructions(self) -> str:
        mode = "COUNT how many tiles satisfy ALL the clues."
        if self.question_mode == "count_attribute" and self.attribute_to_count:
            mode = (
                "First select all tiles that satisfy ALL the clues, "
                f"then COUNT how many of those tiles have {self.attribute_to_count['key']} = {self.attribute_to_count['value']}."
            )
        rules = [
            "- Read all tiles and their attributes.",
            "- Apply every clue; selection is the intersection of all clues.",
            "- No guessing tools; you may request a hint using \\boxed{hint}.",
            "- Submit your numeric answer using \\boxed{answer N} where N is a nonnegative integer.",
        ]
        tiles_text = self._render_tiles_text()
        clues_text = "\n".join([f"- {self._clue_to_text(c)}" for c in self.clues])
        return (
            "Glyph Conclave — Counting Puzzle\n"
            f"Objective: {mode}\n"
            "Clues:\n"
            f"{clues_text}\n\n"
            "Tiles:\n"
            f"{tiles_text}\n\n"
            "Rules:\n"
            f"{chr(10).join(rules)}\n"
            "Actions:\n"
            "- Submit answer: \\boxed{answer N}\n"
            "- Ask for hint: \\boxed{hint}\n"
            f"Hints remaining: {max(0, self.hint_budget - self.hints_used)}\n"
        )

    def get_task_suffix(self) -> str:
        return (
            f"Turns: {self.turn_count}/{self.max_turns} | "
            f"Hints left: {max(0, self.hint_budget - self.hints_used)}\n"
            "Enter your action in \\boxed{...} format. Examples: \\boxed{answer 7}, \\boxed{hint}"
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0
        self.hints_used = 0
        self.solved = False

        # Build instance with feasibility loop
        for _ in range(20):
            self._build_attribute_pool()
            self._sample_tiles()
            self._make_clues()
            self._inject_distractors()
            self._choose_question_mode()
            self._compute_target()
            if self._instance_feasible():
                break

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def _give_hint(self) -> str:
        if self.hints_used >= self.hint_budget:
            return "No hints remaining."
        self.hints_used += 1
        # Hint hierarchy
        if self.hints_used == 1:
            return "Hint: Start by filtering tiles using the simplest equality clue."
        if self.hints_used == 2:
            # Provide a partial selection size after first equality clue
            base_eq = next((c for c in self.clues if c["type"] == "attr_eq"), None)
            if base_eq:
                count = sum(1 for t in self.tiles if t.get(base_eq["key"]) == base_eq["value"])
                return f"Hint: Number of tiles with {base_eq['key']} = {base_eq['value']} is {count}."
            return "Hint: Tackle clues one by one; intersect results."
        # Further hints reveal structure
        if self.question_mode == "count_attribute" and self.attribute_to_count:
            return (
                f"Hint: After filtering, remember to count tiles with "
                f"{self.attribute_to_count['key']} = {self.attribute_to_count['value']}."
            )
        return "Hint: Some clues require uniqueness or meeting at least T of listed attribute subsets."

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1

        parsed = self._parse_action(action)
        if not parsed:
            obs = "INVALID ACTION FORMAT: Use \\boxed{answer N} or \\boxed{hint}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        act = parsed.get("action", "")
        if act not in ("answer", "hint"):
            obs = f"UNSUPPORTED ACTION: {act}. Use 'answer' or 'hint'."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if act == "hint":
            text = self._give_hint()
            if self.turn_count >= self.max_turns:
                obs = f"{text}\nTIMEOUT: Reached max turns."
                return obs, 0.0, True, True, {"suffix": self.get_task_suffix()}
            obs = f"{text}\nContinue."
            return obs, 0.0, False, False, {"suffix": self.get_task_suffix()}

        # act == "answer"
        val = parsed.get("value", None)
        if val is None or not re.fullmatch(r"\d+", val):
            obs = "PROTOCOL VIOLATION: Answer must be a nonnegative integer like \\boxed{answer 4}."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        guess = int(val)
        correct = (guess == self.target_count)
        if correct:
            self.solved = True
            obs = f"Success! Correct count is {self.target_count}."
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}
        else:
            obs = f"Incorrect. Your answer {guess} does not match the correct count. Correct was {self.target_count}."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if action is None:
            return None
        m = re.search(r"\\boxed\{(.+?)\}\s*$", action.strip(), flags=re.DOTALL)
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        parts = inner.split()
        if not parts:
            return None
        if parts[0].lower() == "hint":
            return {"action": "hint"}
        if parts[0].lower() == "answer":
            value = parts[1] if len(parts) > 1 else None
            return {"action": "answer", "value": value}
        return {"action": parts[0].lower()}

    def sample_random_action(self) -> str:
        # Biased to provide an answer within a reasonable range
        guess = random.randint(0, max(3, self.num_tiles))
        return rf"\boxed{{answer {guess}}}"


class GlyphConclaveEnvWithFeedback(GlyphConclaveEnv):
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
            error_detail["issue"] = "missing_or_malformed_boxed"
            hint = "Wrap your action like \\boxed{answer N} or \\boxed{hint}."
        elif "unsupported action" in text:
            error_type = "UnsupportedAction"
            m = re.search(r"unsupported action:\s*([a-z0-9_]+)", text)
            error_detail["action"] = m.group(1) if m else "unknown"
            hint = "Use only 'answer' or 'hint'."
        elif "protocol violation" in text:
            error_type = "ProtocolViolation"
            error_detail["issue"] = "answer_not_integer"
            hint = "Submit a nonnegative integer, e.g., \\boxed{answer 6}."
        elif "incorrect. your answer" in text:
            error_type = "WrongDecision"
            got = None
            exp = None
            mg = re.search(r"your answer\s+(\d+)", text)
            me = re.search(r"correct was\s+(\d+)", text)
            if mg:
                got = int(mg.group(1))
            if me:
                exp = int(me.group(1))
            error_detail["got"] = got
            error_detail["expected"] = exp
            # Provide structured hint based on mode
            if self.feedback_level >= 2:
                if self.question_mode == "count_attribute" and self.attribute_to_count:
                    hint = (
                        f"First intersect all clues, then count tiles with "
                        f"{self.attribute_to_count['key']} = {self.attribute_to_count['value']}."
                    )
                else:
                    hint = "Intersect all clues sequentially; check uniqueness and threshold clues carefully."
        elif "timeout" in text or (truncated and terminated):
            error_type = "Timeout"
            error_detail["turns"] = self.turn_count
            hint = "Plan: filter by equality clues, then apply advanced clues, then answer."
        elif "success!" in text:
            error_type = "OK"
            error_detail["outcome"] = "success"

        diagnostic = {"error_type": error_type}
        if self.feedback_level >= 1:
            diagnostic["error_detail"] = error_detail
            diagnostic["turn"] = getattr(self, "turn_count", None)
            diagnostic["state"] = {
                "hints_left": max(0, self.hint_budget - self.hints_used),
                "num_tiles": getattr(self, "num_tiles", None),
                "num_clues": getattr(self, "num_clues", None),
                "question_mode": getattr(self, "question_mode", None),
            }
        if self.feedback_level >= 2:
            diagnostic["hint"] = hint

        info["diagnostic"] = diagnostic
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        obs, info = super().reset(seed)
        base_hint = "Start by applying any equality clue to reduce the set. Use \\boxed{hint} if stuck."
        info["diagnostic"] = {
            "error_type": "OK",
            "error_detail": {"outcome": "episode_start"},
            "hint": base_hint if self.feedback_level >= 2 else None,
            "turn": 0,
            "state": {
                "hints_left": max(0, self.hint_budget - self.hints_used),
                "num_tiles": self.num_tiles,
                "num_clues": self.num_clues,
                "question_mode": self.question_mode,
            } if self.feedback_level >= 1 else None,
        }
        return obs, info