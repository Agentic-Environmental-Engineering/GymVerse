from gem.core import Env
from gem.utils.constants import LanguageGameReward
import random
import re
from typing import Tuple, Dict, Any, Optional, List


class ArenaStrikeEnv(Env):
    def __init__(
        self,
        complexity: int = 1,
        enable_param_randomization: bool = True,
        max_turns: Optional[int] = 1,
        **_,
    ):
        super().__init__()
        self.complexity = max(1, min(10, complexity))
        self.enable_param_randomization = enable_param_randomization
        self.max_turns = max_turns if max_turns is not None else 1

        # Evolvable parameters
        self.complexity_params = {
            'num_moves': (4, 10),                 # More moves = more options to evaluate = harder
            'num_damage_types': (3, 7),           # More types = more complex resistance mapping = harder
            'num_immunities': (0, 2),             # More immunities = more traps = harder
            'num_vulnerabilities': (3, 1),        # REVERSED: fewer vulnerabilities = harder
            'break_threshold': (30, 80),          # Higher required effective damage = harder
            'stamina_budget': (60, 25),           # REVERSED: less stamina = harder
            'hint_detail_level': (3, 1),          # REVERSED: less numeric detail = harder
        }

        # Variance for each evolvable parameter
        self.param_variance = {
            'num_moves': 1,
            'num_damage_types': 1,
            'num_immunities': 0,
            'num_vulnerabilities': 0,
            'break_threshold': 5,
            'stamina_budget': 4,
            'hint_detail_level': 0,
        }

        # Placeholders
        self.num_moves: int = 0
        self.num_damage_types: int = 0
        self.num_immunities: int = 0
        self.num_vulnerabilities: int = 0
        self.break_threshold: int = 0
        self.stamina_budget: int = 0
        self.hint_detail_level: int = 0

        # Domain-specific state
        self.turn_count: int = 0
        self.enemy_name: str = ""
        self.available_types: List[str] = []
        self.resistances: Dict[str, float] = {}
        self.immune_types: List[str] = []
        self.vulnerable_types: List[str] = []
        self.moves: List[Dict[str, Any]] = []
        self.valid_moves: List[str] = []
        self.index_to_code: Dict[int, str] = {}

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

    def _get_instructions(self) -> str:
        example = self.sample_random_action()
        lines = []
        lines.append("You are in ArenaStrike: a single-turn tactical battle.")
        lines.append("Your goal: choose one attack move that breaks the enemy's guard this turn.")
        lines.append("")
        lines.append("Rules:")
        lines.append("- Submit exactly one available move by its code.")
        lines.append(f"- Your move must fit within your Stamina budget and not target an immune type.")
        lines.append("- Effective damage is calculated as: base_damage * (1 - resistance).")
        lines.append(f"- To break the guard, effective damage must be at least the Break Threshold.")
        lines.append("")
        lines.append("Format your action as \\boxed{move_code}.")
        lines.append(f"For example: {example}")
        lines.append("")
        return "\n".join(lines)

    def get_task_suffix(self) -> str:
        parts = []
        parts.append(f"Enemy: {self.enemy_name}")
        parts.append(f"Stamina: {self.stamina_budget}")
        parts.append(f"Break Threshold: {self.break_threshold}")
        # Resistance hint detail
        if self.hint_detail_level >= 3:
            detail = ", ".join([f"{t}:{int(round(r*100))}%" for t, r in self.resistances.items()])
            parts.append(f"Resistances: {detail}")
            if self.immune_types:
                parts.append(f"Immunities: {', '.join(self.immune_types)}")
            if self.vulnerable_types:
                parts.append(f"Vulnerabilities: {', '.join(self.vulnerable_types)}")
        elif self.hint_detail_level == 2:
            rough_res = ", ".join(sorted(self.immune_types)) if self.immune_types else "None"
            parts.append(f"Immunities: {rough_res}")
            rough_vuln = ", ".join(sorted(self.vulnerable_types)) if self.vulnerable_types else "None"
            parts.append(f"Vulnerable to: {rough_vuln}")
            parts.append("Other types have moderate resistance.")
        else:
            hint_text = []
            if self.immune_types:
                hint_text.append(f"Strongly protected against {', '.join(self.immune_types)}")
            if self.vulnerable_types:
                hint_text.append(f"Seems weak to {', '.join(self.vulnerable_types)}")
            if not hint_text:
                hint_text.append("Defense mix is varied; choose tactically.")
            parts.append("Enemy defense hint: " + "; ".join(hint_text))
        parts.append("Available Moves:")
        for i, m in enumerate(self.moves, start=1):
            if self.hint_detail_level >= 2:
                parts.append(f"{i}. {m['code']}  ({m['name']})  type={m['type']}, dmg={m['damage']}, cost={m['cost']}")
            else:
                parts.append(f"{i}. {m['code']}  ({m['name']})  type={m['type']}, cost={m['cost']}")
        parts.append("Submit your chosen move code in \\boxed{...}.")
        return "\n".join(parts)

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        if seed is not None:
            random.seed(seed)

        self._apply_complexity_params()
        self.turn_count = 0

        enemy_pool = [
            "Stone Golem", "Frost Wyrm", "Ember Sentinel", "Shadow Knight",
            "Storm Titan", "Venom Basilisk", "Sanctum Warden"
        ]
        self.enemy_name = random.choice(enemy_pool)

        all_types = ["slash", "pierce", "blunt", "fire", "ice", "shock", "poison", "holy", "dark"]
        random.shuffle(all_types)
        self.available_types = sorted(all_types[:self.num_damage_types])

        # Base resistances
        self.resistances = {}
        for t in self.available_types:
            base_r = random.uniform(0.1, 0.4)  # 10%-40% resistance typical
            self.resistances[t] = base_r

        # Immunities
        self.immune_types = []
        if self.num_immunities > 0:
            immune_candidates = self.available_types[:]
            random.shuffle(immune_candidates)
            self.immune_types = sorted(immune_candidates[:self.num_immunities])
            for t in self.immune_types:
                self.resistances[t] = 1.0

        # Vulnerabilities (negative resistance)
        self.vulnerable_types = []
        vuln_candidates = [t for t in self.available_types if t not in self.immune_types]
        random.shuffle(vuln_candidates)
        self.vulnerable_types = sorted(vuln_candidates[:max(0, self.num_vulnerabilities)])
        for t in self.vulnerable_types:
            self.resistances[t] = random.uniform(-0.6, -0.3)  # 30%-60% extra damage

        # Generate moves
        self.moves = []
        self.index_to_code = {}
        self.valid_moves = []
        move_name_pool = {
            "slash": ["Crescent Slash", "Twin Slash", "Arc Blade"],
            "pierce": ["Pierce Thrust", "Spear Lunge", "Needle Jab"],
            "blunt": ["Crushing Blow", "Hammerfall", "Skull Bash"],
            "fire": ["Fireball", "Flame Surge", "Inferno Burst"],
            "ice": ["Ice Spike", "Frost Spear", "Glacier Shard"],
            "shock": ["Lightning Arc", "Thunder Lance", "Volt Strike"],
            "poison": ["Toxic Dart", "Venom Bolt", "Corrosive Sting"],
            "holy": ["Radiant Smite", "Lumen Slash", "Blessed Strike"],
            "dark": ["Umbral Cut", "Shadow Rend", "Abyssal Blast"],
        }

        # Ensure a mix of costs/damages
        move_candidates = []
        for t in self.available_types:
            names = move_name_pool.get(t, [])
            copies = names[:] if names else [t.title() + " Strike"]
            while len(copies) < 3:
                copies.append(f"{t.title()} Technique {len(copies)+1}")
            for nm in copies:
                dmg = random.randint(24, 64)
                cost = random.randint(10, 28)
                code = re.sub(r'[^a-z0-9]+', '_', nm.lower()).strip('_')
                move_candidates.append({'code': code, 'name': nm, 'type': t, 'damage': dmg, 'cost': cost})

        random.shuffle(move_candidates)
        selected = move_candidates[:self.num_moves]

        # De-duplicate codes if necessary
        seen_codes = set()
        self.moves = []
        for m in selected:
            code = m['code']
            while code in seen_codes:
                code = f"{code}_{random.randint(1,9)}"
            seen_codes.add(code)
            m['code'] = code
            self.moves.append(m)

        # Index mapping
        for i, m in enumerate(self.moves, start=1):
            self.index_to_code[i] = m['code']

        # Feasibility adjustments: ensure at least one move affordable and able to break threshold
        affordable_moves = [m for m in self.moves if m['cost'] <= self.stamina_budget]
        if not affordable_moves:
            cheapest = min(self.moves, key=lambda x: x['cost'])
            self.stamina_budget = max(self.stamina_budget, cheapest['cost'])
            affordable_moves = [m for m in self.moves if m['cost'] <= self.stamina_budget]

        def effective_damage(m):
            r = self.resistances.get(m['type'], 0.0)
            if r >= 1.0:
                return 0.0
            return m['damage'] * (1.0 - r)

        allowed_nonimmune = [m for m in affordable_moves if self.resistances.get(m['type'], 0.0) < 1.0]
        max_eff = max([effective_damage(m) for m in allowed_nonimmune], default=0.0)
        if max_eff < self.break_threshold:
            if max_eff <= 0:
                # Make one vulnerability if none effective
                choice_types = [t for t in self.available_types if t not in self.immune_types]
                if choice_types:
                    t = random.choice(choice_types)
                    self.resistances[t] = random.uniform(-0.6, -0.4)
                    max_eff = max([effective_damage(m) for m in allowed_nonimmune], default=0.0)
            self.break_threshold = max(10, int(round(max_eff * random.uniform(0.75, 0.95))))  # ensure solvable

        # Precompute valid moves
        self.valid_moves = []
        for m in self.moves:
            r = self.resistances.get(m['type'], 0.0)
            eff = effective_damage(m)
            if m['cost'] <= self.stamina_budget and r < 1.0 and eff >= self.break_threshold:
                self.valid_moves.append(m['code'])

        return self._get_instructions(), {"suffix": self.get_task_suffix()}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self.turn_count += 1
        parsed = self._parse_action(action)
        if parsed is None:
            obs = f"At turn {self.turn_count}, invalid action format. Use \\boxed{{move_code}}."
            return obs, LanguageGameReward.format_error_reward, True, False, {"suffix": self.get_task_suffix()}

        chosen_code = parsed.get("code", "")
        chosen_move = None
        for m in self.moves:
            if m['code'] == chosen_code:
                chosen_move = m
                break

        if chosen_move is None:
            obs = f"At turn {self.turn_count}, unsupported move: '{chosen_code}'. Choose one listed move code."
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        if chosen_move['cost'] > self.stamina_budget:
            obs = (
                f"You used {chosen_move['code']} ({chosen_move['name']}). "
                f"Result: insufficient stamina (cost {chosen_move['cost']} > budget {self.stamina_budget})."
            )
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        r = self.resistances.get(chosen_move['type'], 0.0)
        if r >= 1.0:
            obs = (
                f"You used {chosen_move['code']} ({chosen_move['name']}). "
                f"Result: enemy is immune to {chosen_move['type']}."
            )
            return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

        eff = chosen_move['damage'] * (1.0 - r)
        if eff >= self.break_threshold:
            obs = (
                f"You used {chosen_move['code']} ({chosen_move['name']}). "
                f"Effective damage {int(round(eff))} meets threshold {self.break_threshold}. "
                f"Success! Guard broken."
            )
            return obs, 1.0, True, False, {"suffix": self.get_task_suffix()}

        obs = (
            f"You used {chosen_move['code']} ({chosen_move['name']}). "
            f"Effective damage {int(round(eff))} below threshold {self.break_threshold}. "
            f"Failed to break guard."
        )
        return obs, 0.0, True, False, {"suffix": self.get_task_suffix()}

    def _parse_action(self, action: str) -> Optional[Dict[str, Any]]:
        if not action or not isinstance(action, str):
            return None
        pattern = re.compile(r'\\boxed\{(.+?)\}', re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(action))
        if not matches:
            return None
        content = matches[-1].group(1).strip()
        # Accept either code or index number
        if content.isdigit():
            idx = int(content)
            code = self.index_to_code.get(idx, "")
            if not code:
                return {"code": ""}  # will be treated as unsupported
            return {"code": code}
        # Normalize code-like strings
        code = content.strip().lower()
        return {"code": code}

    def sample_random_action(self) -> str:
        if self.moves:
            m = random.choice(self.moves)
            return f"\\boxed{{{m['code']}}}"
        return "\\boxed{example_move_code}"


class ArenaStrikeEnvWithFeedback(ArenaStrikeEnv):
        def __init__(self, feedback_level: int = 2, **kwargs):
            self.feedback_level = feedback_level
            super().__init__(**kwargs)

        def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
            obs, reward, terminated, truncated, info = super().step(action)

            text = obs.lower()
            error_type = "OK"
            error_detail: Dict[str, Any] = {}
            hint = None

            if "invalid action format" in text or "use \\boxed" in text:
                error_type = "FormatError"
                error_detail["issue"] = "missing_boxed_format"
                hint = "Submit exactly one listed move as \\boxed{move_code}."

            elif "unsupported move" in text:
                error_type = "UnsupportedAction"
                error_detail["issue"] = "unknown_move_code"
                hint = "Pick a code from the Available Moves list in the state suffix."

            elif "insufficient stamina" in text:
                error_type = "ProtocolViolation"
                error_detail["violation"] = "budget_exceeded"
                error_detail["stamina_budget"] = getattr(self, "stamina_budget", None)
                hint = "Choose a move with cost <= your Stamina budget."

            elif "immune to" in text:
                error_type = "WrongDecision"
                # Try to extract type
                m = re.search(r"immune to ([a-z_]+)", text)
                if m:
                    error_detail["wrong_type"] = m.group(1)
                error_detail["immune_types"] = getattr(self, "immune_types", [])
                vuln = getattr(self, "vulnerable_types", [])
                if vuln:
                    hint = f"Avoid immune types. Prefer a type the enemy is weak to: {', '.join(vuln)}."
                else:
                    hint = "Avoid immune types and pick a non-immune type with higher damage."

            elif "below threshold" in text:
                error_type = "WrongDecision"
                error_detail["reason"] = "insufficient_effective_damage"
                error_detail["threshold"] = getattr(self, "break_threshold", None)
                vuln = getattr(self, "vulnerable_types", [])
                if vuln:
                    hint = f"Select a vulnerable type and ensure cost <= Stamina."
                else:
                    hint = "Choose a move with higher base damage or lower resistance."

            elif "success" in text or "guard broken" in text:
                error_type = "OK"
                error_detail["outcome"] = "success"
                hint = None

            if truncated:
                error_type = "Timeout"
                error_detail["outcome"] = "max_turns_reached"
                hint = "Act within the allowed turn limit."

            diagnostic = {"error_type": error_type}
            if self.feedback_level >= 1:
                diagnostic["error_detail"] = error_detail
                diagnostic["turn"] = getattr(self, "turn_count", None)
                diagnostic["enemy"] = getattr(self, "enemy_name", None)
                diagnostic["stamina_budget"] = getattr(self, "stamina_budget", None)
                diagnostic["break_threshold"] = getattr(self, "break_threshold", None)
                diagnostic["immune_types"] = getattr(self, "immune_types", [])
                diagnostic["vulnerable_types"] = getattr(self, "vulnerable_types", [])
            if self.feedback_level >= 2:
                diagnostic["hint"] = hint

            info["diagnostic"] = diagnostic
            return obs, reward, terminated, truncated, info

        def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
            obs, info = super().reset(seed)
            initial_hint = "Pick a move that matches a vulnerable type and fits your stamina budget."
            info["diagnostic"] = {
                "error_type": "OK",
                "error_detail": {"outcome": "episode_start"},
                "hint": initial_hint if self.feedback_level >= 2 else None,
                "turn": 0,
                "enemy": getattr(self, "enemy_name", None),
                "stamina_budget": getattr(self, "stamina_budget", None),
                "break_threshold": getattr(self, "break_threshold", None),
                "immune_types": getattr(self, "immune_types", []),
                "vulnerable_types": getattr(self, "vulnerable_types", []),
            }
            return obs, info